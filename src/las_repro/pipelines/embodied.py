"""Safe prompts and durable 0805 embodied inference orchestration."""

from __future__ import annotations

import json
import hashlib
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from importlib import resources
from numbers import Real
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..cv.entities import NormalizedEntities, normalize_entities
from ..cv.artifacts import CvArtifactStore, CvArtifactHandle, CvArtifactError, cv_cache_key
from ..cv.contracts import CvEvidenceRequest, SamplingPolicy, EvidenceThresholds
from ..cv.timeline import probe_frame_timeline, TimelineError
from ..cv.summary import CvEvidenceSummary, OcclusionCandidate, summarize_cv_evidence, build_cv_prompt_bundle, validate_candidate_identity_evidence
from ..domain import InferenceJob, InferenceJobSpec, TaskRecord
from ..media import TimeSpan, VideoMetadata, probe_video
from ..models.base import VideoSession
from ..pipelines.base import PipelineContext, SafePipelineError
from ..store import SQLiteTaskStore
from ..workers import InferenceJobFailed, JobWaitTimeout, wait_for_jobs
from .output_validation import DEFAULT_OUTPUT_SCHEMAS, NormalizedSchemaOutput
from .occlusion import OcclusionDecisionSet, project_occlusion_events
from .hybrid_result import build_hybrid_result, validate_hybrid_result, build_performance
from .scene_choices import (
    SceneInputPackage, SceneLocationChoice, SceneRelationChoice,
    prepare_scene_choices, authenticate_scene_context, compact_scene_options, project_scene_choices,
)
from .scene_semantics import (
    SceneSemantics,
    trusted_target_skeleton,
    unavailable_scene_semantics,
)
from .validators import (
    BoundaryPlan,
    CoarsePlan,
    EnrichmentResult,
    ObjectInventory,
    TemporalIssue,
    TemporalValidationError,
    validate_coarse_plan,
)


EMBODIED_PROMPT_VERSION = "0805-local-v9"

Probe = Callable[[Path], VideoMetadata]
WaitJobs = Callable[
    [SQLiteTaskStore, str, Sequence[str], float],
    list[dict[str, Any]],
]

_PROMPT_PACKAGE = "las_repro.prompts"
_PROMPT_FILES = {
    "active_objects": "active_objects.txt",
    "embodied_pass_a": "embodied_pass_a.txt",
    "embodied_pass_b": "embodied_pass_b.txt",
    "embodied_enrichment": "embodied_enrichment.txt",
    "scene_semantics": "scene_semantics.txt",
    "occlusion_semantics": "occlusion_semantics.txt",
}
_MARKER = re.compile(r"\{\{(?P<name>[A-Z][A-Z0-9_]*)\}\}")
_MAX_BOUNDARY_SLOTS_PER_ACTION = 10_000
_MAX_ENRICHMENT_RECORDS = _MAX_BOUNDARY_SLOTS_PER_ACTION
_ENRICHMENT_WARNING_FIELD_BY_CODE = (
    ("ENRICHMENT_RESULT_ACTOR_ENUM_VALUE", "actor"),
    ("ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE", "actor_state"),
    ("ENRICHMENT_RESULT_SKILL_ENUM_VALUE", "skill"),
    (
        "ENRICHMENT_RESULT_VISUAL_MOTION_STATE_ENUM_VALUE",
        "visual_motion_state",
    ),
)


class PromptRenderError(ValueError):
    """A prompt asset or its structured variable set is invalid."""


def _with_cv_summary(prompt: str, summary: CvEvidenceSummary | None) -> str:
    if summary is None:
        return prompt
    if type(summary) is not CvEvidenceSummary or summary.status != "available":
        raise PromptRenderError("CV evidence must be an available bounded summary")
    return prompt + "\n\n[CV_EVIDENCE_SUMMARY_JSON]\n" + _canonical_json(summary.prompt_record())


@dataclass(frozen=True)
class _TrustedCanonicalJSON:
    """Internally generated JSON whose exact numeric lexemes must be preserved."""

    text: str


class ActiveObjectPipelineError(SafePipelineError):
    """A stable active-object stage failure safe for persistence."""


class EmbodiedActionPipelineError(SafePipelineError):
    """A stable execution failure in the 0805 action pipeline."""

    def __init__(
        self,
        message: str,
        *,
        repair_history: tuple[str, ...] = ("initial",),
    ) -> None:
        self.repair_history = repair_history
        super().__init__(message)


@dataclass(frozen=True)
class FineSegmentTableRow:
    """A locally-owned fine segment that enrichment cannot mutate."""

    action_index: int
    segment_index: int
    start: float
    end: float
    description: str
    event_type: str
    start_boundary_id: str
    end_boundary_id: str

    def prompt_record(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start": self.start,
            "end": self.end,
            "description": self.description,
        }

    def public_record(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "segment_index": self.segment_index,
            "start": self.start,
            "end": self.end,
            "description": self.description,
            "event_type": self.event_type,
            "start_boundary_id": self.start_boundary_id,
            "end_boundary_id": self.end_boundary_id,
        }


class EmbodiedActionPipeline:
    """Coordinate the 0805 coarse, boundary, and enrichment stages."""

    def __init__(
        self,
        *,
        renderer: PromptRenderer | None = None,
        probe: Probe = probe_video,
        wait_jobs: WaitJobs = wait_for_jobs,
        wait_timeout: float = 300.0,
    ) -> None:
        self._renderer = renderer or PromptRenderer()
        self._probe = probe
        self._wait_jobs = wait_jobs
        self._wait_timeout = _positive_finite(wait_timeout, "wait_timeout")

    def run(self, task: TaskRecord, context: PipelineContext) -> dict[str, Any]:
        started = time.monotonic()
        media_path = _action_media_path(context)
        metadata = self._probe(media_path)
        media_seconds = time.monotonic() - started
        span = TimeSpan(
            0.0,
            _action_positive_finite(metadata.duration, "duration"),
        )
        fps = _action_sampling_fps(task.payload)
        prompt_context = _action_prompt_context(task.payload)

        coarse_data, pass_a_job, _ = self._run_validated_stage(
            task,
            context,
            media_path,
            span,
            fps,
            stage="embodied_pass_a",
            schema_name="CoarsePlan",
            schema_context={"duration": span.end},
            render_prompt=lambda repair: self._renderer.pass_a(
                prompt_context,
                video_duration=span.end,
                repair=repair,
            ),
            affinity_anchor=None,
            metadata=metadata,
        )
        coarse = CoarsePlan.model_validate(coarse_data)
        normalized_entities = normalize_entities(
            coarse.entity_candidates,
            limit=context.settings.cv_entity_limit,
        )
        cv_started = time.monotonic()
        cv_evidence, bundle, timeline, artifact, cv_decode_seconds = self._run_cv_evidence(
            task, context, media_path, span.end, normalized_entities)
        cv_seconds = time.monotonic() - cv_started
        media_seconds += cv_decode_seconds
        summary = bundle.summary if bundle is not None else None
        warnings: list[dict[str, Any]] = []
        if cv_evidence["status"] == "unavailable":
            warnings.extend([{"code": "CV_EVIDENCE_UNAVAILABLE"},
                             {"code": "OCCLUSION_UNAVAILABLE"}])
        for warning in normalized_entities.warnings:
            if warning == "ENTITY_ALIASES_TRUNCATED":
                warnings.append(
                    {
                        "code": warning,
                        "omitted_count": normalized_entities.alias_omitted_count,
                    }
                )
            else:
                warnings.append(
                    {
                        "code": "CV_ENTITY_LIMIT_APPLIED",
                        "omitted_count": normalized_entities.omitted_count,
                        "limit": context.settings.cv_entity_limit,
                        "message": warning,
                    }
                )
        max_fine_segment_seconds = _action_positive_finite(
            context.settings.max_fine_segment_seconds,
            "max_fine_segment_seconds",
        )

        # Feasibility windows guide generation but are intentionally absent
        # from schema_context. The durable output contract remains the strict
        # parent/adjacency/reference/hard-cap validator; local code must not
        # rewrite an otherwise valid evidence-backed timestamp to fit a prompt
        # planning envelope.
        boundary_data, _, boundary_normalization = self._run_validated_stage(
            task,
            context,
            media_path,
            span,
            fps,
            stage="embodied_pass_b",
            schema_name="BoundaryPlan",
            schema_context={
                "coarse_plan": coarse.model_dump(mode="json"),
                "max_segment_seconds": max_fine_segment_seconds,
            },
            render_prompt=lambda repair: self._renderer.pass_b(
                coarse,
                max_fine_segment_seconds=max_fine_segment_seconds,
                repair=repair,
            ),
            affinity_anchor=pass_a_job,
            metadata=metadata,
        )
        boundary = BoundaryPlan.model_validate(boundary_data)
        _guard_enrichment_record_count(boundary)
        segment_table = _fine_segment_table(boundary)
        expected_indices = [row.segment_index for row in segment_table]

        enrichment_data, enrichment_job, enrichment_normalization = self._run_validated_stage(
            task,
            context,
            media_path,
            span,
            fps,
            stage="embodied_enrichment",
            schema_name="EnrichmentResult",
            schema_context={"expected_indices": expected_indices},
            render_prompt=lambda repair: self._renderer.enrichment(
                [row.prompt_record() for row in segment_table],
                expected_indices=expected_indices,
                evidence_summary=summary,
                repair=repair,
            ),
            affinity_anchor=pass_a_job,
            metadata=metadata,
        )
        enrichment = EnrichmentResult.model_validate(enrichment_data)
        if boundary_normalization is not None:
            warnings.append(_boundary_normalization_warning(boundary_normalization))
        if enrichment_normalization is not None:
            warnings.append(_enrichment_normalization_warning(enrichment_normalization))
        segments = _merge_enrichment(segment_table, enrichment)
        scene_input = prepare_scene_choices(summary, segments, duration=span.end)
        scene_status = "available"
        scene_history = ("initial",)
        try:
            scene_data, scene_job, scene_normalization = self._run_validated_stage(
                task,
                context,
                media_path,
                span,
                fps,
                stage="scene_semantics",
                schema_name="SceneSemanticsChoices",
                schema_context=scene_input.context(),
                render_prompt=lambda repair: self._renderer.scene_semantics(
                    segments,
                    video_duration=span.end,
                    evidence_summary=summary,
                    scene_input=scene_input,
                    repair=repair,
                ),
                affinity_anchor=pass_a_job,
                metadata=metadata,
                max_attempts=3,
            )
            scene_history = _repair_history_for(scene_job.ordinal)
            if scene_normalization is not None:
                warnings.append(_scene_normalization_warning(scene_normalization))
            scene_data = project_scene_choices(scene_data, scene_input.context(),
                                               repair_history=scene_history)
        except (TemporalValidationError, EmbodiedActionPipelineError):
            scene_data = unavailable_scene_semantics()
            scene_status = "unavailable"
            warnings.append({"code": "SCENE_SEMANTICS_UNAVAILABLE"})
        scene = SceneSemantics.model_validate(scene_data)
        occlusion_started = time.monotonic()
        occlusion = {"status": cv_evidence["status"], "decisions": [], "events": []}
        if bundle is not None and bundle.candidates:
            decisions, history, occlusion_normalization = self.adjudicate_occlusions(
                task, context, media_path, span, fps, candidates=bundle.candidates,
                normalized_entities=normalized_entities, evidence_summary=summary,
                frame_pts=[frame.timestamp_seconds for frame in timeline.frames],
                affinity_anchor=pass_a_job, metadata=metadata)
            if occlusion_normalization is not None:
                warnings.append(
                    _occlusion_normalization_warning(occlusion_normalization)
                )
            if len(decisions.decisions) != len(bundle.candidates):
                occlusion["status"] = "unavailable"
                warnings.append({"code": "OCCLUSION_UNAVAILABLE"})
            else:
                occlusion["decisions"] = decisions.model_dump(mode="json")["decisions"]
                occlusion["events"] = project_occlusion_events(
                    decisions, bundle.candidates, artifact.tracks, segments, repair_history=history)
        merge_started = time.monotonic()
        occlusion_seconds = merge_started - occlusion_started
        jobs = context.store.list_inference_jobs(task.task_id)
        performance = build_performance(jobs, media_seconds=media_seconds,
            cv_seconds=cv_seconds, occlusion_seconds=occlusion_seconds,
            merge_seconds=0.0, total_seconds=time.monotonic() - started,
            degradation_count=sum(s == "unavailable" for s in
                                  (cv_evidence["status"], scene_status, occlusion["status"])))
        result = build_hybrid_result(task_description=coarse.task_description,
            duration=span.end,
            segments=segments, scene=scene.model_dump(mode="json"), scene_status=scene_status,
            cv_evidence=cv_evidence, evidence_summary=summary, warnings=warnings,
            performance=performance, occlusion=occlusion,
            action_history=_repair_history_for(enrichment_job.ordinal),
            scene_history=scene_history)
        validate_hybrid_result(result, evidence_summary=summary,
            frame_pts=[frame.timestamp_seconds for frame in timeline.frames] if summary is not None else None,
            occlusion_candidates=bundle.candidates if bundle is not None else None)
        result["performance"]["stages"][-1]["elapsed_seconds"] = time.monotonic() - merge_started
        result["performance"]["total_seconds"] = time.monotonic() - started
        return result

    def _run_cv_evidence(self, task, context, media_path, duration, entities):
        settings = context.settings
        if settings.cv_provider == "disabled":
            return {"status": "disabled"}, None, None, None, 0.0
        decode_started = time.monotonic()
        decode_seconds = None
        try:
            timeline = probe_frame_timeline(media_path)
            with media_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            decode_seconds = time.monotonic() - decode_started
            request = CvEvidenceRequest(schema_version="cv_request_v1",
                provider=settings.cv_provider, model_identity=settings.cv_model_alias,
                video_path=media_path, video_sha256=digest, duration_seconds=duration,
                frame_count=len(timeline.frames), checkpoint_sha256=settings.cv_checkpoint_sha256,
                timeline=timeline, entities=entities.entities,
                sampling=SamplingPolicy(short_video_seconds=settings.cv_short_video_seconds,
                    scan_fps=settings.cv_scan_fps, max_fps=settings.cv_max_fps,
                    refinement_radius_seconds=settings.cv_refinement_radius_seconds),
                thresholds=EvidenceThresholds(min_confidence=settings.cv_min_confidence,
                    min_area_fraction=settings.cv_min_area_fraction,
                    occlusion_visibility_drop=settings.cv_occlusion_visibility_drop))
            [job] = context.store.create_inference_jobs(task.task_id, [InferenceJobSpec(
                stage="cv_evidence", ordinal=0, payload=request.model_dump(mode="json"),
                model_name=settings.cv_model_alias)])
            [result] = self._wait_jobs(context.store, task.task_id, [job.job_id], settings.cv_timeout_seconds)
            if (type(result) is not dict or set(result) != {"status", "artifact_key", "manifest_sha256", "cache_hit"}
                    or result["status"] != "available" or type(result["cache_hit"]) is not bool
                    or result["artifact_key"] != cv_cache_key(request)):
                raise ValueError("invalid CV result")
            with CvArtifactStore(settings.cv_cache_root, max_files=settings.cv_cache_max_files,
                                 max_bytes=settings.cv_cache_max_bytes) as store:
                artifact = store.load(CvArtifactHandle(result["artifact_key"], result["manifest_sha256"]))
            if artifact.status != "available":
                raise ValueError("unavailable CV artifact")
            summary = summarize_cv_evidence(
                artifact,
                timeline=timeline,
                thresholds=request.thresholds,
            )
            bundle = build_cv_prompt_bundle(summary, request.thresholds)
            return result, bundle, timeline, artifact, decode_seconds
        except (InferenceJobFailed, JobWaitTimeout, CvArtifactError, TimelineError,
                OSError, ValueError, TypeError):
            return {"status": "unavailable"}, None, None, None, (
                decode_seconds if decode_seconds is not None else time.monotonic() - decode_started)

    def _run_validated_stage(
        self,
        task: TaskRecord,
        context: PipelineContext,
        media_path: Path,
        span: TimeSpan,
        fps: float,
        *,
        stage: str,
        schema_name: str,
        schema_context: Mapping[str, Any],
        render_prompt: Callable[[Mapping[str, Any] | None], str],
        affinity_anchor: InferenceJob | None,
        metadata: VideoMetadata,
        max_attempts: int = 2,
    ) -> tuple[dict[str, Any], InferenceJob, NormalizedSchemaOutput | None]:
        if max_attempts not in (2, 3):
            raise ValueError("max_attempts must be 2 or 3")
        repair: dict[str, Any] | None = None
        first_job: InferenceJob | None = None
        for ordinal in range(max_attempts):
            anchor = affinity_anchor if affinity_anchor is not None else first_job
            affinity_worker_id, affinity_fallback_seconds = _action_affinity(
                anchor,
                context,
                self._wait_timeout,
                stage,
            )
            prompt = render_prompt(repair)
            job_schema_context = dict(schema_context)
            if schema_name == "BoundaryPlan":
                job_schema_context["allow_topology_fallback"] = ordinal == max_attempts - 1
            if schema_name == "EnrichmentResult":
                job_schema_context["allow_enum_unknown_fallback"] = ordinal == max_attempts - 1
            if schema_name == "SceneSemanticsChoices":
                job_schema_context["allow_scene_normalization"] = True
            if schema_name == "OcclusionDecisionSet":
                job_schema_context["allow_occluder_unknown_fallback"] = True
            [job] = context.store.create_inference_jobs(
                task.task_id,
                [
                    InferenceJobSpec(
                        stage=stage,
                        ordinal=ordinal,
                        payload=_job_payload(
                            task,
                            media_path,
                            span,
                            fps,
                            prompt,
                            schema_name=schema_name,
                            schema_context=job_schema_context,
                            metadata=metadata,
                        ),
                        affinity_worker_id=affinity_worker_id,
                        affinity_fallback_seconds=affinity_fallback_seconds,
                    )
                ],
            )
            try:
                [result] = self._wait_jobs(
                    context.store,
                    task.task_id,
                    [job.job_id],
                    self._wait_timeout,
                )
            except InferenceJobFailed:
                raise EmbodiedActionPipelineError(
                    f"{_stage_label(stage)} inference failed",
                    repair_history=_repair_history_for(ordinal),
                ) from None
            except JobWaitTimeout:
                raise EmbodiedActionPipelineError(
                    f"{_stage_label(stage)} inference timed out",
                    repair_history=_repair_history_for(ordinal),
                ) from None

            completed = context.store.get_inference_job(job.job_id)
            if completed is None or completed.completed_by is None:
                raise EmbodiedActionPipelineError(
                    f"{_stage_label(stage)} completion is invalid",
                    repair_history=_repair_history_for(ordinal),
                )
            if ordinal == 0:
                first_job = completed
            sanitized, issue_codes, normalization = _validated_stage_result(
                schema_name,
                result,
                job_schema_context,
            )
            if issue_codes is None:
                return sanitized, completed, normalization
            if ordinal == max_attempts - 1:
                raise _stage_validation_error(stage, issue_codes)
            repair = {"issue_codes": list(issue_codes)}

        raise AssertionError("embodied validation repair loop did not terminate")

    def adjudicate_occlusions(
        self,
        task: TaskRecord,
        context: PipelineContext,
        media_path: Path,
        span: TimeSpan,
        fps: float,
        *,
        candidates: Sequence[Any],
        normalized_entities: Any,
        evidence_summary: Mapping[str, Any] | BaseModel,
        frame_pts: Sequence[float],
        affinity_anchor: InferenceJob | None,
        metadata: VideoMetadata,
    ) -> tuple[OcclusionDecisionSet, tuple[str, ...], NormalizedSchemaOutput | None]:
        """Run the isolated occlusion branch, degrading it conservatively."""
        candidate_tuple = tuple(candidates)
        if not candidate_tuple:
            return OcclusionDecisionSet(decisions=()), ("initial",), None
        try:
            data, completed, normalization = self._run_validated_stage(
                task,
                context,
                media_path,
                span,
                fps,
                stage="occlusion_semantics",
                schema_name="OcclusionDecisionSet",
                schema_context={
                    "duration": span.end,
                    "evidence_summary": evidence_summary.model_dump(mode="json"),
                    "candidates": [
                        candidate.model_dump(mode="json")
                        for candidate in candidate_tuple
                    ],
                },
                render_prompt=lambda repair: self._renderer.occlusion_semantics(
                    candidate_tuple,
                    normalized_entities,
                    evidence_summary,
                    video_duration=span.end,
                    frame_pts=frame_pts,
                    repair=repair,
                ),
                affinity_anchor=affinity_anchor,
                metadata=metadata,
                max_attempts=3,
            )
        except TemporalValidationError:
            return OcclusionDecisionSet(decisions=()), ("initial", "repair", "repair"), None
        except EmbodiedActionPipelineError as error:
            return OcclusionDecisionSet(decisions=()), error.repair_history, None
        history = _repair_history_for(getattr(completed, "ordinal", 0))
        return OcclusionDecisionSet.model_validate(data), history, normalization


class PromptRenderer:
    """Load packaged prompt assets and inject values as canonical JSON data."""

    def render(self, name: str, variables: Mapping[str, Any]) -> str:
        """Render one known asset without interpreting its literal JSON braces."""
        if name not in _PROMPT_FILES:
            raise PromptRenderError("unknown embodied prompt asset")
        if not isinstance(variables, Mapping):
            raise PromptRenderError("prompt variables must be a mapping")
        if any(not isinstance(key, str) for key in variables):
            raise PromptRenderError("prompt variable names must be strings")

        try:
            template = (
                resources.files(_PROMPT_PACKAGE)
                .joinpath(_PROMPT_FILES[name])
                .read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ModuleNotFoundError, OSError) as error:
            raise PromptRenderError("embodied prompt asset is unavailable") from error

        markers = tuple(match.group("name") for match in _MARKER.finditer(template))
        expected = set(markers)
        provided = set(variables)
        if provided != expected:
            raise PromptRenderError("prompt variables do not match asset markers")

        encoded: dict[str, str] = {}
        try:
            for marker in expected:
                encoded[marker] = _canonical_json(variables[marker])
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise PromptRenderError("prompt variable is not finite JSON data") from error

        # Mask every structural marker before inserting any untrusted data. This
        # preserves the required ``str.replace`` rendering strategy without
        # allowing a marker-like string inside one value to trigger a later
        # variable substitution.
        rendered = template
        placeholders: dict[str, str] = {}
        for index, marker in enumerate(sorted(expected)):
            literal = "{{" + marker + "}}"
            occurrence_count = rendered.count(literal)
            placeholder = f"\x00LAS_EMBODIED_VALUE_{index}\x00"
            if occurrence_count == 0 or placeholder in rendered or any(
                placeholder in value for value in encoded.values()
            ):
                raise PromptRenderError("embodied prompt marker layout is invalid")
            rendered = rendered.replace(literal, placeholder)
            if rendered.count(placeholder) != occurrence_count:
                raise PromptRenderError("embodied prompt marker replacement failed")
            placeholders[marker] = placeholder
        if _MARKER.search(rendered) is not None:
            raise PromptRenderError("prompt asset contains an unconsumed marker")
        assert _MARKER.search(rendered) is None

        for marker in sorted(expected):
            placeholder = placeholders[marker]
            rendered = rendered.replace(placeholder, encoded[marker])
            if placeholder in rendered:
                raise PromptRenderError("prompt asset contains an unconsumed marker")
        return rendered

    def active_objects(
        self,
        prompt_context: str | None = None,
        *,
        repair: Mapping[str, Any] | None = None,
    ) -> str:
        """Render active-object instructions with context isolated as JSON data."""
        return self.render(
            "active_objects",
            {
                "NAMING_HINTS_JSON": _naming_hint_data(prompt_context),
                "VALIDATION_REPAIR_JSON": repair,
            },
        )

    def pass_a(
        self,
        prompt_context: str | None = None,
        *,
        video_duration: Any,
        repair: Mapping[str, Any] | None = None,
    ) -> str:
        """Render the latest coarse full-video prompt without executing it."""
        return self.render(
            "embodied_pass_a",
            {
                "NAMING_HINTS_JSON": _naming_hint_data(prompt_context),
                "VIDEO_DURATION_SECONDS_JSON": _prompt_video_duration(
                    video_duration
                ),
                "VALIDATION_REPAIR_JSON": repair,
            },
        )

    def pass_b(
        self,
        coarse_plan: CoarsePlan | Mapping[str, Any],
        *,
        max_fine_segment_seconds: Any,
        repair: Mapping[str, Any] | None = None,
    ) -> str:
        """Render boundary-first instructions with the Pass A plan as JSON data."""
        maximum = _prompt_positive_finite(
            max_fine_segment_seconds,
            "max_fine_segment_seconds",
        )
        plan = _validated_coarse_plan(coarse_plan)
        return self.render(
            "embodied_pass_b",
            {
                "COARSE_PLAN_JSON": plan.model_dump(mode="json"),
                "FINE_SEGMENT_REQUIREMENTS_JSON": _fine_segment_requirements(
                    plan,
                    maximum,
                ),
                "VALIDATION_REPAIR_JSON": repair,
            },
        )

    def enrichment(
        self,
        segments: Sequence[Mapping[str, Any] | BaseModel],
        *,
        expected_indices: Sequence[int],
        evidence_summary: CvEvidenceSummary | None = None,
        repair: Mapping[str, Any] | None = None,
    ) -> str:
        """Render six-field enrichment instructions for an immutable segment table."""
        if isinstance(segments, (str, bytes, bytearray)) or not isinstance(
            segments, Sequence
        ):
            raise PromptRenderError("segments must be a sequence of JSON records")
        if len(segments) > _MAX_ENRICHMENT_RECORDS:
            raise PromptRenderError("enrichment record skeleton is not materializable")
        table = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in segments
        ]
        if any(not isinstance(item, Mapping) for item in table):
            raise PromptRenderError("segments must be a sequence of JSON records")
        indices = _validated_enrichment_indices(expected_indices)
        table_indices = _validated_enrichment_table_indices(table)
        if table_indices != indices:
            raise PromptRenderError(
                "expected_indices must match the immutable segment table"
            )
        return _with_cv_summary(self.render(
            "embodied_enrichment",
            {
                "SEGMENTS_JSON": table,
                "ENRICHMENT_REQUIREMENTS_JSON": _enrichment_requirements(indices),
                "VALIDATION_REPAIR_JSON": repair,
            },
        ), evidence_summary)

    def scene_semantics(
        self,
        segments: Sequence[Mapping[str, Any] | BaseModel],
        *,
        video_duration: Any,
        evidence_summary: CvEvidenceSummary | None = None,
        repair: Mapping[str, Any] | None = None,
        scene_input: SceneInputPackage | None = None,
    ) -> str:
        """Render full-video scene facts with the validated segment table as data."""
        if isinstance(segments, (str, bytes, bytearray)) or not isinstance(
            segments, Sequence
        ):
            raise PromptRenderError("segments must be a sequence of JSON records")
        if len(segments) > _MAX_ENRICHMENT_RECORDS:
            raise PromptRenderError("scene segment table is not materializable")
        table = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in segments
        ]
        if any(not isinstance(item, Mapping) for item in table):
            raise PromptRenderError("segments must be a sequence of JSON records")
        try:
            package = scene_input or prepare_scene_choices(
                evidence_summary, table, duration=float(video_duration))
            trusted = authenticate_scene_context(package.context())
            if trusted["segments"] != table or trusted["duration"] != float(video_duration):
                raise ValueError("scene input does not match renderer arguments")
            if trusted["evidence_summary"] != (evidence_summary.model_dump(mode="json")
                                               if evidence_summary is not None else None):
                raise ValueError("scene input does not match renderer evidence")
            spatial_options = compact_scene_options(trusted)
            summary_json = (_canonical_json(evidence_summary.prompt_record())
                            if evidence_summary is not None else None)
        except ValueError as error:
            raise PromptRenderError(str(error)) from None
        from .scene_model_view import (
            build_scene_model_view, full_model_view, MAX_PROMPT_BYTES, MAX_REPAIR_BYTES,
        )
        if len(_canonical_json(repair).encode("utf-8")) > MAX_REPAIR_BYTES:
            raise PromptRenderError("scene repair data exceeds its byte limit")
        model_view = build_scene_model_view(trusted)

        def render_view(view, repair_data):
            data = view["data"]
            hints = {"metadata": view["metadata"]}
            if data is not None:
                hints.update({k: data[k] for k in (
                    "evidence", "lifecycle", "entities", "completeness")})
            rendered = self.render("scene_semantics", {
                "VIDEO_DURATION_SECONDS_JSON": _prompt_video_duration(video_duration),
                "SEGMENTS_JSON": data["segments"] if data is not None else table,
                "KNOWN_TARGETS_JSON": trusted["known_targets"],
                "CV_EVIDENCE_AVAILABILITY_JSON": {"available": evidence_summary is not None},
                "SCENE_SPATIAL_PROVENANCE_OPTIONS_JSON": spatial_options,
                "SCENE_SPATIAL_FIELDS_JSON": {
                    "location_fields": list(SceneLocationChoice.model_fields),
                    "relation_fields": list(SceneRelationChoice.model_fields),
                },
                "SCENE_MODEL_VIEW_JSON": hints,
                "VALIDATION_REPAIR_JSON": repair_data,
            })
            if data is None and summary_json is not None:
                rendered += "\n\n[CV_EVIDENCE_SUMMARY_JSON]\n" + summary_json
            return rendered

        # Decide from immutable data plus a fixed repair allowance. Never select
        # another mode based on the current attempt's validator issue codes.
        initial = render_view(model_view, None)
        if (model_view["data"] is not None
                and len(initial.encode("utf-8")) + MAX_REPAIR_BYTES > MAX_PROMPT_BYTES):
            model_view = full_model_view(trusted, reason="prompt_bytes",
                                         counts=model_view["metadata"]["counts"])
            initial = render_view(model_view, None)
        prompt = initial if repair is None else render_view(model_view, repair)
        if model_view["data"] is not None and len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise PromptRenderError("compact scene prompt exceeds its byte limit")
        return prompt

    def occlusion_semantics(
        self,
        candidates: tuple[OcclusionCandidate, ...],
        entities: NormalizedEntities,
        evidence_summary: CvEvidenceSummary,
        *,
        video_duration: Any,
        frame_pts: Sequence[float],
        repair: Mapping[str, Any] | None = None,
    ) -> str:
        """Render trusted occlusion skeletons and bounded CV evidence as data."""
        if type(candidates) is not tuple or any(
            type(candidate) is not OcclusionCandidate for candidate in candidates
        ):
            raise PromptRenderError("occlusion candidates must be trusted records")
        if len(candidates) > 256:
            raise PromptRenderError("occlusion candidate count exceeds its bound")
        try:
            validate_candidate_identity_evidence(evidence_summary, candidates)
        except (ValueError, TypeError):
            raise PromptRenderError("occlusion identity evidence is not source authenticated") from None
        candidate_data = []
        for candidate in candidates:
            prompt_record = getattr(candidate, "prompt_record", None)
            if not callable(prompt_record):
                raise PromptRenderError("occlusion candidates must be trusted records")
            candidate_data.append(prompt_record())
        if type(entities) is not NormalizedEntities:
            raise PromptRenderError("entities must be validated normalized entities")
        entity_values = entities.entities
        entity_data = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in entity_values
        ]
        if any(not isinstance(item, Mapping) for item in entity_data):
            raise PromptRenderError("entities must be JSON records")
        if type(evidence_summary) is not CvEvidenceSummary:
            raise PromptRenderError("evidence summary must be a bounded CV summary")
        summary_data = evidence_summary.prompt_record()
        if isinstance(frame_pts, (str, bytes, bytearray)) or not isinstance(
            frame_pts, Sequence
        ):
            raise PromptRenderError("frame_pts must be a sequence")
        pts = list(frame_pts)
        if any(not _finite_nonnegative(value) for value in pts) or pts != sorted(
            set(pts)
        ):
            raise PromptRenderError("frame_pts must be unique ordered finite timestamps")
        if repair is not None and set(repair) != {"issue_codes"}:
            raise PromptRenderError("occlusion repair data may contain only issue codes")
        return self.render(
            "occlusion_semantics",
            {
                "VIDEO_DURATION_SECONDS_JSON": _prompt_video_duration(video_duration),
                "FRAME_PTS_JSON": pts,
                "OCCLUSION_CANDIDATES_JSON": candidate_data,
                "NORMALIZED_ENTITIES_JSON": entity_data,
                "CV_EVIDENCE_SUMMARY_JSON": summary_data,
                "VALIDATION_REPAIR_JSON": repair,
            },
        )


class EmbodiedActiveObjectsPipeline:
    """Run complete-video active-object inference with one schema repair."""

    def __init__(
        self,
        *,
        renderer: PromptRenderer | None = None,
        probe: Probe = probe_video,
        wait_jobs: WaitJobs = wait_for_jobs,
        wait_timeout: float = 300.0,
    ) -> None:
        self._renderer = renderer or PromptRenderer()
        self._probe = probe
        self._wait_jobs = wait_jobs
        self._wait_timeout = _positive_finite(wait_timeout, "wait_timeout")

    def run(self, task: TaskRecord, context: PipelineContext) -> dict[str, Any]:
        """Return only the validated object inventory for the complete video."""
        media_path = _media_path(context)
        metadata = self._probe(media_path)
        span = TimeSpan(0.0, _positive_finite(metadata.duration, "duration"))
        fps = _sampling_fps(task.payload)
        prompt_context = _prompt_context(task.payload)
        first_job: InferenceJob | None = None
        repair: dict[str, Any] | None = None

        for ordinal in range(2):
            affinity_worker_id, affinity_fallback_seconds = _repair_affinity(
                first_job,
                context,
                self._wait_timeout,
            )
            prompt = self._renderer.active_objects(prompt_context, repair=repair)
            [job] = context.store.create_inference_jobs(
                task.task_id,
                [
                    InferenceJobSpec(
                        stage="active_objects",
                        ordinal=ordinal,
                        payload=_job_payload(
                            task,
                            media_path,
                            span,
                            fps,
                            prompt,
                            metadata=metadata,
                        ),
                        affinity_worker_id=affinity_worker_id,
                        affinity_fallback_seconds=affinity_fallback_seconds,
                    )
                ],
            )
            try:
                [result] = self._wait_jobs(
                    context.store,
                    task.task_id,
                    [job.job_id],
                    self._wait_timeout,
                )
            except InferenceJobFailed:
                raise ActiveObjectPipelineError("active object inference failed") from None
            except JobWaitTimeout:
                raise ActiveObjectPipelineError("active object inference timed out") from None

            completed = context.store.get_inference_job(job.job_id)
            if ordinal == 0:
                first_job = completed
            issue_codes = DEFAULT_OUTPUT_SCHEMAS.failure_codes(
                "ObjectInventory",
                result,
            )
            sanitized = result
            if issue_codes is None:
                sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
                    "ObjectInventory",
                    result,
                )
                issue_codes = DEFAULT_OUTPUT_SCHEMAS.failure_codes(
                    "ObjectInventory",
                    sanitized,
                )
            if issue_codes is not None:
                if ordinal == 1:
                    raise ActiveObjectPipelineError(
                        "active object result schema is invalid after repair"
                    ) from None
                repair = {
                    "issue_codes": list(issue_codes),
                }
                continue
            inventory = ObjectInventory.model_validate(sanitized, strict=True)
            return inventory.model_dump(mode="json")

        raise AssertionError("active-object repair loop did not terminate")


def _canonical_json(value: Any) -> str:
    if isinstance(value, _TrustedCanonicalJSON):
        return value.text
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _naming_hint_data(prompt_context: str | None) -> dict[str, str] | None:
    if prompt_context is None:
        return None
    if not isinstance(prompt_context, str):
        raise PromptRenderError("prompt_context must be a string or None")
    if not prompt_context.strip():
        return None
    return {"prompt_context": prompt_context}


def _prompt_video_duration(value: Any) -> float:
    return _prompt_positive_finite(value, "video_duration")


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _prompt_positive_finite(value: Any, name: str) -> float:
    try:
        return _positive_finite(value, name)
    except (TypeError, ValueError):
        raise PromptRenderError(
            f"{name} must be a finite positive number"
        ) from None


def _validated_coarse_plan(value: Any) -> CoarsePlan:
    try:
        plan = CoarsePlan.model_validate(_model_or_mapping(value, "coarse_plan"))
        duration = plan.actions[-1].end if plan.actions else 0.0
        validate_coarse_plan(plan, duration)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise PromptRenderError(
            "pass_b requires a validated coarse_plan"
        ) from None
    return plan


def _validated_enrichment_indices(value: Any) -> list[int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise PromptRenderError(
            "expected_indices must be a sequence of non-negative integers"
        )
    if len(value) > _MAX_ENRICHMENT_RECORDS:
        raise PromptRenderError("enrichment record skeleton is not materializable")

    indices: list[int] = []
    previous: int | None = None
    for index in value:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or (previous is not None and index <= previous)
        ):
            raise PromptRenderError(
                "expected_indices must be strictly increasing non-negative integers"
            )
        indices.append(index)
        previous = index
    return indices


def _validated_enrichment_table_indices(
    table: Sequence[Mapping[str, Any]],
) -> list[int]:
    indices: list[int] = []
    for item in table:
        index = item.get("segment_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PromptRenderError(
                "immutable segment table segment_index values must be "
                "non-negative integers"
            )
        indices.append(index)
    return indices


def _enrichment_requirements(indices: Sequence[int]) -> dict[str, Any]:
    return {
        "exact_record_count": len(indices),
        "expected_indices": list(indices),
        "record_skeleton": [
            {
                "segment_index": index,
                "actor": "unknown",
                "actor_state": "unknown",
                "skill": "unknown",
                "target": "unknown",
                "visual_motion_state": "unknown",
                "confidence": 0.0,
            }
            for index in indices
        ],
    }


def _fine_segment_requirements(
    plan: CoarsePlan,
    maximum: float,
) -> _TrustedCanonicalJSON:
    maximum_fraction = Fraction(Decimal(str(maximum)))
    requirements: list[str] = []
    for action in plan.actions:
        duration = Fraction(Decimal(str(action.end))) - Fraction(
            Decimal(str(action.start))
        )
        minimum_count = _ceiling_fraction_ratio(duration, maximum_fraction)
        requirements.append(
            '{"action_index":'
            f"{action.action_index},"
            '"duration_seconds":'
            f"{_terminating_fraction_json_number(duration)},"
            '"minimum_fine_segment_count":'
            f"{minimum_count}}}"
        )
    return _TrustedCanonicalJSON(
        '{"max_fine_segment_seconds":'
        f"{_canonical_json(maximum)},"
        '"actions":['
        f"{','.join(requirements)}]}}"
    )


def _ceiling_fraction_ratio(numerator: Fraction, denominator: Fraction) -> int:
    ratio_numerator = numerator.numerator * denominator.denominator
    ratio_denominator = numerator.denominator * denominator.numerator
    return (ratio_numerator + ratio_denominator - 1) // ratio_denominator


def _terminating_fraction_json_number(value: Fraction) -> str:
    """Encode a positive finite-decimal fraction as an exact JSON number."""
    if value <= 0:
        raise ValueError("duration must be positive")

    denominator = value.denominator
    powers_of_two = 0
    while denominator % 2 == 0:
        denominator //= 2
        powers_of_two += 1
    powers_of_five = 0
    while denominator % 5 == 0:
        denominator //= 5
        powers_of_five += 1
    if denominator != 1:
        raise ValueError("duration does not have a finite decimal representation")

    scale = max(powers_of_two, powers_of_five)
    coefficient = value.numerator
    coefficient *= 2 ** (scale - powers_of_two)
    coefficient *= 5 ** (scale - powers_of_five)
    while scale and coefficient % 10 == 0:
        coefficient //= 10
        scale -= 1

    number = Decimal(
        (0, tuple(int(digit) for digit in str(coefficient)), -scale)
    )
    encoded = str(number).replace("E", "e")
    return encoded if scale else f"{encoded}.0"


def _model_or_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return value
    raise PromptRenderError(f"{name} must be a model or mapping")


def _media_path(context: PipelineContext) -> Path:
    path = context.media_path
    if path is None or not path.is_absolute() or not path.is_file():
        raise ActiveObjectPipelineError("active object media is unavailable")
    return path.resolve()


def _sampling_fps(payload: Mapping[str, Any]) -> float:
    try:
        return _positive_finite(payload.get("fps", 2.0), "fps")
    except (TypeError, ValueError):
        raise ActiveObjectPipelineError("active object request has invalid fps") from None


def _prompt_context(payload: Mapping[str, Any]) -> str | None:
    task_context = payload.get("task_context")
    if task_context is None:
        return None
    if not isinstance(task_context, Mapping):
        raise ActiveObjectPipelineError("active object naming context is invalid")
    value = task_context.get("prompt_context")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ActiveObjectPipelineError("active object naming context is invalid")
    return value


def _job_payload(
    task: TaskRecord,
    media_path: Path,
    span: TimeSpan,
    fps: float,
    prompt: str,
    *,
    schema_name: str = "ObjectInventory",
    schema_context: Mapping[str, Any] | None = None,
    metadata: VideoMetadata,
) -> dict[str, Any]:
    payload = {
        "video_path": str(media_path),
        "span": {"start": span.start, "end": span.end},
        "fps": fps,
        "prompt": prompt,
        "schema_name": schema_name,
        "video_session_id": task.task_id,
        "video_metadata": {
            "duration": metadata.duration,
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
        },
    }
    if schema_context is not None:
        payload["schema_context"] = dict(schema_context)
    payload.update(
        {
            key: task.payload[key]
            for key in ("media_resolution", "reasoning_effort", "clip_context")
            if key in task.payload
        }
    )
    return payload


def _repair_affinity(
    first_job: InferenceJob | None,
    context: PipelineContext,
    wait_timeout: float,
) -> tuple[str | None, float | None]:
    if first_job is None:
        return None, None
    if first_job.completed_by is None:
        raise ActiveObjectPipelineError("active object inference completion is invalid")
    grace = min(
        _positive_finite(context.settings.lease_seconds, "lease_seconds") / 3.0,
        wait_timeout / 3.0,
    )
    return first_job.completed_by, grace


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _action_media_path(context: PipelineContext) -> Path:
    path = context.media_path
    if path is None or not path.is_absolute() or not path.is_file():
        raise EmbodiedActionPipelineError("embodied action media is unavailable")
    return path.resolve()


def _action_sampling_fps(payload: Mapping[str, Any]) -> float:
    try:
        return _positive_finite(payload.get("fps", 3.0), "fps")
    except (TypeError, ValueError):
        raise EmbodiedActionPipelineError(
            "embodied action request has invalid fps"
        ) from None


def _action_prompt_context(payload: Mapping[str, Any]) -> str | None:
    try:
        return _prompt_context(payload)
    except ActiveObjectPipelineError:
        raise EmbodiedActionPipelineError(
            "embodied action naming context is invalid"
        ) from None


def _action_positive_finite(value: Any, name: str) -> float:
    try:
        return _positive_finite(value, name)
    except (TypeError, ValueError):
        raise EmbodiedActionPipelineError(
            f"embodied action {name} is invalid"
        ) from None


def _action_affinity(
    anchor: InferenceJob | None,
    context: PipelineContext,
    wait_timeout: float,
    stage: str,
) -> tuple[str | None, float | None]:
    if anchor is None:
        return None, None
    if anchor.completed_by is None:
        raise EmbodiedActionPipelineError(
            f"{_stage_label(stage)} affinity source is invalid"
        )
    grace = min(
        _action_positive_finite(context.settings.lease_seconds, "lease_seconds")
        / 3.0,
        wait_timeout / 3.0,
    )
    return anchor.completed_by, grace


def _validated_stage_result(
    schema_name: str,
    result: Mapping[str, Any],
    schema_context: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    tuple[str, ...] | None,
    NormalizedSchemaOutput | None,
]:
    issue_codes = DEFAULT_OUTPUT_SCHEMAS.failure_codes(schema_name, result)
    if issue_codes is not None:
        return dict(result), issue_codes, None
    normalization = DEFAULT_OUTPUT_SCHEMAS.normalized_result(
        schema_name,
        result,
        schema_context,
    )
    if normalization is not None:
        return normalization.data, None, normalization
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        schema_name,
        result,
        schema_context,
    )
    normalization = DEFAULT_OUTPUT_SCHEMAS.normalized_result(
        schema_name,
        sanitized,
        schema_context,
    )
    if normalization is not None:
        return normalization.data, None, normalization
    return (
        sanitized,
        DEFAULT_OUTPUT_SCHEMAS.failure_codes(schema_name, sanitized),
        None,
    )


def _enrichment_normalization_warning(
    normalization: NormalizedSchemaOutput,
) -> dict[str, Any]:
    codes = set(normalization.issue_codes)
    return {
        "code": "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN",
        "fields": [
            field
            for issue_code, field in _ENRICHMENT_WARNING_FIELD_BY_CODE
            if issue_code in codes
        ],
        "count": normalization.normalized_field_count,
    }


def _scene_normalization_warning(
    normalization: NormalizedSchemaOutput,
) -> dict[str, Any]:
    return {
        "code": "SCENE_MECHANICS_NORMALIZED",
        "issue_codes": list(normalization.issue_codes),
        "count": normalization.normalized_field_count,
    }


def _occlusion_normalization_warning(
    normalization: NormalizedSchemaOutput,
) -> dict[str, Any]:
    if "OCCLUSION_BOUNDARIES_COMPLETED" in normalization.issue_codes:
        code = "OCCLUSION_BOUNDARIES_COMPLETED"
    elif "OCCLUSION_EVENTS_NOT_ORDERED" in normalization.issue_codes:
        code = "OCCLUSION_EVENTS_REORDERED"
    else:
        code = "OCCLUSION_OCCLUDER_NORMALIZED"
    return {
        "code": code,
        "count": normalization.normalized_field_count,
    }


def _boundary_normalization_warning(
    normalization: NormalizedSchemaOutput,
) -> dict[str, Any]:
    return {
        "code": "BOUNDARY_TOPOLOGY_NORMALIZED",
        "issue_codes": list(normalization.issue_codes),
        "count": normalization.normalized_field_count,
    }


def _repair_history_for(ordinal: int) -> tuple[str, ...]:
    """Return the closed provenance history for a zero-based attempt ordinal."""
    return ("initial",) + ("repair",) * ordinal


def _stage_validation_error(
    stage: str,
    issue_codes: Sequence[str],
) -> TemporalValidationError:
    return TemporalValidationError(
        TemporalIssue(
            code=code,
            path=(stage,),
            message="stage output remains invalid after one repair",
        )
        for code in issue_codes
    )


def _stage_label(stage: str) -> str:
    try:
        return {
            "embodied_pass_a": "embodied pass A",
            "embodied_pass_b": "embodied pass B",
            "embodied_enrichment": "embodied enrichment",
            "scene_semantics": "scene semantics",
            "occlusion_semantics": "occlusion semantics",
        }[stage]
    except KeyError:
        raise EmbodiedActionPipelineError("embodied stage is invalid") from None


def _fine_segment_table(plan: BoundaryPlan) -> tuple[FineSegmentTableRow, ...]:
    return tuple(
        FineSegmentTableRow(
            action_index=action.action_index,
            segment_index=segment.segment_index,
            start=segment.start,
            end=segment.end,
            description=segment.description,
            event_type=segment.event_type.value,
            start_boundary_id=segment.start_boundary_id,
            end_boundary_id=segment.end_boundary_id,
        )
        for action in plan.actions
        for segment in action.fine_segments
    )


def _guard_enrichment_record_count(plan: BoundaryPlan) -> None:
    count = 0
    for action in plan.actions:
        count += len(action.fine_segments)
        if count > _MAX_ENRICHMENT_RECORDS:
            raise EmbodiedActionPipelineError(
                "embodied enrichment record count exceeds "
                f"{_MAX_ENRICHMENT_RECORDS}"
            )


def _merge_enrichment(
    segment_table: tuple[FineSegmentTableRow, ...],
    enrichment: EnrichmentResult,
) -> list[dict[str, Any]]:
    by_index = {
        segment.segment_index: segment.model_dump(
            mode="json",
            exclude={"segment_index"},
        )
        for segment in enrichment.segments
    }
    merged: list[dict[str, Any]] = []
    for row in segment_table:
        record = row.public_record()
        record.update(by_index[row.segment_index])
        merged.append(record)
    return merged
