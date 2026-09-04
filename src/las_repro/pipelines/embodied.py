"""Safe prompts and durable 0805 embodied inference orchestration."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from importlib import resources
from numbers import Real
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..domain import InferenceJob, InferenceJobSpec, TaskRecord
from ..media import TimeSpan, VideoMetadata, probe_video
from ..models.base import VideoSession
from ..pipelines.base import PipelineContext, SafePipelineError
from ..store import SQLiteTaskStore
from ..workers import InferenceJobFailed, JobWaitTimeout, wait_for_jobs
from .output_validation import DEFAULT_OUTPUT_SCHEMAS, NormalizedSchemaOutput
from .scene_semantics import (
    SceneSemantics,
    trusted_target_skeleton,
    unavailable_scene_semantics,
)
from .semantic_events import build_semantic_events
from .validators import (
    BoundaryPlan,
    CoarsePlan,
    EnrichmentResult,
    ObjectInventory,
    TemporalIssue,
    TemporalValidationError,
    validate_coarse_plan,
)


EMBODIED_PROMPT_VERSION = "0805-local-v2"

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


@dataclass(frozen=True)
class _TrustedCanonicalJSON:
    """Internally generated JSON whose exact numeric lexemes must be preserved."""

    text: str


class ActiveObjectPipelineError(SafePipelineError):
    """A stable active-object stage failure safe for persistence."""


class EmbodiedActionPipelineError(SafePipelineError):
    """A stable execution failure in the 0805 action pipeline."""


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
        media_path = _action_media_path(context)
        metadata = self._probe(media_path)
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

        enrichment_data, _, enrichment_normalization = self._run_validated_stage(
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
                repair=repair,
            ),
            affinity_anchor=pass_a_job,
            metadata=metadata,
        )
        enrichment = EnrichmentResult.model_validate(enrichment_data)
        warnings: list[dict[str, Any]] = []
        if boundary_normalization is not None:
            warnings.append(_boundary_normalization_warning(boundary_normalization))
        if enrichment_normalization is not None:
            warnings.append(_enrichment_normalization_warning(enrichment_normalization))
        segments = _merge_enrichment(segment_table, enrichment)
        trusted_targets = trusted_target_skeleton(segments)
        try:
            scene_data, _, _ = self._run_validated_stage(
                task,
                context,
                media_path,
                span,
                fps,
                stage="scene_semantics",
                schema_name="SceneSemantics",
                schema_context={
                    "duration": span.end,
                    "require_observed_content": bool(trusted_targets),
                    "required_object_ids": [
                        target["object_id"] for target in trusted_targets
                    ],
                },
                render_prompt=lambda repair: self._renderer.scene_semantics(
                    segments,
                    video_duration=span.end,
                    repair=repair,
                ),
                affinity_anchor=pass_a_job,
                metadata=metadata,
            )
        except TemporalValidationError:
            scene_data = unavailable_scene_semantics()
            warnings.append({"code": "SCENE_SEMANTICS_UNAVAILABLE"})
        scene = SceneSemantics.model_validate(scene_data)
        result = {
            "task_description": coarse.task_description,
            "segments": segments,
            "grouped_semantic_events": build_semantic_events(segments),
            **scene.model_dump(mode="json"),
        }
        if warnings:
            result["warnings"] = warnings
        return result

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
    ) -> tuple[dict[str, Any], InferenceJob, NormalizedSchemaOutput | None]:
        repair: dict[str, Any] | None = None
        first_job: InferenceJob | None = None
        for ordinal in range(2):
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
                job_schema_context["allow_topology_fallback"] = ordinal == 1
            if schema_name == "EnrichmentResult":
                job_schema_context["allow_enum_unknown_fallback"] = ordinal == 1
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
                    f"{_stage_label(stage)} inference failed"
                ) from None
            except JobWaitTimeout:
                raise EmbodiedActionPipelineError(
                    f"{_stage_label(stage)} inference timed out"
                ) from None

            completed = context.store.get_inference_job(job.job_id)
            if completed is None or completed.completed_by is None:
                raise EmbodiedActionPipelineError(
                    f"{_stage_label(stage)} completion is invalid"
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
            if ordinal == 1:
                raise _stage_validation_error(stage, issue_codes)
            repair = {"issue_codes": list(issue_codes)}

        raise AssertionError("embodied validation repair loop did not terminate")


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
        return self.render(
            "embodied_enrichment",
            {
                "SEGMENTS_JSON": table,
                "ENRICHMENT_REQUIREMENTS_JSON": _enrichment_requirements(indices),
                "VALIDATION_REPAIR_JSON": repair,
            },
        )

    def scene_semantics(
        self,
        segments: Sequence[Mapping[str, Any] | BaseModel],
        *,
        video_duration: Any,
        repair: Mapping[str, Any] | None = None,
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
        return self.render(
            "scene_semantics",
            {
                "VIDEO_DURATION_SECONDS_JSON": _prompt_video_duration(video_duration),
                "SEGMENTS_JSON": table,
                "KNOWN_TARGETS_JSON": trusted_target_skeleton(table),
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
    planning_target = _planning_target(maximum, maximum_fraction)
    planning_target_fraction = Fraction(Decimal(str(planning_target)))
    requirements: list[str] = []
    for action in plan.actions:
        duration = Fraction(Decimal(str(action.end))) - Fraction(
            Decimal(str(action.start))
        )
        minimum_count = _ceiling_fraction_ratio(duration, maximum_fraction)
        suggested_count = _ceiling_fraction_ratio(
            duration,
            planning_target_fraction,
        )
        boundary_slots = _boundary_slots(
            action_index=action.action_index,
            start=action.start,
            end=action.end,
            count=suggested_count,
            maximum=maximum,
        )
        requirements.append(
            '{"action_index":'
            f"{action.action_index},"
            '"duration_seconds":'
            f"{_terminating_fraction_json_number(duration)},"
            '"minimum_fine_segment_count":'
            f"{minimum_count},"
            '"suggested_fine_segment_count":'
            f"{suggested_count},"
            '"exact_boundary_point_count":'
            f"{suggested_count + 1},"
            '"exact_fine_segment_count":'
            f"{suggested_count},"
            '"boundary_slots":'
            f"{_canonical_json(boundary_slots)}}}"
        )
    return _TrustedCanonicalJSON(
        '{"max_fine_segment_seconds":'
        f"{_canonical_json(maximum)},"
        '"planning_target_seconds":'
        f"{_canonical_json(planning_target)},"
        '"actions":['
        f"{','.join(requirements)}]}}"
    )


def _planning_target(maximum: float, maximum_fraction: Fraction) -> float:
    """Return a representable 90% target, or the hard cap at float underflow."""
    candidate = float(maximum_fraction * Fraction(9, 10))
    if candidate <= 0 or candidate >= maximum:
        next_lower = math.nextafter(maximum, 0.0)
        return next_lower if next_lower > 0 else maximum
    return candidate


def _ceiling_fraction_ratio(numerator: Fraction, denominator: Fraction) -> int:
    ratio_numerator = numerator.numerator * denominator.denominator
    ratio_denominator = numerator.denominator * denominator.numerator
    return (ratio_numerator + ratio_denominator - 1) // ratio_denominator


def _boundary_slots(
    *,
    action_index: int,
    start: float,
    end: float,
    count: int,
    maximum: float,
) -> list[dict[str, Any]]:
    """Build binary64 windows whose worst adjacent choices satisfy the cap.

    For ideal step ``s`` and slack ``d = min(s, cap - s) / 4``, the
    smallest adjacent separation is ``s - 2d > 0`` and the largest possible
    segment is ``s + 2d <= cap``. Rounding both bounds inward can only tighten
    those guarantees. The zero-slack case is necessary when one segment spans
    exactly the smallest representable configured cap.
    """
    if count < 1 or count + 1 > _MAX_BOUNDARY_SLOTS_PER_ACTION:
        raise PromptRenderError("pass_b boundary slot count is not materializable")

    start_fraction = Fraction.from_float(start)
    end_fraction = Fraction.from_float(end)
    maximum_fraction = Fraction.from_float(maximum)
    step = (end_fraction - start_fraction) / count
    if step <= 0 or step > maximum_fraction:
        raise PromptRenderError("pass_b boundary slots are not feasible")
    slack = min(step, maximum_fraction - step) / 4

    slots: list[dict[str, Any]] = []
    for position in range(count + 1):
        if position == 0:
            center = minimum = maximum_time = start
        elif position == count:
            center = minimum = maximum_time = end
        else:
            exact_center = start_fraction + step * position
            exact_minimum = exact_center - slack
            exact_maximum = exact_center + slack
            minimum = _inward_binary64(exact_minimum, lower=True)
            maximum_time = _inward_binary64(exact_maximum, lower=False)
            if minimum > maximum_time:
                raise PromptRenderError("pass_b boundary slots are not representable")
            center = float(exact_center)
            center = min(max(center, minimum), maximum_time)
        slots.append(
            {
                "boundary_position": position,
                "boundary_id": f"a{action_index}_b{position}",
                "ideal_partition_center_seconds": center,
                "inclusive_time_window": {
                    "minimum_seconds": minimum,
                    "maximum_seconds": maximum_time,
                },
            }
        )

    _validate_boundary_slot_guarantees(slots, maximum_fraction)
    return slots


def _inward_binary64(value: Fraction, *, lower: bool) -> float:
    candidate = float(value)
    represented = Fraction.from_float(candidate)
    if lower and represented < value:
        candidate = math.nextafter(candidate, math.inf)
    elif not lower and represented > value:
        candidate = math.nextafter(candidate, -math.inf)
    if not math.isfinite(candidate) or candidate < 0:
        raise PromptRenderError("pass_b boundary slots are not representable")
    return candidate


def _validate_boundary_slot_guarantees(
    slots: Sequence[Mapping[str, Any]],
    maximum: Fraction,
) -> None:
    for previous, following in zip(slots[:-1], slots[1:], strict=True):
        previous_window = previous["inclusive_time_window"]
        following_window = following["inclusive_time_window"]
        assert isinstance(previous_window, Mapping)
        assert isinstance(following_window, Mapping)
        previous_minimum = Fraction.from_float(previous_window["minimum_seconds"])
        previous_maximum = Fraction.from_float(previous_window["maximum_seconds"])
        following_minimum = Fraction.from_float(following_window["minimum_seconds"])
        following_maximum = Fraction.from_float(following_window["maximum_seconds"])
        if (
            following_minimum <= previous_maximum
            or following_maximum - previous_minimum > maximum
        ):
            raise PromptRenderError("pass_b boundary slots are not feasible")


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


def _boundary_normalization_warning(
    normalization: NormalizedSchemaOutput,
) -> dict[str, Any]:
    return {
        "code": "BOUNDARY_TOPOLOGY_NORMALIZED",
        "issue_codes": list(normalization.issue_codes),
        "count": normalization.normalized_field_count,
    }


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
