"""Hierarchical visual-only general video captioning."""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from ..domain import InferenceJob, InferenceJobSpec, TaskRecord
from ..media import TimeSpan, VideoMetadata, plan_segments, probe_video
from ..pipelines.base import PipelineContext, SafePipelineError
from ..store import SQLiteTaskStore
from ..workers import InferenceJobFailed, JobWaitTimeout, wait_for_jobs
from .output_validation import DEFAULT_OUTPUT_SCHEMAS


Probe = Callable[[Path], VideoMetadata]
WaitJobs = Callable[[SQLiteTaskStore, str, Sequence[str], float], list[dict[str, Any]]]
SegmentPlanner = Callable[[float, float, float], list[TimeSpan]]
_UNION_FIELDS = (
    "scene",
    "subjects",
    "actions",
    "visible_text",
    "uncertainty",
    "warnings",
)
EvidenceText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]
EvidenceList = Annotated[list[EvidenceText], Field(max_length=256)]


class GeneralTimelineEvent(BaseModel):
    """One strictly typed visual event retained in the public timeline."""

    model_config = ConfigDict(extra="forbid")

    start_time: Annotated[float, Field(allow_inf_nan=False)]
    end_time: Annotated[float, Field(allow_inf_nan=False)]
    scene: EvidenceList
    subjects: EvidenceList
    actions: EvidenceList
    visible_text: EvidenceList
    uncertainty: EvidenceList
    description: EvidenceText
    warnings: EvidenceList

    @model_validator(mode="after")
    def require_positive_span(self) -> GeneralTimelineEvent:
        if self.start_time >= self.end_time:
            raise ValueError("event start_time must be less than end_time")
        return self


class GeneralSegmentResult(BaseModel):
    """Exact output contract for one visual segment inference."""

    model_config = ConfigDict(extra="forbid")

    segments: Annotated[list[GeneralTimelineEvent], Field(max_length=10_000)]
    warnings: EvidenceList = Field(default_factory=list)


class GeneralSummaryResult(BaseModel):
    """Exact output contract for summary generation over validated evidence."""

    model_config = ConfigDict(extra="forbid")

    summary: EvidenceText
    timeline: Annotated[list[GeneralTimelineEvent], Field(max_length=10_000)]
    warnings: EvidenceList = Field(default_factory=list)


class GeneralCaptionError(SafePipelineError):
    """A stable failure in general-caption request or segment validation."""


class _SummaryValidationError(ValueError):
    def __init__(self, issue: str) -> None:
        self.issue = issue
        super().__init__(issue)


class GeneralCaptionPipeline:
    """Coordinate segment inference and deterministic timeline aggregation."""

    def __init__(
        self,
        *,
        probe: Probe = probe_video,
        planner: SegmentPlanner = plan_segments,
        wait_jobs: WaitJobs = wait_for_jobs,
        wait_timeout: float = 300.0,
    ) -> None:
        self._probe = probe
        self._planner = planner
        self._wait_jobs = wait_jobs
        self._wait_timeout = _positive_finite(wait_timeout, "wait_timeout")

    def run(self, task: TaskRecord, context: PipelineContext) -> dict[str, Any]:
        """Build a validated deterministic timeline and a best-effort summary."""
        media_path = _media_path(context)
        metadata = self._probe(media_path)
        clip = _clip_span(task.payload, metadata.duration)
        sampling_fps = _sampling_fps(task.payload)
        spans = _planned_spans(
            self._planner,
            clip,
            context.settings.segment_seconds,
            context.settings.segment_overlap_seconds,
        )

        segment_jobs = context.store.create_inference_jobs(
            task.task_id,
            [
                InferenceJobSpec(
                    stage="general_segment",
                    ordinal=ordinal,
                    payload=_job_payload(
                        task,
                        media_path,
                        span,
                        sampling_fps,
                        _segment_prompt(task.payload, span),
                        "general_segment",
                        metadata,
                        schema_context={
                            "span": {"start": span.start, "end": span.end}
                        },
                    ),
                )
                for ordinal, span in enumerate(spans)
            ],
        )
        scheduled_model_names = {job.model_name for job in segment_jobs}
        if len(scheduled_model_names) != 1:
            raise GeneralCaptionError("general segment model scheduling is invalid")
        scheduled_model_name = next(iter(scheduled_model_names))
        try:
            segment_results = self._wait_jobs(
                context.store,
                task.task_id,
                [job.job_id for job in segment_jobs],
                self._wait_timeout,
            )
        except InferenceJobFailed as error:
            raise GeneralCaptionError(
                f"general segment {error.ordinal} failed: local inference job failed"
            ) from None
        except JobWaitTimeout:
            raise GeneralCaptionError("general segment inference timed out") from None

        events: list[dict[str, Any]] = []
        warnings: list[str] = []
        for ordinal, (span, result) in enumerate(zip(spans, segment_results, strict=True)):
            validated, result_warnings = _validate_segment_result(result, span, ordinal)
            events.extend(validated)
            warnings.extend(result_warnings)
        timeline = _merge_timeline(events)
        if not timeline:
            raise GeneralCaptionError("general segments returned no visual evidence")
        warnings.extend(
            warning
            for event in timeline
            for warning in event["warnings"]
        )

        summary, summary_warnings = self._generate_summary(
            task,
            context,
            media_path,
            clip,
            sampling_fps,
            metadata,
            timeline,
            segment_jobs,
        )
        warnings.extend(summary_warnings)

        return {
            "summary": summary,
            "timeline": timeline,
            "metadata": {
                "model_name": scheduled_model_name,
                "duration": metadata.duration,
                "width": metadata.width,
                "height": metadata.height,
                "source_fps": metadata.fps,
                "sampling_fps": sampling_fps,
                "clip_start": clip.start,
                "clip_end": clip.end,
                "segment_seconds": context.settings.segment_seconds,
                "segment_overlap_seconds": context.settings.segment_overlap_seconds,
                "segment_count": len(spans),
            },
            "warnings": sorted(set(warnings)),
        }

    def _generate_summary(
        self,
        task: TaskRecord,
        context: PipelineContext,
        media_path: Path,
        clip: TimeSpan,
        sampling_fps: float,
        metadata: VideoMetadata,
        timeline: list[dict[str, Any]],
        segment_jobs: list[InferenceJob],
    ) -> tuple[str, list[str]]:
        completed_segments = [
            job
            for job in (
                context.store.get_inference_job(item.job_id) for item in segment_jobs
            )
            if job is not None
        ]
        affinity_source = next(
            (job for job in completed_segments if job.completed_by is not None),
            None,
        )
        affinity_worker = (
            affinity_source.completed_by if affinity_source is not None else None
        )
        fallback_seconds = None
        if affinity_source is not None:
            fallback_seconds = min(
                _positive_finite(context.settings.lease_seconds, "lease_seconds") / 3.0,
                self._wait_timeout / 3.0,
            )

        issue: str | None = None
        for ordinal in range(2):
            prompt = _summary_prompt(task.payload, timeline, issue)
            [job] = context.store.create_inference_jobs(
                task.task_id,
                [
                    InferenceJobSpec(
                        stage="general_summary",
                        ordinal=ordinal,
                        payload=_job_payload(
                            task,
                            media_path,
                            clip,
                            sampling_fps,
                            prompt,
                            "general_summary",
                            metadata,
                            schema_context={
                                "span": {"start": clip.start, "end": clip.end},
                                "expected_timeline": timeline,
                            },
                        ),
                        affinity_worker_id=affinity_worker,
                        affinity_fallback_seconds=fallback_seconds,
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
                return _validate_summary_result(result, clip, timeline)
            except InferenceJobFailed:
                issue = "summary_job_failed"
            except JobWaitTimeout:
                return _fallback_summary(timeline, clip), ["summary_generation_failed"]
            except _SummaryValidationError as error:
                issue = error.issue

        return _fallback_summary(timeline, clip), ["summary_generation_failed"]


def _media_path(context: PipelineContext) -> Path:
    path = context.media_path
    if path is None or not path.is_absolute() or not path.is_file():
        raise GeneralCaptionError("general caption media is unavailable")
    return path.resolve()


def _clip_span(payload: Mapping[str, Any], duration: float) -> TimeSpan:
    try:
        duration_value = _positive_finite(duration, "duration")
        start = payload.get("start")
        end = payload.get("end")
        if (start is None) != (end is None):
            raise ValueError
        if start is None:
            return TimeSpan(0.0, duration_value)
        start_value = _finite_number(start, "start")
        end_value = _finite_number(end, "end")
        if start_value < 0 or start_value >= end_value or end_value > duration_value:
            raise ValueError
        return TimeSpan(start_value, end_value)
    except (TypeError, ValueError):
        raise GeneralCaptionError("general caption request has invalid clip bounds") from None


def _sampling_fps(payload: Mapping[str, Any]) -> float:
    try:
        return _positive_finite(payload.get("fps", 2.0), "fps")
    except (TypeError, ValueError):
        raise GeneralCaptionError("general caption request has invalid fps") from None


def _planned_spans(
    planner: SegmentPlanner,
    clip: TimeSpan,
    segment_seconds: float,
    overlap_seconds: float,
) -> list[TimeSpan]:
    local = planner(clip.end - clip.start, segment_seconds, overlap_seconds)
    if not local:
        raise GeneralCaptionError("general segment planner returned no visual spans")
    clip_duration = clip.end - clip.start
    if (
        local[0].start != 0.0
        or local[-1].end != clip_duration
        or any(
            not isinstance(span, TimeSpan)
            or span.end > clip_duration
            or (index > 0 and span.start > local[index - 1].end)
            for index, span in enumerate(local)
        )
    ):
        raise GeneralCaptionError("general segment planner returned invalid visual spans")
    result: list[TimeSpan] = []
    for span in local:
        end = clip.end if span.end == local[-1].end else clip.start + span.end
        result.append(TimeSpan(clip.start + span.start, end))
    return result


def _job_payload(
    task: TaskRecord,
    media_path: Path,
    span: TimeSpan,
    fps: float,
    prompt: str,
    schema_name: str,
    metadata: VideoMetadata,
    *,
    schema_context: Mapping[str, Any],
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
        "schema_context": dict(schema_context),
    }
    payload.update(
        {
            key: task.payload[key]
            for key in ("media_resolution", "reasoning_effort", "clip_context")
            if key in task.payload
        }
    )
    return payload


def _segment_prompt(payload: Mapping[str, Any], span: TimeSpan) -> str:
    document: dict[str, Any] = {
        "evidence": "visual_frames_only",
        "instructions": (
            "Return exactly one JSON object matching output_schema. Describe only "
            "directly visible events in this span and do not use any other evidence "
            "source. Preserve scene/environment, visible subjects, actions/events, "
            "visible on-screen text, and explicit uncertainty for every event."
        ),
        "output_schema": GeneralSegmentResult.model_json_schema(),
        "segment": {"start_time": span.start, "end_time": span.end},
    }
    query = payload.get("query")
    if isinstance(query, str) and query.strip():
        document["query"] = query.strip()
    task_context = payload.get("task_context")
    if isinstance(task_context, Mapping):
        hint = task_context.get("prompt_context")
        if isinstance(hint, str) and hint.strip():
            document["naming_context"] = hint.strip()
    tuning = _prompt_tuning(payload)
    if tuning:
        document["tuning"] = tuning
    return _canonical_json(document)


def _summary_prompt(
    payload: Mapping[str, Any],
    timeline: list[dict[str, Any]],
    issue: str | None,
) -> str:
    document: dict[str, Any] = {
        "evidence": "validated_visual_timeline_only",
        "instructions": (
            "Return exactly one JSON object matching output_schema. Summarize only "
            "the supplied validated visual timeline and echo that timeline exactly, "
            "without changing timestamps, descriptions, or evidence fields."
        ),
        "output_schema": GeneralSummaryResult.model_json_schema(),
        "timeline": timeline,
    }
    query = payload.get("query")
    if isinstance(query, str) and query.strip():
        document["query"] = query.strip()
    tuning = _prompt_tuning(payload)
    if tuning:
        document["tuning"] = tuning
    if issue is not None:
        document["repair"] = {
            "attempt": 1,
            "issues": [issue],
            "instruction": "Correct the schema issue without adding unobserved evidence.",
        }
    return _canonical_json(document)


def _prompt_tuning(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("clip_context", "media_resolution", "reasoning_effort")
        if key in payload
    }


def _validate_segment_result(
    result: Mapping[str, Any],
    span: TimeSpan,
    ordinal: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    failure_codes = DEFAULT_OUTPUT_SCHEMAS.failure_codes("general_segment", result)
    if failure_codes is not None:
        if "GENERAL_SEGMENT_TIMESTAMP_OUT_OF_SPAN" in failure_codes:
            raise GeneralCaptionError(
                f"general segment {ordinal} failed: event timestamps are outside its visual span"
            )
        raise GeneralCaptionError(
            f"general segment {ordinal} failed: result schema is invalid"
        )
    try:
        parsed = GeneralSegmentResult.model_validate(result, strict=True)
    except ValidationError:
        raise GeneralCaptionError(
            f"general segment {ordinal} failed: result schema is invalid"
        ) from None
    events: list[dict[str, Any]] = []
    for event in parsed.segments:
        start = event.start_time
        end = event.end_time
        if start < span.start or end > span.end or start >= end:
            raise GeneralCaptionError(
                f"general segment {ordinal} failed: event timestamps are outside its visual span"
            )
        value = event.model_dump(mode="json")
        value["description"] = " ".join(event.description.split())
        for field in _UNION_FIELDS:
            value[field] = sorted(set(value[field]))
        events.append(value)
    return events, sorted(set(parsed.warnings))


def _merge_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        events,
        key=lambda item: (
            item["start_time"],
            item["end_time"],
            _normalized_description(item["description"]),
        ),
    )
    merged: list[dict[str, Any]] = []
    latest_by_description: dict[str, int] = {}
    for event in ordered:
        normalized = _normalized_description(event["description"])
        previous_index = latest_by_description.get(normalized)
        if previous_index is not None:
            previous = merged[previous_index]
            if event["start_time"] < previous["end_time"]:
                previous["end_time"] = max(previous["end_time"], event["end_time"])
                for field in _UNION_FIELDS:
                    previous[field] = sorted(
                        set(previous[field]) | set(event[field])
                    )
                continue
        latest_by_description[normalized] = len(merged)
        merged.append(dict(event))
    return sorted(
        merged,
        key=lambda item: (item["start_time"], item["end_time"], item["description"]),
    )


def _normalized_description(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _validate_summary_result(
    result: Mapping[str, Any],
    clip: TimeSpan,
    expected_timeline: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise _SummaryValidationError("summary_non_blank")
    try:
        parsed = GeneralSummaryResult.model_validate(result, strict=True)
    except ValidationError:
        raise _SummaryValidationError("summary_schema_invalid") from None
    previous_start = -1.0
    timeline = parsed.model_dump(mode="json")["timeline"]
    for event in timeline:
        start = event["start_time"]
        end = event["end_time"]
        if start < clip.start or end > clip.end or start >= end or start < previous_start:
            raise _SummaryValidationError("summary_timeline_order")
        previous_start = start
    if timeline != expected_timeline:
        raise _SummaryValidationError("summary_timeline_mismatch")
    return parsed.summary, sorted(set(parsed.warnings))


def _fallback_summary(timeline: list[dict[str, Any]], clip: TimeSpan) -> str:
    return (
        f"Visual timeline contains {len(timeline)} events from "
        f"{clip.start:.3f} to {clip.end:.3f} seconds."
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _positive_finite(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
