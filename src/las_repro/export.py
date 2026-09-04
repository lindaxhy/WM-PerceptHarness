"""Deterministic, durable exports for completed embodied action captions."""

from __future__ import annotations

import errno
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import Literal, TypeVar, final

from .pipelines.scene_semantics import (
    SceneSemantics,
    trusted_target_skeleton,
    unavailable_scene_semantics,
    validate_scene_semantics,
)
from .pipelines.validators import (
    Actor,
    ActorState,
    FineEventType,
    Skill,
    VisualMotionState,
)
from .pipelines.semantic_events import validate_semantic_events


_ANNOTATION_STAGE = "boundary_fine_segments_0805"
_TIME_EPSILON = 1e-9
_MAX_FINE_SEGMENT_SECONDS = 1.0
_MAX_CAPTION_CHARACTERS = 60
_CAPTION_ID = re.compile(r".+_cap_[0-9]{4,}")
_UNSUPPORTED_DIRECTORY_SYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }
)
_REQUIRED_RESULT_KEYS = frozenset({"task_description", "segments"})
_SCENE_RESULT_KEYS = frozenset(
    {"objects", "initial_state", "final_state", "outcome", "semantic_events"}
)
_OPTIONAL_RESULT_KEYS = frozenset(
    {"warnings", "grouped_semantic_events"}
) | _SCENE_RESULT_KEYS
_NORMALIZATION_WARNING_KEYS = frozenset({"code", "fields", "count"})
_NORMALIZATION_WARNING_CODE = "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN"
_BOUNDARY_WARNING_KEYS = frozenset({"code", "issue_codes", "count"})
_BOUNDARY_WARNING_CODE = "BOUNDARY_TOPOLOGY_NORMALIZED"
_SCENE_WARNING_CODE = "SCENE_SEMANTICS_UNAVAILABLE"
_BOUNDARY_WARNING_ISSUE_CODES = (
    "SEGMENT_TOO_LONG",
    "SEGMENT_BOUNDARY_NOT_ADJACENT",
    "SEGMENT_DESCRIPTION_INVALID",
)
_NORMALIZABLE_ENUM_FIELDS = (
    "actor",
    "actor_state",
    "skill",
    "visual_motion_state",
)
_SEGMENT_KEYS = frozenset(
    {
        "action_index",
        "segment_index",
        "start",
        "end",
        "description",
        "event_type",
        "start_boundary_id",
        "end_boundary_id",
        "actor",
        "actor_state",
        "skill",
        "target",
        "visual_motion_state",
        "confidence",
    }
)
_StrEnum = TypeVar("_StrEnum", bound=StrEnum)


class ActionCaptionExportError(ValueError):
    """A stable export-boundary error which does not expose supplied content."""


@final
@dataclass(frozen=True, slots=True)
class ActionCaption:
    """One flattened 0805 action caption ready for a JSONL training corpus."""

    caption_id: str
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int
    duration_frames: int
    caption: str
    caption_char_count: int
    annotation_stage: Literal["boundary_fine_segments_0805"]
    source_segment_index: int
    parent_source_segment_index: int
    needs_refinement: bool
    over_caption_char_limit: bool
    actor: Actor
    actor_state: ActorState
    skill: Skill
    target: str
    visual_motion_state: VisualMotionState
    confidence: float
    schema_parse_ok: bool
    parse_errors: tuple[str, ...]
    needs_review: bool
    source_description: str

    def __post_init__(self) -> None:
        _validate_action_caption(self)

    def to_dict(self) -> dict[str, object]:
        """Return an independent JSON-ready record in the documented field order."""
        _validate_action_caption(self)
        return {
            "caption_id": self.caption_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "caption": self.caption,
            "caption_char_count": self.caption_char_count,
            "annotation_stage": self.annotation_stage,
            "source_segment_index": self.source_segment_index,
            "parent_source_segment_index": self.parent_source_segment_index,
            "needs_refinement": self.needs_refinement,
            "over_caption_char_limit": self.over_caption_char_limit,
            "actor": self.actor.value,
            "actor_state": self.actor_state.value,
            "skill": self.skill.value,
            "target": self.target,
            "visual_motion_state": self.visual_motion_state.value,
            "confidence": self.confidence,
            "schema_parse_ok": self.schema_parse_ok,
            "parse_errors": list(self.parse_errors),
            "needs_review": self.needs_review,
            "source_description": self.source_description,
        }


@dataclass(frozen=True, slots=True)
class _CompletedSegment:
    action_index: int
    segment_index: int
    start: float
    end: float
    description: str
    actor: Actor
    actor_state: ActorState
    skill: Skill
    target: str
    visual_motion_state: VisualMotionState
    confidence: float


def iter_action_captions(
    video_id: str,
    embodied_result: dict[str, object],
    *,
    source_fps: float,
) -> Iterator[ActionCaption]:
    """Yield deterministic caption rows from one completed 0805 action result.

    The result must be an exact built-in ``dict`` and is validated from one
    shallow snapshot without modifying it; custom mappings are rejected at the
    trust boundary.  Frame ranges are half-open: ``[start_frame, end_frame)``.
    A source span that rounds to no source frame is rejected rather than
    stretched, because stretching would invent timing.
    """
    checked_video_id = _video_id(video_id)
    checked_fps = _source_fps(source_fps)
    segments = _completed_segments(embodied_result)

    frame_spans: list[tuple[int, int]] = []
    previous_end_frame: int | None = None
    for segment in segments:
        start_frame = _rounded_frame(segment.start, checked_fps)
        end_frame = _rounded_frame(segment.end, checked_fps)
        if end_frame <= start_frame:
            raise ActionCaptionExportError(
                "completed embodied result frame span collapsed"
            )
        if previous_end_frame is None and start_frame != 0:
            raise ActionCaptionExportError(
                "completed embodied result frame topology is invalid"
            )
        if previous_end_frame is not None and start_frame != previous_end_frame:
            raise ActionCaptionExportError(
                "completed embodied result frame topology is invalid"
            )
        frame_spans.append((start_frame, end_frame))
        previous_end_frame = end_frame

    for ordinal, (segment, (start_frame, end_frame)) in enumerate(
        zip(segments, frame_spans, strict=True)
    ):
        caption = segment.description
        yield ActionCaption(
            caption_id=f"{checked_video_id}_cap_{ordinal:04d}",
            start_sec=segment.start,
            end_sec=segment.end,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=end_frame - start_frame,
            caption=caption,
            caption_char_count=len(caption),
            annotation_stage=_ANNOTATION_STAGE,
            source_segment_index=segment.segment_index,
            parent_source_segment_index=segment.action_index,
            needs_refinement=False,
            over_caption_char_limit=False,
            actor=segment.actor,
            actor_state=segment.actor_state,
            skill=segment.skill,
            target=segment.target,
            visual_motion_state=segment.visual_motion_state,
            confidence=segment.confidence,
            schema_parse_ok=True,
            parse_errors=(),
            needs_review=False,
            source_description=_source_description(segment),
        )


def write_action_captions_jsonl(
    path: str | os.PathLike[str], captions: Iterable[ActionCaption]
) -> None:
    """Atomically replace *path* with deterministic UTF-8 JSON Lines.

    A private temporary file is created beside the destination, so replacement
    is same-directory and atomic.  Any producer or write failure removes that
    temporary file and leaves the prior destination untouched.
    """
    destination = _destination_path(path)
    temporary_path: str | None = None
    descriptor = -1
    try:
        try:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                descriptor = -1
                for caption in captions:
                    _validate_action_caption(caption)
                    output.write(_jsonl_line(caption))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
            _fsync_directory(destination.parent)
        except OSError:
            raise ActionCaptionExportError("caption export write failed") from None
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _completed_segments(
    embodied_result: dict[str, object],
) -> tuple[_CompletedSegment, ...]:
    try:
        if type(embodied_result) is not dict:
            raise ValueError
        result_data = embodied_result.copy()
        result_keys = set(result_data)
        if not _REQUIRED_RESULT_KEYS <= result_keys or not result_keys <= (
            _REQUIRED_RESULT_KEYS | _OPTIONAL_RESULT_KEYS
        ):
            raise ValueError
        task_description = result_data["task_description"]
        raw_segments = result_data["segments"]
        if not _non_blank_string(task_description) or not isinstance(raw_segments, list):
            raise ValueError
        if not raw_segments:
            raise ValueError

        segments = tuple(_segment(value) for value in raw_segments)
        _validate_segment_order(segments)
        if "grouped_semantic_events" in result_data:
            validate_semantic_events(
                result_data["grouped_semantic_events"], raw_segments
            )
        present_scene_keys = result_keys & _SCENE_RESULT_KEYS
        if present_scene_keys and present_scene_keys != _SCENE_RESULT_KEYS:
            raise ValueError
        if present_scene_keys:
            scene = SceneSemantics.model_validate_json(
                json.dumps(
                    {key: result_data[key] for key in _SCENE_RESULT_KEYS},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                strict=True,
            )
            raw_warnings = result_data.get("warnings", [])
            scene_unavailable = isinstance(raw_warnings, list) and any(
                isinstance(warning, Mapping)
                and warning.get("code") == _SCENE_WARNING_CODE
                for warning in raw_warnings
            )
            required_targets = trusted_target_skeleton(raw_segments)
            validate_scene_semantics(
                scene,
                raw_segments[-1]["end"],
                require_observed_content=(
                    not scene_unavailable
                    and bool(required_targets)
                ),
                required_object_ids=(
                    ()
                    if scene_unavailable
                    else tuple(target["object_id"] for target in required_targets)
                ),
            )
        if "warnings" in result_data:
            _validate_normalization_warnings(
                result_data["warnings"], segments, result_data
            )
        return segments
    except Exception:
        raise ActionCaptionExportError("completed embodied result is invalid") from None


def _validate_normalization_warnings(
    value: object,
    segments: tuple[_CompletedSegment, ...],
    result: Mapping[str, object],
) -> None:
    if type(value) is not list or not 1 <= len(value) <= 3:
        raise ValueError
    codes: list[str] = []
    for warning in value:
        if not isinstance(warning, Mapping):
            raise ValueError
        code = warning.get("code")
        if type(code) is not str or code in codes:
            raise ValueError
        codes.append(code)
        if code == _NORMALIZATION_WARNING_CODE:
            _validate_enrichment_warning(warning, segments)
        elif code == _BOUNDARY_WARNING_CODE:
            _validate_boundary_warning(warning, segments)
        elif code == _SCENE_WARNING_CODE:
            if dict(warning) != {"code": _SCENE_WARNING_CODE}:
                raise ValueError
            if set(result) & _SCENE_RESULT_KEYS != _SCENE_RESULT_KEYS:
                raise ValueError
            if {
                key: result[key] for key in _SCENE_RESULT_KEYS
            } != unavailable_scene_semantics():
                raise ValueError
        else:
            raise ValueError


def _validate_enrichment_warning(
    warning: Mapping[str, object],
    segments: tuple[_CompletedSegment, ...],
) -> None:
    warning_data = dict(warning)
    if set(warning_data) != _NORMALIZATION_WARNING_KEYS:
        raise ValueError
    code = warning_data["code"]
    if type(code) is not str or code != _NORMALIZATION_WARNING_CODE:
        raise ValueError

    fields = warning_data["fields"]
    if type(fields) is not list or not fields:
        raise ValueError
    if any(type(field) is not str for field in fields):
        raise ValueError
    canonical_fields = [
        field for field in _NORMALIZABLE_ENUM_FIELDS if field in fields
    ]
    if fields != canonical_fields:
        raise ValueError

    count = warning_data["count"]
    if type(count) is not int or count <= 0:
        raise ValueError
    if not len(fields) <= count <= len(fields) * len(segments):
        raise ValueError

    unknown_counts = {
        field: sum(
            1
            for segment in segments
            if getattr(segment, field).value == "unknown"
        )
        for field in fields
    }
    if any(field_count == 0 for field_count in unknown_counts.values()):
        raise ValueError
    if count > sum(unknown_counts.values()):
        raise ValueError


def _validate_boundary_warning(
    warning: Mapping[str, object],
    segments: tuple[_CompletedSegment, ...],
) -> None:
    warning_data = dict(warning)
    if set(warning_data) != _BOUNDARY_WARNING_KEYS:
        raise ValueError
    code = warning_data["code"]
    if type(code) is not str or code != _BOUNDARY_WARNING_CODE:
        raise ValueError
    issue_codes = warning_data["issue_codes"]
    if type(issue_codes) is not list or not issue_codes:
        raise ValueError
    canonical = [
        value for value in _BOUNDARY_WARNING_ISSUE_CODES if value in issue_codes
    ]
    if issue_codes != canonical:
        raise ValueError
    count = warning_data["count"]
    if type(count) is not int or count != len(segments):
        raise ValueError


def _segment(value: object) -> _CompletedSegment:
    if not isinstance(value, Mapping) or set(value) != _SEGMENT_KEYS:
        raise ValueError
    action_index = _nonnegative_index(value["action_index"])
    segment_index = _nonnegative_index(value["segment_index"])
    start = _nonnegative_float(value["start"])
    end = _nonnegative_float(value["end"])
    if end <= start:
        raise ValueError
    description = _final_caption(value["description"])
    _enum(FineEventType, value["event_type"])
    _required_string(value["start_boundary_id"])
    _required_string(value["end_boundary_id"])
    actor = _enum(Actor, value["actor"])
    actor_state = _enum(ActorState, value["actor_state"])
    skill = _enum(Skill, value["skill"])
    target = _required_string(value["target"])
    visual_motion_state = _enum(VisualMotionState, value["visual_motion_state"])
    confidence = _finite_float(value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError
    return _CompletedSegment(
        action_index=action_index,
        segment_index=segment_index,
        start=start,
        end=end,
        description=description,
        actor=actor,
        actor_state=actor_state,
        skill=skill,
        target=target,
        visual_motion_state=visual_motion_state,
        confidence=confidence,
    )


def _validate_segment_order(segments: tuple[_CompletedSegment, ...]) -> None:
    previous: _CompletedSegment | None = None
    for segment in segments:
        if segment.end - segment.start > _MAX_FINE_SEGMENT_SECONDS + _TIME_EPSILON:
            raise ValueError
        if previous is None and segment.start != 0.0:
            raise ValueError
        if previous is not None and (
            segment.action_index < previous.action_index
            or segment.segment_index <= previous.segment_index
            or segment.start != previous.end
        ):
            raise ValueError
        previous = segment


def _video_id(value: object) -> str:
    if not _non_blank_string(value):
        raise ActionCaptionExportError("video_id is invalid")
    return value


def _source_fps(value: object) -> float:
    try:
        rate = _finite_float(value)
    except Exception:
        raise ActionCaptionExportError("source_fps is invalid") from None
    if rate <= 0:
        raise ActionCaptionExportError("source_fps is invalid")
    return rate


def _destination_path(value: str | os.PathLike[str]) -> Path:
    try:
        raw_path = os.fspath(value)
        if type(raw_path) is not str or not raw_path:
            raise ValueError
        return Path(raw_path)
    except Exception:
        raise ActionCaptionExportError("destination path is invalid") from None


def _non_blank_string(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _required_string(value: object) -> str:
    if not _non_blank_string(value):
        raise ValueError
    assert isinstance(value, str)
    return value


def _nonnegative_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError
    return value


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError
    return result


def _nonnegative_float(value: object) -> float:
    result = _finite_float(value)
    if result < 0:
        raise ValueError
    return result


def _enum(enum_type: type[_StrEnum], value: object) -> _StrEnum:
    if type(value) is not str:
        raise ValueError
    return enum_type(value)


def _final_caption(value: object) -> str:
    caption = _required_string(value)
    words = caption.split()
    allowed_subjects = ("left hand", "right hand", "both hands", "neither hand")
    if (
        caption != caption.lower()
        or not 2 <= len(words) <= 10
        or len(caption) > _MAX_CAPTION_CHARACTERS
        or not any(caption.startswith(subject + " ") for subject in allowed_subjects)
    ):
        raise ValueError
    return caption


def _rounded_frame(seconds: float, source_fps: float) -> int:
    try:
        scaled = seconds * source_fps
        if not math.isfinite(scaled):
            raise ValueError
        return round(scaled)
    except Exception:
        raise ActionCaptionExportError(
            "completed embodied result frame span is invalid"
        ) from None


def _validate_action_caption(caption: object) -> None:
    """Reject forged or internally inconsistent rows before they reach JSON."""
    try:
        if type(caption) is not ActionCaption:
            raise ValueError
        if (
            type(caption.caption_id) is not str
            or _CAPTION_ID.fullmatch(caption.caption_id) is None
            or type(caption.start_sec) is not float
            or type(caption.end_sec) is not float
            or type(caption.start_frame) is not int
            or type(caption.end_frame) is not int
            or type(caption.duration_frames) is not int
            or type(caption.caption) is not str
            or type(caption.caption_char_count) is not int
            or type(caption.annotation_stage) is not str
            or type(caption.source_segment_index) is not int
            or type(caption.parent_source_segment_index) is not int
            or type(caption.needs_refinement) is not bool
            or type(caption.over_caption_char_limit) is not bool
            or type(caption.actor) is not Actor
            or type(caption.actor_state) is not ActorState
            or type(caption.skill) is not Skill
            or type(caption.target) is not str
            or type(caption.visual_motion_state) is not VisualMotionState
            or type(caption.confidence) is not float
            or type(caption.schema_parse_ok) is not bool
            or type(caption.parse_errors) is not tuple
            or type(caption.needs_review) is not bool
            or type(caption.source_description) is not str
        ):
            raise ValueError
        if (
            caption.annotation_stage != _ANNOTATION_STAGE
            or not _non_blank_string(caption.caption)
            or not _non_blank_string(caption.target)
            or caption.caption_char_count != len(caption.caption)
            or caption.source_segment_index < 0
            or caption.parent_source_segment_index < 0
            or caption.start_frame < 0
            or caption.end_frame <= caption.start_frame
            or caption.duration_frames != caption.end_frame - caption.start_frame
            or not math.isfinite(caption.start_sec)
            or not math.isfinite(caption.end_sec)
            or not math.isfinite(caption.confidence)
            or caption.start_sec < 0
            or caption.end_sec <= caption.start_sec
            or not 0.0 <= caption.confidence <= 1.0
            or any(type(error) is not str or not error for error in caption.parse_errors)
        ):
            raise ValueError
        _final_caption(caption.caption)

        needs_refinement = (
            caption.end_sec - caption.start_sec
            > _MAX_FINE_SEGMENT_SECONDS + _TIME_EPSILON
        )
        over_caption_limit = len(caption.caption) > _MAX_CAPTION_CHARACTERS
        if (
            caption.needs_refinement != needs_refinement
            or caption.over_caption_char_limit != over_caption_limit
            or (caption.schema_parse_ok and caption.parse_errors)
            or (not caption.schema_parse_ok and not caption.parse_errors)
            or caption.needs_review
            != (not caption.schema_parse_ok or needs_refinement or over_caption_limit)
        ):
            raise ValueError
        expected_description = _source_description_from_values(
            actor=caption.actor,
            actor_state=caption.actor_state,
            skill=caption.skill,
            target=caption.target,
            visual_motion_state=caption.visual_motion_state,
            action=caption.caption,
            confidence=caption.confidence,
        )
        if caption.source_description != expected_description:
            raise ValueError
    except ActionCaptionExportError:
        raise
    except Exception:
        raise ActionCaptionExportError("action caption is invalid") from None


def _source_description(segment: _CompletedSegment) -> str:
    return _source_description_from_values(
        actor=segment.actor,
        actor_state=segment.actor_state,
        skill=segment.skill,
        target=segment.target,
        visual_motion_state=segment.visual_motion_state,
        action=segment.description,
        confidence=segment.confidence,
    )


def _source_description_from_values(
    *,
    actor: Actor,
    actor_state: ActorState,
    skill: Skill,
    target: str,
    visual_motion_state: VisualMotionState,
    action: str,
    confidence: float,
) -> str:
    return " | ".join(
        (
            f"actor={actor.value}",
            f"actor_state={actor_state.value}",
            f"skill={skill.value}",
            f"target={target}",
            f"visual_motion_state={visual_motion_state.value}",
            f"action={action}",
            f"confidence={confidence}",
        )
    )


def _jsonl_line(caption: ActionCaption) -> str:
    return json.dumps(
        caption.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _fsync_directory(directory: Path) -> None:
    """Sync the containing directory after replacement when that is supported.

    If this raises, ``os.replace`` has already succeeded: callers must treat the
    new destination as present but its directory-entry durability as uncertain.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if _directory_sync_unsupported(error):
            return
        raise ActionCaptionExportError(
            "caption export durability is uncertain after atomic replacement"
        ) from None
    try:
        os.fsync(descriptor)
    except OSError as error:
        if not _directory_sync_unsupported(error):
            raise ActionCaptionExportError(
                "caption export durability is uncertain after atomic replacement"
            ) from None
    finally:
        os.close(descriptor)


def _directory_sync_unsupported(error: OSError) -> bool:
    return error.errno in _UNSUPPORTED_DIRECTORY_SYNC_ERRNOS
