"""Strict LAS-aligned scene facts and overlapping semantic events."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .validators import Actor, TemporalIssue, TemporalValidationError


Timestamp = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
Confidence = Annotated[
    float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)
]
Index = Annotated[int, Field(ge=0, strict=True)]
ObjectId = Annotated[
    str, Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$", strict=True)
]


class SceneEventType(StrEnum):
    MOVE = "move"
    TRANSPORT = "transport"
    GRASP = "grasp"
    REACH = "reach"
    RELEASE = "release"
    LIFT = "lift"
    PLACE = "place"
    APPROACH = "approach"
    CONTACT = "contact"
    PUSH = "push"
    PULL = "pull"
    ROTATE = "rotate"
    STOP = "stop"
    AUTONOMOUS_MOTION = "autonomous_motion"
    STATE_CHANGE = "state_change"
    OCCLUSION_ENTER = "occlusion_enter"
    OCCLUDED = "occluded"
    OCCLUSION_EXIT = "occlusion_exit"
    UNKNOWN = "unknown"


class OutcomeStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class _SceneModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def reject_blank_strings(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        return value


class SceneObject(_SceneModel):
    object_id: ObjectId
    name: str
    description: str


class SceneState(_SceneModel):
    object_id: ObjectId
    state: str
    visual_evidence: str
    confidence: Confidence


class SceneOutcome(_SceneModel):
    status: OutcomeStatus
    description: str
    confidence: Confidence


class SceneSemanticEvent(_SceneModel):
    event_index: Index
    start: Timestamp
    end: Timestamp
    event_type: SceneEventType
    actor: Actor
    target_object_id: ObjectId
    description: str
    confidence: Confidence


class SceneSemantics(_SceneModel):
    objects: list[SceneObject]
    initial_state: list[SceneState]
    final_state: list[SceneState]
    outcome: SceneOutcome
    semantic_events: list[SceneSemanticEvent]


def trusted_target_skeleton(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Create stable ASCII IDs for concrete targets in a trusted fine track."""
    names: list[str] = []
    seen_names: set[str] = set()
    for segment in segments:
        target = segment.get("target")
        if not isinstance(target, str):
            raise ValueError("scene segment target is invalid")
        name = " ".join(target.strip().split())
        key = name.casefold()
        if not name or key in {"unknown", "none"} or key in seen_names:
            continue
        names.append(name)
        seen_names.add(key)
    result: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for index, name in enumerate(names):
        base = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
        base = base or f"target_{index}"
        object_id = base
        suffix = 2
        while object_id in used_ids:
            object_id = f"{base}_{suffix}"
            suffix += 1
        used_ids.add(object_id)
        result.append({"object_id": object_id, "name": name})
    return result


def validate_scene_semantics(
    result: SceneSemantics,
    duration: float,
    *,
    require_observed_content: bool = False,
    required_object_ids: tuple[str, ...] = (),
) -> None:
    """Validate references and time bounds while allowing event overlap."""
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("duration must be finite and positive")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not isinstance(require_observed_content, bool):
        raise ValueError("require_observed_content must be a boolean")
    if (
        not isinstance(required_object_ids, tuple)
        or len(set(required_object_ids)) != len(required_object_ids)
        or any(not isinstance(value, str) or not value for value in required_object_ids)
    ):
        raise ValueError("required_object_ids must be unique non-blank strings")
    issues: list[TemporalIssue] = []
    if require_observed_content and not result.objects:
        issues.append(
            TemporalIssue(
                "EMPTY_SCENE_OBJECTS",
                ("objects",),
                "an action video with known targets must declare scene objects",
            )
        )
    if require_observed_content and not result.semantic_events:
        issues.append(
            TemporalIssue(
                "EMPTY_SCENE_EVENTS",
                ("semantic_events",),
                "an action video with known targets must declare semantic events",
            )
        )
    object_ids: set[str] = set()
    for index, item in enumerate(result.objects):
        if item.object_id in object_ids:
            issues.append(
                TemporalIssue(
                    "SCENE_OBJECT_ID_DUPLICATE",
                    ("objects", index, "object_id"),
                    "scene object IDs must be unique",
                )
            )
        object_ids.add(item.object_id)
    for object_id in required_object_ids:
        if object_id not in object_ids:
            issues.append(
                TemporalIssue(
                    "SCENE_REQUIRED_OBJECT_MISSING",
                    ("objects",),
                    "scene objects must retain every trusted target object ID",
                )
            )
    for collection_name, states in (
        ("initial_state", result.initial_state),
        ("final_state", result.final_state),
    ):
        seen: set[str] = set()
        for index, state in enumerate(states):
            if state.object_id not in object_ids:
                issues.append(
                    TemporalIssue(
                        "SCENE_STATE_UNKNOWN_OBJECT",
                        (collection_name, index, "object_id"),
                        "scene state must reference a declared object",
                    )
                )
            if state.object_id in seen:
                issues.append(
                    TemporalIssue(
                        "SCENE_STATE_DUPLICATE_OBJECT",
                        (collection_name, index, "object_id"),
                        "each object has at most one state record",
                    )
                )
            seen.add(state.object_id)
    previous_start: float | None = None
    for index, event in enumerate(result.semantic_events):
        path = ("semantic_events", index)
        if event.event_index != index:
            issues.append(
                TemporalIssue(
                    "SCENE_EVENT_INDEX_NOT_CONTIGUOUS",
                    path + ("event_index",),
                    "scene event indexes must start at zero and be contiguous",
                )
            )
        if not event.start < event.end:
            issues.append(
                TemporalIssue(
                    "SCENE_EVENT_NONPOSITIVE_DURATION",
                    path,
                    "scene events must have positive duration",
                )
            )
        if event.end > duration:
            issues.append(
                TemporalIssue(
                    "SCENE_EVENT_OUTSIDE_VIDEO",
                    path,
                    "scene event must remain within the probed video",
                )
            )
        if previous_start is not None and event.start < previous_start:
            issues.append(
                TemporalIssue(
                    "SCENE_EVENT_START_NOT_ORDERED",
                    path + ("start",),
                    "scene events must be ordered by start time",
                )
            )
        if event.target_object_id != "unknown" and event.target_object_id not in object_ids:
            issues.append(
                TemporalIssue(
                    "SCENE_EVENT_UNKNOWN_OBJECT",
                    path + ("target_object_id",),
                    "scene event must reference a declared object or unknown",
                )
            )
        previous_start = event.start
    if issues:
        raise TemporalValidationError(issues)


def unavailable_scene_semantics() -> dict[str, object]:
    """Return an explicit conservative result after schema repair exhaustion."""
    return {
        "objects": [],
        "initial_state": [],
        "final_state": [],
        "outcome": {
            "status": "unknown",
            "description": "scene semantics unavailable",
            "confidence": 0.0,
        },
        "semantic_events": [],
    }
