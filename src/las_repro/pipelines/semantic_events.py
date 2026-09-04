"""Deterministic projection from strict fine segments to longer action events."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


_ACTOR_PROJECTION = {
    "left_hand": "left_hand", "left_gripper": "left_hand",
    "right_hand": "right_hand", "right_gripper": "right_hand",
    "both_hands": "both_hands", "both_grippers": "both_hands",
    "robot_arm": "robot_arm", "unknown": "unknown",
}
_ACTION_PROJECTION = {
    "grasp": "grasp", "pick": "grasp",
    "move": "motion", "lift": "motion", "push": "motion",
    "pull": "motion", "rotate": "motion", "place": "motion",
    "reach": "reach", "release": "release", "hold": "hold",
    "touch": "contact", "open": "open", "close": "close",
    "retract": "retract", "unknown": "unknown",
}
_EVENT_KEYS = {
    "event_index", "start", "end", "actor", "action", "target",
    "description", "confidence", "source_segment_indices",
}


def build_semantic_events(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group adjacent segments with the same broad actor/action/target key."""
    checked = [
        _checked_segment(segment, position)
        for position, segment in enumerate(segments)
    ]
    if not checked:
        return []
    for previous, current in zip(checked, checked[1:], strict=False):
        if current["segment_index"] != previous["segment_index"] + 1:
            raise ValueError("semantic event segment indices must be contiguous")
        if current["start"] != previous["end"]:
            raise ValueError("semantic event segment times must be contiguous")

    groups: list[list[dict[str, Any]]] = []
    for segment in checked:
        key = (segment["actor"], segment["action"], segment["target_key"])
        if groups and groups[-1][0]["group_key"] == key:
            groups[-1].append(segment)
        else:
            segment["group_key"] = key
            groups.append([segment])

    events: list[dict[str, Any]] = []
    for event_index, group in enumerate(groups):
        first = group[0]
        descriptions = list(dict.fromkeys(item["description"] for item in group))
        duration = sum(item["end"] - item["start"] for item in group)
        confidence_values = {item["confidence"] for item in group}
        confidence = (
            first["confidence"]
            if len(confidence_values) == 1
            else math.fsum(
                item["confidence"] * (item["end"] - item["start"])
                for item in group
            )
            / duration
        )
        events.append(
            {
                "event_index": event_index,
                "start": first["start"],
                "end": group[-1]["end"],
                "actor": first["actor"],
                "action": first["action"],
                "target": first["target"],
                "description": "; ".join(descriptions),
                "confidence": confidence,
                "source_segment_indices": [item["segment_index"] for item in group],
            }
        )
    return events


def validate_semantic_events(
    events: object,
    segments: Sequence[Mapping[str, Any]],
) -> None:
    """Require the exact deterministic projection at serialization boundaries."""
    if type(events) is not list or any(
        not isinstance(event, Mapping) or set(event) != _EVENT_KEYS
        for event in events
    ):
        raise ValueError("semantic events are invalid")
    if events != build_semantic_events(segments):
        raise ValueError("semantic events do not match source segments")


def _checked_segment(value: Mapping[str, Any], position: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("semantic event segment must be an object")
    index = value.get("segment_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("semantic event segment_index is invalid")
    if position and index < 1:
        raise ValueError("semantic event segment indices must be contiguous")
    start = _finite_real(value.get("start"), "start")
    end = _finite_real(value.get("end"), "end")
    if end <= start:
        raise ValueError("semantic event segment duration is invalid")
    confidence = _finite_real(value.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("semantic event confidence is invalid")
    actor = value.get("actor")
    skill = value.get("skill")
    if type(actor) is not str or actor not in _ACTOR_PROJECTION:
        raise ValueError("semantic event actor is invalid")
    if type(skill) is not str or skill not in _ACTION_PROJECTION:
        raise ValueError("semantic event skill is invalid")
    target = value.get("target")
    description = value.get("description")
    if type(target) is not str or not target.strip():
        raise ValueError("semantic event target is invalid")
    if type(description) is not str or not description.strip():
        raise ValueError("semantic event description is invalid")
    return {
        "segment_index": index,
        "start": start,
        "end": end,
        "actor": _ACTOR_PROJECTION[actor],
        "action": _ACTION_PROJECTION[skill],
        "target": target,
        "target_key": " ".join(target.casefold().split()),
        "description": description,
        "confidence": confidence,
        "group_key": None,
    }


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"semantic event {name} is invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"semantic event {name} is invalid")
    return number
