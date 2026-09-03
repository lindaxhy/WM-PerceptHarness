"""Deterministic, offline visual-model fixture for local tests."""

from __future__ import annotations

import copy
import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from .base import ModelOutputError, ModelRequest, ModelRequestError, VideoModel


class UnknownModelStageError(ModelRequestError):
    """The requested stage has no deterministic fake fixture."""


ScriptedResult = Mapping[str, Any] | BaseException


class FakeVideoModel(VideoModel):
    """Return stable visual-only fixtures without loading weights or using a GPU.

    ``failure_script`` is a per-stage FIFO.  Each scripted response is consumed
    exactly once, allowing repair-path tests to provide an invalid structured
    response followed by a valid one.  Script inputs and returned mappings are
    copied so callers cannot alter later results by mutation.
    """

    _STAGES = frozenset(
        {
            "general_segment",
            "active_objects",
            "embodied_pass_a",
            "embodied_pass_b",
            "embodied_enrichment",
            "general_summary",
        }
    )

    def __init__(
        self,
        *,
        failure_script: Mapping[str, Sequence[ScriptedResult]] | None = None,
    ) -> None:
        self._failure_script = {
            stage: deque(copy.deepcopy(list(results)))
            for stage, results in (failure_script or {}).items()
        }

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        """Return the next scripted result or the stable fixture for ``stage``."""
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        if request.stage not in self._STAGES:
            raise UnknownModelStageError(
                f"unsupported fake model stage: {request.stage}"
            )

        scripted = self._failure_script.get(request.stage)
        if scripted:
            result = scripted.popleft()
            if isinstance(result, BaseException):
                raise result
            if not isinstance(result, Mapping):
                raise ModelOutputError("scripted fake result must be a JSON object")
            return copy.deepcopy(dict(result))

        return copy.deepcopy(self._fixture(request))

    def _fixture(self, request: ModelRequest) -> dict[str, Any]:
        if request.stage == "general_segment":
            return {
                "segments": [
                    _general_event(request.span.start, request.span.end)
                ],
                "warnings": [],
            }
        if request.stage == "active_objects":
            return {
                "objects": [
                    {
                        "category": "container",
                        "instances": [
                            {
                                "name": "red container",
                                "description": "a visible red rectangular container",
                            }
                        ],
                    }
                ]
            }
        if request.stage == "embodied_pass_a":
            return {
                "task_description": "move the red container",
                "actions": _coarse_actions(request.span.start, request.span.end),
            }
        if request.stage == "embodied_pass_b":
            return {
                "task_description": "move the red container",
                "actions": _boundary_actions(
                    request.span.start,
                    request.span.end,
                    requirements=_pass_b_requirements(request.prompt),
                ),
            }
        if request.stage == "embodied_enrichment":
            return {
                "segments": [
                    {
                        "segment_index": index,
                        "actor": "right_gripper",
                        "actor_state": "holding",
                        "skill": "move",
                        "target": "red container",
                        "visual_motion_state": "active",
                        "confidence": 0.9,
                    }
                    for index in _requested_segment_indexes(request.prompt)
                ]
            }
        if request.stage == "general_summary":
            return {
                "summary": "deterministic video summary",
                "timeline": _summary_timeline(request),
                "warnings": [],
            }
        raise AssertionError("validated fake stage had no fixture")


def _general_event(start: float, end: float) -> dict[str, Any]:
    return {
        "start_time": start,
        "end_time": end,
        "scene": ["deterministic indoor scene"],
        "subjects": ["deterministic visible subject"],
        "actions": ["deterministic visible action"],
        "visible_text": [],
        "uncertainty": [],
        "description": "deterministic visual event",
        "warnings": [],
    }


def _summary_timeline(request: ModelRequest) -> list[dict[str, Any]]:
    try:
        prompt = json.loads(request.prompt)
    except json.JSONDecodeError:
        prompt = None
    if isinstance(prompt, dict) and isinstance(prompt.get("timeline"), list):
        return prompt["timeline"]
    return [_general_event(request.span.start, request.span.end)]


def _coarse_actions(start: float, end: float) -> list[dict[str, Any]]:
    midpoint = (start + end) / 2
    return [
        {
            "action_index": 0,
            "start": start,
            "end": midpoint,
            "description": "right hand reaches toward red container",
            "event_type": "reach_and_grasp",
        },
        {
            "action_index": 1,
            "start": midpoint,
            "end": end,
            "description": "right hand moves red container",
            "event_type": "transport",
        },
    ]


def _boundary_actions(
    start: float,
    end: float,
    *,
    requirements: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    segment_index = 0
    for coarse in _coarse_actions(start, end):
        required_boundaries = _required_fake_boundaries(
            requirements,
            coarse["action_index"],
        )
        if required_boundaries is None:
            pieces = _split_at_visible_changes(
                coarse["start"],
                coarse["end"],
                phase=coarse["action_index"],
            )
            boundary_ids = [
                f"a{coarse['action_index']}_b{position}"
                for position in range(len(pieces) + 1)
            ]
            boundary_times = [piece[0] for piece in pieces] + [pieces[-1][1]]
        else:
            boundary_ids, boundary_times = required_boundaries
            pieces = list(zip(boundary_times[:-1], boundary_times[1:], strict=True))
        boundary_points = [
            {
                "boundary_id": boundary_ids[ordinal],
                "time": timestamp,
                "event_type": (
                    "action_start"
                    if ordinal == 0
                    else "action_end"
                    if ordinal == len(boundary_times) - 1
                    else "transport_continue"
                ),
                "visual_evidence": "the visible action state changes",
            }
            for ordinal, timestamp in enumerate(boundary_times)
        ]
        fine_segments = []
        for ordinal, (fine_start, fine_end) in enumerate(pieces):
            fine_segments.append(
                {
                    "segment_index": segment_index,
                    "start": fine_start,
                    "end": fine_end,
                    "description": "right hand moves red container",
                    "event_type": "transport_continue",
                    "start_boundary_id": boundary_points[ordinal]["boundary_id"],
                    "end_boundary_id": boundary_points[ordinal + 1]["boundary_id"],
                }
            )
            segment_index += 1
        actions.append(
            {
                **coarse,
                "boundary_points": boundary_points,
                "fine_segments": fine_segments,
            }
        )
    return actions


def _pass_b_requirements(prompt: str) -> dict[int, Mapping[str, Any]] | None:
    try:
        section = prompt.split(
            "[trusted fine-segmentation requirements JSON data]\n",
            1,
        )[1].split("\n\n", 1)[0]
        value = json.loads(section.splitlines()[-1])
        actions = value["actions"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(actions, list):
        return None
    return {
        action["action_index"]: action
        for action in actions
        if isinstance(action, Mapping)
        and isinstance(action.get("action_index"), int)
        and not isinstance(action.get("action_index"), bool)
    }


def _required_fake_boundaries(
    requirements: Mapping[int, Mapping[str, Any]] | None,
    action_index: int,
) -> tuple[list[str], list[float]] | None:
    if requirements is None:
        return None
    requirement = requirements.get(action_index)
    if requirement is None:
        return None
    slots = requirement.get("boundary_slots")
    if not isinstance(slots, list) or len(slots) < 2:
        return None

    boundary_ids: list[str] = []
    boundary_times: list[float] = []
    ratios = (0.31, 0.67, 0.43, 0.79)
    for position, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            return None
        boundary_id = slot.get("boundary_id")
        window = slot.get("inclusive_time_window")
        if not isinstance(boundary_id, str) or not isinstance(window, Mapping):
            return None
        minimum = window.get("minimum_seconds")
        maximum = window.get("maximum_seconds")
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or minimum > maximum
        ):
            return None
        ratio = ratios[(position - 1) % len(ratios)]
        timestamp = float(minimum + (maximum - minimum) * ratio)
        timestamp = min(max(timestamp, float(minimum)), float(maximum))
        boundary_ids.append(boundary_id)
        boundary_times.append(timestamp)
    return boundary_ids, boundary_times


def _split_at_visible_changes(
    start: float,
    end: float,
    *,
    phase: int,
) -> list[tuple[float, float]]:
    """Return a deterministic non-uniform fixture whose intervals stay <= 1s."""
    duration = end - start
    if duration <= 1.0:
        ratio = 0.41 if phase % 2 == 0 else 0.63
        middle = start + duration * ratio
        if middle <= start or middle >= end:
            return [(start, end)]
        return [(start, middle), (middle, end)]

    step_pattern = (0.73, 0.61, 0.89)
    pieces: list[tuple[float, float]] = []
    cursor = start
    pattern_index = phase % len(step_pattern)
    while end - cursor > 1.0:
        next_cursor = cursor + step_pattern[pattern_index]
        pieces.append((cursor, next_cursor))
        cursor = next_cursor
        pattern_index = (pattern_index + 1) % len(step_pattern)
    pieces.append((cursor, end))
    return pieces


def _requested_segment_indexes(prompt: str) -> list[int]:
    """Prefer the trusted skeleton, with legacy discovery only when it is absent."""
    trusted = _trusted_enrichment_indexes(prompt)
    if trusted is not None:
        return trusted

    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        parsed = None

    indexes: list[int] = []
    if parsed is not None:
        _collect_segment_indexes(parsed, indexes)
    if not indexes:
        indexes.extend(int(match) for match in re.findall(r'"segment_index"\s*:\s*(\d+)', prompt))

    seen: set[int] = set()
    return [index for index in indexes if not (index in seen or seen.add(index))]


def _trusted_enrichment_indexes(prompt: str) -> list[int] | None:
    header = "[trusted enrichment output requirements JSON data]\n"
    if header not in prompt:
        return None

    try:
        if prompt.count(header) != 1:
            raise ValueError
        section = prompt.split(header, 1)[1].split("\n\n", 1)[0]
        json_lines = [line for line in section.splitlines() if line.startswith("{")]
        if len(json_lines) != 1:
            raise ValueError
        requirements = json.loads(json_lines[0])
        if not isinstance(requirements, dict) or set(requirements) != {
            "exact_record_count",
            "expected_indices",
            "record_skeleton",
        }:
            raise ValueError
        count = requirements["exact_record_count"]
        expected = requirements["expected_indices"]
        skeleton = requirements["record_skeleton"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(expected, list)
            or not isinstance(skeleton, list)
            or len(expected) != count
            or len(skeleton) != count
        ):
            raise ValueError

        previous: int | None = None
        for index, record in zip(expected, skeleton, strict=True):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or (previous is not None and index <= previous)
                or not _is_conservative_enrichment_record(record, index)
            ):
                raise ValueError
            previous = index
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ModelRequestError("fake enrichment requirements are invalid") from None
    return list(expected)


def _is_conservative_enrichment_record(value: Any, index: int) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "segment_index",
            "actor",
            "actor_state",
            "skill",
            "target",
            "visual_motion_state",
            "confidence",
        }
        and type(value["segment_index"]) is int
        and value["segment_index"] == index
        and value["actor"] == "unknown"
        and value["actor_state"] == "unknown"
        and value["skill"] == "unknown"
        and value["target"] == "unknown"
        and value["visual_motion_state"] == "unknown"
        and type(value["confidence"]) is float
        and value["confidence"] == 0.0
    )


def _collect_segment_indexes(value: Any, indexes: list[int]) -> None:
    if isinstance(value, dict):
        index = value.get("segment_index")
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
            indexes.append(index)
        for child in value.values():
            _collect_segment_indexes(child, indexes)
    elif isinstance(value, list):
        for child in value:
            _collect_segment_indexes(child, indexes)
