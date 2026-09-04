"""Strict schemas and non-mutating temporal checks for embodied annotations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Iterable, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator


Timestamp: TypeAlias = Annotated[
    float, Field(ge=0, strict=True, allow_inf_nan=False)
]
NonnegativeIndex: TypeAlias = Annotated[int, Field(ge=0, strict=True)]
Confidence: TypeAlias = Annotated[
    float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)
]

# Arithmetic operations on decimal JSON timestamps can produce a tiny binary tail.
# This epsilon is only numerical hygiene; it is not a semantic timing tolerance.
_ARITHMETIC_EPSILON = 1e-9


class Actor(StrEnum):
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    BOTH_HANDS = "both_hands"
    LEFT_GRIPPER = "left_gripper"
    RIGHT_GRIPPER = "right_gripper"
    BOTH_GRIPPERS = "both_grippers"
    ROBOT_ARM = "robot_arm"
    UNKNOWN = "unknown"


class ActorState(StrEnum):
    IDLE = "idle"
    REACHING = "reaching"
    CONTACTING = "contacting"
    GRASPING = "grasping"
    HOLDING = "holding"
    TRANSPORTING = "transporting"
    PLACING = "placing"
    RELEASING = "releasing"
    RETRACTING = "retracting"
    UNKNOWN = "unknown"


class Skill(StrEnum):
    HOLD = "hold"
    REACH = "reach"
    GRASP = "grasp"
    PICK = "pick"
    LIFT = "lift"
    MOVE = "move"
    PLACE = "place"
    RELEASE = "release"
    PUSH = "push"
    PULL = "pull"
    ROTATE = "rotate"
    OPEN = "open"
    CLOSE = "close"
    RETRACT = "retract"
    TOUCH = "touch"
    UNKNOWN = "unknown"


class VisualMotionState(StrEnum):
    STATIC = "static"
    LOW = "low"
    ACTIVE = "active"
    UNKNOWN = "unknown"


class CoarseEventType(StrEnum):
    IDLE = "idle"
    REACH_AND_GRASP = "reach_and_grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    LOWER_AND_PLACE = "lower_and_place"
    RELEASE = "release"
    RETRACT = "retract"
    SEARCH_OR_ADJUST = "search_or_adjust"
    UNKNOWN_ACTION = "unknown_action"


class FineEventType(StrEnum):
    ACTION_START = "action_start"
    IDLE = "idle"
    REACH_START = "reach_start"
    APPROACH = "approach"
    PRE_CONTACT = "pre_contact"
    CONTACT_START = "contact_start"
    GRASP_SECURED = "grasp_secured"
    LIFT_START = "lift_start"
    LIFT_CONTINUE = "lift_continue"
    TRANSPORT_START = "transport_start"
    TRANSPORT_CONTINUE = "transport_continue"
    LOWER_START = "lower_start"
    DESTINATION_CONTACT = "destination_contact"
    PLACE_CONTINUE = "place_continue"
    RELEASE_START = "release_start"
    RELEASE_END = "release_end"
    RETRACT_START = "retract_start"
    ACTION_END = "action_end"
    UNKNOWN_TRANSITION = "unknown_transition"


class SchemaModel(BaseModel):
    """Models must reject fields outside the current 0805 output schemas."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def reject_blank_strings(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        return value


class ObjectInstance(SchemaModel):
    name: str
    description: str


class ObjectGroup(SchemaModel):
    category: str
    instances: list[ObjectInstance]


class ObjectInventory(SchemaModel):
    objects: list[ObjectGroup]


class CoarseAction(SchemaModel):
    action_index: NonnegativeIndex
    start: Timestamp
    end: Timestamp
    description: str
    event_type: CoarseEventType


class CoarsePlan(SchemaModel):
    task_description: str
    actions: list[CoarseAction]


class BoundaryPoint(SchemaModel):
    boundary_id: str
    time: Timestamp
    event_type: FineEventType
    visual_evidence: str


class FineSegment(SchemaModel):
    segment_index: NonnegativeIndex
    start: Timestamp
    end: Timestamp
    description: str
    event_type: FineEventType
    start_boundary_id: str
    end_boundary_id: str


class BoundaryAction(CoarseAction):
    boundary_points: list[BoundaryPoint]
    fine_segments: list[FineSegment]


class BoundaryPlan(SchemaModel):
    task_description: str
    actions: list[BoundaryAction]


class EnrichmentSegment(SchemaModel):
    """The only six model-generated fields after local temporal finalization."""

    segment_index: NonnegativeIndex
    actor: Actor
    actor_state: ActorState
    skill: Skill
    target: str
    visual_motion_state: VisualMotionState
    confidence: Confidence


class EnrichmentResult(SchemaModel):
    segments: list[EnrichmentSegment]


@dataclass(frozen=True)
class TemporalIssue:
    """A deterministic, machine-readable validation finding for one field."""

    code: str
    path: tuple[str | int, ...]
    message: str

    @property
    def location(self) -> tuple[str | int, ...]:
        """Alias for callers that use Pydantic-style location terminology."""
        return self.path


class TemporalValidationError(ValueError):
    """Raised after collecting all independently discoverable temporal issues."""

    def __init__(self, issues: Iterable[TemporalIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(
            "temporal validation failed: "
            + "; ".join(f"{issue.code} at {issue.path}" for issue in self.issues)
        )


def validate_coarse_plan(
    plan: CoarsePlan,
    duration: float,
    tolerance: float = _ARITHMETIC_EPSILON,
) -> None:
    """Validate exact topology, allowing only capped floating-point noise."""
    _require_nonnegative_finite("duration", duration)
    _require_nonnegative_finite("tolerance", tolerance)
    comparison_epsilon = min(tolerance, _ARITHMETIC_EPSILON)
    issues: list[TemporalIssue] = []
    actions = plan.actions
    if not actions:
        _issue(issues, "EMPTY_ACTIONS", ("actions",), "at least one action is required")
    else:
        if not _equal(actions[0].start, 0.0, comparison_epsilon):
            _issue(
                issues,
                "ACTION_START_NOT_ZERO",
                ("actions", 0, "start"),
                "first action must start at 0",
            )
        _validate_action_indices(actions, issues)
        for index, action in enumerate(actions):
            if not _strictly_before(action.start, action.end):
                _issue(
                    issues,
                    "ACTION_NONPOSITIVE_DURATION",
                    ("actions", index),
                    "action end must be greater than start",
                )
            if index:
                _validate_adjacency(
                    issues,
                    previous_end=actions[index - 1].end,
                    current_start=action.start,
                    tolerance=comparison_epsilon,
                    path=("actions", index, "start"),
                    gap_code="ACTION_GAP",
                    overlap_code="ACTION_OVERLAP",
                    noun="actions",
                )
        if not _equal(actions[-1].end, duration, comparison_epsilon):
            _issue(
                issues,
                "ACTION_END_MISMATCH_DURATION",
                ("actions", len(actions) - 1, "end"),
                "last action must end at the video duration",
            )
    _raise_if_any(issues)


def validate_boundary_plan(
    plan: BoundaryPlan,
    coarse: CoarsePlan,
    max_segment_seconds: float = 1.0,
    tolerance: float = _ARITHMETIC_EPSILON,
) -> None:
    """Validate exact Pass-B topology while preserving all supplied values."""
    _require_positive_finite("max_segment_seconds", max_segment_seconds)
    _require_nonnegative_finite("tolerance", tolerance)
    comparison_epsilon = min(tolerance, _ARITHMETIC_EPSILON)
    issues: list[TemporalIssue] = []
    if plan.task_description != coarse.task_description:
        _issue(
            issues,
            "TASK_DESCRIPTION_MISMATCH",
            ("task_description",),
            "Pass B task_description must equal Pass A task_description",
        )

    actions = plan.actions
    if not actions:
        _issue(issues, "EMPTY_ACTIONS", ("actions",), "at least one action is required")

    _validate_action_indices(actions, issues)
    _validate_action_topology(actions, comparison_epsilon, issues)
    coarse_by_index = {action.action_index: action for action in coarse.actions}
    seen_boundary_ids: dict[str, tuple[str | int, ...]] = {}
    previous_segment_index: int | None = None
    seen_segment_indexes: set[int] = set()

    for action_position, action in enumerate(actions):
        action_path = ("actions", action_position)
        parent = coarse_by_index.get(action.action_index)
        if parent is None:
            _issue(
                issues,
                "UNKNOWN_COARSE_ACTION",
                action_path + ("action_index",),
                "Pass B action_index does not resolve to Pass A",
            )
        else:
            _validate_parent_copy(
                action,
                parent,
                action_path,
                comparison_epsilon,
                issues,
            )
        _validate_boundary_points(
            action,
            action_path,
            comparison_epsilon,
            seen_boundary_ids,
            issues,
        )
        previous_segment_index = _validate_fine_segments(
            action,
            action_path,
            max_segment_seconds,
            comparison_epsilon,
            previous_segment_index,
            seen_segment_indexes,
            issues,
        )

    plan_indexes = {action.action_index for action in actions}
    for coarse_action in coarse.actions:
        if coarse_action.action_index not in plan_indexes:
            _issue(
                issues,
                "MISSING_COARSE_ACTION",
                ("actions",),
                f"Pass B is missing Pass A action_index {coarse_action.action_index}",
            )
    _raise_if_any(issues)


def validate_enrichment(
    result: EnrichmentResult, expected_indices: Iterable[int]
) -> None:
    """Require one ordered six-field enrichment record for each fine segment."""
    expected = _expected_index_set(expected_indices)
    issues: list[TemporalIssue] = []
    actual: set[int] = set()
    previous: int | None = None
    for position, segment in enumerate(result.segments):
        index = segment.segment_index
        path = ("segments", position, "segment_index")
        if index in actual:
            _issue(
                issues,
                "DUPLICATE_ENRICHMENT_INDEX",
                path,
                f"segment_index {index} appears more than once",
            )
        if previous is not None and index <= previous:
            _issue(
                issues,
                "ENRICHMENT_INDEX_NOT_ORDERED",
                path,
                "segment indexes must be strictly increasing",
            )
        actual.add(index)
        previous = index

    missing = expected - actual
    unexpected = actual - expected
    if missing:
        _issue(
            issues,
            "MISSING_ENRICHMENT_INDEX",
            ("segments",),
            "missing enrichment indices: " + ", ".join(map(str, sorted(missing))),
        )
    if unexpected:
        _issue(
            issues,
            "UNEXPECTED_ENRICHMENT_INDEX",
            ("segments",),
            "unexpected enrichment indices: "
            + ", ".join(map(str, sorted(unexpected))),
        )
    _raise_if_any(issues)


def _validate_action_indices(
    actions: list[CoarseAction] | list[BoundaryAction], issues: list[TemporalIssue]
) -> None:
    seen: set[int] = set()
    previous: int | None = None
    for position, action in enumerate(actions):
        path = ("actions", position, "action_index")
        if action.action_index in seen:
            _issue(
                issues,
                "DUPLICATE_ACTION_INDEX",
                path,
                f"action_index {action.action_index} appears more than once",
            )
        if previous is not None and action.action_index <= previous:
            _issue(
                issues,
                "ACTION_INDEX_NOT_ORDERED",
                path,
                "action indexes must be strictly increasing",
            )
        seen.add(action.action_index)
        previous = action.action_index


def _validate_action_topology(
    actions: list[BoundaryAction], tolerance: float, issues: list[TemporalIssue]
) -> None:
    """Check Pass B action chronology independently of tolerant parent copying."""
    for position, action in enumerate(actions):
        if not _strictly_before(action.start, action.end):
            _issue(
                issues,
                "ACTION_NONPOSITIVE_DURATION",
                ("actions", position),
                "action end must be greater than start",
            )
        if position:
            _validate_adjacency(
                issues,
                previous_end=actions[position - 1].end,
                current_start=action.start,
                tolerance=tolerance,
                path=("actions", position, "start"),
                gap_code="ACTION_GAP",
                overlap_code="ACTION_OVERLAP",
                noun="actions",
            )


def _validate_parent_copy(
    action: BoundaryAction,
    parent: CoarseAction,
    action_path: tuple[str | int, ...],
    tolerance: float,
    issues: list[TemporalIssue],
) -> None:
    if not _equal(action.start, parent.start, tolerance):
        _issue(
            issues,
            "PARENT_START_MISMATCH",
            action_path + ("start",),
            "Pass B action start must equal its Pass A parent",
        )
    if not _equal(action.end, parent.end, tolerance):
        _issue(
            issues,
            "PARENT_END_MISMATCH",
            action_path + ("end",),
            "Pass B action end must equal its Pass A parent",
        )
    if action.description != parent.description:
        _issue(
            issues,
            "PARENT_DESCRIPTION_MISMATCH",
            action_path + ("description",),
            "Pass B action description must equal its Pass A parent",
        )
    if action.event_type != parent.event_type:
        _issue(
            issues,
            "PARENT_EVENT_TYPE_MISMATCH",
            action_path + ("event_type",),
            "Pass B action event_type must equal its Pass A parent",
        )


def _validate_boundary_points(
    action: BoundaryAction,
    action_path: tuple[str | int, ...],
    tolerance: float,
    seen_ids: dict[str, tuple[str | int, ...]],
    issues: list[TemporalIssue],
) -> None:
    points = action.boundary_points
    if not points:
        _issue(
            issues,
            "MISSING_BOUNDARY_POINTS",
            action_path + ("boundary_points",),
            "each action needs boundary_points",
        )
        return
    if not _equal(points[0].time, action.start, tolerance):
        _issue(
            issues,
            "BOUNDARY_START_MISMATCH",
            action_path + ("boundary_points", 0, "time"),
            "first boundary point must equal the parent action start",
        )
    if not _equal(points[-1].time, action.end, tolerance):
        _issue(
            issues,
            "BOUNDARY_END_MISMATCH",
            action_path + ("boundary_points", len(points) - 1, "time"),
            "last boundary point must equal the parent action end",
        )
    previous: float | None = None
    for position, point in enumerate(points):
        path = action_path + ("boundary_points", position)
        if point.boundary_id in seen_ids:
            _issue(
                issues,
                "DUPLICATE_BOUNDARY_ID",
                path + ("boundary_id",),
                f"boundary_id {point.boundary_id!r} is not unique",
            )
        else:
            seen_ids[point.boundary_id] = path
        if point.time < action.start - tolerance or point.time > action.end + tolerance:
            _issue(
                issues,
                "BOUNDARY_OUTSIDE_PARENT",
                path + ("time",),
                "boundary point must be inside its parent action",
            )
        if previous is not None and point.time < previous - tolerance:
            _issue(
                issues,
                "BOUNDARY_NOT_ORDERED",
                path + ("time",),
                "boundary points must be chronological",
            )
        previous = point.time


def _validate_fine_segments(
    action: BoundaryAction,
    action_path: tuple[str | int, ...],
    max_segment_seconds: float,
    tolerance: float,
    previous_global_index: int | None,
    seen_indexes: set[int],
    issues: list[TemporalIssue],
) -> int | None:
    segments = action.fine_segments
    if not segments:
        _issue(
            issues,
            "MISSING_FINE_SEGMENTS",
            action_path + ("fine_segments",),
            "each action needs fine_segments",
        )
        return previous_global_index
    boundaries = {point.boundary_id: point.time for point in action.boundary_points}
    boundary_positions = {
        point.boundary_id: position
        for position, point in enumerate(action.boundary_points)
    }
    if not _equal(segments[0].start, action.start, tolerance):
        _issue(
            issues,
            "SEGMENT_START_MISMATCH_PARENT",
            action_path + ("fine_segments", 0, "start"),
            "first fine segment must start at the parent action start",
        )
    if not _equal(segments[-1].end, action.end, tolerance):
        _issue(
            issues,
            "SEGMENT_END_MISMATCH_PARENT",
            action_path + ("fine_segments", len(segments) - 1, "end"),
            "last fine segment must end at the parent action end",
        )
    previous_end: float | None = None
    previous_index = previous_global_index
    previous_end_boundary_id: str | None = None
    for position, segment in enumerate(segments):
        path = action_path + ("fine_segments", position)
        if not valid_fine_description(segment.description):
            _issue(
                issues,
                "SEGMENT_DESCRIPTION_INVALID",
                path + ("description",),
                "fine segment description violates the public caption contract",
            )
        if segment.segment_index in seen_indexes:
            _issue(
                issues,
                "DUPLICATE_SEGMENT_INDEX",
                path + ("segment_index",),
                f"segment_index {segment.segment_index} appears more than once",
            )
        if previous_index is not None and segment.segment_index <= previous_index:
            _issue(
                issues,
                "SEGMENT_INDEX_NOT_ORDERED",
                path + ("segment_index",),
                "fine segment indexes must be strictly increasing",
            )
        expected_index = 0 if previous_index is None else previous_index + 1
        if segment.segment_index != expected_index:
            _issue(
                issues,
                "SEGMENT_INDEX_NOT_CONTIGUOUS",
                path + ("segment_index",),
                "fine segment indexes must start at 0 and increase by exactly 1",
            )
        seen_indexes.add(segment.segment_index)
        previous_index = segment.segment_index
        if not _strictly_before(segment.start, segment.end):
            _issue(
                issues,
                "SEGMENT_NONPOSITIVE_DURATION",
                path,
                "fine segment end must be greater than start",
            )
        if segment.start < action.start - tolerance or segment.end > action.end + tolerance:
            _issue(
                issues,
                "SEGMENT_OUTSIDE_PARENT",
                path,
                "fine segment must be inside its parent action",
            )
        if segment.end - segment.start > max_segment_seconds + _ARITHMETIC_EPSILON:
            _issue(
                issues,
                "SEGMENT_TOO_LONG",
                path,
                "fine segment exceeds max_segment_seconds",
            )
        if previous_end is not None:
            _validate_adjacency(
                issues,
                previous_end=previous_end,
                current_start=segment.start,
                tolerance=tolerance,
                path=path + ("start",),
                gap_code="SEGMENT_GAP",
                overlap_code="SEGMENT_OVERLAP",
                noun="fine segments",
            )
        _validate_boundary_reference(
            boundaries,
            segment.start_boundary_id,
            segment.start,
            path + ("start_boundary_id",),
            tolerance,
            issues,
        )
        _validate_boundary_reference(
            boundaries,
            segment.end_boundary_id,
            segment.end,
            path + ("end_boundary_id",),
            tolerance,
            issues,
        )
        _validate_adjacent_boundary_references(
            boundary_positions,
            segment.start_boundary_id,
            segment.end_boundary_id,
            path,
            issues,
        )
        if (
            previous_end_boundary_id is not None
            and segment.start_boundary_id != previous_end_boundary_id
        ):
            _issue(
                issues,
                "SEGMENT_BOUNDARY_DISCONTINUITY",
                path + ("start_boundary_id",),
                "successive fine segments must share their boundary ID",
            )
        previous_end = segment.end
        previous_end_boundary_id = segment.end_boundary_id
    return previous_index


def valid_fine_description(value: str) -> bool:
    """Return whether a caption satisfies the shared prompt/export contract."""
    words = value.split()
    subjects = ("left hand", "right hand", "both hands", "neither hand")
    return (
        value == value.lower()
        and 2 <= len(words) <= 10
        and len(value) <= 60
        and any(value.startswith(subject + " ") for subject in subjects)
    )


def _validate_boundary_reference(
    boundaries: dict[str, float],
    boundary_id: str,
    segment_time: float,
    path: tuple[str | int, ...],
    tolerance: float,
    issues: list[TemporalIssue],
) -> None:
    boundary_time = boundaries.get(boundary_id)
    if boundary_time is None:
        _issue(
            issues,
            "UNKNOWN_BOUNDARY_ID",
            path,
            f"boundary_id {boundary_id!r} does not resolve inside this parent action",
        )
    elif not _equal(boundary_time, segment_time, tolerance):
        _issue(
            issues,
            "BOUNDARY_TIME_MISMATCH",
            path,
            "fine segment time must equal its referenced boundary point time",
        )


def _validate_adjacent_boundary_references(
    positions: dict[str, int],
    start_boundary_id: str,
    end_boundary_id: str,
    path: tuple[str | int, ...],
    issues: list[TemporalIssue],
) -> None:
    """Ensure a fine segment is derived from one adjacent boundary pair."""
    start_position = positions.get(start_boundary_id)
    end_position = positions.get(end_boundary_id)
    if (
        start_position is not None
        and end_position is not None
        and end_position != start_position + 1
    ):
        _issue(
            issues,
            "SEGMENT_BOUNDARY_NOT_ADJACENT",
            path,
            "fine segment boundaries must be adjacent points in the parent action",
        )


def _validate_adjacency(
    issues: list[TemporalIssue],
    *,
    previous_end: float,
    current_start: float,
    tolerance: float,
    path: tuple[str | int, ...],
    gap_code: str,
    overlap_code: str,
    noun: str,
) -> None:
    if current_start > previous_end + tolerance:
        _issue(issues, gap_code, path, f"{noun} must be adjacent without a gap")
    elif current_start < previous_end - tolerance:
        _issue(issues, overlap_code, path, f"{noun} must not overlap")


def _expected_index_set(expected_indices: Iterable[int]) -> set[int]:
    expected: set[int] = set()
    for index in expected_indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("expected_indices must contain non-negative integer indexes")
        expected.add(index)
    return expected


def _equal(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def _strictly_before(start: float, end: float) -> bool:
    return end > start


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_finite(name: str, value: float) -> None:
    _require_nonnegative_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _issue(
    issues: list[TemporalIssue], code: str, path: tuple[str | int, ...], message: str
) -> None:
    issues.append(TemporalIssue(code=code, path=path, message=message))


def _raise_if_any(issues: list[TemporalIssue]) -> None:
    if issues:
        raise TemporalValidationError(issues)
