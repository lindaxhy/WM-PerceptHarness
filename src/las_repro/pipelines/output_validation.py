"""Pre-persistence validation for model outputs with declared local schemas."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from .scene_semantics import SceneSemantics, validate_scene_semantics
from .validators import (
    BoundaryPlan,
    CoarsePlan,
    EnrichmentResult,
    ObjectInventory,
    TemporalValidationError,
    validate_boundary_plan,
    validate_coarse_plan,
    validate_enrichment,
    valid_fine_description,
)


SCHEMA_VALIDATION_FIELD = "_schema_validation"
_INVENTORY_GENERIC_REPAIR_CODE = "OBJECT_INVENTORY_SCHEMA_INVALID"
_INVENTORY_REPAIR_CODE_BY_ERROR_TYPE = (
    ("missing", "OBJECT_INVENTORY_MISSING_FIELD"),
    ("extra_forbidden", "OBJECT_INVENTORY_EXTRA_FIELD"),
    ("list_type", "OBJECT_INVENTORY_LIST_TYPE"),
    ("model_type", "OBJECT_INVENTORY_OBJECT_TYPE"),
    ("dict_type", "OBJECT_INVENTORY_OBJECT_TYPE"),
    ("string_type", "OBJECT_INVENTORY_STRING_TYPE"),
    ("value_error", "OBJECT_INVENTORY_BLANK_STRING"),
)
_INVENTORY_REPAIR_CODES = tuple(
    dict.fromkeys(
        code for _, code in _INVENTORY_REPAIR_CODE_BY_ERROR_TYPE
    )
) + (_INVENTORY_GENERIC_REPAIR_CODE,)

_SCHEMA_ERROR_SUFFIX_BY_ERROR_TYPE = (
    ("missing", "MISSING_FIELD"),
    ("extra_forbidden", "EXTRA_FIELD"),
    ("list_type", "LIST_TYPE"),
    ("model_type", "OBJECT_TYPE"),
    ("dict_type", "OBJECT_TYPE"),
    ("string_type", "STRING_TYPE"),
    ("float_type", "NUMBER_TYPE"),
    ("int_type", "NUMBER_TYPE"),
    ("finite_number", "NUMBER_TYPE"),
    ("greater_than_equal", "NUMBER_RANGE"),
    ("less_than_equal", "NUMBER_RANGE"),
    ("enum", "ENUM_VALUE"),
    ("value_error", "BLANK_STRING"),
)

_COARSE_TEMPORAL_CODES = (
    "EMPTY_ACTIONS",
    "ACTION_START_NOT_ZERO",
    "DUPLICATE_ACTION_INDEX",
    "ACTION_INDEX_NOT_ORDERED",
    "ACTION_NONPOSITIVE_DURATION",
    "ACTION_GAP",
    "ACTION_OVERLAP",
    "ACTION_END_MISMATCH_DURATION",
)
_BOUNDARY_TEMPORAL_CODES = (
    "TASK_DESCRIPTION_MISMATCH",
    "EMPTY_ACTIONS",
    "DUPLICATE_ACTION_INDEX",
    "ACTION_INDEX_NOT_ORDERED",
    "ACTION_NONPOSITIVE_DURATION",
    "ACTION_GAP",
    "ACTION_OVERLAP",
    "UNKNOWN_COARSE_ACTION",
    "PARENT_START_MISMATCH",
    "PARENT_END_MISMATCH",
    "PARENT_DESCRIPTION_MISMATCH",
    "PARENT_EVENT_TYPE_MISMATCH",
    "MISSING_BOUNDARY_POINTS",
    "BOUNDARY_START_MISMATCH",
    "BOUNDARY_END_MISMATCH",
    "DUPLICATE_BOUNDARY_ID",
    "BOUNDARY_OUTSIDE_PARENT",
    "BOUNDARY_NOT_ORDERED",
    "MISSING_FINE_SEGMENTS",
    "SEGMENT_START_MISMATCH_PARENT",
    "SEGMENT_END_MISMATCH_PARENT",
    "DUPLICATE_SEGMENT_INDEX",
    "SEGMENT_INDEX_NOT_ORDERED",
    "SEGMENT_INDEX_NOT_CONTIGUOUS",
    "SEGMENT_NONPOSITIVE_DURATION",
    "SEGMENT_OUTSIDE_PARENT",
    "SEGMENT_TOO_LONG",
    "SEGMENT_GAP",
    "SEGMENT_OVERLAP",
    "UNKNOWN_BOUNDARY_ID",
    "BOUNDARY_TIME_MISMATCH",
    "SEGMENT_BOUNDARY_NOT_ADJACENT",
    "SEGMENT_BOUNDARY_DISCONTINUITY",
    "SEGMENT_DESCRIPTION_INVALID",
    "MISSING_COARSE_ACTION",
)
_ENRICHMENT_TEMPORAL_CODES = (
    "DUPLICATE_ENRICHMENT_INDEX",
    "ENRICHMENT_INDEX_NOT_ORDERED",
    "MISSING_ENRICHMENT_INDEX",
    "UNEXPECTED_ENRICHMENT_INDEX",
)
_SCENE_TEMPORAL_CODES = (
    "EMPTY_SCENE_OBJECTS",
    "EMPTY_SCENE_EVENTS",
    "SCENE_REQUIRED_OBJECT_MISSING",
    "SCENE_OBJECT_ID_DUPLICATE",
    "SCENE_STATE_UNKNOWN_OBJECT",
    "SCENE_STATE_DUPLICATE_OBJECT",
    "SCENE_EVENT_INDEX_NOT_CONTIGUOUS",
    "SCENE_EVENT_NONPOSITIVE_DURATION",
    "SCENE_EVENT_OUTSIDE_VIDEO",
    "SCENE_EVENT_START_NOT_ORDERED",
    "SCENE_EVENT_UNKNOWN_OBJECT",
)
_ENRICHMENT_ENUM_FIELDS = (
    "actor",
    "actor_state",
    "skill",
    "visual_motion_state",
)
_ENRICHMENT_ENUM_FIELD_CODES = {
    "actor": "ENRICHMENT_RESULT_ACTOR_ENUM_VALUE",
    "actor_state": "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE",
    "skill": "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
    "visual_motion_state": "ENRICHMENT_RESULT_VISUAL_MOTION_STATE_ENUM_VALUE",
}
_BOUNDARY_NORMALIZABLE_CODES = (
    "SEGMENT_TOO_LONG",
    "SEGMENT_BOUNDARY_NOT_ADJACENT",
    "SEGMENT_DESCRIPTION_INVALID",
)
_GENERAL_SEGMENT_TEMPORAL_CODES = (
    "GENERAL_SEGMENT_TIMESTAMP_OUT_OF_SPAN",
)
_GENERAL_SUMMARY_TEMPORAL_CODES = (
    "GENERAL_SUMMARY_TIMELINE_OUT_OF_SPAN",
    "GENERAL_SUMMARY_TIMELINE_ORDER",
    "GENERAL_SUMMARY_TIMELINE_MISMATCH",
)


class DeclaredSchemaOutputError(ValueError):
    """A declared output schema rejected model data before persistence."""

    def __init__(self, issue_codes: tuple[str, ...]) -> None:
        self.issue_codes = issue_codes
        super().__init__("declared model output schema is invalid")


@dataclass(frozen=True)
class NormalizedSchemaOutput:
    """Canonical data and bounded audit metadata from a trusted normalization."""

    data: dict[str, Any]
    issue_codes: tuple[str, ...]
    normalized_field_count: int


OutputValidator = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None],
    dict[str, Any] | NormalizedSchemaOutput,
]


@dataclass(frozen=True)
class _SchemaEntry:
    validator: OutputValidator
    allowed_issue_codes: tuple[str, ...]
    generic_issue_code: str
    preserve_issue_code_order: bool


class OutputSchemaRegistry:
    """Validate known schemas and pass unknown schemas through unchanged.

    Validators are registered here so invalid raw model data is replaced before
    it reaches durable storage. Pipeline modules are imported lazily by the two
    general validators to avoid a worker/orchestrator import cycle.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _SchemaEntry] = {}

    def register(
        self,
        schema_name: str,
        validator: OutputValidator,
        *,
        allowed_issue_codes: tuple[str, ...],
        generic_issue_code: str,
        preserve_issue_code_order: bool = False,
    ) -> None:
        if not isinstance(schema_name, str) or not schema_name.strip():
            raise ValueError("schema_name must be a non-blank string")
        if schema_name in self._entries:
            raise ValueError(f"output schema {schema_name!r} is already registered")
        if not callable(validator):
            raise TypeError("output schema validator must be callable")
        if (
            not allowed_issue_codes
            or len(set(allowed_issue_codes)) != len(allowed_issue_codes)
            or any(
                not isinstance(code, str) or not code.strip()
                for code in allowed_issue_codes
            )
            or generic_issue_code not in allowed_issue_codes
        ):
            raise ValueError("allowed issue codes must be unique non-blank strings")
        if not isinstance(preserve_issue_code_order, bool):
            raise TypeError("preserve_issue_code_order must be a boolean")
        self._entries[schema_name] = _SchemaEntry(
            validator=validator,
            allowed_issue_codes=allowed_issue_codes,
            generic_issue_code=generic_issue_code,
            preserve_issue_code_order=preserve_issue_code_order,
        )

    def sanitize(
        self,
        schema_name: str,
        result: Mapping[str, Any],
        validation_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return canonical valid data or a closed, model-free failure envelope."""
        entry = self._entries.get(schema_name)
        if entry is None:
            return dict(result)
        try:
            validated = entry.validator(result, validation_context)
            if isinstance(validated, NormalizedSchemaOutput):
                return {
                    SCHEMA_VALIDATION_FIELD: {
                        "schema_name": schema_name,
                        "status": "normalized",
                        "issue_codes": list(validated.issue_codes),
                        "normalized_field_count": (
                            validated.normalized_field_count
                        ),
                    },
                    "data": validated.data,
                }
            return validated
        except DeclaredSchemaOutputError as error:
            if entry.preserve_issue_code_order:
                issue_codes = list(
                    dict.fromkeys(
                        code
                        for code in error.issue_codes
                        if code in entry.allowed_issue_codes
                    )
                )
            else:
                requested = set(error.issue_codes)
                issue_codes = [
                    code for code in entry.allowed_issue_codes if code in requested
                ]
            if not issue_codes:
                issue_codes = [entry.generic_issue_code]
            return {
                SCHEMA_VALIDATION_FIELD: {
                    "schema_name": schema_name,
                    "status": "invalid",
                    "issue_codes": issue_codes,
                }
            }

    def normalized_result(
        self,
        schema_name: str,
        result: Mapping[str, Any],
        validation_context: Mapping[str, Any] | None = None,
    ) -> NormalizedSchemaOutput | None:
        """Parse and independently revalidate an exact normalized envelope."""
        entry = self._entries.get(schema_name)
        if entry is None:
            return None
        if schema_name == "BoundaryPlan":
            return _normalized_boundary_envelope(
                entry.validator,
                result,
                validation_context,
            )
        if schema_name != "EnrichmentResult":
            return None
        try:
            if set(result) != {SCHEMA_VALIDATION_FIELD, "data"}:
                return None
            envelope = result.get(SCHEMA_VALIDATION_FIELD)
            data = result.get("data")
            if not isinstance(envelope, Mapping) or set(envelope) != {
                "schema_name",
                "status",
                "issue_codes",
                "normalized_field_count",
            }:
                return None
            codes = envelope.get("issue_codes")
            count = envelope.get("normalized_field_count")
            if (
                envelope.get("schema_name") != schema_name
                or envelope.get("status") != "normalized"
                or not isinstance(codes, list)
                or not codes
                or any(not isinstance(code, str) for code in codes)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or not isinstance(data, Mapping)
            ):
                return None
            requested_codes = set(codes)
            normalized_fields = tuple(
                field
                for field in _ENRICHMENT_ENUM_FIELDS
                if _ENRICHMENT_ENUM_FIELD_CODES[field] in requested_codes
            )
            canonical_codes = tuple(
                _ENRICHMENT_ENUM_FIELD_CODES[field]
                for field in normalized_fields
            )
            if list(canonical_codes) != codes or count < len(canonical_codes):
                return None
            context = _enrichment_validation_context(validation_context)
            if context["allow_enum_unknown_fallback"] is not True:
                return None
            strict_context = {
                "expected_indices": context["expected_indices"],
                "allow_enum_unknown_fallback": False,
            }
            canonical = entry.validator(data, strict_context)
            if not isinstance(canonical, dict):
                return None
            segments = canonical.get("segments")
            if not isinstance(segments, list):
                return None
            unknown_counts = tuple(
                sum(segment[field] == "unknown" for segment in segments)
                for field in normalized_fields
            )
            if any(field_count == 0 for field_count in unknown_counts) or count > sum(
                unknown_counts
            ):
                return None
            return NormalizedSchemaOutput(
                data=canonical,
                issue_codes=canonical_codes,
                normalized_field_count=count,
            )
        except Exception:
            return None

    def failure_codes(
        self, schema_name: str, result: Mapping[str, Any]
    ) -> tuple[str, ...] | None:
        """Read only an exact trusted envelope emitted by this registry."""
        entry = self._entries.get(schema_name)
        if entry is None or set(result) != {SCHEMA_VALIDATION_FIELD}:
            return None
        envelope = result.get(SCHEMA_VALIDATION_FIELD)
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "schema_name",
            "status",
            "issue_codes",
        }:
            return None
        codes = envelope.get("issue_codes")
        if (
            envelope.get("schema_name") != schema_name
            or envelope.get("status") != "invalid"
            or not isinstance(codes, list)
            or not codes
            or any(not isinstance(code, str) for code in codes)
        ):
            return None
        if entry.preserve_issue_code_order:
            if len(set(codes)) != len(codes) or any(
                code not in entry.allowed_issue_codes for code in codes
            ):
                return None
            return tuple(codes)
        canonical = tuple(
            code for code in entry.allowed_issue_codes if code in set(codes)
        )
        if list(canonical) != codes:
            return None
        return canonical

    def model_output_failure(self, schema_name: str) -> dict[str, Any] | None:
        """Return a generic closed envelope for a typed parser/output failure."""
        entry = self._entries.get(schema_name)
        if entry is None:
            return None
        return {
            SCHEMA_VALIDATION_FIELD: {
                "schema_name": schema_name,
                "status": "invalid",
                "issue_codes": [entry.generic_issue_code],
            }
        }


def _validate_object_inventory(
    result: Mapping[str, Any], validation_context: Mapping[str, Any] | None
) -> dict[str, Any]:
    if validation_context is not None:
        raise ValueError("ObjectInventory validation context is invalid")
    try:
        inventory = _model_from_json(ObjectInventory, result)
    except ValidationError as error:
        raise DeclaredSchemaOutputError(_inventory_issue_codes(error)) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError((_INVENTORY_GENERIC_REPAIR_CODE,)) from None
    return inventory.model_dump(mode="json")


def _inventory_issue_codes(error: ValidationError) -> tuple[str, ...]:
    """Map Pydantic types to constants without retaining model keys or values."""
    error_types = {
        str(issue["type"])
        for issue in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    }
    codes: list[str] = []
    for error_type, code in _INVENTORY_REPAIR_CODE_BY_ERROR_TYPE:
        if error_type in error_types and code not in codes:
            codes.append(code)
    known_types = {
        error_type for error_type, _ in _INVENTORY_REPAIR_CODE_BY_ERROR_TYPE
    }
    if error_types - known_types:
        codes.append(_INVENTORY_GENERIC_REPAIR_CODE)
    return tuple(codes or [_INVENTORY_GENERIC_REPAIR_CODE])


def _schema_codes(prefix: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"{prefix}_{suffix}"
            for _, suffix in _SCHEMA_ERROR_SUFFIX_BY_ERROR_TYPE
        )
    ) + (f"{prefix}_SCHEMA_INVALID",)


def _pydantic_issue_codes(error: ValidationError, prefix: str) -> tuple[str, ...]:
    error_types = {
        str(issue["type"])
        for issue in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    }
    codes: list[str] = []
    for error_type, suffix in _SCHEMA_ERROR_SUFFIX_BY_ERROR_TYPE:
        code = f"{prefix}_{suffix}"
        if error_type in error_types and code not in codes:
            codes.append(code)
    known_types = {
        error_type for error_type, _ in _SCHEMA_ERROR_SUFFIX_BY_ERROR_TYPE
    }
    if error_types - known_types:
        codes.append(f"{prefix}_SCHEMA_INVALID")
    return tuple(codes or [f"{prefix}_SCHEMA_INVALID"])


def _enrichment_pydantic_issue_codes(error: ValidationError) -> tuple[str, ...]:
    """Return closed enrichment diagnostics without retaining Pydantic locations."""
    codes: list[str] = []
    for issue in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        error_type = str(issue["type"])
        if error_type == "enum":
            location = issue.get("loc")
            field = (
                location[-1]
                if isinstance(location, (tuple, list)) and location
                else None
            )
            code = (
                _ENRICHMENT_ENUM_FIELD_CODES.get(field)
                if isinstance(field, str)
                else None
            )
            code = code or "ENRICHMENT_RESULT_ENUM_VALUE"
        else:
            code = _pydantic_issue_codes_for_type(
                error_type, "ENRICHMENT_RESULT"
            )
        if code not in codes:
            codes.append(code)
    return tuple(codes or ["ENRICHMENT_RESULT_SCHEMA_INVALID"])


def _pydantic_issue_codes_for_type(error_type: str, prefix: str) -> str:
    """Map one Pydantic type through the shared closed-code policy."""
    for known_type, suffix in _SCHEMA_ERROR_SUFFIX_BY_ERROR_TYPE:
        if error_type == known_type:
            return f"{prefix}_{suffix}"
    return f"{prefix}_SCHEMA_INVALID"


def _validate_coarse_output(
    result: Mapping[str, Any], validation_context: Mapping[str, Any] | None
) -> dict[str, Any]:
    duration = _context_finite_number(validation_context, "duration", positive=True)
    try:
        plan = _model_from_json(CoarsePlan, result)
    except ValidationError as error:
        raise DeclaredSchemaOutputError(
            _pydantic_issue_codes(error, "COARSE_PLAN")
        ) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("COARSE_PLAN_SCHEMA_INVALID",)) from None
    try:
        validate_coarse_plan(plan, duration)
    except TemporalValidationError as error:
        raise DeclaredSchemaOutputError(
            tuple(dict.fromkeys(issue.code for issue in error.issues))
        ) from None
    return plan.model_dump(mode="json")


def _validate_boundary_output(
    result: Mapping[str, Any], validation_context: Mapping[str, Any] | None
) -> dict[str, Any] | NormalizedSchemaOutput:
    context = _boundary_validation_context(validation_context)
    coarse_value = context["coarse_plan"]
    if not isinstance(coarse_value, Mapping):
        raise ValueError("BoundaryPlan validation context is invalid")
    try:
        coarse = _model_from_json(CoarsePlan, coarse_value)
    except ValidationError:
        raise ValueError("BoundaryPlan validation context is invalid") from None
    maximum = _finite_real(context["max_segment_seconds"], positive=True)
    allow_fallback = context["allow_topology_fallback"]
    try:
        snapshot = _finite_json_snapshot(result)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("BOUNDARY_PLAN_SCHEMA_INVALID",)) from None
    try:
        plan = _model_from_json(BoundaryPlan, snapshot)
    except ValidationError as error:
        raise DeclaredSchemaOutputError(
            _pydantic_issue_codes(error, "BOUNDARY_PLAN")
        ) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("BOUNDARY_PLAN_SCHEMA_INVALID",)) from None
    try:
        validate_boundary_plan(plan, coarse, max_segment_seconds=maximum)
    except TemporalValidationError as error:
        issue_codes = tuple(dict.fromkeys(issue.code for issue in error.issues))
        if allow_fallback and set(issue_codes) <= set(_BOUNDARY_NORMALIZABLE_CODES):
            normalized = _normalize_boundary_topology(
                plan,
                coarse,
                maximum,
                issue_codes,
            )
            if normalized is not None:
                return normalized
        raise DeclaredSchemaOutputError(
            issue_codes
        ) from None
    return plan.model_dump(mode="json")


def _boundary_validation_context(
    validation_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Parse the strict boundary context while retaining old strict callers."""
    if not isinstance(validation_context, Mapping):
        raise ValueError("BoundaryPlan validation context is invalid")
    keys = set(validation_context)
    legacy = {"coarse_plan", "max_segment_seconds"}
    current = legacy | {"allow_topology_fallback"}
    if keys not in (legacy, current):
        raise ValueError("BoundaryPlan validation context is invalid")
    enabled = validation_context.get("allow_topology_fallback", False)
    if not isinstance(enabled, bool):
        raise ValueError("BoundaryPlan validation context is invalid")
    return {
        "coarse_plan": validation_context["coarse_plan"],
        "max_segment_seconds": validation_context["max_segment_seconds"],
        "allow_topology_fallback": enabled,
    }


def _normalize_boundary_topology(
    plan: BoundaryPlan,
    coarse: CoarsePlan,
    maximum: float,
    issue_codes: tuple[str, ...],
) -> NormalizedSchemaOutput | None:
    """Rebuild references from trusted ordered points after the sole repair."""
    try:
        actions: list[dict[str, Any]] = []
        next_segment_index = 0
        total_segments = 0
        for action in plan.actions:
            original_points = [point.model_dump(mode="json") for point in action.boundary_points]
            original_segments = [
                segment.model_dump(mode="json") for segment in action.fine_segments
            ]
            points: list[dict[str, Any]] = []
            existing_ids = {point["boundary_id"] for point in original_points}
            for point_index, (left, right) in enumerate(
                zip(original_points, original_points[1:], strict=False)
            ):
                points.append(left)
                duration = right["time"] - left["time"]
                piece_count = _minimum_safe_piece_count(
                    left["time"], right["time"], maximum
                )
                if piece_count > 10_000:
                    return None
                for piece_index in range(1, piece_count):
                    boundary_id = (
                        f"local_a{action.action_index}_p{point_index}_s{piece_index}"
                    )
                    if boundary_id in existing_ids:
                        return None
                    existing_ids.add(boundary_id)
                    points.append(
                        {
                            "boundary_id": boundary_id,
                            "time": left["time"]
                            + duration * piece_index / piece_count,
                            "event_type": "unknown_transition",
                            "visual_evidence": "local hard-cap subdivision",
                        }
                    )
            points.append(original_points[-1])
            if len(points) > 10_001:
                return None

            fine_segments: list[dict[str, Any]] = []
            for left, right in zip(points, points[1:], strict=False):
                source = _maximum_overlap_segment(
                    original_segments,
                    left["time"],
                    right["time"],
                )
                if source is None:
                    return None
                description = source["description"]
                if not valid_fine_description(description):
                    description = action.description
                if not valid_fine_description(description):
                    return None
                fine_segments.append(
                    {
                        "segment_index": next_segment_index,
                        "start": left["time"],
                        "end": right["time"],
                        "description": description,
                        "event_type": source["event_type"],
                        "start_boundary_id": left["boundary_id"],
                        "end_boundary_id": right["boundary_id"],
                    }
                )
                next_segment_index += 1
            total_segments += len(fine_segments)
            actions.append(
                {
                    "action_index": action.action_index,
                    "start": action.start,
                    "end": action.end,
                    "description": action.description,
                    "event_type": action.event_type.value,
                    "boundary_points": points,
                    "fine_segments": fine_segments,
                }
            )

        normalized = BoundaryPlan.model_validate(
            {"task_description": plan.task_description, "actions": actions}
        )
        validate_boundary_plan(normalized, coarse, max_segment_seconds=maximum)
    except (ValidationError, TemporalValidationError, ValueError, TypeError, OverflowError):
        return None
    return NormalizedSchemaOutput(
        data=normalized.model_dump(mode="json"),
        issue_codes=tuple(
            code for code in _BOUNDARY_NORMALIZABLE_CODES if code in issue_codes
        ),
        normalized_field_count=total_segments,
    )


def _maximum_overlap_segment(
    segments: list[dict[str, Any]], start: float, end: float
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_overlap = 0.0
    for segment in segments:
        overlap = min(end, segment["end"]) - max(start, segment["start"])
        if overlap > best_overlap:
            best = segment
            best_overlap = overlap
    return best


def _minimum_safe_piece_count(start: float, end: float, maximum: float) -> int:
    """Use the fewest equal pieces whose materialized floats pass the hard cap."""
    duration = end - start
    piece_count = max(1, math.ceil(duration / maximum))
    while piece_count <= 10_000:
        times = [
            start + duration * index / piece_count
            for index in range(piece_count + 1)
        ]
        if all(
            right - left <= maximum
            for left, right in zip(times, times[1:], strict=False)
        ):
            return piece_count
        piece_count += 1
    return piece_count


def _normalized_boundary_envelope(
    validator: OutputValidator,
    result: Mapping[str, Any],
    validation_context: Mapping[str, Any] | None,
) -> NormalizedSchemaOutput | None:
    """Revalidate an exact repair-only BoundaryPlan normalization envelope."""
    try:
        if set(result) != {SCHEMA_VALIDATION_FIELD, "data"}:
            return None
        envelope = result.get(SCHEMA_VALIDATION_FIELD)
        data = result.get("data")
        if not isinstance(envelope, Mapping) or set(envelope) != {
            "schema_name",
            "status",
            "issue_codes",
            "normalized_field_count",
        }:
            return None
        codes = envelope.get("issue_codes")
        count = envelope.get("normalized_field_count")
        if (
            envelope.get("schema_name") != "BoundaryPlan"
            or envelope.get("status") != "normalized"
            or not isinstance(codes, list)
            or not codes
            or codes
            != [code for code in _BOUNDARY_NORMALIZABLE_CODES if code in codes]
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(data, Mapping)
        ):
            return None
        context = _boundary_validation_context(validation_context)
        if context["allow_topology_fallback"] is not True:
            return None
        strict_context = {
            "coarse_plan": context["coarse_plan"],
            "max_segment_seconds": context["max_segment_seconds"],
            "allow_topology_fallback": False,
        }
        canonical = validator(data, strict_context)
        if not isinstance(canonical, dict):
            return None
        actions = canonical.get("actions")
        if not isinstance(actions, list):
            return None
        segment_count = sum(
            len(action["fine_segments"])
            for action in actions
            if isinstance(action, dict) and isinstance(action.get("fine_segments"), list)
        )
        if segment_count != count:
            return None
        return NormalizedSchemaOutput(
            data=canonical,
            issue_codes=tuple(codes),
            normalized_field_count=count,
        )
    except Exception:
        return None


def _validate_enrichment_output(
    result: Mapping[str, Any], validation_context: Mapping[str, Any] | None
) -> dict[str, Any] | NormalizedSchemaOutput:
    context = _enrichment_validation_context(validation_context)
    indices = context["expected_indices"]
    allow_fallback = context["allow_enum_unknown_fallback"]
    try:
        snapshot = _finite_json_snapshot(result)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("ENRICHMENT_RESULT_SCHEMA_INVALID",)) from None
    try:
        enrichment = _model_from_json(EnrichmentResult, snapshot)
    except ValidationError as error:
        if allow_fallback:
            normalized = _normalize_enrichment_enums(snapshot, error, indices)
            if normalized is not None:
                return normalized
        raise DeclaredSchemaOutputError(
            _enrichment_pydantic_issue_codes(error)
        ) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("ENRICHMENT_RESULT_SCHEMA_INVALID",)) from None
    try:
        validate_enrichment(enrichment, indices)
    except TemporalValidationError as error:
        raise DeclaredSchemaOutputError(
            tuple(dict.fromkeys(issue.code for issue in error.issues))
        ) from None
    return enrichment.model_dump(mode="json")


def _validate_scene_semantics_output(
    result: Mapping[str, Any], validation_context: Mapping[str, Any] | None
) -> dict[str, Any]:
    context = _exact_context(
        validation_context,
        {"duration", "require_observed_content", "required_object_ids"},
    )
    duration = _finite_real(context["duration"], positive=True)
    require_content = context["require_observed_content"]
    if not isinstance(require_content, bool):
        raise ValueError("SceneSemantics validation context is invalid")
    required_ids = context["required_object_ids"]
    if (
        not isinstance(required_ids, list)
        or len(set(required_ids)) != len(required_ids)
        or any(not isinstance(value, str) or not value for value in required_ids)
    ):
        raise ValueError("SceneSemantics validation context is invalid")
    try:
        scene = _model_from_json(SceneSemantics, result)
    except ValidationError as error:
        raise DeclaredSchemaOutputError(
            _pydantic_issue_codes(error, "SCENE_SEMANTICS")
        ) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("SCENE_SEMANTICS_SCHEMA_INVALID",)) from None
    try:
        validate_scene_semantics(
            scene,
            duration,
            require_observed_content=require_content,
            required_object_ids=tuple(required_ids),
        )
    except TemporalValidationError as error:
        raise DeclaredSchemaOutputError(
            tuple(dict.fromkeys(issue.code for issue in error.issues))
        ) from None
    return scene.model_dump(mode="json")


def _enrichment_validation_context(
    validation_context: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    context = _exact_context(
        validation_context,
        {"expected_indices", "allow_enum_unknown_fallback"},
    )
    indices = context["expected_indices"]
    allow_fallback = context["allow_enum_unknown_fallback"]
    if (
        not isinstance(indices, list)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or not isinstance(allow_fallback, bool)
    ):
        raise ValueError("EnrichmentResult validation context is invalid")
    return context


def _normalize_enrichment_enums(
    snapshot: dict[str, Any],
    error: ValidationError,
    expected_indices: list[int],
) -> NormalizedSchemaOutput | None:
    locations: list[tuple[int, str]] = []
    for issue in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = issue.get("loc")
        if (
            issue.get("type") != "enum"
            or not isinstance(location, tuple)
            or len(location) != 3
            or location[0] != "segments"
            or isinstance(location[1], bool)
            or not isinstance(location[1], int)
            or location[1] < 0
            or location[2] not in _ENRICHMENT_ENUM_FIELDS
        ):
            return None
        locations.append((location[1], location[2]))

    if not locations:
        return None
    try:
        segments = snapshot.get("segments")
        if not isinstance(segments, list):
            return None
        for offset, field in locations:
            if offset >= len(segments) or not isinstance(segments[offset], dict):
                return None
            segments[offset][field] = "unknown"
        enrichment = _model_from_json(EnrichmentResult, snapshot)
    except ValidationError as normalized_error:
        raise DeclaredSchemaOutputError(
            _enrichment_pydantic_issue_codes(normalized_error)
        ) from None
    except Exception:
        return None

    try:
        validate_enrichment(enrichment, expected_indices)
    except TemporalValidationError as temporal_error:
        raise DeclaredSchemaOutputError(
            tuple(dict.fromkeys(issue.code for issue in temporal_error.issues))
        ) from None

    fields = {field for _, field in locations}
    return NormalizedSchemaOutput(
        data=enrichment.model_dump(mode="json"),
        issue_codes=tuple(
            _ENRICHMENT_ENUM_FIELD_CODES[field]
            for field in _ENRICHMENT_ENUM_FIELDS
            if field in fields
        ),
        normalized_field_count=len(locations),
    )


def _validate_general_segment_output(
    result: Mapping[str, Any], validation_context: Mapping[str, Any] | None
) -> dict[str, Any]:
    from .general import GeneralSegmentResult

    start, end = _context_span(validation_context, {"span"})
    try:
        parsed = _model_from_json(GeneralSegmentResult, result)
    except ValidationError as error:
        raise DeclaredSchemaOutputError(
            _pydantic_issue_codes(error, "GENERAL_SEGMENT")
        ) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("GENERAL_SEGMENT_SCHEMA_INVALID",)) from None
    if any(
        event.start_time < start or event.end_time > end
        for event in parsed.segments
    ):
        raise DeclaredSchemaOutputError(
            ("GENERAL_SEGMENT_TIMESTAMP_OUT_OF_SPAN",)
        )
    return parsed.model_dump(mode="json")


def _validate_general_summary_output(
    result: Mapping[str, Any], validation_context: Mapping[str, Any] | None
) -> dict[str, Any]:
    from .general import GeneralSegmentResult, GeneralSummaryResult

    context = _exact_context(validation_context, {"span", "expected_timeline"})
    start, end = _span_value(context["span"])
    expected_value = context["expected_timeline"]
    if not isinstance(expected_value, list):
        raise ValueError("general summary validation context is invalid")
    try:
        expected = _model_from_json(
            GeneralSegmentResult,
            {"segments": expected_value, "warnings": []},
        ).model_dump(mode="json")["segments"]
    except (ValidationError, TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("general summary validation context is invalid") from None
    try:
        parsed = _model_from_json(GeneralSummaryResult, result)
    except ValidationError as error:
        raise DeclaredSchemaOutputError(
            _pydantic_issue_codes(error, "GENERAL_SUMMARY")
        ) from None
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise DeclaredSchemaOutputError(("GENERAL_SUMMARY_SCHEMA_INVALID",)) from None
    timeline = parsed.model_dump(mode="json")["timeline"]
    if any(item["start_time"] < start or item["end_time"] > end for item in timeline):
        raise DeclaredSchemaOutputError(
            ("GENERAL_SUMMARY_TIMELINE_OUT_OF_SPAN",)
        )
    if any(
        timeline[index]["start_time"] < timeline[index - 1]["start_time"]
        for index in range(1, len(timeline))
    ):
        raise DeclaredSchemaOutputError(("GENERAL_SUMMARY_TIMELINE_ORDER",))
    if timeline != expected:
        raise DeclaredSchemaOutputError(("GENERAL_SUMMARY_TIMELINE_MISMATCH",))
    return parsed.model_dump(mode="json")


def _exact_context(
    value: Mapping[str, Any] | None, expected_keys: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("declared output validation context is invalid")
    return value


def _context_span(
    context: Mapping[str, Any] | None, expected_keys: set[str]
) -> tuple[float, float]:
    return _span_value(_exact_context(context, expected_keys)["span"])


def _span_value(value: Any) -> tuple[float, float]:
    if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
        raise ValueError("declared output validation context is invalid")
    start = _finite_real(value["start"], positive=False)
    end = _finite_real(value["end"], positive=True)
    if start < 0 or start >= end:
        raise ValueError("declared output validation context is invalid")
    return start, end


def _context_finite_number(
    context: Mapping[str, Any] | None,
    key: str,
    *,
    positive: bool,
) -> float:
    value = _exact_context(context, {key})[key]
    return _finite_real(value, positive=positive)


def _finite_real(value: Any, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("declared output validation context is invalid")
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0):
        raise ValueError("declared output validation context is invalid")
    return numeric


def _model_from_json(
    model: type[BaseModel], result: Mapping[str, Any]
) -> BaseModel:
    return model.model_validate_json(_finite_json_text(result), strict=True)


def _finite_json_snapshot(result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        snapshot = json.loads(_finite_json_text(result))
    except Exception as error:
        raise TypeError("model result is not finite JSON data") from error
    if not isinstance(snapshot, dict):
        raise TypeError("model result must be a JSON object")
    return snapshot


def _finite_json_text(result: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except Exception as error:
        raise TypeError("model result is not finite JSON data") from error


DEFAULT_OUTPUT_SCHEMAS = OutputSchemaRegistry()
DEFAULT_OUTPUT_SCHEMAS.register(
    "ObjectInventory",
    _validate_object_inventory,
    allowed_issue_codes=_INVENTORY_REPAIR_CODES,
    generic_issue_code=_INVENTORY_GENERIC_REPAIR_CODE,
)
DEFAULT_OUTPUT_SCHEMAS.register(
    "CoarsePlan",
    _validate_coarse_output,
    allowed_issue_codes=_schema_codes("COARSE_PLAN") + _COARSE_TEMPORAL_CODES,
    generic_issue_code="COARSE_PLAN_SCHEMA_INVALID",
)
DEFAULT_OUTPUT_SCHEMAS.register(
    "BoundaryPlan",
    _validate_boundary_output,
    allowed_issue_codes=_schema_codes("BOUNDARY_PLAN") + _BOUNDARY_TEMPORAL_CODES,
    generic_issue_code="BOUNDARY_PLAN_SCHEMA_INVALID",
)
DEFAULT_OUTPUT_SCHEMAS.register(
    "EnrichmentResult",
    _validate_enrichment_output,
    allowed_issue_codes=_schema_codes("ENRICHMENT_RESULT")
    + tuple(_ENRICHMENT_ENUM_FIELD_CODES.values())
    + _ENRICHMENT_TEMPORAL_CODES,
    generic_issue_code="ENRICHMENT_RESULT_SCHEMA_INVALID",
    preserve_issue_code_order=True,
)
DEFAULT_OUTPUT_SCHEMAS.register(
    "SceneSemantics",
    _validate_scene_semantics_output,
    allowed_issue_codes=_schema_codes("SCENE_SEMANTICS") + _SCENE_TEMPORAL_CODES,
    generic_issue_code="SCENE_SEMANTICS_SCHEMA_INVALID",
)
DEFAULT_OUTPUT_SCHEMAS.register(
    "general_segment",
    _validate_general_segment_output,
    allowed_issue_codes=_schema_codes("GENERAL_SEGMENT")
    + _GENERAL_SEGMENT_TEMPORAL_CODES,
    generic_issue_code="GENERAL_SEGMENT_SCHEMA_INVALID",
)
DEFAULT_OUTPUT_SCHEMAS.register(
    "general_summary",
    _validate_general_summary_output,
    allowed_issue_codes=_schema_codes("GENERAL_SUMMARY")
    + _GENERAL_SUMMARY_TEMPORAL_CODES,
    generic_issue_code="GENERAL_SUMMARY_SCHEMA_INVALID",
)
