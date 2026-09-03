"""Contract tests for deterministic embodied temporal validation."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from pydantic import ValidationError

import las_repro.pipelines.output_validation as output_validation_module
from las_repro.pipelines.output_validation import (
    DEFAULT_OUTPUT_SCHEMAS,
    _enrichment_pydantic_issue_codes,
)
from las_repro.pipelines.validators import (
    BoundaryPlan,
    CoarsePlan,
    EnrichmentResult,
    ObjectInventory,
    Skill,
    TemporalValidationError,
    validate_boundary_plan,
    validate_coarse_plan,
    validate_enrichment,
)


@pytest.fixture
def coarse_plan() -> CoarsePlan:
    return CoarsePlan.model_validate(
        {
            "task_description": "move the red container",
            "actions": [
                {
                    "action_index": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "description": "right hand reaches container",
                    "event_type": "reach_and_grasp",
                },
                {
                    "action_index": 1,
                    "start": 1.0,
                    "end": 2.0,
                    "description": "right hand moves container",
                    "event_type": "transport",
                },
            ],
        }
    )


@pytest.fixture
def valid_boundary_plan(coarse_plan: CoarsePlan) -> BoundaryPlan:
    actions = []
    segment_index = 0
    for action in coarse_plan.actions:
        midpoint = (action.start + action.end) / 2
        start_id = f"a{action.action_index}-start"
        middle_id = f"a{action.action_index}-middle"
        end_id = f"a{action.action_index}-end"
        actions.append(
            {
                **action.model_dump(),
                "boundary_points": [
                    {
                        "boundary_id": start_id,
                        "time": action.start,
                        "event_type": "action_start",
                        "visual_evidence": "hand begins visible motion",
                    },
                    {
                        "boundary_id": middle_id,
                        "time": midpoint,
                        "event_type": "approach",
                        "visual_evidence": "hand approaches container",
                    },
                    {
                        "boundary_id": end_id,
                        "time": action.end,
                        "event_type": "action_end",
                        "visual_evidence": "action phase visibly ends",
                    },
                ],
                "fine_segments": [
                    {
                        "segment_index": segment_index,
                        "start": action.start,
                        "end": midpoint,
                        "description": "right hand approaches container",
                        "event_type": "approach",
                        "start_boundary_id": start_id,
                        "end_boundary_id": middle_id,
                    },
                    {
                        "segment_index": segment_index + 1,
                        "start": midpoint,
                        "end": action.end,
                        "description": "right hand transports container",
                        "event_type": "transport_continue",
                        "start_boundary_id": middle_id,
                        "end_boundary_id": end_id,
                    },
                ],
            }
        )
        segment_index += 2
    return BoundaryPlan.model_validate(
        {"task_description": coarse_plan.task_description, "actions": actions}
    )


def test_object_inventory_requires_exact_visible_object_shape():
    """Inventory shape is strict while truthful no-object results remain representable."""
    inventory = ObjectInventory.model_validate(
        {
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
    )

    assert inventory.objects[0].instances[0].name == "red container"
    with pytest.raises(ValidationError):
        ObjectInventory.model_validate(
            {
                "objects": [
                    {
                        "category": "container",
                        "instances": [{"name": "", "description": "visible"}],
                    }
                ]
            }
        )
    assert ObjectInventory.model_validate({"objects": []}).objects == []
    assert ObjectInventory.model_validate(
        {"objects": [{"category": "container", "instances": []}]}
    ).objects[0].instances == []
    with pytest.raises(ValidationError):
        ObjectInventory.model_validate({"objects": [], "guessed": True})


def test_models_reject_nonfinite_times_invalid_enums_and_enrichment_shape():
    """Schema validation must block values that aggregate checks cannot compare."""
    with pytest.raises(ValidationError):
        CoarsePlan.model_validate(
            {
                "task_description": "move container",
                "actions": [
                    {
                        "action_index": 0,
                        "start": float("nan"),
                        "end": 1.0,
                        "description": "move container",
                        "event_type": "transport",
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        EnrichmentResult.model_validate(
            {
                "segments": [
                    {
                        "segment_index": 0,
                        "actor": "hand",
                        "actor_state": "holding",
                        "skill": "move",
                        "target": "container",
                        "visual_motion_state": "active",
                        "confidence": 0.9,
                    }
                ]
            }
        )
    with pytest.raises(ValidationError):
        EnrichmentResult.model_validate(
            {
                "segments": [
                    {
                        "segment_index": 0,
                        "actor": "right_gripper",
                        "actor_state": "holding",
                        "skill": "move",
                        "target": "container",
                        "visual_motion_state": "active",
                        "confidence": 0.9,
                        "start": 0.0,
                    }
                ]
            }
        )


def test_enrichment_accepts_proven_touch_skill_without_widening_skill_vocabulary():
    """Missing ``touch`` would reject the replayed payload or admit unproven aliases."""
    enrichment = EnrichmentResult.model_validate(
        {
            "segments": [
                {
                    "segment_index": 0,
                    "actor": "right_gripper",
                    "actor_state": "contacting",
                    "skill": "touch",
                    "target": "container",
                    "visual_motion_state": "low",
                    "confidence": 0.9,
                }
            ]
        }
    )

    assert enrichment.segments[0].skill is Skill.TOUCH
    assert {skill.name: skill.value for skill in Skill} == {
        "HOLD": "hold",
        "REACH": "reach",
        "GRASP": "grasp",
        "PICK": "pick",
        "LIFT": "lift",
        "MOVE": "move",
        "PLACE": "place",
        "RELEASE": "release",
        "PUSH": "push",
        "PULL": "pull",
        "ROTATE": "rotate",
        "OPEN": "open",
        "CLOSE": "close",
        "RETRACT": "retract",
        "UNKNOWN": "unknown",
        "TOUCH": "touch",
    }
    with pytest.raises(ValidationError):
        EnrichmentResult.model_validate(
            {
                "segments": [
                    {
                        "segment_index": 0,
                        "actor": "right_gripper",
                        "actor_state": "contacting",
                        "skill": "tap_contact",
                        "target": "container",
                        "visual_motion_state": "low",
                        "confidence": 0.9,
                    }
                ]
            }
        )


@pytest.mark.parametrize("invalid_time", ["0.0", False, True])
def test_models_reject_coercive_timestamp_inputs(invalid_time: object):
    """Parsing strings or booleans would silently repair model-supplied times."""
    with pytest.raises(ValidationError):
        CoarsePlan.model_validate(
            {
                "task_description": "move container",
                "actions": [
                    {
                        "action_index": 0,
                        "start": invalid_time,
                        "end": 1.0,
                        "description": "move container",
                        "event_type": "transport",
                    }
                ],
            }
        )


@pytest.mark.parametrize("invalid_confidence", ["0.9", False, True])
def test_enrichment_rejects_coercive_confidence_inputs(invalid_confidence: object):
    """Stage 4 confidence must be a supplied finite number, not a coerced value."""
    with pytest.raises(ValidationError):
        EnrichmentResult.model_validate(
            {
                "segments": [
                    {
                        "segment_index": 0,
                        "actor": "right_gripper",
                        "actor_state": "holding",
                        "skill": "move",
                        "target": "container",
                        "visual_motion_state": "active",
                        "confidence": invalid_confidence,
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("actor", "ENRICHMENT_RESULT_ACTOR_ENUM_VALUE"),
        ("actor_state", "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE"),
        ("skill", "ENRICHMENT_RESULT_SKILL_ENUM_VALUE"),
        (
            "visual_motion_state",
            "ENRICHMENT_RESULT_VISUAL_MOTION_STATE_ENUM_VALUE",
        ),
    ],
)
def test_enrichment_enum_errors_expose_only_closed_field_family(
    field: str, expected: str
) -> None:
    """Enum diagnostics may identify a closed field family, never rejected data."""
    invalid_token = "private-enrichment-enum-token"
    raw_result = {
        "segments": [
            {
                "segment_index": 17,
                "actor": "right_gripper",
                "actor_state": "holding",
                "skill": "move",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.9,
            }
        ]
    }
    raw_result["segments"][0][field] = invalid_token

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        raw_result,
        {
            "expected_indices": [17],
            "allow_enum_unknown_fallback": False,
        },
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "invalid",
            "issue_codes": [expected],
        }
    }
    rendered_envelope = str(sanitized)
    for private_detail in (
        invalid_token,
        "17",
        "segments",
        field,
        "target",
        "red container",
    ):
        assert private_detail not in rendered_envelope


def test_enrichment_enum_errors_deduplicate_closed_codes_in_encounter_order() -> None:
    """A repair pass needs each affected field family once, in stable error order."""
    raw_result = {
        "segments": [
            {
                "segment_index": 17,
                "actor": "private-actor-token",
                "actor_state": "private-actor-state-token",
                "skill": "private-skill-token",
                "target": "red container",
                "visual_motion_state": "private-visual-motion-state-token",
                "confidence": 0.9,
            },
            {
                "segment_index": 18,
                "actor": "private-duplicate-actor-token",
                "actor_state": "holding",
                "skill": "move",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.9,
            },
        ]
    }

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        raw_result,
        {
            "expected_indices": [17, 18],
            "allow_enum_unknown_fallback": False,
        },
    )

    assert sanitized["_schema_validation"]["issue_codes"] == [
        "ENRICHMENT_RESULT_ACTOR_ENUM_VALUE",
        "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE",
        "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
        "ENRICHMENT_RESULT_VISUAL_MOTION_STATE_ENUM_VALUE",
    ]


def test_enrichment_enum_errors_preserve_cross_record_encounter_order() -> None:
    """The envelope must not reorder closed codes after Pydantic emits them."""
    raw_result = {
        "segments": [
            {
                "segment_index": 17,
                "actor": "right_gripper",
                "actor_state": "holding",
                "skill": "private-skill-token",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.9,
            },
            {
                "segment_index": 18,
                "actor": "private-actor-token",
                "actor_state": "holding",
                "skill": "move",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.9,
            },
        ]
    }

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        raw_result,
        {
            "expected_indices": [17, 18],
            "allow_enum_unknown_fallback": False,
        },
    )

    assert sanitized["_schema_validation"]["issue_codes"] == [
        "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
        "ENRICHMENT_RESULT_ACTOR_ENUM_VALUE",
    ]


def test_enrichment_enum_errors_use_generic_code_for_unrecognized_location() -> None:
    """Only allowlisted final location components may identify a field family."""
    error = ValidationError.from_exception_data(
        "EnrichmentResult",
        [
            {
                "type": "enum",
                "loc": ("segments", 0, "unrecognized_enum_field"),
                "input": "private-enrichment-enum-token",
                "ctx": {"expected": "'known'"},
            }
        ],
    )

    assert _enrichment_pydantic_issue_codes(error) == (
        "ENRICHMENT_RESULT_ENUM_VALUE",
    )


def _fallback_enrichment_result() -> dict[str, Any]:
    return {
        "segments": [
            {
                "segment_index": 17,
                "actor": "right_gripper",
                "actor_state": "holding",
                "skill": "private-skill-one",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.9,
            },
            {
                "segment_index": 18,
                "actor": "left_gripper",
                "actor_state": "private-state-one",
                "skill": "private-skill-two",
                "target": "blue container",
                "visual_motion_state": "low",
                "confidence": 0.8,
            },
        ]
    }


def _fallback_context(enabled: bool) -> dict[str, Any]:
    return {
        "expected_indices": [17, 18],
        "allow_enum_unknown_fallback": enabled,
    }


def _canonical_normalized_enrichment() -> dict[str, Any]:
    return {
        "segments": [
            {
                "segment_index": 17,
                "actor": "right_gripper",
                "actor_state": "holding",
                "skill": "unknown",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.9,
            },
            {
                "segment_index": 18,
                "actor": "left_gripper",
                "actor_state": "unknown",
                "skill": "unknown",
                "target": "blue container",
                "visual_motion_state": "low",
                "confidence": 0.8,
            },
        ]
    }


def _normalized_enrichment_envelope() -> dict[str, Any]:
    return {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "normalized",
            "issue_codes": [
                "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE",
                "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
            ],
            "normalized_field_count": 3,
        },
        "data": _canonical_normalized_enrichment(),
    }


def test_enrichment_enum_fallback_is_disabled_for_initial_output() -> None:
    """Enabling normalization before the repair ordinal would skip model repair."""
    raw_result = _fallback_enrichment_result()

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        raw_result,
        _fallback_context(False),
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "invalid",
            "issue_codes": [
                "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
                "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE",
            ],
        }
    }


def test_enrichment_enum_fallback_normalizes_only_rejected_occurrences() -> None:
    """A repair containing only allowlisted enum errors becomes audited unknowns."""
    raw_result = _fallback_enrichment_result()

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        raw_result,
        _fallback_context(True),
    )

    assert sanitized == _normalized_enrichment_envelope()
    assert sanitized["data"]["segments"][0]["actor_state"] == "holding"
    assert sanitized["data"]["segments"][1]["actor"] == "left_gripper"
    metadata_text = json.dumps(
        sanitized["_schema_validation"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    complete_text = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
    for rejected_token in (
        "private-skill-one",
        "private-state-one",
        "private-skill-two",
    ):
        assert rejected_token not in complete_text
    for forbidden_location in (
        '["segments",0,"skill"]',
        '["segments",1,"actor_state"]',
        '["segments",1,"skill"]',
    ):
        assert forbidden_location not in complete_text
    assert (
        json.dumps(raw_result, ensure_ascii=False, separators=(",", ":"))
        not in complete_text
    )
    assert "segments" not in metadata_text
    assert "17" not in metadata_text
    assert "18" not in metadata_text


def _missing_enrichment_field(raw: dict[str, Any]) -> None:
    _restore_valid_enrichment_enums(raw)
    del raw["segments"][0]["target"]


def _add_enrichment_field(raw: dict[str, Any]) -> None:
    _restore_valid_enrichment_enums(raw)
    raw["segments"][0]["description"] = "must not be copied"


def _invalidate_enrichment_confidence(raw: dict[str, Any]) -> None:
    _restore_valid_enrichment_enums(raw)
    raw["segments"][0]["confidence"] = 2.0


def _replace_enrichment_segments(raw: dict[str, Any]) -> None:
    raw["segments"] = {"private": "wrong type"}


def _restore_valid_enrichment_enums(raw: dict[str, Any]) -> None:
    raw["segments"][0]["actor_state"] = "holding"
    raw["segments"][0]["skill"] = "move"
    raw["segments"][1]["actor_state"] = "holding"
    raw["segments"][1]["skill"] = "move"


def _mismatch_enrichment_indices(raw: dict[str, Any]) -> None:
    raw["segments"][1]["segment_index"] = 19


def _mix_enrichment_enum_and_non_enum_errors(raw: dict[str, Any]) -> None:
    raw["segments"][0]["confidence"] = 2.0


@pytest.mark.parametrize(
    ("mutation", "expected_codes"),
    [
        (_missing_enrichment_field, ["ENRICHMENT_RESULT_MISSING_FIELD"]),
        (_add_enrichment_field, ["ENRICHMENT_RESULT_EXTRA_FIELD"]),
        (_invalidate_enrichment_confidence, ["ENRICHMENT_RESULT_NUMBER_RANGE"]),
        (_replace_enrichment_segments, ["ENRICHMENT_RESULT_LIST_TYPE"]),
        (
            _mismatch_enrichment_indices,
            ["MISSING_ENRICHMENT_INDEX", "UNEXPECTED_ENRICHMENT_INDEX"],
        ),
        (
            _mix_enrichment_enum_and_non_enum_errors,
            [
                "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
                "ENRICHMENT_RESULT_NUMBER_RANGE",
                "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE",
            ],
        ),
    ],
    ids=(
        "missing-field",
        "extra-field",
        "invalid-confidence",
        "wrong-segments-type",
        "index-mismatch",
        "mixed-enum-and-non-enum",
    ),
)
def test_enrichment_enum_fallback_fails_closed_for_non_allowlisted_failures(
    mutation: Any,
    expected_codes: list[str],
) -> None:
    """Fallback must never hide shape, type, confidence, or index failures."""
    raw_result = _fallback_enrichment_result()
    mutation(raw_result)

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        raw_result,
        _fallback_context(True),
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "invalid",
            "issue_codes": expected_codes,
        }
    }
    assert "data" not in sanitized


@pytest.mark.parametrize(
    ("location", "expected_code"),
    [
        (
            ("segments", 0, "future_enum_field"),
            "ENRICHMENT_RESULT_ENUM_VALUE",
        ),
        (("segments", True, "skill"), "ENRICHMENT_RESULT_SKILL_ENUM_VALUE"),
        (("segments", -1, "skill"), "ENRICHMENT_RESULT_SKILL_ENUM_VALUE"),
        (("segments", 0.0, "skill"), "ENRICHMENT_RESULT_SKILL_ENUM_VALUE"),
        (("segments", "0", "skill"), "ENRICHMENT_RESULT_SKILL_ENUM_VALUE"),
        (("segments", 99, "skill"), "ENRICHMENT_RESULT_SKILL_ENUM_VALUE"),
    ],
)
def test_enrichment_enum_fallback_rejects_an_unrecognized_error_path_or_offset(
    monkeypatch: pytest.MonkeyPatch,
    location: tuple[Any, ...],
    expected_code: str,
) -> None:
    """Only real in-range Pydantic list offsets at known fields may normalize."""
    fabricated = ValidationError.from_exception_data(
        "EnrichmentResult",
        [
            {
                "type": "enum",
                "loc": location,
                "input": "private-future-token",
                "ctx": {"expected": "'known'"},
            }
        ],
    )

    def reject_with_unknown_path(
        model: type[Any], result: Mapping[str, Any]
    ) -> Any:
        del model, result
        raise fabricated

    monkeypatch.setattr(
        output_validation_module,
        "_model_from_json",
        reject_with_unknown_path,
    )

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        _fallback_enrichment_result(),
        _fallback_context(True),
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "invalid",
            "issue_codes": [expected_code],
        }
    }


@pytest.mark.parametrize(
    "context",
    [
        {"expected_indices": [17, 18]},
        {
            "expected_indices": [17, 18],
            "allow_enum_unknown_fallback": 1,
        },
        {
            "expected_indices": [17, 18],
            "allow_enum_unknown_fallback": "false",
        },
        {
            "expected_indices": [17, 18],
            "allow_enum_unknown_fallback": False,
            "extra": False,
        },
    ],
)
def test_enrichment_fallback_context_requires_exact_keys_and_real_boolean(
    context: dict[str, Any],
) -> None:
    """A truthy local configuration value must not silently enable fallback."""
    with pytest.raises(ValueError, match="context"):
        DEFAULT_OUTPUT_SCHEMAS.sanitize(
            "EnrichmentResult",
            _canonical_normalized_enrichment(),
            context,
        )


@pytest.mark.parametrize(
    "schema_name",
    [
        "ObjectInventory",
        "CoarsePlan",
        "BoundaryPlan",
        "general_segment",
        "general_summary",
    ],
)
def test_enum_fallback_context_is_rejected_by_every_other_schema(
    schema_name: str,
) -> None:
    """The fallback switch is local configuration for enrichment alone."""
    with pytest.raises(ValueError, match="context"):
        DEFAULT_OUTPUT_SCHEMAS.sanitize(
            schema_name,
            {},
            {"allow_enum_unknown_fallback": False},
        )


def test_normalized_envelope_parser_revalidates_and_returns_canonical_metadata(
) -> None:
    """Only strict canonical embedded data may cross the trusted unwrap boundary."""
    parser = getattr(DEFAULT_OUTPUT_SCHEMAS, "normalized_result", None)
    assert parser is not None
    parsed = parser(
        "EnrichmentResult",
        _normalized_enrichment_envelope(),
        _fallback_context(True),
    )

    assert parsed is not None
    assert parsed.data == _canonical_normalized_enrichment()
    assert parsed.issue_codes == (
        "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE",
        "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
    )
    assert parsed.normalized_field_count == 3


def test_normalized_envelope_parser_requires_enabled_repair_context() -> None:
    """An ordinal-zero context must never trust a fabricated normalized envelope."""
    parser = getattr(DEFAULT_OUTPUT_SCHEMAS, "normalized_result", None)
    assert parser is not None

    assert (
        parser(
            "EnrichmentResult",
            _normalized_enrichment_envelope(),
            _fallback_context(False),
        )
        is None
    )


def _normalized_envelope_variants() -> list[Any]:
    variants: list[Any] = []

    missing_outer = _normalized_enrichment_envelope()
    del missing_outer["data"]
    variants.append(missing_outer)

    extra_outer = _normalized_enrichment_envelope()
    extra_outer["raw"] = "private"
    variants.append(extra_outer)

    extra_metadata = _normalized_enrichment_envelope()
    extra_metadata["_schema_validation"]["offsets"] = [0, 1]
    variants.append(extra_metadata)

    wrong_schema = _normalized_enrichment_envelope()
    wrong_schema["_schema_validation"]["schema_name"] = "CoarsePlan"
    variants.append(wrong_schema)

    wrong_status = _normalized_enrichment_envelope()
    wrong_status["_schema_validation"]["status"] = "invalid"
    variants.append(wrong_status)

    duplicate_codes = _normalized_enrichment_envelope()
    duplicate_codes["_schema_validation"]["issue_codes"] = [
        "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
        "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
    ]
    variants.append(duplicate_codes)

    wrong_code_order = _normalized_enrichment_envelope()
    wrong_code_order["_schema_validation"]["issue_codes"] = [
        "ENRICHMENT_RESULT_SKILL_ENUM_VALUE",
        "ENRICHMENT_RESULT_ACTOR_STATE_ENUM_VALUE",
    ]
    variants.append(wrong_code_order)

    generic_code = _normalized_enrichment_envelope()
    generic_code["_schema_validation"]["issue_codes"] = [
        "ENRICHMENT_RESULT_SCHEMA_INVALID"
    ]
    variants.append(generic_code)

    unsubstantiated_code = _normalized_enrichment_envelope()
    unsubstantiated_code["_schema_validation"]["issue_codes"] = [
        "ENRICHMENT_RESULT_ACTOR_ENUM_VALUE"
    ]
    unsubstantiated_code["_schema_validation"]["normalized_field_count"] = 1
    variants.append(unsubstantiated_code)

    for invalid_count in (True, 0, -1, 1.0, 5):
        wrong_count = _normalized_enrichment_envelope()
        wrong_count["_schema_validation"]["normalized_field_count"] = invalid_count
        variants.append(wrong_count)

    invalid_embedded_data = _normalized_enrichment_envelope()
    invalid_embedded_data["data"]["segments"][0]["actor_state"] = "still-private"
    variants.append(invalid_embedded_data)

    mismatched_embedded_index = _normalized_enrichment_envelope()
    mismatched_embedded_index["data"]["segments"][1]["segment_index"] = 19
    variants.append(mismatched_embedded_index)

    nonfinite_embedded_data = _normalized_enrichment_envelope()
    nonfinite_embedded_data["data"]["segments"][0]["confidence"] = float("nan")
    variants.append(nonfinite_embedded_data)

    unserializable_embedded_data = _normalized_enrichment_envelope()
    unserializable_embedded_data["data"]["segments"][0]["target"] = object()
    variants.append(unserializable_embedded_data)

    recursive_embedded_data = _normalized_enrichment_envelope()
    recursive_embedded_data["data"] = _normalized_enrichment_envelope()
    variants.append(recursive_embedded_data)

    cyclic_embedded_data = _normalized_enrichment_envelope()
    cyclic_embedded_data["data"] = cyclic_embedded_data
    variants.append(cyclic_embedded_data)

    return variants


@pytest.mark.parametrize("malformed", _normalized_envelope_variants())
def test_normalized_envelope_parser_rejects_hostile_or_malformed_data(
    malformed: Any,
) -> None:
    """Malformed audit data must never be mistaken for trusted canonical output."""
    parser = getattr(DEFAULT_OUTPUT_SCHEMAS, "normalized_result", None)
    assert parser is not None
    assert (
        parser(
            "EnrichmentResult",
            malformed,
            _fallback_context(True),
        )
        is None
    )


class _RaisingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("hostile mapping access")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("hostile mapping iteration")

    def __len__(self) -> int:
        raise RuntimeError("hostile mapping length")


def test_normalized_envelope_parser_contains_arbitrary_mapping_failures() -> None:
    """An adversarial Mapping implementation must fail closed without escaping."""
    parser = getattr(DEFAULT_OUTPUT_SCHEMAS, "normalized_result", None)
    assert parser is not None
    assert (
        parser(
            "EnrichmentResult",
            _RaisingMapping(),
            _fallback_context(True),
        )
        is None
    )


def test_enrichment_fallback_rejects_non_json_mapping_without_leaking() -> None:
    """A non-JSON Mapping result must remain a generic closed failure."""
    class DelegatingMapping(Mapping[str, Any]):
        def __init__(self, value: dict[str, Any]) -> None:
            self._value = value

        def __getitem__(self, key: str) -> Any:
            return self._value[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._value)

        def __len__(self) -> int:
            return len(self._value)

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        DelegatingMapping(_fallback_enrichment_result()),
        _fallback_context(True),
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "invalid",
            "issue_codes": ["ENRICHMENT_RESULT_SCHEMA_INVALID"],
        }
    }


def test_enrichment_fallback_contains_hostile_dict_subclass_failures() -> None:
    """A Mapping serializer failure must become the generic closed envelope."""
    class RaisingItemsDict(dict[str, Any]):
        def items(self) -> Any:
            raise RuntimeError("hostile dict items")

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        RaisingItemsDict(_fallback_enrichment_result()),
        _fallback_context(True),
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "invalid",
            "issue_codes": ["ENRICHMENT_RESULT_SCHEMA_INVALID"],
        }
    }


def test_enrichment_fallback_does_not_request_a_second_mapping_view() -> None:
    """Normalization must succeed entirely from the finite initial snapshot."""
    class RaisingSecondItemsDict(dict[str, Any]):
        item_calls = 0

        def items(self) -> Any:
            self.item_calls += 1
            if self.item_calls == 2:
                raise RuntimeError("hostile second dict items")
            return super().items()

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        RaisingSecondItemsDict(_fallback_enrichment_result()),
        _fallback_context(True),
    )

    assert sanitized == _normalized_enrichment_envelope()


def test_enrichment_fallback_ignores_changed_second_mapping_view() -> None:
    """A later target/confidence view must not enter enum-normalized output."""
    class ChangingSecondItemsDict(dict[str, Any]):
        item_calls = 0

        def items(self) -> Any:
            self.item_calls += 1
            if self.item_calls == 1:
                return super().items()
            altered = copy.deepcopy(dict(dict.items(self)))
            altered["segments"][0]["target"] = "private-second-view-target"
            altered["segments"][0]["confidence"] = 0.1
            altered["segments"][1]["target"] = "private-second-view-container"
            altered["segments"][1]["confidence"] = 0.2
            return altered.items()

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        ChangingSecondItemsDict(_fallback_enrichment_result()),
        _fallback_context(True),
    )

    assert sanitized == _normalized_enrichment_envelope()
    assert "private-second-view" not in json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("confidence", float("nan")),
        ("target", object()),
    ],
)
def test_enrichment_fallback_snapshot_rejects_nonfinite_or_unserializable_data(
    field: str,
    invalid_value: Any,
) -> None:
    """Snapshot creation failures must use the generic closed envelope."""
    raw_result = _fallback_enrichment_result()
    raw_result["segments"][0][field] = invalid_value

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "EnrichmentResult",
        raw_result,
        _fallback_context(True),
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "EnrichmentResult",
            "status": "invalid",
            "issue_codes": ["ENRICHMENT_RESULT_SCHEMA_INVALID"],
        }
    }


@pytest.mark.parametrize("negative_time", [-1e-9, -0.01])
def test_models_reject_negative_timestamps_even_inside_comparison_tolerance(
    negative_time: float,
):
    """Timestamp domain errors are schema errors, not tolerant topology differences."""
    with pytest.raises(ValidationError):
        CoarsePlan.model_validate(
            {
                "task_description": "move container",
                "actions": [
                    {
                        "action_index": 0,
                        "start": negative_time,
                        "end": 1.0,
                        "description": "move container",
                        "event_type": "transport",
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        BoundaryPlan.model_validate(
            {
                "task_description": "move container",
                "actions": [
                    {
                        "action_index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "description": "move container",
                        "event_type": "transport",
                        "boundary_points": [
                            {
                                "boundary_id": "start",
                                "time": negative_time,
                                "event_type": "action_start",
                                "visual_evidence": "hand is visible",
                            }
                        ],
                        "fine_segments": [],
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        BoundaryPlan.model_validate(
            {
                "task_description": "move container",
                "actions": [
                    {
                        "action_index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "description": "move container",
                        "event_type": "transport",
                        "boundary_points": [],
                        "fine_segments": [
                            {
                                "segment_index": 0,
                                "start": negative_time,
                                "end": 1.0,
                                "description": "move container",
                                "event_type": "transport_continue",
                                "start_boundary_id": "start",
                                "end_boundary_id": "end",
                            }
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize("negative_time", [-1e-9, -0.01])
def test_models_reject_negative_timestamp_ends_even_inside_comparison_tolerance(
    negative_time: float,
):
    """End fields share the same nonnegative time domain as start fields."""
    with pytest.raises(ValidationError):
        CoarsePlan.model_validate(
            {
                "task_description": "move container",
                "actions": [
                    {
                        "action_index": 0,
                        "start": 0.0,
                        "end": negative_time,
                        "description": "move container",
                        "event_type": "transport",
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        BoundaryPlan.model_validate(
            {
                "task_description": "move container",
                "actions": [
                    {
                        "action_index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "description": "move container",
                        "event_type": "transport",
                        "boundary_points": [],
                        "fine_segments": [
                            {
                                "segment_index": 0,
                                "start": 0.0,
                                "end": negative_time,
                                "description": "move container",
                                "event_type": "transport_continue",
                                "start_boundary_id": "start",
                                "end_boundary_id": "end",
                            }
                        ],
                    }
                ],
            }
        )


def test_coarse_plan_reports_all_coverage_index_and_duration_issues():
    """Returning on the first coarse error would hide repair instructions."""
    plan = CoarsePlan.model_validate(
        {
            "task_description": "move container",
            "actions": [
                {
                    "action_index": 1,
                    "start": 0.1,
                    "end": 0.1,
                    "description": "wait still",
                    "event_type": "idle",
                },
                {
                    "action_index": 1,
                    "start": 0.0,
                    "end": 1.2,
                    "description": "move container",
                    "event_type": "transport",
                },
            ],
        }
    )

    with pytest.raises(TemporalValidationError) as error:
        validate_coarse_plan(plan, duration=1.0)

    assert [issue.code for issue in error.value.issues] == [
        "ACTION_START_NOT_ZERO",
        "DUPLICATE_ACTION_INDEX",
        "ACTION_INDEX_NOT_ORDERED",
        "ACTION_NONPOSITIVE_DURATION",
        "ACTION_OVERLAP",
        "ACTION_END_MISMATCH_DURATION",
    ]
    assert error.value.issues[0].path == ("actions", 0, "start")


def test_boundary_plan_rejects_gap_and_invalid_reference(
    valid_boundary_plan: BoundaryPlan, coarse_plan: CoarsePlan
):
    """A gap and a missing boundary must both be available to one repair pass."""
    broken = copy.deepcopy(valid_boundary_plan)
    broken.actions[0].fine_segments[1].start += 0.2
    broken.actions[0].fine_segments[1].start_boundary_id = "missing"

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan, max_segment_seconds=1.0)

    assert {issue.code for issue in error.value.issues} >= {
        "SEGMENT_GAP",
        "UNKNOWN_BOUNDARY_ID",
    }
    assert broken.actions[0].fine_segments[1].start == pytest.approx(0.7)


def test_boundary_plan_reports_parent_boundary_and_index_violations_together(
    valid_boundary_plan: BoundaryPlan, coarse_plan: CoarsePlan
):
    """Cross-record validation must retain every independently discoverable defect."""
    broken = copy.deepcopy(valid_boundary_plan)
    first = broken.actions[0]
    second = broken.actions[1]
    first.boundary_points[0].boundary_id = second.boundary_points[0].boundary_id
    first.boundary_points[0].time = 0.2
    first.fine_segments[0].end = 1.2
    first.fine_segments[0].segment_index = 3
    second.fine_segments[0].segment_index = 3
    second.start = 1.1

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan, max_segment_seconds=1.0)

    codes = {issue.code for issue in error.value.issues}
    assert {
        "DUPLICATE_BOUNDARY_ID",
        "BOUNDARY_START_MISMATCH",
        "SEGMENT_OUTSIDE_PARENT",
        "BOUNDARY_TIME_MISMATCH",
        "DUPLICATE_SEGMENT_INDEX",
        "PARENT_START_MISMATCH",
    } <= codes


@pytest.mark.parametrize(
    ("segment_indexes", "existing_codes"),
    [
        ([2, 7, 8, 9], set()),
        ([0, 2, 3, 4], set()),
        ([0, 1, 1, 2], {"DUPLICATE_SEGMENT_INDEX", "SEGMENT_INDEX_NOT_ORDERED"}),
        ([0, 2, 1, 3], {"SEGMENT_INDEX_NOT_ORDERED"}),
    ],
    ids=("starts-at-two-and-jumps-to-seven", "gap", "duplicate", "reordered"),
)
def test_boundary_plan_requires_zero_based_globally_contiguous_segment_indexes(
    valid_boundary_plan: BoundaryPlan,
    coarse_plan: CoarsePlan,
    segment_indexes: list[int],
    existing_codes: set[str],
) -> None:
    """Unique increasing values are insufficient when rows map by global position."""
    broken = copy.deepcopy(valid_boundary_plan)
    segments = [
        segment
        for action in broken.actions
        for segment in action.fine_segments
    ]
    for segment, replacement in zip(segments, segment_indexes, strict=True):
        segment.segment_index = replacement

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan)

    codes = {issue.code for issue in error.value.issues}
    assert "SEGMENT_INDEX_NOT_CONTIGUOUS" in codes
    assert existing_codes <= codes


def test_boundary_plan_accepts_contiguous_indexes_across_action_boundaries(
    valid_boundary_plan: BoundaryPlan,
    coarse_plan: CoarsePlan,
) -> None:
    """The counter continues across parents instead of resetting per action."""
    validate_boundary_plan(valid_boundary_plan, coarse_plan)

    assert [
        segment.segment_index
        for action in valid_boundary_plan.actions
        for segment in action.fine_segments
    ] == [0, 1, 2, 3]


def test_boundary_plan_rejects_noncontiguous_parent_actions_and_long_segments(
    valid_boundary_plan: BoundaryPlan, coarse_plan: CoarsePlan
):
    """Fine output cannot conceal an out-of-order coarse parent or overlong interval."""
    broken = copy.deepcopy(valid_boundary_plan)
    broken.actions[1].action_index = 0
    broken.actions[1].fine_segments[0].end = 2.0

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan, max_segment_seconds=0.25)

    assert {issue.code for issue in error.value.issues} >= {
        "DUPLICATE_ACTION_INDEX",
        "SEGMENT_TOO_LONG",
    }


@pytest.mark.parametrize(
    ("replacement_start", "expected_code"),
    [(1.11, "ACTION_GAP"), (0.89, "ACTION_OVERLAP")],
)
def test_boundary_plan_rejects_cross_action_gaps_and_overlaps_beyond_tolerance(
    valid_boundary_plan: BoundaryPlan,
    coarse_plan: CoarsePlan,
    replacement_start: float,
    expected_code: str,
):
    """Pass B actions need direct topology checks, not only tolerant parent copies."""
    broken = copy.deepcopy(valid_boundary_plan)
    broken.actions[1].start = replacement_start

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan, tolerance=0.05)

    assert expected_code in {issue.code for issue in error.value.issues}


@pytest.mark.parametrize(
    ("delta", "expected_code"),
    [(0.000001, "ACTION_GAP"), (-0.000001, "ACTION_OVERLAP")],
)
def test_coarse_plan_rejects_real_topology_errors_inside_legacy_tolerance(
    coarse_plan: CoarsePlan, delta: float, expected_code: str
):
    """Comparison tolerance must never authorize a real final-output hole or overlap."""
    broken = copy.deepcopy(coarse_plan)
    broken.actions[1].start += delta

    with pytest.raises(TemporalValidationError) as error:
        validate_coarse_plan(broken, duration=2.0, tolerance=0.05)

    assert expected_code in {issue.code for issue in error.value.issues}
    assert broken.actions[1].start == 1.0 + delta


def test_coarse_plan_rejects_small_video_endpoint_drift(coarse_plan: CoarsePlan):
    broken = copy.deepcopy(coarse_plan)
    broken.actions[-1].end = 1.999999

    with pytest.raises(TemporalValidationError) as error:
        validate_coarse_plan(broken, duration=2.0, tolerance=0.05)

    assert "ACTION_END_MISMATCH_DURATION" in {
        issue.code for issue in error.value.issues
    }


def test_boundary_plan_rejects_small_segment_gap_inside_legacy_tolerance(
    valid_boundary_plan: BoundaryPlan, coarse_plan: CoarsePlan
):
    broken = copy.deepcopy(valid_boundary_plan)
    broken.actions[0].fine_segments[1].start += 0.000001

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan, tolerance=0.05)

    assert "SEGMENT_GAP" in {issue.code for issue in error.value.issues}


def test_topology_allows_only_binary_arithmetic_noise_without_modifying_values():
    boundary = 0.1 + 0.2
    plan = CoarsePlan.model_validate(
        {
            "task_description": "move container",
            "actions": [
                {
                    "action_index": 0,
                    "start": 0.0,
                    "end": boundary,
                    "description": "reach toward container",
                    "event_type": "reach_and_grasp",
                },
                {
                    "action_index": 1,
                    "start": 0.3,
                    "end": 1.0,
                    "description": "move the container",
                    "event_type": "transport",
                },
            ],
        }
    )

    validate_coarse_plan(plan, duration=1.0, tolerance=0.05)

    assert plan.actions[0].end == boundary
    assert plan.actions[1].start == 0.3


def test_boundary_plan_rejects_reversed_short_parent_action(
    valid_boundary_plan: BoundaryPlan, coarse_plan: CoarsePlan
):
    """A reversed action remains invalid even when endpoint deltas fit tolerance."""
    broken = copy.deepcopy(valid_boundary_plan)
    broken.actions[0].start = 0.04
    broken.actions[0].end = 0.0

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan, tolerance=0.05)

    assert "ACTION_NONPOSITIVE_DURATION" in {issue.code for issue in error.value.issues}


def test_boundary_plan_enforces_max_segment_duration_without_topology_tolerance(
    valid_boundary_plan: BoundaryPlan, coarse_plan: CoarsePlan
):
    """A 0.50 second segment must fail a configured 0.49 second hard cap."""
    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(
            valid_boundary_plan,
            coarse_plan,
            max_segment_seconds=0.49,
            tolerance=0.05,
        )

    assert "SEGMENT_TOO_LONG" in {issue.code for issue in error.value.issues}


def test_empty_boundary_plan_reports_every_missing_coarse_action(
    coarse_plan: CoarsePlan,
):
    """Empty Pass B output must retain all repairable missing-action diagnostics."""
    empty = BoundaryPlan.model_validate(
        {"task_description": coarse_plan.task_description, "actions": []}
    )

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(empty, coarse_plan)

    assert [issue.code for issue in error.value.issues] == [
        "EMPTY_ACTIONS",
        "MISSING_COARSE_ACTION",
        "MISSING_COARSE_ACTION",
    ]
    assert [issue.message for issue in error.value.issues[1:]] == [
        "Pass B is missing Pass A action_index 0",
        "Pass B is missing Pass A action_index 1",
    ]


def test_boundary_plan_requires_fine_segments_from_adjacent_boundary_points(
    valid_boundary_plan: BoundaryPlan, coarse_plan: CoarsePlan
):
    """An unused intermediate boundary must not be skipped by a fine segment."""
    broken = copy.deepcopy(valid_boundary_plan)
    original_middle = broken.actions[0].boundary_points[1]
    broken.actions[0].boundary_points.insert(
        1,
        original_middle.model_copy(
            update={"boundary_id": "a0-observable-change", "time": 0.25}
        ),
    )

    with pytest.raises(TemporalValidationError) as error:
        validate_boundary_plan(broken, coarse_plan, max_segment_seconds=1.0)

    assert "SEGMENT_BOUNDARY_NOT_ADJACENT" in {
        issue.code for issue in error.value.issues
    }


def test_enrichment_requires_exact_ordered_unique_index_set_and_six_fields():
    """A partial or reordered enrichment result must not be merged by position."""
    result = EnrichmentResult.model_validate(
        {
            "segments": [
                {
                    "segment_index": 2,
                    "actor": "right_gripper",
                    "actor_state": "holding",
                    "skill": "move",
                    "target": "red container",
                    "visual_motion_state": "active",
                    "confidence": 0.9,
                },
                {
                    "segment_index": 2,
                    "actor": "left_gripper",
                    "actor_state": "idle",
                    "skill": "unknown",
                    "target": "unknown",
                    "visual_motion_state": "static",
                    "confidence": 0.0,
                },
                {
                    "segment_index": 5,
                    "actor": "both_grippers",
                    "actor_state": "transporting",
                    "skill": "move",
                    "target": "red container",
                    "visual_motion_state": "active",
                    "confidence": 1.0,
                },
            ]
        }
    )

    with pytest.raises(TemporalValidationError) as error:
        validate_enrichment(result, expected_indices=[1, 2])

    assert [issue.code for issue in error.value.issues] == [
        "DUPLICATE_ENRICHMENT_INDEX",
        "ENRICHMENT_INDEX_NOT_ORDERED",
        "MISSING_ENRICHMENT_INDEX",
        "UNEXPECTED_ENRICHMENT_INDEX",
    ]
    assert error.value.issues[-1].message == "unexpected enrichment indices: 5"
