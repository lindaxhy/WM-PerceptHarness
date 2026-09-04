"""Contracts for overlapping LAS-aligned scene semantics."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
from las_repro.pipelines.scene_semantics import (
    SceneSemantics,
    validate_scene_semantics,
)
from las_repro.pipelines.validators import TemporalValidationError


def valid_scene_semantics() -> dict[str, object]:
    return {
        "objects": [
            {
                "object_id": "panel",
                "name": "white occluding panel",
                "description": "white panel moving across the visible workspace",
            },
            {
                "object_id": "apple",
                "name": "pink apple",
                "description": "pink apple manipulated by the right hand",
            },
        ],
        "initial_state": [
            {
                "object_id": "panel",
                "state": "panel is at the right side",
                "visual_evidence": "panel is visible at the right edge",
                "confidence": 0.9,
            }
        ],
        "final_state": [
            {
                "object_id": "panel",
                "state": "panel is at the left side",
                "visual_evidence": "panel is visible at the left edge",
                "confidence": 0.8,
            }
        ],
        "outcome": {
            "status": "unknown",
            "description": "task success is not fully visible",
            "confidence": 0.4,
        },
        "semantic_events": [
            {
                "event_index": 0,
                "start": 0.0,
                "end": 1.4,
                "event_type": "move",
                "actor": "right_hand",
                "target_object_id": "panel",
                "description": "right hand moves the white panel",
                "confidence": 0.9,
            },
            {
                "event_index": 1,
                "start": 0.8,
                "end": 1.8,
                "event_type": "occluded",
                "actor": "unknown",
                "target_object_id": "apple",
                "description": "pink apple is fully hidden by panel",
                "confidence": 0.8,
            },
        ],
    }


def test_scene_semantics_accepts_overlapping_action_and_occlusion_events():
    """Forcing a single non-overlapping track would preserve the main LAS gap."""
    parsed = SceneSemantics.model_validate(valid_scene_semantics())

    validate_scene_semantics(parsed, duration=2.0)

    assert [event.event_type.value for event in parsed.semantic_events] == [
        "move",
        "occluded",
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda value: value["semantic_events"][1].update({"end": 2.1}),
            "SCENE_EVENT_OUTSIDE_VIDEO",
        ),
        (
            lambda value: value["semantic_events"][1].update(
                    {"target_object_id": "private_object"}
            ),
            "SCENE_EVENT_UNKNOWN_OBJECT",
        ),
        (
            lambda value: value["initial_state"][0].update(
                    {"object_id": "private_object"}
            ),
            "SCENE_STATE_UNKNOWN_OBJECT",
        ),
    ],
)
def test_scene_semantics_rejects_bounds_and_unknown_object_references(
    mutation: object,
    expected_code: str,
) -> None:
    raw = valid_scene_semantics()
    mutation(raw)
    parsed = SceneSemantics.model_validate(raw)

    with pytest.raises(TemporalValidationError) as error:
        validate_scene_semantics(parsed, duration=2.0)

    assert expected_code in {issue.code for issue in error.value.issues}


def test_scene_semantics_output_registry_returns_closed_temporal_failure():
    """Raw object IDs and descriptions must not enter persisted repair metadata."""
    raw = valid_scene_semantics()
    raw["semantic_events"][0]["target_object_id"] = "private_object"

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "SceneSemantics",
        raw,
        {
            "duration": 2.0,
            "require_observed_content": True,
            "required_object_ids": [],
        },
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "SceneSemantics",
            "status": "invalid",
            "issue_codes": ["SCENE_EVENT_UNKNOWN_OBJECT"],
        }
    }


def test_scene_semantics_rejects_degenerate_empty_track_when_content_is_required():
    raw = valid_scene_semantics()
    raw["objects"] = []
    raw["initial_state"] = []
    raw["final_state"] = []
    raw["semantic_events"] = []
    parsed = SceneSemantics.model_validate(raw)

    with pytest.raises(TemporalValidationError) as error:
        validate_scene_semantics(parsed, duration=2.0, require_observed_content=True)

    assert {issue.code for issue in error.value.issues} == {
        "EMPTY_SCENE_OBJECTS",
        "EMPTY_SCENE_EVENTS",
    }


def test_scene_semantics_requires_the_trusted_target_object_ids():
    parsed = SceneSemantics.model_validate(valid_scene_semantics())

    with pytest.raises(TemporalValidationError) as error:
        validate_scene_semantics(
            parsed,
            duration=2.0,
            require_observed_content=True,
            required_object_ids=("apple", "ramp"),
        )

    assert {issue.code for issue in error.value.issues} == {
        "SCENE_REQUIRED_OBJECT_MISSING"
    }


@pytest.mark.parametrize("invalid_id", ["BadId", "bad id", "bad/id", "对象"])
def test_scene_semantics_rejects_noncanonical_object_ids(invalid_id: str):
    raw = valid_scene_semantics()
    raw["objects"][0]["object_id"] = invalid_id
    raw["initial_state"][0]["object_id"] = invalid_id
    raw["final_state"][0]["object_id"] = invalid_id
    raw["semantic_events"][0]["target_object_id"] = invalid_id

    with pytest.raises(ValidationError):
        SceneSemantics.model_validate(raw)
