"""Behavior tests for LAS-aligned deterministic semantic-event projection."""

from __future__ import annotations

import pytest

from percept_harness.pipelines.semantic_events import build_semantic_events


def _segment(
    index: int,
    start: float,
    end: float,
    *,
    actor: str = "right_gripper",
    skill: str = "move",
    target: str = "Red Container",
    description: str = "right hand moves red container",
    confidence: float = 0.8,
) -> dict[str, object]:
    return {
        "segment_index": index,
        "start": start,
        "end": end,
        "actor": actor,
        "skill": skill,
        "target": target,
        "description": description,
        "confidence": confidence,
    }


def test_compatible_segments_merge_with_weighted_confidence_and_hand_projection():
    """Same-skill adjacent segments must merge with duration-weighted confidence."""
    segments = [
        _segment(0, 0.0, 0.4, confidence=0.5),
        _segment(
            1,
            0.4,
            1.0,
            actor="right_hand",
            skill="move",
            target=" red   container ",
            description="right hand slides red container left",
            confidence=1.0,
        ),
    ]

    assert build_semantic_events(segments) == [
        {
            "event_index": 0,
            "start": 0.0,
            "end": 1.0,
            "actor": "right_hand",
            "action": "move",
            "target": "Red Container",
            "description": (
                "right hand moves red container; right hand slides red container left"
            ),
            "confidence": pytest.approx(0.8),
            "source_segment_indices": [0, 1],
        }
    ]


def test_actor_action_and_target_changes_split_semantic_events():
    """A broad action family must never hide a real participant or object change."""
    segments = [
        _segment(0, 0.0, 0.25),
        _segment(1, 0.25, 0.5, actor="left_gripper"),
        _segment(2, 0.5, 0.75, actor="left_hand", skill="grasp"),
        _segment(3, 0.75, 1.0, actor="left_hand", target="blue cup"),
    ]

    events = build_semantic_events(segments)

    assert [event["event_index"] for event in events] == [0, 1, 2, 3]
    assert [(event["actor"], event["action"], event["target"]) for event in events] == [
        ("right_hand", "move", "Red Container"),
        ("left_hand", "move", "Red Container"),
        ("left_hand", "grasp", "Red Container"),
        ("left_hand", "move", "blue cup"),
    ]
    assert [event["source_segment_indices"] for event in events] == [[0], [1], [2], [3]]


def test_autonomous_motion_passes_through_without_projection():
    """Object self-motion keeps its official label instead of a family alias."""
    events = build_semantic_events(
        [
            _segment(
                0, 0.0, 0.5,
                actor="unknown",
                skill="autonomous_motion",
                target="red apple",
                description="red apple rolls down the ramp",
            ),
            _segment(
                1, 0.5, 1.0,
                actor="unknown",
                skill="stop",
                target="red apple",
                description="red apple comes to rest in the container",
            ),
        ]
    )

    assert [event["action"] for event in events] == ["autonomous_motion", "stop"]
    assert [event["source_segment_indices"] for event in events] == [[0], [1]]


def test_projection_rejects_overlap_or_unknown_input_values():
    """The additive output must fail closed if called before segment validation."""
    with pytest.raises(ValueError, match="overlap"):
        build_semantic_events([_segment(0, 0.0, 0.6), _segment(1, 0.5, 1.0)])
    with pytest.raises(ValueError, match="actor"):
        build_semantic_events([_segment(0, 0.0, 1.0, actor="private_actor")])
    with pytest.raises(ValueError, match="skill"):
        build_semantic_events([_segment(0, 0.0, 1.0, skill="private_skill")])


def test_projection_keeps_gap_separated_same_key_segments_apart():
    """A visible pause must split events even when actor, action, and target repeat."""
    events = build_semantic_events(
        [_segment(0, 0.0, 0.4), _segment(1, 0.7, 1.0)]
    )

    assert [event["source_segment_indices"] for event in events] == [[0], [1]]
    assert [(event["start"], event["end"]) for event in events] == [
        (0.0, 0.4),
        (0.7, 1.0),
    ]
