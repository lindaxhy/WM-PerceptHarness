"""Behavior tests for LAS-aligned deterministic semantic-event projection."""

from __future__ import annotations

import pytest

from las_repro.pipelines.semantic_events import build_semantic_events


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
    """Losing broad action continuity would preserve the measured fragmentation gap."""
    segments = [
        _segment(0, 0.0, 0.4, confidence=0.5),
        _segment(
            1,
            0.4,
            1.0,
            actor="right_hand",
            skill="lift",
            target=" red   container ",
            description="right hand lifts red container",
            confidence=1.0,
        ),
    ]

    assert build_semantic_events(segments) == [
        {
            "event_index": 0,
            "start": 0.0,
            "end": 1.0,
            "actor": "right_hand",
            "action": "motion",
            "target": "Red Container",
            "description": (
                "right hand moves red container; right hand lifts red container"
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
        ("right_hand", "motion", "Red Container"),
        ("left_hand", "motion", "Red Container"),
        ("left_hand", "grasp", "Red Container"),
        ("left_hand", "motion", "blue cup"),
    ]
    assert [event["source_segment_indices"] for event in events] == [[0], [1], [2], [3]]


def test_projection_rejects_noncontiguous_or_unknown_input_values():
    """The additive output must fail closed if called before segment validation."""
    with pytest.raises(ValueError, match="contiguous"):
        build_semantic_events([_segment(0, 0.0, 0.4), _segment(1, 0.5, 1.0)])
    with pytest.raises(ValueError, match="actor"):
        build_semantic_events([_segment(0, 0.0, 1.0, actor="private_actor")])
    with pytest.raises(ValueError, match="skill"):
        build_semantic_events([_segment(0, 0.0, 1.0, skill="private_skill")])
