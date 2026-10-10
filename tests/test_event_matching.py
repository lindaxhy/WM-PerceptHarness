"""Hand-calculated event matching regressions, independent of experiment data."""

import pytest

from percept_harness.evaluation.event_matching import Event, match_events, temporal_iou


def test_one_to_one_matching_maximizes_cardinality_then_iou():
    refs = [Event(0, 2, "move"), Event(2, 4, "move")]
    preds = [Event(0, 4, "move"), Event(0, 2, "move")]
    matches = match_events(refs, preds, temporal_iou_threshold=0.3)
    assert {(m.reference_index, m.prediction_index) for m in matches} == {
        (0, 1),
        (1, 0),
    }


def test_half_open_threshold_equality_types_empty_and_deterministic_ties():
    a, b = Event(0, 2, "move"), Event(2, 4, "move")
    assert temporal_iou(a, b) == 0
    assert temporal_iou(a, Event(0, 4, "move")) == 0.5
    assert len(match_events([a], [Event(0, 4, "move")], 0.5)) == 1
    assert match_events([a], [Event(0, 4, "grasp")], 0.3) == []
    assert match_events([], [a], 0.3) == []
    assert match_events([a], [], 0.3) == []
    assert [
        (m.reference_index, m.prediction_index)
        for m in match_events([a, a], [a, a], 0.3)
    ] == [(0, 0), (1, 1)]
    assert len(match_events([a, a], [a], 0.3)) == 1


@pytest.mark.parametrize(
    "args", [(0, 0, "move"), (-1, 2, "move"), (0, float("inf"), "move")]
)
def test_invalid_events_rejected(args):
    with pytest.raises(ValueError):
        Event(*args)


def test_iou_secondary_objective_selects_best_of_equal_cardinality():
    refs = [Event(0, 2, "move"), Event(1, 3, "move")]
    preds = [Event(1, 3, "move"), Event(0, 2, "move")]
    matches = match_events(refs, preds, 0.3)
    assert [(m.reference_index, m.prediction_index) for m in matches] == [
        (0, 1),
        (1, 0),
    ]
    assert sum(m.iou for m in matches) == 2


def test_unknown_events_are_not_positive_matches():
    assert match_events([Event(0, 1, "unknown")], [Event(0, 1, "unknown")], 0.3) == []
