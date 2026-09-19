"""Hand-calculated fidelity examples; the full_0211 numbers are the ones in the
2026-09-19 Feishu record and were reproduced by the original prototype."""

import json
import math

import pytest

from percept_harness.evaluation.wm_fidelity import (
    FAMILIES,
    FPS,
    FidelityEvent,
    Sample,
    aggregate,
    bootstrap_ci,
    family_frame_sets,
    family_recall_precision,
    frame_miou,
    frame_overlap,
    load_sample,
    pearson,
    score_groups,
    score_pair,
    stratify,
)


def ev(start, end, kind, desc=""):
    from percept_harness.evaluation.wm_fidelity import FAMILY_OF

    return FidelityEvent(start, end, kind, FAMILY_OF[kind], desc)


def sample(sid, duration, events, outcome="success"):
    return Sample(sid, duration, tuple(events), outcome)


# --- full_0211: GT vs wan vs h3, values from the recorded run ---------------
GT_0211 = sample("full_0211", 3.77, [
    ev(0.67, 1.00, "reach"), ev(1.00, 2.00, "move"), ev(2.00, 2.33, "release"),
])
WAN_0211 = sample("full_0211", 3.79, [
    ev(0.33, 1.00, "approach"), ev(1.00, 1.33, "grasp"), ev(1.33, 2.00, "move"),
    ev(2.00, 2.33, "release"), ev(2.33, 2.67, "move"),
])
H3_0211 = sample("full_0211", 3.79, [
    ev(1.00, 1.33, "approach"), ev(1.33, 2.00, "grasp"), ev(2.00, 3.40, "move"),
    ev(3.40, 3.67, "release"),
])


def test_full_0211_frame_miou_matches_recorded_values():
    wan = score_pair(GT_0211, WAN_0211)
    h3 = score_pair(GT_0211, H3_0211)
    assert wan.miou == pytest.approx(0.51, abs=0.01)
    assert h3.miou == pytest.approx(0.01, abs=0.01)
    # per family, wan: release interval identical -> place IoU 1.0
    i, u = wan.overlap["place"]
    assert i == u and i > 0
    # h3 has no frame in common with GT on place at all
    assert h3.overlap["place"][0] == 0


def test_full_0211_outcome_agrees_while_timing_does_not():
    h3 = score_pair(GT_0211, H3_0211)
    assert h3.outcome == ("success", "success")
    assert h3.miou < 0.05


# --- frame bucketing semantics ------------------------------------------------
def test_frame_sets_use_inclusive_end_bucket():
    s = family_frame_sets([ev(0.0, 1.0, "move")], duration=2.0)
    assert s["displace"] == set(range(0, FPS + 1))
    assert all(not s[f] for f in FAMILIES if f != "displace")


def test_frame_sets_are_clipped_to_duration():
    s = family_frame_sets([ev(0.0, 5.0, "move")], duration=1.0)
    assert max(s["displace"]) == FPS


def test_family_miou_is_insensitive_to_segmentation_granularity():
    one = sample("x", 4.0, [ev(0.0, 2.0, "place")])
    two = sample("x", 4.0, [ev(0.0, 1.0, "transport"), ev(1.0, 2.0, "place")])
    merged = sample("x", 4.0, [ev(0.0, 2.0, "place")])
    # same family on the same frames -> identical timeline
    assert frame_miou(frame_overlap(one, merged)) == 1.0
    # transport/place split moves 0-1s into a different family -> penalised
    assert frame_miou(frame_overlap(one, two)) < 1.0
    # but splitting inside one family costs nothing
    split = sample("x", 4.0, [ev(0.0, 1.0, "place"), ev(1.0, 2.0, "release")])
    assert frame_miou(frame_overlap(one, split)) == 1.0


def test_family_miou_ignores_actor_and_description():
    a = sample("x", 3.0, [FidelityEvent(0, 1, "move", "displace", "left hand pushes")])
    b = sample("x", 3.0, [FidelityEvent(0, 1, "move", "displace", "right hand slides")])
    assert frame_miou(frame_overlap(a, b)) == 1.0


def test_family_miou_penalises_missing_family():
    a = sample("x", 3.0, [ev(0, 1, "move"), ev(1, 2, "state_change")])
    b = sample("x", 3.0, [ev(0, 1, "move")])
    ov = frame_overlap(a, b)
    assert ov["passive"][0] == 0 and ov["passive"][1] > 0
    assert frame_miou(ov) == pytest.approx(0.5, abs=0.02)


# --- event level + semantic ---------------------------------------------------
def test_semantic_column_weights_similarity_by_tiou():
    ref = sample("x", 4.0, [ev(0, 2, "place", "sets the cube on the tower")])
    pred = sample("x", 4.0, [ev(0, 2, "release", "lets go of the cube")])
    calls = []

    def sim(a, b):
        calls.append((a, b))
        return 0.5

    s = score_pair(ref, pred, sim)
    assert calls == [("sets the cube on the tower", "lets go of the cube")]
    assert s.iou_sum == 1.0
    assert s.sim_weighted_sum == 0.5
    assert s.sem_f1 == pytest.approx(0.5)
    assert s.soft_f1 == pytest.approx(1.0)


def test_semantic_column_absent_without_similarity_fn():
    s = score_pair(GT_0211, WAN_0211)
    assert s.sim_weighted_sum is None and s.sem_f1 is None
    agg = aggregate([s])
    assert "sem_f1" not in agg


def test_aggregate_micro_pools_frames_and_events():
    a = score_pair(GT_0211, WAN_0211)
    b = score_pair(GT_0211, H3_0211)
    agg = aggregate([a, b])
    inter = sum(v[0] for s in (a, b) for v in s.overlap.values())
    union = sum(v[1] for s in (a, b) for v in s.overlap.values())
    assert agg["frame_miou_micro"] == pytest.approx(inter / union)
    assert agg["reference_events"] == 6 and agg["prediction_events"] == 9
    assert agg["outcome_agreement"] == {"agree": 2, "total": 2}
    assert agg["frame_miou_per_sample"] == [a.miou, b.miou]


def test_score_pair_rejects_identity_mismatch():
    with pytest.raises(ValueError):
        score_pair(GT_0211, sample("other", 3.0, []))


# --- breakdown helpers --------------------------------------------------------
def test_family_recall_precision():
    ref = sample("x", 3.0, [ev(0, 2, "move")])
    pred = sample("x", 3.0, [ev(1, 3, "move")])
    rp = family_recall_precision([score_pair(ref, pred)], {"x": ref}, {"x": pred})
    # overlap 1-2s of 0-2s reference and 1-3s prediction
    assert rp["displace"]["recall"] == pytest.approx(0.5, abs=0.02)
    assert rp["displace"]["precision"] == pytest.approx(0.5, abs=0.02)
    assert rp["place"]["recall"] is None


def test_stratify_reports_wins_for_two_systems():
    rows = [
        {"sample_id": "a", "duration": 3.0, "event_count": 2, "task": "", "wan": 0.5, "h3": 0.2},
        {"sample_id": "b", "duration": 12.0, "event_count": 9, "task": "", "wan": 0.1, "h3": 0.6},
    ]
    out = stratify(rows, "duration", ((0, 6), (6, math.inf)))
    assert [e["n"] for e in out] == [1, 1]
    assert out[0]["h3_wins"] == 0 and out[1]["h3_wins"] == 1


def test_pearson_and_bootstrap_are_deterministic():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert bootstrap_ci([0.2, 0.4, 0.6], iterations=200) == bootstrap_ci([0.2, 0.4, 0.6], iterations=200)
    assert bootstrap_ci([]) is None


# --- loading -----------------------------------------------------------------
def _write(tmp_path, sid, data):
    d = tmp_path / sid
    d.mkdir()
    p = d / f"{sid}.json"
    p.write_text(json.dumps({"status": "completed", "data": data}))
    return p


def test_load_sample_drops_occlusion_unknown_and_failed(tmp_path):
    p = _write(tmp_path, "full_0001", {
        "duration": 5.0,
        "segments": [{"start": 0, "end": 1}],
        "outcome": {"status": "failure"},
        "task_description": "topple the tower",
        "semantic_events": [
            {"start": 0.0, "end": 1.0, "event_type": "place", "description": "a"},
            {"start": 1.0, "end": 2.0, "event_type": "occluded", "description": "b"},
            {"start": 2.0, "end": 3.0, "event_type": "unknown", "description": "c"},
            {"start": 3.0, "end": 3.0, "event_type": "move", "description": "zero length"},
            {"start": 3.0, "end": 4.0, "event_type": "state_change", "description": "d"},
        ],
    })
    s = load_sample(p)
    assert s.sample_id == "full_0001" and s.outcome == "failure"
    assert [e.family for e in s.events] == ["place", "passive"]
    assert s.task_description == "topple the tower"
    failed = _write(tmp_path, "full_0002", {"duration": 5.0, "segments": [], "semantic_events": []})
    assert load_sample(failed) is None


def test_score_groups_pairs_only_common_ids(tmp_path):
    ref_dir, pred_dir = tmp_path / "ref", tmp_path / "pred"
    ref_dir.mkdir(); pred_dir.mkdir()
    base = {"duration": 2.0, "segments": [1], "semantic_events": [{"start": 0, "end": 1, "event_type": "move"}]}
    for sid in ("a", "b"):
        _write(ref_dir, sid, base)
    _write(pred_dir, "b", base)
    _write(pred_dir, "c", base)
    from percept_harness.evaluation.wm_fidelity import load_group

    scores = score_groups(load_group(ref_dir), load_group(pred_dir))
    assert [s.sample_id for s in scores] == ["b"]
    assert scores[0].miou == 1.0
