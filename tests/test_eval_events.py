import json
import math

import pytest

from percept_harness.evaluation import events, events_cli
from percept_harness.evaluation.event_matching import Match, match_weight_matrix, temporal_iou
from percept_harness.pipelines.event_schemas import GroundedEvents

TIOU_MESSAGE = r"tIoU threshold must be finite in \(0,1\]"


def grounded(*spans, duration=10.0):
    return GroundedEvents.model_validate({"duration": duration, "events": [
        {"start": start, "end": end, "description": text} for start, end, text in spans]})


def same_text(reference, prediction):
    return {"match": reference == prediction, "reason": "compared"}


def test_temporal_iou():
    a, b = grounded((0, 2, "a"), (1, 3, "b")).events
    assert math.isclose(temporal_iou(a, b), 1 / 3)
    c, d = grounded((0, 1, "a"), (2, 3, "b")).events
    assert temporal_iou(c, d) == 0


def test_match_weight_matrix_cardinality_first():
    # Greedy on IoU would pick (0,0)=0.95 and leave row 1 unmatched.
    assert match_weight_matrix([[0.95, 0.8], [0.9, 0.0]], 0.75) == [Match(0, 1, 0.8), Match(1, 0, 0.9)]
    assert match_weight_matrix([[0.5]], 0.75) == []
    assert match_weight_matrix([[0.75]], 0.75) == [Match(0, 0, 0.75)]
    assert match_weight_matrix([], 0.5) == []
    assert match_weight_matrix([[]], 0.5) == []
    assert match_weight_matrix([[0.8, 0.9]], 1) == []


@pytest.mark.parametrize("scores, threshold, message", [
    ([[0.5]], 0, "invalid temporal IoU threshold"),
    ([[0.5]], 1.5, "invalid temporal IoU threshold"),
    ([[0.5]], float("nan"), "invalid finite number"),
    ([[0.5]], True, "invalid finite number"),
    ([[0.5]], "0.5", "invalid finite number"),
    ([], -1, "invalid temporal IoU threshold"),
    ([[0.5], [0.5, 0.5]], 0.5, "ragged IoU matrix"),
    ([[1.5]], 0.5, "invalid IoU"),
    ([[float("inf")]], 0.5, "invalid finite number"),
    ([[True]], 0.5, "invalid finite number"),
])
def test_match_weight_matrix_errors(scores, threshold, message):
    with pytest.raises(ValueError, match=message):
        match_weight_matrix(scores, threshold)


def test_score_pair():
    reference = grounded((0, 4, "pick cup"), (5, 9, "drop cup"))
    prediction = grounded((0, 4, "pick cup"), (5, 8, "lift cup"), (9, 10, "extra"))
    calls = []
    def judge(a, b):
        calls.append((a, b))
        return same_text(a, b)
    score = events.score_pair(reference, prediction, judge, 0.7)
    assert calls == [("pick cup", "pick cup"), ("drop cup", "lift cup")]
    assert {key: score[key] for key in (
        "reference_events", "prediction_events", "temporal_match_count", "semantic_passes_on_temporal_matches",
        "joint_match_count", "segment_precision", "segment_recall", "segment_f1", "semantic_match_rate",
        "joint_precision", "joint_recall", "joint_f1", "temporal_matches", "joint_matches",
        "unmatched_temporal_reference_indices", "unmatched_temporal_prediction_indices",
        "unmatched_joint_reference_indices", "unmatched_joint_prediction_indices")} == {
        "reference_events": 2, "prediction_events": 3, "temporal_match_count": 2,
        "semantic_passes_on_temporal_matches": 1, "joint_match_count": 1,
        "segment_precision": 2 / 3, "segment_recall": 1.0, "segment_f1": 0.8, "semantic_match_rate": 0.5,
        "joint_precision": 1 / 3, "joint_recall": 0.5, "joint_f1": 0.4,
        "temporal_matches": [{"reference_index": 0, "prediction_index": 0, "iou": 1.0},
                             {"reference_index": 1, "prediction_index": 1, "iou": 0.75}],
        "joint_matches": [{"reference_index": 0, "prediction_index": 0, "iou": 1.0}],
        "unmatched_temporal_reference_indices": [], "unmatched_temporal_prediction_indices": [2],
        "unmatched_joint_reference_indices": [1], "unmatched_joint_prediction_indices": [1, 2]}
    assert score["candidate_judgments"][1] == {
        "reference_index": 1, "prediction_index": 1, "iou": 0.75, "reference_description": "drop cup",
        "prediction_description": "lift cup", "match": False, "reason": "compared"}


def test_score_pair_empty():
    score = events.score_pair(grounded(), grounded(), same_text)
    assert score["segment_f1"] is None and score["joint_f1"] is None and score["semantic_match_rate"] is None


@pytest.mark.parametrize("threshold", [0, -0.1, 1.01, float("nan"), float("inf"), True, "0.5", None])
def test_score_pair_threshold_errors(threshold):
    with pytest.raises(ValueError, match=TIOU_MESSAGE):
        events.score_pair(grounded(), grounded(), same_text, threshold)


def test_score_pair_judge_errors():
    reference = grounded((0, 1, "a"))
    with pytest.raises(events.JudgeError, match=r"candidate \(0,0\) failed: judge must return strict"):
        events.score_pair(reference, reference, lambda a, b: {"match": 1, "reason": "x"})


def write_annotation(path, *spans, duration=10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"duration": duration, "events": [
        {"start": s, "end": e, "description": d} for s, e, d in spans]}))


@pytest.fixture
def dataset(tmp_path):
    ref, sys_dir = tmp_path / "ref", tmp_path / "sys"
    write_annotation(ref / "a.json", (0, 4, "pick cup"))
    write_annotation(sys_dir / "a.json", (0, 4, "pick cup"))
    write_annotation(ref / "b" / "result.json", (0, 2, "wave"))
    write_annotation(sys_dir / "b" / "result.json", (5, 6, "wave"))
    for root in (ref, sys_dir):
        (root / ".percept").mkdir()
        (root / ".percept" / "junk.json").write_text("not json")
        (root / "shard-0.status.json").write_text("{}")
        (root / "nested").mkdir()
    return ref, sys_dir


def test_load_group_skips_and_identity(dataset):
    ref, _ = dataset
    write_annotation(ref / "nested" / "shard-1.status.json", (0, 1, "kept"))
    group = events.load_group(ref)
    assert sorted(group) == ["a", "b", "nested"]
    assert group["b"]["path"] == str(ref / "b" / "result.json")


def test_load_group_audit_sidecar_redirect_and_skip(dataset, tmp_path):
    ref, _ = dataset
    (ref / "c.json").write_text(json.dumps({"duration": 3.0, "events": []}))
    (ref / "c.audit.json").write_text(json.dumps({"status": "completed", "sample_id": "c-id"}))
    # A non-annotation audit next to a skipped shard status name stays skipped.
    (ref / "shard-2.status.audit.json").write_text(json.dumps({"status": "running"}))
    group = events.load_group(ref)
    assert sorted(group) == ["a", "b", "c-id"]
    assert group["c-id"]["record"] == {"status": "completed", "sample_id": "c-id", "data": {"duration": 3.0, "events": []}}


def test_load_group_errors(tmp_path):
    with pytest.raises(ValueError, match="annotation directory missing"):
        events.load_group(tmp_path / "none")
    write_annotation(tmp_path / "x" / "v.json", (0, 1, "a"))
    write_annotation(tmp_path / "x" / "v" / "result.json", (0, 1, "a"))
    with pytest.raises(ValueError, match="duplicate or ambiguous sample ID 'v'"):
        events.load_group(tmp_path / "x")
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / "bad.json").write_text(json.dumps({"sample_id": " ", "duration": 1, "events": []}))
    with pytest.raises(ValueError, match="malformed primary annotation .*invalid explicit sample/job ID"):
        events.load_group(tmp_path / "y")


def test_evaluate(dataset):
    ref, sys_dir = dataset
    report = events.evaluate(ref, {"m": sys_dir}, same_text, 0.5)
    system = report["systems"]["m"]
    assert report["protocol"]["tiou_threshold"] == 0.5
    assert system["valid_count"] == 2 and system["invalid_count"] == 0
    assert system["aggregate"]["totals"] == {"samples": 2, "reference_events": 2, "prediction_events": 2,
                                            "temporal_matches": 1, "semantic_passes_on_temporal_matches": 1,
                                            "joint_matches": 1}
    assert system["aggregate"]["segment"]["micro"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert system["aggregate"]["segment"]["macro"]["f1"] == {"value": 0.5, "contributing_count": 2}


def test_evaluate_missing_and_incomplete(dataset):
    ref, sys_dir = dataset
    (sys_dir / "a.json").unlink()
    (sys_dir / "b" / "result.audit.json").write_text(json.dumps({"status": "failed", "error": "boom"}))
    report = events.evaluate(ref, {"m": sys_dir}, same_text)
    assert report["protocol"]["tiou_threshold"] == 0.5
    samples = report["systems"]["m"]["samples"]
    assert samples["a"] == {"status": "error", "error": "missing prediction"}
    assert samples["b"]["error"] == "ValueError: annotation status is not completed: boom"
    assert report["systems"]["m"]["aggregate"] is None


@pytest.mark.parametrize("threshold", [0, 2, float("nan"), True])
def test_evaluate_threshold_checked_before_loading(tmp_path, threshold):
    with pytest.raises(ValueError, match=TIOU_MESSAGE):
        events.evaluate(tmp_path / "missing", {}, same_text, threshold)


def test_evaluate_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="completely empty dataset"):
        events.evaluate(tmp_path, {"m": tmp_path}, same_text)


def test_events_cli(dataset, tmp_path, capsys):
    ref, sys_dir = dataset
    out = tmp_path / "report"
    assert events_cli.main(["--reference", str(ref), "--system", f"m={sys_dir}", "--out", str(out),
                            "--tiou-threshold", "0.5"], judge=same_text) == 0
    report = json.loads((out / "events.json").read_text())
    assert report["judge"] == {"protocol": "injected callable"}
    assert capsys.readouterr().out == f"event score report: {out / 'events.json'}\n"
    assert events_cli.main(["--reference", str(ref), "--system", f"m={sys_dir}", "--out", str(out),
                            "--tiou-threshold", "2"], judge=same_text) == 1
    report = json.loads((out / "events.json").read_text())
    assert report["status"] == "error"
    assert report["error"] == "ValueError: tIoU threshold must be finite in (0,1]"
