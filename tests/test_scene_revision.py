"""Full review results replace the draft and fail closed on disagreement."""
import copy

import pytest

from test_scene_choices import fixture
from test_runner import silent_video
from percept_harness.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS as registry


def revision_fixture():
    draft, ctx = fixture()
    return {"scene": copy.deepcopy(draft), "decisions": []}, {
        "scene_context": ctx, "draft": draft,
        "occlusion_context": {"duration": ctx["duration"],
            "candidates": [], "evidence_summary": ctx["evidence_summary"]},
    }


def test_review_without_candidates_returns_revised_scene():
    result, ctx = revision_fixture()
    result["scene"]["semantic_events"][0]["event_type"] = "out_of_frame"
    assert registry.sanitize("SceneRevision", result, ctx) == result
    assert ctx["draft"]["semantic_events"][0]["event_type"] == "move"


def test_out_of_frame_review_rejects_remaining_occluded_claim():
    from test_cv_summary import _candidate_artifact
    from percept_harness.cv.contracts import EvidenceThresholds
    from percept_harness.cv.summary import summarize_cv_evidence, build_cv_prompt_bundle
    from percept_harness.pipelines.scene_choices import prepare_scene_choices
    artifact, timeline = _candidate_artifact()
    summary = summarize_cv_evidence(artifact, timeline=timeline)
    bundle = build_cv_prompt_bundle(summary, EvidenceThresholds(
        min_confidence=0.5, min_area_fraction=0.01, occlusion_visibility_drop=0.5))
    candidate = next(c for c in bundle.candidates if c.target_entity_id == "edge_target")
    gap = next(o for o in candidate.allowed_event_intervals if o.event_type == "occluded")
    segments = [{"segment_index": 0, "start": 0.0, "end": 0.5, "target": "edge target"}]
    context = prepare_scene_choices(summary, segments, duration=0.5).context()
    scene = {"objects": [{"object_id": "edge_target", "name": "edge target", "description": "visible target"}],
        "locations": [], "relations": [], "initial_state": [], "final_state": [],
        "outcome": {"status": "unknown", "description": "not established", "confidence": 0.2},
        "semantic_events": [{"event_index": 0, "start": gap.start, "end": gap.end,
            "event_type": "occluded", "actor": "unknown", "target_object_id": "edge_target",
            "description": "target disappears", "confidence": 0.8}]}
    decisions = [{"candidate_id": c.candidate_id, "target_entity_id": c.target_entity_id,
        "classification": "out_of_frame" if c.candidate_id == candidate.candidate_id else "unknown",
        "occluder_entity_id": "unknown", "events": [], "visual_evidence": "target exits at the edge",
        "confidence": 0.8} for c in bundle.candidates]
    ctx = {"scene_context": context, "draft": copy.deepcopy(scene), "occlusion_context": {
        "duration": 0.5, "evidence_summary": summary.model_dump(mode="json"),
        "candidates": [c.model_dump(mode="json") for c in bundle.candidates]}}
    result = {"scene": scene, "decisions": decisions}
    rejected = registry.sanitize("SceneRevision", result, ctx)
    assert rejected["_schema_validation"]["issue_codes"] == ["SCENE_REVISION_VISIBILITY_CONFLICT"]
    result["scene"]["semantic_events"][0]["event_type"] = "out_of_frame"
    assert registry.sanitize("SceneRevision", result, ctx) == result


@pytest.mark.parametrize("fault", ["shape", "objects", "bounds"])
def test_invalid_revision_is_not_published_as_success(fault):
    result, ctx = revision_fixture()
    if fault == "shape":
        del result["scene"]
    elif fault == "objects":
        result["scene"]["objects"] = []
    else:
        result["scene"]["semantic_events"][0]["end"] = ctx["scene_context"]["duration"] + 1
    assert registry.sanitize("SceneRevision", result, ctx)["_schema_validation"]["status"] == "invalid"


def test_fake_cv_runs_complete_review_without_candidates(tmp_path, silent_video, monkeypatch):
    import json
    from percept_harness.cli import main
    monkeypatch.setenv("PERCEPT_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("PERCEPT_CV_CACHE_ROOT", str(tmp_path / "cv-cache"))
    output = tmp_path / "out"
    assert main(["annotate", "--videos", str(silent_video), "--template",
                 "embodied_action_captioning", "--backend", "fake", "--cv", "fake",
                 "--output", str(output)]) == 0
    data = json.loads((output / "demo.json").read_text())["data"]
    assert any(s.get("model_stage") == "occlusion_semantics" for s in data["performance"]["stages"])
    scene = data["annotation_branches"]["scene_facts"]["events"]
    assert all(e["model_stage"] == "occlusion_semantics" for e in scene)
    assert [e["event_type"] for e in scene] == [e["event_type"] for e in data["semantic_events"]]


def test_exhausted_review_fails_annotation_instead_of_publishing_draft(tmp_path, silent_video, monkeypatch):
    import json
    from percept_harness import cli
    from percept_harness.models.fake import FakeVideoModel
    model = FakeVideoModel(failure_script={"occlusion_semantics": [{}, {}, {}]})
    monkeypatch.setattr(cli, "_load_backend", lambda args, settings: (model, "qwen3-vl-8b-instruct", lambda: None))
    monkeypatch.setenv("PERCEPT_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("PERCEPT_CV_CACHE_ROOT", str(tmp_path / "cache"))
    output = tmp_path / "out"
    assert cli.main(["annotate", "--videos", str(silent_video), "--template",
                     "embodied_action_captioning", "--backend", "fake", "--cv", "fake",
                     "--output", str(output)]) == 1
    result = json.loads((output / "demo.json").read_text())
    assert result["status"] == "failed"
    assert result["data"] is None
