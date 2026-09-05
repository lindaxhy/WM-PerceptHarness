"""Hand-calculated alignment examples and hostile input boundaries."""

import copy
import hashlib
from pathlib import Path

import pytest

from las_repro.evaluation.las_alignment import (
    Annotation,
    Event,
    aggregate_metrics,
    canonical_json,
    evaluate_sample,
    load_mapping,
    match_events,
    strict_json,
    temporal_iou,
)


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


def mapping():
    return load_mapping("evaluation/config/las_alignment_mapping_v1.json")


def test_hand_calculated_metrics_and_null_denominators():
    reference = Annotation(
        "s",
        (
            Event(0, 2, "motion", actor="right hand", target="cup", event_id="r1"),
            Event(3, 5, "grasp", event_id="r2"),
        ),
        (Event(0, 2, "occluded", event_id="o1"),),
    )
    prediction = Annotation(
        "s",
        (
            Event(0, 4, "motion", actor="left hand", target="cup", event_id="p1"),
            Event(6, 8, "grasp", event_id="p2"),
        ),
        (Event(1, 3, "occluded", event_id="o2"),),
        component_statuses=("unavailable", "disabled", "available"),
        repair_count=2,
    )
    metrics = evaluate_sample(reference, prediction, mapping())
    assert metrics.action["0.3"]["f1"]["value"] == 0.5
    assert metrics.action["0.5"]["precision"]["value"] == 0.5
    assert metrics.occlusion["0.3"]["f1"]["value"] == 1
    assert metrics.occlusion["0.5"]["f1"]["value"] == 0
    assert metrics.occlusion_intervals["0.3"]["f1"]["value"] == 1
    assert metrics.occlusion_intervals["0.5"]["f1"]["value"] == 0
    assert metrics.occlusion_iou["value"] == pytest.approx(1 / 3)
    assert metrics.boundary_errors["enter"]["mean"]["value"] == 1
    assert metrics.boundary_errors["exit"]["median"]["value"] == 1
    assert metrics.fields["actor"]["macro_f1"]["value"] == 0
    assert metrics.fields["target"]["macro_f1"]["value"] == 1
    assert metrics.fields["state"]["macro_f1"]["value"] is None
    assert metrics.factual_precision["value"] is None
    total = aggregate_metrics([metrics])
    assert total.rates["degradation"]["value"] == pytest.approx(1 / 3)
    assert total.rates["repair"]["value"] == 1
    assert total.rates["completion"]["value"] == 1
    assert "NaN" not in canonical_json(total.to_dict())
    empty = evaluate_sample(Annotation("e"), Annotation("e"), mapping())
    assert empty.action["0.3"]["recall"] == {
        "value": None,
        "numerator": 0,
        "denominator": 0,
        "reason": "no reference events",
    }


@pytest.mark.parametrize(
    "raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}']
)
def test_strict_json_rejects_ambiguous_nonfinite_inputs(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_pretty_historical_json_is_accepted_without_changing_bytes():
    assert strict_json('{\n  "x": 1\n}') == {"x": 1}
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}\n'


def test_frozen_references_verify_bytes_and_closed_references(tmp_path):
    from las_repro.evaluation import las_alignment as las

    manifest = Path(
        "evaluation/references/las_official_english_2026-09-04/manifest.json"
    )
    refs, entries = las.load_references(manifest)
    assert len(refs) == 5
    assert len(refs["full_0001"].actions) == 6
    assert len(refs["full_0001"].occlusions) == 3
    assert len(refs["full_0001"].intervals) == 1
    raw = strict_json(
        (manifest.parent / entries["full_0001"]["reference_file"]).read_bytes()
    )
    for change in (
        lambda r: r["semantic_events"][0].update(object_ids=["foreign"]),
        lambda r: r["semantic_events"][0].update(confidence=float("nan")),
        lambda r: r["semantic_events"][0].update(end_s=99),
        lambda r: r.update(extra=1),
    ):
        bad = copy.deepcopy(raw)
        change(bad)
        with pytest.raises(ValueError):
            las.adapt_reference(bad, sample_id="s", duration=10.94)
    wrong = tmp_path / "manifest.json"
    wrong.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen manifest"):
        las.load_references(wrong)


def test_legacy_adapter_uses_existing_primary_projection_only():
    from test_hybrid_result import source_segments

    from las_repro.evaluation import las_alignment as las
    from las_repro.pipelines.semantic_events import build_semantic_events

    segments = source_segments()
    result = {
        "task_description": "move",
        "segments": segments,
        "grouped_semantic_events": build_semantic_events(segments),
        "objects": [],
        "initial_state": [],
        "final_state": [],
        "outcome": {"status": "unknown", "description": "unknown", "confidence": 0.0},
        "semantic_events": [],
        "warnings": [],
    }
    a = las.adapt_legacy(result, sample_id="s", duration=1)
    assert len(a.actions) == 1
    assert a.occlusions == ()
    assert a.repair_count is None
    bad = copy.deepcopy(result)
    bad["grouped_semantic_events"][0]["end"] = 0.5
    with pytest.raises(ValueError):
        las.adapt_legacy(bad, sample_id="s", duration=1)


def test_reviews_must_cover_exact_positive_events_and_digest():
    ref = Annotation("s", occlusions=(Event(0, 2, "occluded", event_id="r"),))
    pred = Annotation(
        "s",
        occlusions=(
            Event(
                0,
                2,
                "occluded",
                event_id="p",
                target_entity_id="cup",
                occluder_entity_id="board",
            ),
        ),
        provenance={"prediction_sha256": "a" * 64},
    )
    review = {
        "schema_version": "las_occlusion_review_v1",
        "sample_id": "s",
        "prediction_sha256": "a" * 64,
        "reviewer": "human-1",
        "reviewer_kind": "human",
        "claims": [
            {
                "event_id": "p",
                "correct": True,
                "target_entity_id": "cup",
                "occluder_entity_id": "board",
                "event_type": "occluded",
                "start": 0.0,
                "end": 2.0,
                "visual_reason": "The board covers the cup until it becomes visible again.",
            }
        ],
    }
    assert evaluate_sample(ref, pred, mapping(), review).factual_precision["value"] == 1
    assert evaluate_sample(ref, pred, mapping(), review).provenance[
        "human_review_sha256"
    ]
    for change in (
        lambda r: r.update(claims=[]),
        lambda r: r.update(prediction_sha256="b" * 64),
        lambda r: r["claims"][0].update(correct=None),
        lambda r: r.update(reviewer_kind="ai"),
        lambda r: r["claims"][0].update(target_entity_id="foreign"),
        lambda r: r["claims"][0].update(occluder_entity_id="foreign"),
        lambda r: r["claims"][0].update(event_type="occlusion_enter"),
        lambda r: r["claims"][0].update(start=0.1),
        lambda r: r["claims"][0].update(end=1.9),
        lambda r: r["claims"][0].update(start=False),
        lambda r: r["claims"][0].update(visual_reason=" "),
        lambda r: r["claims"][0].update(visual_reason="x" * 1025),
    ):
        bad = copy.deepcopy(review)
        change(bad)
        with pytest.raises(ValueError):
            evaluate_sample(ref, pred, mapping(), bad)


def test_metadata_binds_configuration_models_and_all_samples():
    from las_repro.evaluation import las_alignment as las

    metadata = metadata_fixture()
    las.validate_metadata(metadata, {"s"}, "qwen")
    for change in (
        lambda m: m["configuration"].update(fps=3),
        lambda m: m["samples"][0].update(model_identity="doubao-seed"),
        lambda m: m["runtime"].update(api_key="secret"),
        lambda m: m.update(samples=[]),
    ):
        bad = copy.deepcopy(metadata)
        change(bad)
        with pytest.raises(ValueError):
            las.validate_metadata(bad, {"s"}, "qwen")


def metadata_fixture():
    from las_repro.evaluation.las_alignment import digest

    config = {
        "fps": 2,
        "media_resolution": "medium",
        "clip_context": "high",
        "reasoning_effort": "high",
        "max_fine_segment_seconds": 1.0,
        "query_sha256": "a" * 64,
        "cv": None,
    }
    return {
        "schema_version": "las_evaluation_run_v1",
        "model_identity": "qwen3-vl-8b-instruct",
        "provider": "qwen3_vl",
        "configuration": config,
        "configuration_sha256": digest(config),
        "runtime": {
            "gpu_devices": ["0"],
            "model_revision": None,
            "checkpoint_sha256": None,
        },
        "samples": [
            {
                "sample_id": "s",
                "result_sha256": "b" * 64,
                "source_video_sha256": "c" * 64,
                "model_identity": "qwen3-vl-8b-instruct",
                "configuration_sha256": digest(config),
                "status": "COMPLETED",
                "wall_seconds": None,
                "stage_seconds": None,
            }
        ],
    }


def test_gate_requires_both_controls_same_model_and_all_positive_reviews():
    from las_repro.evaluation import las_alignment as las

    def aggregate(action=0.8, occ=0.5, reviewed=5, precision=0.8):
        return {
            "sample_count": 5,
            "action": {"0.3": {"f1": {"value": action}}},
            "occlusion": {"0.3": {"f1": {"value": occ}, "prediction_count": 5}},
            "factual_precision": {"value": precision, "denominator": reviewed},
            "rates": {"completion": {"value": 1}},
        }

    good = las.acceptance_gates(
        aggregate(), aggregate(), aggregate(), same_model=True, provenance_complete=True
    )
    assert all(good.values())
    assert not all(
        las.acceptance_gates(
            aggregate(),
            aggregate(),
            aggregate(reviewed=4),
            same_model=True,
            provenance_complete=True,
        ).values()
    )
    assert not all(
        las.acceptance_gates(
            aggregate(),
            aggregate(),
            aggregate(action=0.74),
            same_model=True,
            provenance_complete=True,
        ).values()
    )
    assert not all(
        las.acceptance_gates(
            aggregate(), None, aggregate(), same_model=False, provenance_complete=True
        ).values()
    )


def test_available_evidence_rebuilt_with_bound_config_and_real_artifact(
    tmp_path, monkeypatch
):
    from test_cv_artifacts import artifact_for, cv_request
    from test_hybrid_result import source_segments

    from las_repro.cv.artifacts import CvArtifactStore, cv_cache_key
    from las_repro.cv.summary import summarize_cv_evidence
    from las_repro.evaluation import las_alignment as las
    from las_repro.pipelines.hybrid_result import build_hybrid_result
    from las_repro.pipelines.scene_semantics import unavailable_scene_semantics

    media_bytes = b"decoder fixture with verified bytes"
    request = cv_request.__wrapped__().model_copy(
        update={"video_sha256": hashlib.sha256(media_bytes).hexdigest()}
    )
    artifact = artifact_for(request, b"unused").model_copy(
        update={"tracks": (), "files": ()}
    )
    root = tmp_path / "artifacts"
    with CvArtifactStore(root) as store, store.staging(cv_cache_key(request)) as stage:
        handle = store.publish(request, stage, artifact)
    config = {
        "sampling": request.sampling.model_dump(mode="json"),
        "thresholds": request.thresholds.model_dump(mode="json"),
        "summary_limits": {
            "max_tracks": 64,
            "max_observations_per_track": 64,
            "max_relations": 512,
            "max_overlays": 24,
            "max_prompt_chars": 200000,
        },
        "bundle_limits": {"max_candidates": 256, "max_prompt_chars": 200000},
    }
    result = build_hybrid_result(
        task_description="move",
        segments=source_segments(),
        scene=unavailable_scene_semantics(),
        scene_status="unavailable",
        cv_evidence={
            "status": "available",
            "artifact_key": handle.key,
            "manifest_sha256": handle.manifest_sha256,
            "cache_hit": False,
        },
        warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}],
        performance={
            "stages": [],
            "total_seconds": 0.0,
            "repair_count": 0,
            "degradation_count": 1,
        },
        evidence_summary=summarize_cv_evidence(artifact, timeline=request.timeline),
    )
    adapted = las.adapt_hybrid(
        result,
        sample_id="s",
        duration=3,
        configuration={"cv": config},
        artifact_root=root,
        timeline=request.timeline,
        video_sha256=request.video_sha256,
    )
    assert adapted.occlusions == ()
    assert adapted.component_statuses == ("available", "unavailable", "available")
    assert adapted.provenance["cv_model"]["provider"] == "fake"
    with pytest.raises(ValueError, match="SAM 3.1"):
        las.adapt_hybrid(
            result,
            sample_id="s",
            duration=3,
            configuration={"cv": config},
            artifact_root=root,
            timeline=request.timeline,
            video_sha256=request.video_sha256,
            require_sam31=True,
        )
    # Only replace the decoder: artifact storage, hashes, sidecar validation,
    # run adapter, and production SAM policy execute unchanged.
    monkeypatch.setattr(
        "las_repro.cv.timeline.probe_frame_timeline", lambda _: request.timeline
    )
    results_dir, media_dir = tmp_path / "results", tmp_path / "media"
    results_dir.mkdir()
    media_dir.mkdir()
    result_bytes = canonical_json(result).encode()
    (results_dir / "s.json").write_bytes(result_bytes)
    (media_dir / "s.mp4").write_bytes(media_bytes)
    metadata = metadata_fixture()
    metadata["configuration"]["cv"] = config
    from las_repro.evaluation.las_alignment import digest

    metadata["configuration_sha256"] = digest(metadata["configuration"])
    metadata["samples"][0].update(
        configuration_sha256=metadata["configuration_sha256"],
        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        source_video_sha256=request.video_sha256,
    )
    with pytest.raises(ValueError, match="SAM 3.1"):
        las.evaluate_run(
            results_dir,
            metadata,
            {"s": Annotation("s")},
            {
                "s": {
                    "source_video": {
                        "sha256": request.video_sha256,
                        "duration_seconds": 3.0,
                    }
                }
            },
            mapping(),
            role="hybrid",
            media_dir=media_dir,
            artifact_root=root,
        )
    bad = copy.deepcopy(config)
    bad["thresholds"]["min_confidence"] = 0.7
    with pytest.raises(ValueError, match="artifact configuration"):
        las.adapt_hybrid(
            result,
            sample_id="s",
            duration=3,
            configuration={"cv": bad},
            artifact_root=root,
            timeline=request.timeline,
            video_sha256=request.video_sha256,
        )
    with pytest.raises(ValueError):
        las.adapt_hybrid(
            result,
            sample_id="s",
            duration=3,
            configuration={"cv": config},
            artifact_root=None,
            timeline=request.timeline,
            video_sha256=request.video_sha256,
        )
    bad = copy.deepcopy(result)
    bad["annotation_branches"]["action_events"][0]["source_track_ids"] = ["foreign"]
    with pytest.raises(ValueError):
        las.adapt_hybrid(
            bad,
            sample_id="s",
            duration=3,
            configuration={"cv": config},
            artifact_root=root,
            timeline=request.timeline,
            video_sha256=request.video_sha256,
        )


def test_report_writer_is_canonical_idempotent_and_refuses_overwrite(tmp_path):
    from las_repro.evaluation import las_alignment as las

    path = tmp_path / "report.json"
    las.write_report(path, {"b": 1, "a": 2})
    assert path.read_bytes() == b'{"a":2,"b":1}\n'
    las.write_report(path, {"a": 2, "b": 1})
    with pytest.raises(ValueError):
        las.write_report(path, {"a": 3})
    assert path.read_bytes() == b'{"a":2,"b":1}\n'
    las.write_report(path, {"a": 3}, replace=True)
    assert strict_json(path.read_bytes()) == {"a": 3}


def test_cli_refuses_missing_samples_without_output(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_las_alignment.py",
            "--reference-manifest",
            "evaluation/references/las_official_english_2026-09-04/manifest.json",
            "--qwen-results",
            str(tmp_path),
            "--hybrid-results",
            str(tmp_path),
            "--qwen-metadata",
            str(tmp_path / "missing.json"),
            "--hybrid-metadata",
            str(tmp_path / "missing.json"),
            "--mapping",
            "evaluation/config/las_alignment_mapping_v1.json",
            "--output",
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "evaluation input validation failed" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_mapping_is_frozen_and_rejects_output_tuned_changes(tmp_path):
    bad = mapping()
    bad["target"]["red apple"] = "apple"
    path = tmp_path / "mapping.json"
    path.write_text(canonical_json(bad))
    with pytest.raises(ValueError):
        load_mapping(path)


def test_no_raw_masks_in_legacy_metric_inputs():
    from las_repro.evaluation import las_alignment as las

    with pytest.raises(ValueError):
        las.Event(0, 1, "motion", target="masks/0.npz")


def test_state_result_macro_f1_uses_comparable_labels():
    ref = Annotation(
        "s",
        actions=(
            Event(0, 1, "move", state="open", result="success"),
            Event(1, 2, "move", state="closed", result="failure"),
        ),
    )
    pred = Annotation(
        "s",
        actions=(
            Event(0, 1, "move", state="open", result="success"),
            Event(1, 2, "move", state="open", result="success"),
        ),
    )
    metrics = evaluate_sample(ref, pred, mapping())
    assert metrics.fields["state"]["macro_f1"]["value"] == pytest.approx(1 / 3)
    assert metrics.fields["result"]["macro_f1"]["value"] == pytest.approx(1 / 3)


def test_iou_secondary_objective_selects_best_of_equal_cardinality():
    refs = [Event(0, 2, "move"), Event(1, 3, "move")]
    preds = [Event(1, 3, "move"), Event(0, 2, "move")]
    matches = match_events(refs, preds, 0.3)
    assert [(m.reference_index, m.prediction_index) for m in matches] == [
        (0, 1),
        (1, 0),
    ]
    assert sum(m.iou for m in matches) == 2


def test_wrong_action_labels_reduce_field_f1_without_type_compatible_events():
    ref = Annotation("s", actions=(Event(0, 2, "grasp"),), outcome="success")
    pred = Annotation("s", actions=(Event(0, 2, "move"),), outcome="failure")
    metrics = evaluate_sample(ref, pred, mapping())
    assert metrics.action["0.3"]["matched_count"] == 0
    assert metrics.fields["action"]["macro_f1"]["value"] == 0
    assert metrics.fields["action"]["matched_pairs"] == 1
    assert metrics.fields["actor"]["excluded_pairs"] == 1
    assert metrics.fields["actor"]["exclusion_reason"] == "unknown or unavailable label"
    assert metrics.fields["result"]["macro_f1"]["value"] == 0


def test_field_assignment_unmatched_ids_are_distinct_from_event_matching():
    refs = Annotation(
        "s",
        actions=(
            Event(0, 1, "grasp", event_id="r1"),
            Event(3, 4, "grasp", event_id="r2"),
        ),
    )
    preds = Annotation(
        "s",
        actions=(
            Event(0, 1, "move", event_id="p1"),
            Event(6, 7, "move", event_id="p2"),
        ),
    )
    metrics = evaluate_sample(refs, preds, mapping())
    assert metrics.action["0.3"]["unmatched_reference_ids"] == ["r1", "r2"]
    assert metrics.action["0.3"]["unmatched_prediction_ids"] == ["p1", "p2"]
    assert metrics.field_temporal_assignment == {
        "threshold": 0.3,
        "matched_count": 1,
        "unmatched_reference_ids": ["r2"],
        "unmatched_prediction_ids": ["p2"],
    }
    assert metrics.to_dict()["field_temporal_assignment"][
        "unmatched_prediction_ids"
    ] == ["p2"]


def test_unknown_events_are_not_positive_matches():
    assert match_events([Event(0, 1, "unknown")], [Event(0, 1, "unknown")], 0.3) == []


def test_frozen_qwen_identity_cannot_be_relabelled(tmp_path):
    from las_repro.evaluation import las_alignment as las

    metadata = metadata_fixture()
    metadata["model_identity"] = metadata["samples"][0]["model_identity"] = "qwen-other"
    with pytest.raises(ValueError, match="frozen Qwen metadata"):
        las.evaluate_run(
            tmp_path,
            metadata,
            {"s": Annotation("s")},
            {},
            mapping(),
            role="qwen",
            media_dir=tmp_path,
        )


def test_offline_five_sample_cli_with_local_frozen_inputs(tmp_path):
    """Optional integration: existing frozen bytes, synthetic unavailable hybrid."""
    from las_repro.evaluation import las_alignment as las
    from las_repro.pipelines.hybrid_result import build_hybrid_result
    from las_repro.pipelines.scene_semantics import unavailable_scene_semantics

    qwen = Path("outputs/five-demo/qwen-only")
    if not all(
        (qwen / f"{sid}.json").exists()
        and (Path("evaluation/viewer/media") / f"{sid}.mp4").exists()
        for sid in las.FROZEN_QWEN
    ):
        pytest.skip("local frozen result/media fixtures unavailable")
    refs, entries = las.load_references(
        "evaluation/references/las_official_english_2026-09-04/manifest.json"
    )
    baseline = metadata_fixture()
    baseline["configuration"]["query_sha256"] = (
        "bf94e33ffb1dc1ccd40f225766281ad39b3c7f18d4b3b6d6280e2b04d37f9f2a"
    )
    baseline["configuration_sha256"] = las.digest(baseline["configuration"])
    baseline["samples"] = []
    hybrid = copy.deepcopy(baseline)
    result_dir = tmp_path / "hybrid"
    result_dir.mkdir()
    for sid in refs:
        old = strict_json((qwen / f"{sid}.json").read_bytes())
        result = build_hybrid_result(
            task_description="fixture",
            segments=old["segments"],
            scene=unavailable_scene_semantics(),
            scene_status="unavailable",
            cv_evidence={"status": "disabled"},
            warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}],
            performance={
                "stages": [],
                "total_seconds": 0.0,
                "repair_count": 0,
                "degradation_count": 1,
            },
        )
        raw = canonical_json(result).encode()
        (result_dir / f"{sid}.json").write_bytes(raw)
        item = {
            "sample_id": sid,
            "result_sha256": las.FROZEN_QWEN[sid],
            "source_video_sha256": entries[sid]["source_video"]["sha256"],
            "model_identity": baseline["model_identity"],
            "configuration_sha256": baseline["configuration_sha256"],
            "status": "COMPLETED",
            "wall_seconds": None,
            "stage_seconds": None,
        }
        baseline["samples"].append(item)
        hybrid["samples"].append(
            {**item, "result_sha256": hashlib.sha256(raw).hexdigest()}
        )
    bpath, hpath = tmp_path / "baseline-meta.json", tmp_path / "hybrid-meta.json"
    bpath.write_text(canonical_json(baseline))
    hpath.write_text(canonical_json(hybrid))
    output = tmp_path / "report.json"
    args = [
        "--reference-manifest",
        "evaluation/references/las_official_english_2026-09-04/manifest.json",
        "--qwen-results",
        str(qwen),
        "--hybrid-results",
        str(result_dir),
        "--qwen-metadata",
        str(bpath),
        "--hybrid-metadata",
        str(hpath),
        "--mapping",
        "evaluation/config/las_alignment_mapping_v1.json",
        "--output",
        str(output),
    ]
    assert las.main(args) == 0
    report = strict_json(output.read_bytes())
    assert report["accepted"] is False
    assert report["acceptance_gates"]["reviewed_precision_at_least_0_80"] is False
    assert report["runs"]["qwen"]["aggregate"]["sample_count"] == 5
    assert report["runs"]["hybrid"]["aggregate"]["rates"]["degradation"][
        "value"
    ] == pytest.approx(1 / 3)
    before = output.read_bytes()
    assert las.main(args) == 0
    assert output.read_bytes() == before


def test_reference_rejects_wrong_collection_types():
    from las_repro.evaluation import las_alignment as las

    path = Path(
        "evaluation/references/las_official_english_2026-09-04/references/full_0001.json"
    )
    raw = strict_json(path.read_bytes())
    raw["occlusions"] = {}
    with pytest.raises(ValueError):
        las.adapt_reference(raw, sample_id="s", duration=11)


def test_metadata_rejects_unsafe_identity_and_unknown_timing_stages():
    from las_repro.evaluation import las_alignment as las

    for change in (
        lambda m: m["samples"][0].update(stage_seconds={"api_key": 1.0}),
        lambda m: m.update(model_identity="qwen/private/path"),
    ):
        bad = metadata_fixture()
        change(bad)
        if bad["model_identity"] != bad["samples"][0]["model_identity"]:
            bad["samples"][0]["model_identity"] = bad["model_identity"]
        with pytest.raises(ValueError):
            las.validate_metadata(bad, {"s"}, "qwen")


@pytest.mark.parametrize("drift", [None, "model_identity", "checkpoint_sha256"])
def test_run_requires_consistent_cv_identity_across_two_verified_artifacts(
    tmp_path, monkeypatch, drift
):
    from test_cv_artifacts import artifact_for, cv_request
    from test_hybrid_result import source_segments

    from las_repro.cv.artifacts import CvArtifactStore, cv_cache_key
    from las_repro.evaluation import las_alignment as las
    from las_repro.pipelines.hybrid_result import build_hybrid_result
    from las_repro.pipelines.scene_semantics import unavailable_scene_semantics

    template = cv_request.__wrapped__().model_copy(
        update={"provider": "sam31", "model_identity": "sam31-pinned"}
    )
    config = {
        "sampling": template.sampling.model_dump(mode="json"),
        "thresholds": template.thresholds.model_dump(mode="json"),
        "summary_limits": {
            "max_tracks": 64,
            "max_observations_per_track": 64,
            "max_relations": 512,
            "max_overlays": 24,
            "max_prompt_chars": 200000,
        },
        "bundle_limits": {"max_candidates": 256, "max_prompt_chars": 200000},
    }
    metadata = metadata_fixture()
    metadata["configuration"]["cv"] = config
    metadata["configuration_sha256"] = las.digest(metadata["configuration"])
    sample_template = metadata["samples"][0]
    metadata["samples"] = []
    root, results, media = (
        tmp_path / "artifacts",
        tmp_path / "results",
        tmp_path / "media",
    )
    results.mkdir()
    media.mkdir()
    references, entries = {}, {}
    # The decoder alone is stubbed; source bytehashes, both artifact manifests,
    # cache identity reconstruction, metadata, and run validation are real.
    monkeypatch.setattr(
        "las_repro.cv.timeline.probe_frame_timeline", lambda _: template.timeline
    )
    for sid in ("s1", "s2"):
        source = sid.encode()
        changes = {"video_sha256": hashlib.sha256(source).hexdigest()}
        if sid == "s2" and drift:
            changes[drift] = "other-sam31" if drift == "model_identity" else "c" * 64
        request = template.model_copy(update=changes)
        artifact = artifact_for(request, b"unused").model_copy(
            update={"tracks": (), "files": ()}
        )
        with (
            CvArtifactStore(root) as store,
            store.staging(cv_cache_key(request)) as stage,
        ):
            handle = store.publish(request, stage, artifact)
        result = build_hybrid_result(
            task_description="fixture",
            segments=source_segments(),
            scene=unavailable_scene_semantics(),
            scene_status="unavailable",
            cv_evidence={
                "status": "available",
                "artifact_key": handle.key,
                "manifest_sha256": handle.manifest_sha256,
                "cache_hit": False,
            },
            warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}],
            performance={
                "stages": [],
                "total_seconds": 0.0,
                "repair_count": 0,
                "degradation_count": 1,
            },
        )
        raw = canonical_json(result).encode()
        (results / f"{sid}.json").write_bytes(raw)
        (media / f"{sid}.mp4").write_bytes(source)
        metadata["samples"].append(
            {
                **sample_template,
                "sample_id": sid,
                "result_sha256": hashlib.sha256(raw).hexdigest(),
                "source_video_sha256": request.video_sha256,
                "configuration_sha256": metadata["configuration_sha256"],
            }
        )
        references[sid] = Annotation(sid)
        entries[sid] = {
            "source_video": {"sha256": request.video_sha256, "duration_seconds": 3.0}
        }
    if drift:
        with pytest.raises(ValueError, match="mixed CV model"):
            las.evaluate_run(
                results,
                metadata,
                references,
                entries,
                mapping(),
                role="hybrid",
                media_dir=media,
                artifact_root=root,
            )
    else:
        run = las.evaluate_run(
            results,
            metadata,
            references,
            entries,
            mapping(),
            role="hybrid",
            media_dir=media,
            artifact_root=root,
        )
        assert run["cv_model"] == {
            "provider": "sam31",
            "model_identity": "sam31-pinned",
            "checkpoint_sha256": "b" * 64,
        }
