"""Display-only projection preserves evidence without publishing private payloads."""

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import test_hybrid_result as hybrid_fixtures
from test_hybrid_result import source_segments

from las_repro.evaluation.las_alignment import canonical_json
from las_repro.pipelines.hybrid_result import build_hybrid_result
from las_repro.pipelines.scene_semantics import unavailable_scene_semantics

available_result = hybrid_fixtures.available_result


@pytest.fixture
def result():
    return build_hybrid_result(
        task_description="move cup",
        segments=source_segments(),
        scene=unavailable_scene_semantics(),
        scene_status="unavailable",
        cv_evidence={"status": "disabled"},
        warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}],
        performance={
            "stages": [],
            "total_seconds": 2.5,
            "repair_count": 0,
            "degradation_count": 1,
        },
    )


def project(result, **kwargs):
    from las_repro.evaluation.viewer_projection import project_hybrid_viewer_data

    return project_hybrid_viewer_data(
        "full_0001", 1.0, result, source_sha256="c" * 64, **kwargs
    )


def test_projection_retains_canonical_branches_and_distinct_digest_meanings(result):
    before = copy.deepcopy(result)
    output = project(
        result,
        source_result_sha256="d" * 64,
        model_identity="doubao-seed-2-1-pro-260628",
    )
    assert set(output) == {
        "schema_version",
        "sample",
        "layers",
        "provenance",
        "warnings",
    }
    assert output["schema_version"] == "comparison_viewer_hybrid_v1"
    assert output["sample"] == {"sample_id": "full_0001", "duration_seconds": 1.0}
    assert output["provenance"]["source_result_sha256"] == "d" * 64
    assert (
        output["provenance"]["canonical_result_sha256"]
        == hashlib.sha256(canonical_json(result).encode()).hexdigest()
    )
    assert output["provenance"]["source_video_sha256"] == "c" * 64
    assert output["provenance"]["model_identity"] == "doubao-seed-2-1-pro-260628"
    assert output["provenance"]["performance"] == {
        "total_seconds": 2.5,
        "repair_count": 0,
        "degradation_count": 1,
    }
    event = output["layers"]["action_events"]["events"][0]
    assert event["id"] == "action_0"
    assert event["confidence"] == 0.8
    assert event["evidence_mode"] == "vlm_only"
    assert event["source_segment_indices"] == [0]
    assert event["review_status"] == "not_required"
    assert output["layers"]["occlusion_events"] == {"status": "disabled", "events": []}
    assert output["layers"]["scene_facts"]["status"] == "unavailable"
    assert "fine_segments" not in output["layers"]
    assert "stages" not in output["provenance"]["performance"]
    assert output["warnings"] == [{"code": "SCENE_SEMANTICS_UNAVAILABLE"}]
    assert result == before


def test_fine_layer_is_opt_in_and_retains_source_identity(result):
    event = project(result, include_fine_segments=True)["layers"]["fine_segments"][
        "events"
    ][0]
    assert event["id"] == "fine_0"
    assert event["segment_index"] == 0
    assert event["skill"] == "move"
    assert event["source_segment_indices"] == [0]
    assert event["review_status"] == "not_required"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(job_payload={"secret": "private"}),
        lambda r: r["segments"][0].update(prompt="secret prompt"),
        lambda r: r["segments"][0].update(mask=[[0, 1]]),
        lambda r: r.update(task_description="/root/private"),
        lambda r: r["annotation_branches"]["action_events"][0].update(start=-1),
        lambda r: r["annotation_branches"]["action_events"][0].update(
            confidence=float("nan")
        ),
        lambda r: r["cv_evidence"].update(error="secret"),
    ],
)
def test_private_unknown_and_invalid_source_fields_fail_closed(result, mutation):
    mutation(result)
    with pytest.raises(ValueError):
        project(result)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_result_sha256": "bad"},
        {"model_identity": "/root/model"},
        {"include_fine_segments": 1},
        {"overlay_references": [{"path": "/private/overlay.png"}]},
    ],
)
def test_invalid_optional_projection_context_is_rejected(result, kwargs):
    with pytest.raises(ValueError):
        project(result, **kwargs)


@pytest.mark.parametrize(
    "sample,duration,sha",
    [
        ("../s", 1.0, "c" * 64),
        ("s", 0.9, "c" * 64),
        ("s", 2.0, "c" * 64),
        ("s", True, "c" * 64),
        ("s", float("nan"), "c" * 64),
        ("s", 1.0, "C" * 64),
    ],
)
def test_sample_duration_and_source_digest_are_bound(result, sample, duration, sha):
    from las_repro.evaluation.viewer_projection import project_hybrid_viewer_data

    with pytest.raises(ValueError):
        project_hybrid_viewer_data(sample, duration, result, source_sha256=sha)


def test_available_scene_preserves_spatial_evidence(available_result):
    result, _ = available_result
    output = project(result)
    row = output["layers"]["scene_facts"]["locations"][0]
    assert row["location"] == "center"
    assert row["source_track_ids"] == ["cup_1"]
    assert output["layers"]["scene_facts"]["events"][0]["id"] == "scene_0"
    assert output["provenance"]["cv_evidence"]["manifest_sha256"] == "b" * 64
    assert "/root/" not in json.dumps(output)


@pytest.fixture
def positive_result(result):
    decision = {
        "candidate_id": "occ_aaaaaaaaaaaa_0000",
        "classification": "occlusion",
        "target_entity_id": "cup",
        "occluder_entity_id": "board",
        "visual_evidence": "Board covers cup",
        "confidence": 0.8,
        "events": [{"event_type": "occluded", "start": 0.2, "end": 0.8}],
    }
    event = {
        "event_index": 0,
        "start": 0.2,
        "end": 0.8,
        "event_type": "occluded",
        "target_entity_id": "cup",
        "occluder_entity_id": "board",
        "description": "Board covers cup",
        "confidence": 0.8,
        "source_candidate_id": "occ_aaaaaaaaaaaa_0000",
        "branch": "occlusion",
        "model_stage": "occlusion_semantics",
        "source_segment_indices": [0],
        "source_track_ids": ["cup_1"],
        "source_keyframe_ids": ["frame_0002"],
        "evidence_mode": "hybrid",
        "repair_history": ["initial"],
        "review_status": "unreviewed",
    }
    result["cv_evidence"] = {
        "status": "available",
        "artifact_key": "a" * 64,
        "manifest_sha256": "b" * 64,
        "cache_hit": False,
    }
    result["annotation_branches"]["occlusion"] = {
        "status": "available",
        "events": [event],
        "decisions": [decision],
    }
    return result


def review_for(result):
    return {
        "schema_version": "las_occlusion_review_v1",
        "sample_id": "full_0001",
        "prediction_sha256": hashlib.sha256(
            canonical_json(result).encode()
        ).hexdigest(),
        "reviewer": "human-reviewer",
        "reviewer_kind": "human",
        "claims": [
            {
                "event_id": "occlusion_0",
                "correct": True,
                "target_entity_id": "cup",
                "occluder_entity_id": "board",
                "event_type": "occluded",
                "start": 0.2,
                "end": 0.8,
                "visual_reason": "The board visibly covers the cup.",
            }
        ],
    }


def test_complete_human_review_only_changes_matching_positive_claim(positive_result):
    review = review_for(positive_result)
    before = copy.deepcopy(positive_result)
    output = project(positive_result, review=review)
    assert (
        output["layers"]["occlusion_events"]["events"][0]["review_status"]
        == "supported"
    )
    assert output["layers"]["occlusion_events"]["events"][0]["review"] == {
        "reviewer": "human-reviewer",
        "visual_reason": "The board visibly covers the cup.",
    }
    review["claims"][0]["correct"] = False
    assert (
        project(positive_result, review=review)["layers"]["occlusion_events"]["events"][
            0
        ]["review_status"]
        == "unsupported"
    )
    assert (
        project(positive_result)["layers"]["occlusion_events"]["events"][0][
            "review_status"
        ]
        == "unreviewed"
    )
    assert positive_result == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(claims=[]),
        lambda r: r["claims"].append(copy.deepcopy(r["claims"][0])),
        lambda r: r["claims"][0].update(event_id="foreign"),
        lambda r: r.update(prediction_sha256="e" * 64),
        lambda r: r.update(reviewer_kind="ai"),
        lambda r: r["claims"][0].update(start=0.0),
        lambda r: r["claims"][0].update(target_entity_id="board"),
        lambda r: r["claims"][0].update(correct=1),
        lambda r: r["claims"][0].update(visual_reason="/root/private"),
    ],
)
def test_review_requires_exact_identity_coverage_and_safe_evidence(
    positive_result, mutation
):
    review = review_for(positive_result)
    mutation(review)
    with pytest.raises(ValueError):
        project(positive_result, review=review)


def overlay():
    return {
        "keyframe_id": "frame_0002",
        "track_id": "cup_1",
        "frame_index": 2,
        "timestamp_seconds": 0.2,
        "path": "evaluation/viewer/data/hybrid/overlays/" + "e" * 64 + ".png",
        "sha256": "e" * 64,
        "size_bytes": 100,
    }


def test_only_referenced_digest_named_png_metadata_is_published(positive_result):
    reference = overlay()
    output = project(positive_result, overlay_references=[reference])
    assert output["provenance"]["overlays"] == [reference]
    assert "mask" not in json.dumps(output)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(path="/root/frame.png"),
        lambda r: r.update(path="evaluation/../frame.png"),
        lambda r: r.update(path="evaluation/%2e%2e/frame.png"),
        lambda r: r.update(path="evaluation/frame.npz"),
        lambda r: r.update(path="https://example.com/frame.png"),
        lambda r: r.update(sha256="f" * 64),
        lambda r: r.update(keyframe_id="foreign"),
        lambda r: r.update(track_id="foreign"),
        lambda r: r.update(timestamp_seconds=1.0),
        lambda r: r.update(size_bytes=0),
        lambda r: r.update(frame_index=True),
        lambda r: r.update(mask=[]),
    ],
)
def test_overlay_references_are_closed_bounded_and_bound_to_events(
    positive_result, mutation
):
    reference = overlay()
    mutation(reference)
    with pytest.raises(ValueError):
        project(positive_result, overlay_references=[reference])


@pytest.mark.parametrize("keyframe", ["frame_0002", "frame.0002", "frame_" + "a" * 150])
def test_python_projection_is_accepted_by_browser_model(positive_result, keyframe):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the cross-language viewer contract")
    positive_result["annotation_branches"]["occlusion"]["events"][0][
        "source_keyframe_ids"
    ] = [keyframe]
    reference = overlay()
    reference["keyframe_id"] = keyframe
    projected = project(
        positive_result,
        include_fine_segments=True,
        overlay_references=[reference],
        review=review_for(positive_result),
    )
    checked = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            """
import {normalizeHybrid} from './evaluation/viewer/js/model.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const value = normalizeHybrid(JSON.parse(input), 1, 'full_0001');
console.log(JSON.stringify({modes: value.availableModes, review: value.occlusion[0].meta.reviewStatus}));
""",
        ],
        input=canonical_json(projected),
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        timeout=10,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == {
        "modes": ["grouped", "occlusion", "fine"],
        "review": "supported",
    }
