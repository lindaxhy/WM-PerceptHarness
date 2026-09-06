"""Canonical hybrid result projections and trust boundaries."""

import copy

import pytest

from las_repro.pipelines import hybrid_result
from las_repro.pipelines.scene_semantics import unavailable_scene_semantics


def source_segments():
    return [
        {
            "segment_index": 0,
            "start": 0.0,
            "end": 1.0,
            "actor": "right_hand",
            "skill": "move",
            "target": "cup",
            "description": "hand moves cup",
            "confidence": 0.8,
        }
    ]


def test_canonical_projection_preserves_legacy_fields_and_empty_evidence():
    result = hybrid_result.build_hybrid_result(
        task_description="move cup",
        segments=source_segments(),
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
    hybrid_result.validate_hybrid_result(result)
    action = result["annotation_branches"]["action_events"][0]
    assert action["evidence_mode"] == "vlm_only"
    assert action["source_track_ids"] == []
    assert hybrid_result.legacy_action_projection(action) == {
        "event_index": 0,
        "start": 0.0,
        "end": 1.0,
        "actor": "right_hand",
        "action": "motion",
        "target": "cup",
        "description": "hand moves cup",
        "confidence": 0.8,
        "source_segment_indices": [0],
    }
    assert result["semantic_events"] == []
    assert result["annotation_branches"]["occlusion"] == {
        "status": "disabled",
        "decisions": [],
        "events": [],
    }
    for mutation in (
        lambda r: r["grouped_semantic_events"][0].update(confidence=0.1),
        lambda r: r["annotation_branches"]["action_events"][0].update(
            source_track_ids=["foreign"]
        ),
        lambda r: r["cv_evidence"].update(mask_path="/private/mask.npz"),
        lambda r: r["annotation_branches"]["action_events"][0].update(
            source_segment_indices=[9]
        ),
    ):
        bad = copy.deepcopy(result)
        mutation(bad)
        with pytest.raises(ValueError):
            hybrid_result.validate_hybrid_result(bad)


@pytest.fixture
def available_result():
    from test_cv_summary import _artifact, _observation, _track

    from las_repro.cv.summary import summarize_cv_evidence

    artifact = _artifact(
        (_track("cup_1", "cup", tuple(_observation(i) for i in (0, 5, 10))),)
    )
    summary = summarize_cv_evidence(artifact)
    scene = unavailable_scene_semantics()
    scene["objects"] = [
        {"object_id": "cup", "name": "cup", "description": "visible cup"}
    ]
    scene["semantic_events"] = [
        {
            "event_index": 0,
            "start": 0.0,
            "end": 1.0,
            "event_type": "move",
            "actor": "right_hand",
            "target_object_id": "cup",
            "description": "cup moves",
            "confidence": 0.8,
        }
    ]
    scene["locations"] = [
        {
            "object_id": "cup",
            "location": "center",
            "start": 0.0,
            "end": 1.0,
            "visual_evidence": "cup visible in center",
            "confidence": 0.8,
            "branch": "scene",
            "model_stage": "scene_semantics",
            "evidence_mode": "hybrid",
            "source_track_ids": ["cup_1"],
            "source_keyframe_ids": [],
            "source_segment_indices": [0],
            "repair_history": ["initial"],
            "review_status": "not_required",
        }
    ]
    result = hybrid_result.build_hybrid_result(
        task_description="move cup",
        segments=source_segments(),
        scene=scene,
        scene_status="available",
        evidence_summary=summary,
        cv_evidence={
            "status": "available",
            "artifact_key": "a" * 64,
            "manifest_sha256": "b" * 64,
            "cache_hit": False,
        },
        warnings=[],
        performance={
            "stages": [],
            "total_seconds": 0.0,
            "repair_count": 0,
            "degradation_count": 0,
        },
    )
    return result, summary


@pytest.mark.parametrize("fault", ["ordering", "provenance"])
def test_scene_spatial_failure_survives_registry(available_result, fault):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
    from las_repro.pipelines.embodied import _validated_stage_result

    result, summary = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    context = {
        "duration": 1.0,
        "require_observed_content": True,
        "required_object_ids": ["cup"],
        "segments": source_segments(),
        "evidence_summary": summary.model_dump(mode="json"),
    }
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context) == scene
    if fault == "ordering":
        later = copy.deepcopy(scene["locations"][0])
        later["start"] = 0.5
        scene["locations"].insert(0, later)
    else:
        scene["locations"][0]["source_segment_indices"] = [5]
    before = copy.deepcopy(scene)
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context)
    expected_codes = ["SCENE_SPATIAL_INVALID", (
        "SCENE_SPATIAL_ORDER_INVALID" if fault == "ordering"
        else "SCENE_SPATIAL_SOURCE_SEGMENTS_INVALID"
    )]
    assert sanitized == {
        "_schema_validation": {
            "schema_name": "SceneSemantics",
            "status": "invalid",
            "issue_codes": expected_codes,
        }
    }
    assert scene == before
    _, codes, _ = _validated_stage_result("SceneSemantics", sanitized, context)
    assert list(codes) == expected_codes


def test_scene_repair_aggregates_independent_faults_without_leaking(available_result):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
    from las_repro.pipelines.embodied import _validated_stage_result

    result, summary = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    later = copy.deepcopy(scene["locations"][0])
    later["start"] = 0.5
    scene["locations"].insert(0, later)
    scene["locations"][1].update(
        end=0.93, source_segment_indices=[], source_track_ids=[],
        visual_evidence="private_marker left/right",
    )
    context = {
        "duration": 1.0, "require_observed_content": True,
        "required_object_ids": ["cup"], "segments": source_segments(),
        "evidence_summary": summary.model_dump(mode="json"),
    }
    before = copy.deepcopy(scene)
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context)
    expected = [
        "SCENE_SPATIAL_INVALID", "SCENE_SPATIAL_ORDER_INVALID",
        "SCENE_SPATIAL_TIME_NOT_OBSERVED", "SCENE_SPATIAL_SOURCE_SEGMENTS_INVALID",
        "SCENE_SPATIAL_TRACKS_INVALID", "SCENE_SPATIAL_PROVENANCE_INVALID",
        "SCENE_SPATIAL_PROHIBITED_CONTENT",
    ]
    assert sanitized == {"_schema_validation": {
        "schema_name": "SceneSemantics", "status": "invalid", "issue_codes": expected,
    }}
    _, codes, _ = _validated_stage_result("SceneSemantics", sanitized, context)
    assert list(codes) == expected
    assert scene == before


@pytest.mark.parametrize("field,value,code", [
    ("end", 0.93, "TIME_NOT_OBSERVED"),
    ("end", 1.1, "TIME_BOUNDS_INVALID"),
    ("object_id", "missing", "OBJECT_REFERENCE_INVALID"),
    ("source_track_ids", [], "TRACKS_INVALID"),
    ("source_track_ids", ["foreign_track"], "TRACKS_INVALID"),
    ("source_track_ids", ["cup_1", "cup_1"], "PROVENANCE_INVALID"),
    ("source_keyframe_ids", ["foreign_frame"], "KEYFRAMES_INVALID"),
    ("source_segment_indices", [], "SOURCE_SEGMENTS_INVALID"),
    ("repair_history", [], "PROVENANCE_INVALID"),
    ("visual_evidence", "moves left/right", "PROHIBITED_CONTENT"),
])
def test_scene_repair_reports_specific_closed_fault(available_result, field, value, code):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    result, summary = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    scene["locations"][0][field] = value
    context = {
        "duration": 1.0, "require_observed_content": True,
        "required_object_ids": ["cup"], "segments": source_segments(),
        "evidence_summary": summary.model_dump(mode="json"),
    }
    before = copy.deepcopy(scene)
    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context)
    codes = sanitized["_schema_validation"]["issue_codes"]
    assert "SCENE_SPATIAL_INVALID" in codes
    assert "SCENE_SPATIAL_" + code in codes
    assert scene == before


def test_scene_repair_keeps_abstention_and_unavailable_evidence_rules(available_result):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    result, _ = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    context = {"duration": 1.0, "require_observed_content": True,
               "required_object_ids": ["cup"]}
    invalid = DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context)
    assert "SCENE_SPATIAL_EVIDENCE_UNAVAILABLE" in invalid["_schema_validation"]["issue_codes"]
    scene["locations"] = []
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context) == scene
    scene["objects"][0]["description"] = "private/path"
    invalid = DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context)
    assert "SCENE_SPATIAL_PROHIBITED_CONTENT" in invalid["_schema_validation"]["issue_codes"]


@pytest.mark.parametrize("error", [
    hybrid_result.ProvenanceValidationError(("private/path arbitrary code",)),
    ValueError("private/path raw exception"),
])
def test_scene_repair_filters_unknown_provenance_failures(available_result, monkeypatch, error):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    result, summary = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    context = {
        "duration": 1.0, "require_observed_content": True,
        "required_object_ids": ["cup"], "segments": source_segments(),
        "evidence_summary": summary.model_dump(mode="json"),
    }

    # Fault-inject only the validator failure; exercise the real transport boundary.
    def fail_provenance(*args, **kwargs):
        raise error

    monkeypatch.setattr(hybrid_result, "validate_event_provenance", fail_provenance)
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context) == {
        "_schema_validation": {
            "schema_name": "SceneSemantics", "status": "invalid",
            "issue_codes": ["SCENE_SPATIAL_INVALID"],
        }
    }


def test_scene_video_end_is_not_implicitly_an_observed_frame(available_result):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    result, summary = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    context = {
        "duration": 1.03, "require_observed_content": True,
        "required_object_ids": ["cup"], "segments": source_segments(),
        "evidence_summary": summary.model_dump(mode="json"),
    }
    scene["locations"][0]["end"] = 1.03
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context) == {
        "_schema_validation": {
            "schema_name": "SceneSemantics", "status": "invalid",
            "issue_codes": ["SCENE_SPATIAL_INVALID", "SCENE_SPATIAL_TIME_NOT_OBSERVED"],
        }
    }
    scene["locations"] = []
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context) == scene


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_track_ids", ["foreign_track"]),
        ("source_keyframe_ids", ["foreign_frame"]),
        ("end", 0.93),
    ],
)
def test_external_references_require_trusted_context(available_result, field, value):
    result, summary = available_result
    hybrid_result.validate_hybrid_result(
        result, evidence_summary=summary, frame_pts=[i / 10 for i in range(11)]
    )
    result["locations"][0][field] = value
    result["annotation_branches"]["scene_facts"]["locations"][0][field] = value
    hybrid_result.validate_hybrid_result(result)
    with pytest.raises(ValueError):
        hybrid_result.validate_hybrid_result(
            result, evidence_summary=summary, frame_pts=[i / 10 for i in range(11)]
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["performance"].update(degradation_count=9),
        lambda r: r["performance"].update(repair_count=9),
        lambda r: r.update(warnings=[{"code": "PRIVATE_WARNING", "path": "/secret"}]),
        lambda r: r["grouped_semantic_events"][0].update(end=1),
        lambda r: r["annotation_branches"]["action_events"][0].update(
            repair_history=["initial", "repair"]
        ),
        lambda r: r.update(task_description="/private/masks.npy"),
        lambda r: r["annotation_branches"]["action_events"][0].update(
            source_track_ids=["NOT_A_TRACK"]
        ),
        lambda r: r["annotation_branches"]["occlusion"].update(
            decisions=[
                {
                    "candidate_id": "occ_aaaaaaaaaaaa_0000",
                    "classification": "occlusion",
                    "target_entity_id": "cup",
                    "occluder_entity_id": "unknown",
                    "events": [],
                    "visual_evidence": "cup hidden",
                    "confidence": 0.8,
                }
            ]
        ),
    ],
)
def test_public_validation_rejects_internally_forged_metadata(
    available_result, mutation
):
    result, _ = available_result
    mutation(result)
    with pytest.raises(ValueError):
        hybrid_result.validate_hybrid_result(result)


@pytest.mark.parametrize("semantic_occluder", ["unknown", "flicker_target"])
def test_positive_occlusion_is_a_checked_projection_of_real_candidates(
    semantic_occluder,
):
    from test_cv_summary import _candidate_artifact

    from las_repro.cv.contracts import EvidenceThresholds
    from las_repro.cv.summary import build_cv_prompt_bundle, summarize_cv_evidence
    from las_repro.pipelines.occlusion import (
        OcclusionDecisionSet,
        project_occlusion_events,
    )

    artifact, timeline = _candidate_artifact()
    summary = summarize_cv_evidence(artifact, timeline=timeline)
    bundle = build_cv_prompt_bundle(
        summary,
        EvidenceThresholds(
            min_confidence=0.5, min_area_fraction=0.01, occlusion_visibility_drop=0.5
        ),
    )
    decisions = []
    for candidate in bundle.candidates:
        positive = candidate.target_entity_id == "behind_target"
        decisions.append(
            {
                "candidate_id": candidate.candidate_id,
                "classification": "occlusion" if positive else "unknown",
                "target_entity_id": candidate.target_entity_id,
                "occluder_entity_id": semantic_occluder if positive else "unknown",
                "visual_evidence": "target visibly hidden behind board"
                if positive
                else "insufficient evidence",
                "confidence": 0.8 if positive else 0.2,
                "events": [
                    {
                        "event_type": "occluded",
                        "start": candidate.allowed_start_times[0],
                        "end": candidate.allowed_end_times[-1],
                    }
                ]
                if positive
                else [],
            }
        )
    parsed = OcclusionDecisionSet.model_validate({"decisions": decisions})
    segments = source_segments()
    projected = project_occlusion_events(
        parsed,
        bundle.candidates,
        artifact.tracks,
        segments,
        repair_history=("initial",),
    )
    assert len(projected) == 1
    result = hybrid_result.build_hybrid_result(
        task_description="move cup",
        segments=segments,
        scene=unavailable_scene_semantics(),
        scene_status="unavailable",
        evidence_summary=summary,
        cv_evidence={
            "status": "available",
            "artifact_key": "a" * 64,
            "manifest_sha256": "b" * 64,
            "cache_hit": False,
        },
        warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}],
        performance={
            "stages": [],
            "total_seconds": 0.0,
            "repair_count": 0,
            "degradation_count": 1,
        },
        occlusion={"status": "available", "decisions": decisions, "events": projected},
    )
    context = {
        "evidence_summary": summary,
        "frame_pts": [f.timestamp_seconds for f in timeline.frames],
        "occlusion_candidates": bundle.candidates,
    }
    hybrid_result.validate_hybrid_result(result, **context)
    event = result["annotation_branches"]["occlusion"]["events"][0]
    assert event["occluder_entity_id"] == semantic_occluder
    assert event["source_track_ids"] == [
        "behind_target_1",
        "board_1",
        "flicker_target_1",
        "partial_target_1",
        "permanent_target_1",
        "stable_target_1",
    ]
    from las_repro.evaluation.viewer_projection import project_hybrid_viewer_data

    viewer = project_hybrid_viewer_data(
        "candidate-context",
        1.0,
        result,
        source_sha256="c" * 64,
        overlay_references=[
            {
                "keyframe_id": "board-1-00000001",
                "track_id": "board_1",
                "frame_index": 1,
                "timestamp_seconds": 0.1,
                "path": "evaluation/viewer/data/hybrid/overlays/" + "d" * 64 + ".png",
                "sha256": "d" * 64,
                "size_bytes": 1,
            }
        ],
    )
    assert viewer["layers"]["occlusion_events"]["events"][0][
        "occluder_entity_id"
    ] == semantic_occluder
    assert viewer["provenance"]["overlays"][0]["track_id"] == "board_1"
    empty = copy.deepcopy(result)
    empty["annotation_branches"]["occlusion"].update(decisions=[], events=[])
    with pytest.raises(ValueError):
        hybrid_result.validate_hybrid_result(empty, **context)
    empty["annotation_branches"]["occlusion"]["status"] = "unavailable"
    empty["warnings"].append({"code": "OCCLUSION_UNAVAILABLE"})
    empty["performance"]["degradation_count"] = 2
    hybrid_result.validate_hybrid_result(empty, **context)
    result["annotation_branches"]["occlusion"]["events"][0][
        "source_keyframe_ids"
    ] = ["foreign"]
    hybrid_result.validate_hybrid_result(result)
    with pytest.raises(ValueError):
        hybrid_result.validate_hybrid_result(result, **context)


def test_scene_registry_checks_spatial_rows_against_summary(available_result):
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    result, summary = available_result
    scene = {key: copy.deepcopy(result[key]) for key in hybrid_result.SCENE_KEYS}
    scene["relations"] = [
        {
            **{
                k: v
                for k, v in scene["locations"][0].items()
                if k not in {"location", "object_id"}
            },
            "subject_object_id": "cup",
            "object_object_id": "cup",
            "relation": "unknown",
        }
    ]
    context = {
        "duration": 1.0,
        "require_observed_content": True,
        "required_object_ids": ["cup"],
        "segments": source_segments(),
        "evidence_summary": summary.model_dump(mode="json"),
    }
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", scene, context) == scene
    for mutation in (
        lambda r: r["relations"][0].update(object_object_id="foreign"),
        lambda r: r["relations"][0].update(relation="private"),
        lambda r: r["relations"][0].update(start=0.03),
        lambda r: r["relations"][0].update(source_track_ids=["foreign"]),
        lambda r: r["relations"][0].update(source_segment_indices=[5]),
        lambda r: r["relations"][0].update(visual_evidence=" "),
        lambda r: r["relations"][0].update(visual_evidence="/tmp/private.mask"),
    ):
        bad = copy.deepcopy(scene)
        mutation(bad)
        sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize("SceneSemantics", bad, context)
        assert DEFAULT_OUTPUT_SCHEMAS.failure_codes("SceneSemantics", sanitized)


@pytest.mark.parametrize("status", ["unavailable", "disabled"])
def test_nonavailable_scene_rejects_successful_outcome(status):
    result = hybrid_result.build_hybrid_result(
        task_description="move cup",
        segments=source_segments(),
        scene=unavailable_scene_semantics(),
        scene_status=status,
        cv_evidence={"status": "disabled"},
        warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}]
        if status == "unavailable"
        else [],
        performance={
            "stages": [],
            "total_seconds": 0.0,
            "repair_count": 0,
            "degradation_count": int(status == "unavailable"),
        },
    )
    hybrid_result.validate_hybrid_result(result)
    outcome = {"status": "success", "description": "task achieved", "confidence": 1.0}
    result["outcome"] = outcome
    result["annotation_branches"]["scene_facts"]["outcome"] = outcome.copy()
    with pytest.raises(ValueError):
        hybrid_result.validate_hybrid_result(result)


@pytest.mark.parametrize(
    "warning",
    [
        {"code": "ENTITY_ALIASES_TRUNCATED", "omitted_count": -100},
        {"code": "ENTITY_ALIASES_TRUNCATED", "omitted_count": True},
        {"code": "ENTITY_ALIASES_TRUNCATED", "omitted_count": 16385},
        {
            "code": "CV_ENTITY_LIMIT_APPLIED",
            "omitted_count": 1,
            "limit": 16,
            "message": "false audit",
        },
        {
            "code": "CV_ENTITY_LIMIT_APPLIED",
            "omitted_count": 63,
            "limit": 16,
            "message": "63 entity candidates omitted by limit 16",
        },
        {
            "code": "BOUNDARY_TOPOLOGY_NORMALIZED",
            "issue_codes": ["SEGMENT_TOO_LONG"],
            "count": -1,
        },
        {
            "code": "BOUNDARY_TOPOLOGY_NORMALIZED",
            "issue_codes": ["private"],
            "count": 1,
        },
        {
            "code": "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN",
            "fields": ["skill"],
            "count": 1,
        },
        {
            "code": "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN",
            "fields": ["private"],
            "count": 1,
        },
    ],
)
def test_audit_warning_values_must_match_their_semantics(available_result, warning):
    result, _ = available_result
    result["warnings"] = [warning]
    with pytest.raises(ValueError):
        hybrid_result.validate_hybrid_result(result)
