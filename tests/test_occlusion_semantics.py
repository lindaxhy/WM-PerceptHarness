"""Evidence-constrained occlusion decisions and projection."""

from __future__ import annotations

import copy

import pytest

from las_repro.cv.summary import (
    OccluderProvenance,
    OcclusionCandidate,
    _candidate_identity,
)
from las_repro.media import TimeSpan
from las_repro.pipelines.embodied import EmbodiedActionPipeline, PromptRenderer
from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
from las_repro.pipelines.occlusion import (
    OcclusionDecisionSet,
    project_occlusion_events,
    validate_occlusion_decisions,
)
from las_repro.pipelines.validators import TemporalValidationError


def _candidate() -> OcclusionCandidate:
    # Candidate identity is sealed by the CV summary builder.  This unit fixture
    # isolates the adjudicator from that already-tested construction path.
    values = dict(
        target_entity_id="apple",
        target_track_id="apple_1",
        possible_occluders=(
            OccluderProvenance(
                entity_id="board", track_id="board_1", supporting_frames=(3, 4)
            ),
        ),
        possible_occluder_entity_ids=("board",),
        allowed_start_times=(1.0, 1.2),
        allowed_end_times=(2.0, 2.2),
        last_visible_frame=2,
        first_revisible_frame=5,
        edge_departure=False,
        low_confidence=False,
        overlay_refs=("overlays/apple_1_000002.png",),
        observation_support_complete=True,
        relation_support_complete=True,
        overlay_support_complete=True,
        support_complete=True,
        source_search_complete=True,
    )
    provisional = OcclusionCandidate.model_construct(
        candidate_id="occ_000000000000_0001", **values
    )
    return OcclusionCandidate(
        candidate_id=_candidate_identity(provisional, 1), **values
    )


def _positive_raw(candidate: OcclusionCandidate) -> dict[str, object]:
    return {
        "decisions": [
            {
                "candidate_id": candidate.candidate_id,
                "classification": "occlusion",
                "target_entity_id": candidate.target_entity_id,
                "occluder_entity_id": candidate.possible_occluder_entity_ids[0],
                "events": [
                    {"event_type": "occluded", "start": 1.0, "end": 2.0}
                ],
                "visual_evidence": "target disappears behind the board and returns",
                "confidence": 0.9,
            }
        ]
    }


def test_occlusion_decision_must_use_candidate_and_observed_boundaries():
    candidate = _candidate()
    parsed = OcclusionDecisionSet.model_validate(_positive_raw(candidate))

    validate_occlusion_decisions(parsed, (candidate,), duration=3.0)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda raw: raw["decisions"].append(copy.deepcopy(raw["decisions"][0])), "OCCLUSION_DECISION_CARDINALITY"),
        (lambda raw: raw["decisions"][0].update({"target_entity_id": "pear"}), "OCCLUSION_TARGET_MISMATCH"),
        (lambda raw: raw["decisions"][0].update({"occluder_entity_id": "hand"}), "OCCLUSION_OCCLUDER_NOT_PROPOSED"),
        (lambda raw: raw["decisions"][0]["events"][0].update({"start": 1.1}), "OCCLUSION_START_NOT_OBSERVED"),
        (lambda raw: raw["decisions"][0]["events"][0].update({"end": 3.1}), "OCCLUSION_END_NOT_OBSERVED"),
    ],
)
def test_occlusion_decisions_reject_injected_references_and_times(mutation, expected_code):
    candidate = _candidate()
    raw = _positive_raw(candidate)
    mutation(raw)
    parsed = OcclusionDecisionSet.model_validate(raw)

    with pytest.raises(TemporalValidationError) as error:
        validate_occlusion_decisions(parsed, (candidate,), duration=3.0)

    assert expected_code in {issue.code for issue in error.value.issues}


@pytest.mark.parametrize("classification", ["out_of_frame", "detector_loss", "unknown"])
def test_non_occlusion_classifications_cannot_emit_positive_events(classification):
    candidate = _candidate()
    raw = _positive_raw(candidate)
    raw["decisions"][0]["classification"] = classification
    parsed = OcclusionDecisionSet.model_validate(raw)

    with pytest.raises(TemporalValidationError) as error:
        validate_occlusion_decisions(parsed, (candidate,), duration=3.0)

    assert {issue.code for issue in error.value.issues} == {
        "NON_OCCLUSION_HAS_EVENTS"
    }


def test_projection_emits_only_positive_events_with_closed_provenance():
    candidate = _candidate()
    parsed = OcclusionDecisionSet.model_validate(_positive_raw(candidate))

    events = project_occlusion_events(
        parsed,
        (candidate,),
        (),
        (
            {"segment_index": 4, "start": 0.5, "end": 1.1},
            {"segment_index": 5, "start": 1.1, "end": 2.5},
        ),
        repair_history=("initial", "repair"),
    )

    assert events == [
        {
            "event_index": 0,
            "start": 1.0,
            "end": 2.0,
            "event_type": "occluded",
            "target_entity_id": "apple",
            "occluder_entity_id": "board",
            "description": "target disappears behind the board and returns",
            "confidence": 0.9,
            "branch": "occlusion",
            "model_stage": "occlusion_semantics",
            "source_candidate_id": candidate.candidate_id,
            "source_segment_indices": [4, 5],
            "source_track_ids": ["apple_1", "board_1"],
            "source_keyframe_ids": ["apple_1_000002"],
            "evidence_mode": "hybrid",
            "repair_history": ["initial", "repair"],
            "review_status": "unreviewed",
        }
    ]


def test_occlusion_prompt_isolates_trusted_data_and_repair_codes():
    candidate = _candidate()
    prompt = PromptRenderer().occlusion_semantics(
        (candidate,),
        [{"entity_id": "apple", "canonical_label": "apple", "aliases": [], "role": "manipulated_object"}],
        {"schema_version": "cv_summary_v1", "status": "available"},
        video_duration=3.0,
        frame_pts=(0.0, 1.0, 2.0, 3.0),
        repair={"issue_codes": ["OCCLUSION_START_NOT_OBSERVED"]},
    )

    assert "[trusted occlusion candidate JSON data]" in prompt
    assert candidate.candidate_id in prompt
    assert "OCCLUSION_START_NOT_OBSERVED" in prompt
    assert "invent" in prompt.lower()


def test_output_registry_replaces_arbitrary_timestamp_with_closed_issue_code():
    candidate = _candidate()
    raw = _positive_raw(candidate)
    raw["decisions"][0]["events"][0]["start"] = 1.1

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "OcclusionDecisionSet",
        raw,
        {"duration": 3.0, "candidates": [candidate.model_dump(mode="json")]},
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "OcclusionDecisionSet",
            "status": "invalid",
            "issue_codes": ["OCCLUSION_START_NOT_OBSERVED"],
        }
    }


def test_empty_candidate_tuple_skips_model_stage(monkeypatch):
    pipeline = EmbodiedActionPipeline()

    monkeypatch.setattr(
        pipeline,
        "_run_validated_stage",
        lambda *args, **kwargs: pytest.fail("empty candidates must skip Qwen"),
    )

    decisions, history = pipeline.adjudicate_occlusions(
        None,
        None,
        None,
        TimeSpan(0.0, 3.0),
        1.0,
        candidates=(),
        normalized_entities=(),
        evidence_summary={},
        frame_pts=(),
        affinity_anchor=None,
        metadata=None,
    )

    assert decisions == OcclusionDecisionSet(decisions=())
    assert history == ("initial",)


def test_invalid_repair_degrades_only_occlusion_branch(monkeypatch):
    pipeline = EmbodiedActionPipeline()
    issue = TemporalValidationError([])

    def invalid_stage(*args, **kwargs):
        raise issue

    monkeypatch.setattr(pipeline, "_run_validated_stage", invalid_stage)

    decisions, history = pipeline.adjudicate_occlusions(
        None,
        None,
        None,
        TimeSpan(0.0, 3.0),
        1.0,
        candidates=(_candidate(),),
        normalized_entities=(),
        evidence_summary={},
        frame_pts=(0.0, 1.0),
        affinity_anchor=None,
        metadata=None,
    )

    assert decisions == OcclusionDecisionSet(decisions=())
    assert history == ("initial", "repair")
