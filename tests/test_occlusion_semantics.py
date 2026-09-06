"""Evidence-constrained occlusion decisions and projection."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from las_repro.config import Settings
from las_repro.cv.contracts import (
    CvEvidenceArtifact,
    CvTrack,
    EntityPrompt,
    EntityRole,
    FrameTimeline,
    FrameTimestamp,
    TrackObservation,
)
from las_repro.cv.entities import NormalizedEntities
from las_repro.cv.summary import (
    CvEvidenceSummary,
    OccluderProvenance,
    OcclusionCandidate,
    _candidate_identity,
    summarize_cv_evidence,
)
from las_repro.domain import InferenceStatus
from las_repro.media import MediaResolver, TimeSpan, VideoMetadata
from las_repro.models.fake import FakeVideoModel
from las_repro.pipelines.base import PipelineContext
from las_repro.pipelines.embodied import (
    EmbodiedActionPipeline,
    PromptRenderer,
    PromptRenderError,
)
from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
from las_repro.pipelines.occlusion import (
    OcclusionDecisionSet,
    project_occlusion_events,
    validate_occlusion_decisions,
)
from las_repro.pipelines.validators import TemporalValidationError
from las_repro.store import SQLiteTaskStore
from las_repro.workers import GPUWorker, JobWaitTimeout, wait_for_jobs


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


def _second_candidate() -> OcclusionCandidate:
    values = dict(
        target_entity_id="pear",
        target_track_id="pear_1",
        possible_occluders=(
            OccluderProvenance(
                entity_id="board", track_id="board_1", supporting_frames=(6, 7)
            ),
        ),
        possible_occluder_entity_ids=("board",),
        allowed_start_times=(2.3,),
        allowed_end_times=(2.8,),
        last_visible_frame=5,
        first_revisible_frame=8,
        edge_departure=False,
        low_confidence=False,
        overlay_refs=("overlays/pear_1_000005.png",),
        observation_support_complete=True,
        relation_support_complete=True,
        overlay_support_complete=True,
        support_complete=True,
        source_search_complete=True,
    )
    provisional = OcclusionCandidate.model_construct(
        candidate_id="occ_000000000000_0002", **values
    )
    return OcclusionCandidate(
        candidate_id=_candidate_identity(provisional, 2), **values
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


def _trusted_prompt_inputs() -> tuple[NormalizedEntities, CvEvidenceSummary]:
    entity = EntityPrompt(
        entity_id="apple",
        canonical_label="apple",
        aliases=(),
        role=EntityRole.MANIPULATED_OBJECT,
    )
    observation = TrackObservation(
        frame_index=0,
        timestamp_seconds=0.0,
        bbox_xyxy=(0.2, 0.2, 0.4, 0.4),
        mask_ref=None,
        visible=True,
        confidence=0.9,
        area_fraction=0.04,
        center_xy=(0.3, 0.3),
    )
    timeline = FrameTimeline(
        frames=(FrameTimestamp(frame_index=0, timestamp_seconds=0.0),)
    )
    artifact = CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status="available",
        provider="fake",
        model_identity="fake-sam31-v1",
        video_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        processed_timeline=timeline,
        entities=(entity,),
        tracks=(CvTrack(track_id="apple_1", entity_id="apple", observations=(observation,)),),
        files=(),
        overlay_records=(),
        warnings=(),
    )
    summary = summarize_cv_evidence(artifact, timeline=timeline)
    return (
        NormalizedEntities(entities=(entity,), omitted_count=0),
        summary,
    )


def _occlusion_runtime(tmp_path: Path, model, *, timeout: bool = False):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    video_path = allowed / "video.mp4"
    video_path.write_bytes(b"deterministic silent visual fixture")
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store.initialize()
    settings = Settings(
        database_path=store.database_path,
        work_root=tmp_path / "work",
        allowed_media_roots=(allowed,),
        lease_seconds=6,
    )
    task = store.create_task(
        {
            "video_url": str(video_path),
            "task_template": "embodied_action_captioning",
            "model_name": "qwen3-vl-8b-instruct",
        }
    )
    worker = GPUWorker(store, model, "gpu-0", "cuda:0", lease_seconds=6.0)

    def wait(store_arg, task_id, job_ids, requested_timeout):
        if timeout:
            raise JobWaitTimeout("private timeout detail")
        while worker.run_once():
            pass
        return wait_for_jobs(
            store_arg,
            task_id,
            job_ids,
            0.0,
            monotonic=lambda: 0.0,
            sleep=lambda _: pytest.fail("terminal job must not sleep"),
        )

    context = PipelineContext(
        store=store,
        media_resolver=MediaResolver(settings),
        settings=settings,
        task_dir=settings.work_root / task.task_id,
        media_path=video_path.resolve(),
    )
    return (
        EmbodiedActionPipeline(wait_jobs=wait, wait_timeout=0.75),
        task,
        context,
        video_path.resolve(),
        VideoMetadata(duration=3.0, width=320, height=180, fps=10.0),
    )


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


def test_occlusion_decisions_preserve_trusted_candidate_order():
    first = _candidate()
    second = _second_candidate()
    first_raw = _positive_raw(first)["decisions"][0]
    second_raw = {
        "candidate_id": second.candidate_id,
        "classification": "unknown",
        "target_entity_id": second.target_entity_id,
        "occluder_entity_id": "unknown",
        "events": [],
        "visual_evidence": "visible evidence is insufficient",
        "confidence": 0.2,
    }
    parsed = OcclusionDecisionSet.model_validate(
        {"decisions": [second_raw, first_raw]}
    )

    with pytest.raises(TemporalValidationError) as error:
        validate_occlusion_decisions(parsed, (first, second), duration=3.0)

    assert "OCCLUSION_CANDIDATE_ORDER" in {
        issue.code for issue in error.value.issues
    }


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


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda events: events.clear(), "OCCLUSION_EVENTS_EMPTY"),
        (
            lambda events: events.extend(
                [
                    {"event_type": "occluded", "start": 1.2, "end": 2.2},
                    {"event_type": "occluded", "start": 1.0, "end": 2.0},
                ]
            ),
            "OCCLUSION_EVENTS_NOT_ORDERED",
        ),
        (
            lambda events: events.append(
                {"event_type": "occluded", "start": 1.2, "end": 2.2}
            ),
            "OCCLUSION_EVENT_OVERLAP",
        ),
    ],
)
def test_positive_occlusion_requires_ordered_nonoverlapping_events(
    mutation, expected_code
):
    candidate = _candidate()
    raw = _positive_raw(candidate)
    mutation(raw["decisions"][0]["events"])
    parsed = OcclusionDecisionSet.model_validate(raw)

    with pytest.raises(TemporalValidationError) as error:
        validate_occlusion_decisions(parsed, (candidate,), duration=3.0)

    assert expected_code in {issue.code for issue in error.value.issues}


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
    entities, summary = _trusted_prompt_inputs()
    prompt = PromptRenderer().occlusion_semantics(
        (candidate,),
        entities,
        summary,
        video_duration=3.0,
        frame_pts=(0.0, 1.0, 2.0, 3.0),
        repair={"issue_codes": ["OCCLUSION_START_NOT_OBSERVED"]},
    )

    assert "[trusted occlusion candidate JSON data]" in prompt
    assert candidate.candidate_id in prompt
    assert "OCCLUSION_START_NOT_OBSERVED" in prompt
    assert "invent" in prompt.lower()


def test_occlusion_prompt_examples_are_distinct_valid_shapes():
    candidate = _candidate()
    entities, summary = _trusted_prompt_inputs()
    prompt = PromptRenderer().occlusion_semantics(
        (candidate,),
        entities,
        summary,
        video_duration=3.0,
        frame_pts=(0.0, 1.0, 2.0, 3.0),
        repair=None,
    )
    output_schema = prompt.split("[output schema]\n", 1)[1].split(
        "\n[validation repair data]", 1
    )[0]
    examples = [
        json.loads(line)
        for line in output_schema.splitlines()
        if line.startswith('{"decisions":')
    ]

    assert any(
        decision["classification"] == "unknown" and decision["events"] == []
        for example in examples
        for decision in example["decisions"]
    )
    assert any(
        decision["classification"] == "occlusion" and decision["events"]
        for example in examples
        for decision in example["decisions"]
    )
    for example in examples:
        for decision in example["decisions"]:
            if decision["candidate_id"] == "occ_example_0001":
                decision["candidate_id"] = candidate.candidate_id
            if decision["target_entity_id"] == "object":
                decision["target_entity_id"] = candidate.target_entity_id
            if decision["occluder_entity_id"] == "panel":
                decision["occluder_entity_id"] = (
                    candidate.possible_occluder_entity_ids[0]
                )
        parsed = OcclusionDecisionSet.model_validate(example)
        validate_occlusion_decisions(parsed, (candidate,), duration=3.0)


@pytest.mark.parametrize(
    ("bad_event", "expected_codes"),
    [
        (
            {"event_type": "occluded", "timestamp": 1.0},
            [
                "OCCLUSION_DECISION_SET_MISSING_FIELD",
                "OCCLUSION_DECISION_SET_EXTRA_FIELD",
            ],
        ),
        (
            {"event_type": "occluded", "start": 1.0},
            ["OCCLUSION_DECISION_SET_MISSING_FIELD"],
        ),
        (
            {"event_type": "occluded", "end": 2.0},
            ["OCCLUSION_DECISION_SET_MISSING_FIELD"],
        ),
        (
            {
                "event_type": "occluded",
                "start": 1.0,
                "end": 2.0,
                "timestamp": 1.0,
            },
            ["OCCLUSION_DECISION_SET_EXTRA_FIELD"],
        ),
    ],
)
def test_occlusion_registry_redacts_malformed_events(bad_event, expected_codes):
    candidate = _candidate()
    raw = _positive_raw(candidate)
    raw["decisions"][0]["events"] = [bad_event]

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "OcclusionDecisionSet",
        raw,
        {"duration": 3.0, "candidates": [candidate.model_dump(mode="json")]},
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "OcclusionDecisionSet",
            "status": "invalid",
            "issue_codes": expected_codes,
        }
    }
    assert not any(key in json.dumps(sanitized) for key in bad_event)


def test_occlusion_prompt_rejects_unvalidated_summary_and_entity_mappings():
    candidate = _candidate()
    entities, summary = _trusted_prompt_inputs()

    with pytest.raises(PromptRenderError):
        PromptRenderer().occlusion_semantics(
            (candidate,),
            entities,
            {"raw_masks": [[1, 0], [0, 1]]},
            video_duration=3.0,
            frame_pts=(0.0, 1.0),
        )
    with pytest.raises(PromptRenderError):
        PromptRenderer().occlusion_semantics(
            (candidate,),
            [{"entity_id": "apple", "private_path": "/tmp/private.npy"}],
            summary,
            video_duration=3.0,
            frame_pts=(0.0, 1.0),
        )


@pytest.mark.parametrize(
    "evidence",
    [
        json.dumps({"mask": [[1, 0], [0, 1]]}),
        "/tmp/private.npy",
        "masks/private.npz",
        "The mask is private.npy and shows the target.",
        "mask pixels:\n1,0\n0,1",
        "The object moves left/right.",
        "The object moves left\\right.",
        "The object is [hidden].",
    ],
)
def test_schema_rejects_mask_or_path_content_before_persistence(evidence):
    candidate = _candidate()
    raw = _positive_raw(candidate)
    raw["decisions"][0]["visual_evidence"] = evidence

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "OcclusionDecisionSet",
        raw,
        {"duration": 3.0, "candidates": [candidate.model_dump(mode="json")]},
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "OcclusionDecisionSet",
            "status": "invalid",
            "issue_codes": ["OCCLUSION_EVIDENCE_PROHIBITED_CONTENT"],
        }
    }
    assert evidence not in json.dumps(sanitized)


def test_plain_words_pass_without_rewriting_occlusion_evidence():
    candidate = _candidate()
    raw = _positive_raw(candidate)
    raw["decisions"][0]["visual_evidence"] = "The object moves left or right."
    assert DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "OcclusionDecisionSet", raw,
        {"duration": 3.0, "candidates": [candidate.model_dump(mode="json")]},
    ) == raw


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


def test_invalid_initial_creates_exactly_one_repair_then_degrades(tmp_path):
    candidate = _candidate()
    invalid = _positive_raw(candidate)
    invalid["decisions"][0]["events"][0]["start"] = 1.1
    model = FakeVideoModel(
        failure_script={"occlusion_semantics": [invalid, invalid]}
    )
    pipeline, task, context, video_path, metadata = _occlusion_runtime(
        tmp_path, model
    )
    entities, summary = _trusted_prompt_inputs()

    decisions, history = pipeline.adjudicate_occlusions(
        task,
        context,
        video_path,
        TimeSpan(0.0, 3.0),
        10.0,
        candidates=(candidate,),
        normalized_entities=entities,
        evidence_summary=summary,
        frame_pts=(0.0, 1.0, 2.0, 3.0),
        affinity_anchor=None,
        metadata=metadata,
    )

    jobs = context.store.list_inference_jobs(task.task_id)
    assert decisions == OcclusionDecisionSet(decisions=())
    assert history == ("initial", "repair")
    assert [job.ordinal for job in jobs] == [0, 1]
    assert all(job.status is InferenceStatus.COMPLETED for job in jobs)


@pytest.mark.parametrize("mode", ["failure", "timeout"])
def test_occlusion_inference_failure_or_timeout_degrades_with_truthful_history(
    tmp_path, mode
):
    model = FakeVideoModel(
        failure_script={
            "occlusion_semantics": [RuntimeError("private model detail")]
        }
    )
    pipeline, task, context, video_path, metadata = _occlusion_runtime(
        tmp_path, model, timeout=mode == "timeout"
    )
    entities, summary = _trusted_prompt_inputs()

    decisions, history = pipeline.adjudicate_occlusions(
        task,
        context,
        video_path,
        TimeSpan(0.0, 3.0),
        10.0,
        candidates=(_candidate(),),
        normalized_entities=entities,
        evidence_summary=summary,
        frame_pts=(0.0, 1.0, 2.0, 3.0),
        affinity_anchor=None,
        metadata=metadata,
    )

    jobs = context.store.list_inference_jobs(task.task_id)
    assert decisions == OcclusionDecisionSet(decisions=())
    assert history == ("initial",)
    assert len(jobs) == 1
