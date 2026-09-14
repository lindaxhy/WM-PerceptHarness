from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources
from pathlib import Path
import re
import zipfile

import pytest

import las_repro
from las_repro.cv.contracts import (
    CvEvidenceArtifact,
    CvTrack,
    EntityPrompt,
    EntityRole,
    FrameTimeline,
    FrameTimestamp,
    TrackObservation,
)
from las_repro.cv.summary import (
    CvEvidenceSummary,
    OccluderProvenance,
    OcclusionCandidate,
    AllowedEventInterval,
    _candidate_identity,
    summarize_cv_evidence,
)
from las_repro.pipelines.embodied import PromptRenderer
from las_repro.pipelines.scene_choices import prepare_scene_choices
from scripts.reverify_semantic_stages import (
    OperatorError,
    main,
    rebuild_prompt,
    run_stage,
)


MODEL_IDENTITY = "doubao-seed-2-1-pro-260628"
MODEL_ALIAS = "doubao-pro"
SOURCE_COMMIT = "7" * 40


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prompt_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _current_template(stage: str) -> str:
    return resources.files("las_repro.prompts").joinpath(f"{stage}.txt").read_text()


def _summary() -> CvEvidenceSummary:
    entity = EntityPrompt(
        entity_id="panel",
        canonical_label="panel",
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
        tracks=(
            CvTrack(
                track_id="panel_1",
                entity_id="panel",
                observations=(observation,),
            ),
        ),
        files=(),
        overlay_records=(),
        warnings=(),
    )
    return summarize_cv_evidence(artifact, timeline=timeline)


def _scene_variables(summary: CvEvidenceSummary | None = None) -> dict[str, object]:
    return {
        "VIDEO_DURATION_SECONDS_JSON": 2.0,
        "SEGMENTS_JSON": [
            {
                "segment_index": 0,
                "start": 0.0,
                "end": 2.0,
                "target": "unknown",
                "description": "literal {{VALIDATION_REPAIR_JSON}} data",
            }
        ],
        "KNOWN_TARGETS_JSON": [],
        "CV_EVIDENCE_AVAILABILITY_JSON": {"available": summary is not None},
        "SCENE_SPATIAL_PROVENANCE_OPTIONS_JSON": {"options": [], "options_complete": True},
        "SCENE_SPATIAL_FIELDS_JSON": {
            "location_fields": ["option_id", "location", "visual_evidence", "confidence"],
            "relation_fields": ["option_id", "direction", "relation", "visual_evidence", "confidence"],
        },
        "VALIDATION_REPAIR_JSON": None,
    }


def _scene_prompt_pair(summary: CvEvidenceSummary | None = None) -> tuple[str, str, dict[str, object]]:
    variables = _scene_variables(summary)
    current_template = (Path(__file__).parent / "fixtures/prompts/scene_semantics_v2.txt").read_text()
    old_template = current_template.replace(
        "You extract overlapping scene semantics",
        "You previously extracted overlapping scene semantics",
        1,
    )
    from scripts.reverify_semantic_stages import _render_exact_template
    current_prompt = _render_exact_template(current_template, variables)
    old_prompt = current_prompt.replace(
        "You extract overlapping scene semantics",
        "You previously extracted overlapping scene semantics",
        1,
    )
    if summary is not None:
        suffix = "\n\n[CV_EVIDENCE_SUMMARY_JSON]\n" + _prompt_json(summary.prompt_record())
        current_prompt += suffix
        old_prompt += suffix
    return old_template, old_prompt, variables


def _valid_scene() -> dict[str, object]:
    return {
        "objects": [],
        "initial_state": [],
        "final_state": [],
        "locations": [],
        "relations": [],
        "outcome": {
            "status": "unknown",
            "description": "visible result is uncertain",
            "confidence": 0.2,
        },
        "semantic_events": [],
    }


def _candidate() -> OcclusionCandidate:
    values = dict(
        target_entity_id="panel",
        target_track_id="panel_1",
        possible_occluders=(
            OccluderProvenance(
                entity_id="board", track_id="board_1", supporting_frames=(1,)
            ),
        ),
        possible_occluder_entity_ids=("board",),
        allowed_event_intervals=(AllowedEventInterval(event_type="occluded",start=0.5,end=1.5),),
        last_visible_frame=0,
        first_revisible_frame=2,
        edge_departure=False,
        low_confidence=False,
        overlay_refs=(),
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


def _scene_stage(tmp_path: Path, *, summary: CvEvidenceSummary | None = None) -> dict[str, object]:
    video = tmp_path / "private-video.mp4"
    video.write_bytes(b"trusted-video")
    template, prompt, _ = _scene_prompt_pair(summary)
    context = prepare_scene_choices(summary, _scene_variables(summary)["SEGMENTS_JSON"],
                                    duration=2.0).context()
    payload = {
        "video_path": str(video.resolve()),
        "span": {"start": 0.0, "end": 2.0},
        "fps": 2.0,
        "prompt": prompt,
        "schema_name": "SceneSemanticsChoices",
        "schema_context": context,
        "video_session_id": "private-database-id",
    }
    return {
        "sample_id": "full_0001",
        "stage": "scene_semantics",
        "model_name": MODEL_ALIAS,
        "payload": payload,
        "original_template": template,
        "source_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "original_job_payload_sha256": _digest(payload),
    }


class _Model:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests = []
        self._usage = {"input_tokens": 12, "output_tokens": 8}

    def generate(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return copy.deepcopy(outcome)

    def request_metrics(self):
        return dict(self._usage)

    def release_request(self, request):
        self._usage = {}


def test_rebuild_prompt_preserves_historical_scene_template_and_changes_only_repair() -> None:
    summary = _summary()
    original_template, original_prompt, variables = _scene_prompt_pair(summary)

    initial, initial_digest = rebuild_prompt(
        original_prompt, original_template, "scene_semantics"
    )
    repaired, repaired_digest = rebuild_prompt(
        original_prompt,
        original_template,
        "scene_semantics",
        repair={"issue_codes": ["SCENE_EVENT_TYPE_ENUM_VALUE"]},
    )

    from scripts.reverify_semantic_stages import _render_exact_template
    expected_initial = _render_exact_template(original_template, variables)
    repaired_variables = dict(variables)
    repaired_variables["VALIDATION_REPAIR_JSON"] = {
        "issue_codes": ["SCENE_EVENT_TYPE_ENUM_VALUE"]
    }
    expected_repaired = _render_exact_template(original_template, repaired_variables)
    suffix = "\n\n[CV_EVIDENCE_SUMMARY_JSON]\n" + _prompt_json(summary.prompt_record())
    assert initial == expected_initial + suffix
    assert repaired == expected_repaired + suffix
    assert initial_digest == repaired_digest
    assert original_prompt.endswith(suffix)


@pytest.mark.parametrize(
    ("stage", "template", "prompt"),
    [
        ("occlusion_semantics", "{{VALIDATION_REPAIR_JSON}}", "null\nextra"),
        ("scene_semantics", "{{VALIDATION_REPAIR_JSON}} trailing", "null trailing junk"),
        ("scene_semantics", "{{A}}{{A}}", "nullnull"),
    ],
)
def test_rebuild_prompt_rejects_suffixes_unconsumed_text_and_ambiguous_markers(
    stage: str, template: str, prompt: str
) -> None:
    with pytest.raises(OperatorError):
        rebuild_prompt(prompt, template, stage)


def test_run_stage_accepts_current_positive_output_in_one_call_and_keeps_raw_private(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "private-output"
    output_dir.mkdir(mode=0o700)
    stage = _scene_stage(tmp_path)
    positive = _valid_scene()
    model = _Model([positive])

    report = run_stage(stage, output_dir=output_dir, model=model)

    assert report["status"] == "valid"
    assert report["call_count"] == 1
    assert report["semantic_event_count"] == 0
    assert report["attempts"][0]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 8,
    }
    assert report["attempts"][0]["elapsed_seconds"] >= 0
    assert re.fullmatch(r"[0-9a-f]{64}", report["attempts"][0]["request_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", report["attempts"][0]["response_sha256"])
    assert len(model.requests) == 1
    assert model.requests[0].prompt == stage["payload"]["prompt"]
    raw = json.loads((output_dir / "full_0001.scene_semantics.call-1.raw.private.json").read_text())
    validated = json.loads((output_dir / "full_0001.scene_semantics.validated.private.json").read_text())
    assert raw == positive == validated
    for path in output_dir.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
    public = (output_dir / "full_0001.scene_semantics.report.json").read_text()
    assert "visible result is uncertain" not in public
    assert "private-video" not in public
    assert "private-database-id" not in public


def test_run_stage_records_occlusion_decision_and_classification_counts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(mode=0o700)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"trusted")
    candidate = _candidate()
    variables = {
        "VIDEO_DURATION_SECONDS_JSON": 2.0,
        "FRAME_PTS_JSON": [0.0, 1.0],
        "OCCLUSION_CANDIDATES_JSON": [candidate.prompt_record()],
        "NORMALIZED_ENTITIES_JSON": [],
        "CV_EVIDENCE_SUMMARY_JSON": _summary().prompt_record(),
        "VALIDATION_REPAIR_JSON": None,
    }
    prompt = PromptRenderer().render("occlusion_semantics", variables)
    payload = {
        "video_path": str(video.resolve()),
        "span": {"start": 0.0, "end": 2.0},
        "fps": 2.0,
        "prompt": prompt,
        "schema_name": "OcclusionDecisionSet",
        "schema_context": {
            "duration": 2.0,
            "candidates": [candidate.model_dump(mode="json")],
        },
    }
    stage = {
        "sample_id": "full_0002",
        "stage": "occlusion_semantics",
        "model_name": MODEL_ALIAS,
        "payload": payload,
        "original_template": _current_template("occlusion_semantics"),
        "original_job_payload_sha256": _digest(payload),
    }
    positive = {
        "decisions": [
            {
                "candidate_id": candidate.candidate_id,
                "classification": "unknown",
                "target_entity_id": "panel",
                "occluder_entity_id": "unknown",
                "events": [],
                "visual_evidence": "visible evidence is insufficient",
                "confidence": 0.2,
            }
        ]
    }

    report = run_stage(stage, output_dir=output_dir, model=_Model([positive]))

    assert report["status"] == "valid"
    assert report["decision_count"] == 1
    assert report["classification_counts"] == {"unknown": 1}


def test_run_stage_repairs_once_using_only_closed_codes(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(mode=0o700)
    model = _Model([{"private": "raw first output"}, _valid_scene()])

    report = run_stage(_scene_stage(tmp_path), output_dir=output_dir, model=model)

    assert report["status"] == "valid"
    assert report["call_count"] == 2
    assert report["repair_issue_codes"] == [
        "SCENE_SEMANTICS_CHOICES_MISSING_FIELD",
        "SCENE_SEMANTICS_CHOICES_EXTRA_FIELD",
    ]
    assert "raw first output" not in _canonical(report)
    assert model.requests[0].prompt != model.requests[1].prompt
    assert _scene_variables()["SEGMENTS_JSON"] == [
        {
            "segment_index": 0,
            "start": 0.0,
            "end": 2.0,
            "target": "unknown",
            "description": "literal {{VALIDATION_REPAIR_JSON}} data",
        }
    ]


def test_run_stage_final_invalid_is_failed_after_exactly_two_calls(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(mode=0o700)
    model = _Model([{}, {}])

    report = run_stage(_scene_stage(tmp_path), output_dir=output_dir, model=model)

    assert report["status"] == "invalid"
    assert report["call_count"] == 2
    assert report["final_issue_codes"] == ["SCENE_SEMANTICS_CHOICES_MISSING_FIELD"]
    assert len(model.requests) == 2


def test_run_stage_transport_failure_is_closed_and_never_retried(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(mode=0o700)
    model = _Model([RuntimeError("secret transport payload /private/path")])

    report = run_stage(_scene_stage(tmp_path), output_dir=output_dir, model=model)

    assert report["status"] == "error"
    assert report["error_code"] == "MODEL_GENERATION_FAILED"
    assert report["call_count"] == 1
    assert "secret" not in _canonical(report)
    assert len(model.requests) == 1


def test_existing_reservation_refuses_a_repeat_without_calling_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(mode=0o700)
    stage = _scene_stage(tmp_path)
    first = _Model([_valid_scene()])
    run_stage(stage, output_dir=output_dir, model=first)
    second = _Model([RuntimeError("reserved stage must not call model")])

    with pytest.raises(OperatorError, match="reserved"):
        run_stage(stage, output_dir=output_dir, model=second)

    assert second.requests == []


def _build_test_wheel(path: Path, *, alter_domain: bool = False) -> str:
    package = Path(las_repro.__file__).resolve().parent
    with zipfile.ZipFile(path, "w") as archive:
        for source in sorted(package.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
                destination = "las_repro/" + source.relative_to(package).as_posix()
                if alter_domain and destination == "las_repro/domain.py":
                    archive.writestr(destination, "altered")
                else:
                    archive.write(source, destination)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, str]:
    scene = _scene_stage(tmp_path, summary=_summary())
    video = tmp_path / "occlusion-video.mp4"
    video.write_bytes(b"trusted-occlusion-video")
    candidate = _candidate()
    variables = {
        "VIDEO_DURATION_SECONDS_JSON": 2.0,
        "FRAME_PTS_JSON": [0.0, 1.0],
        "OCCLUSION_CANDIDATES_JSON": [candidate.prompt_record()],
        "NORMALIZED_ENTITIES_JSON": [],
        "CV_EVIDENCE_SUMMARY_JSON": _summary().prompt_record(),
        "VALIDATION_REPAIR_JSON": None,
    }
    prompt = PromptRenderer().render("occlusion_semantics", variables)
    payload = {
        "video_path": str(video.resolve()),
        "span": {"start": 0.0, "end": 2.0},
        "fps": 2.0,
        "prompt": prompt,
        "schema_name": "OcclusionDecisionSet",
        "schema_context": {
            "duration": 2.0,
            "candidates": [candidate.model_dump(mode="json")],
        },
        "video_session_id": "private-occlusion-job-id",
    }
    occlusion = {
        "sample_id": "full_0002",
        "stage": "occlusion_semantics",
        "model_name": MODEL_ALIAS,
        "payload": payload,
        "original_template": _current_template("occlusion_semantics"),
        "source_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "original_job_payload_sha256": _digest(payload),
    }
    bundle = {
        "schema": "semantic_reverification_input_v1",
        "model_identity": MODEL_IDENTITY,
        "settings": {
            "timeout_seconds": 180.0,
            "max_frames": 128,
            "max_request_bytes": 32 * 1024 * 1024,
            "max_output_chars": 1_000_000,
        },
        "stages": [scene, occlusion],
    }
    path = tmp_path / "private-bundle.json"
    path.write_text(_canonical(bundle))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _cli_args(tmp_path: Path, bundle: Path, bundle_hash: str, wheel: Path, wheel_hash: str) -> list[str]:
    return [
        "--bundle", str(bundle),
        "--bundle-sha256", bundle_hash,
        "--wheel", str(wheel),
        "--wheel-sha256", wheel_hash,
        "--source-commit", SOURCE_COMMIT,
        "--output-dir", str(tmp_path / "result"),
        "--api-key-file", str(tmp_path / "must-not-open.key"),
    ]


def test_cli_dry_run_verifies_bundle_wheel_and_inputs_without_credentials_or_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, bundle_hash = _bundle(tmp_path)
    wheel = tmp_path / "runtime.whl"
    wheel_hash = _build_test_wheel(wheel)
    calls = []

    assert main(
        _cli_args(tmp_path, bundle, bundle_hash, wheel, wheel_hash),
        model_factory=lambda **kwargs: calls.append(kwargs),
    ) == 0

    output = capsys.readouterr()
    assert calls == []
    assert not (tmp_path / "result").exists()
    assert "preflight_valid" in output.out
    assert "private" not in output.out
    assert output.err == ""


def test_cli_hash_mismatch_precedes_credentials_model_and_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, _ = _bundle(tmp_path)
    wheel = tmp_path / "runtime.whl"
    wheel_hash = _build_test_wheel(wheel)
    calls = []
    args = _cli_args(tmp_path, bundle, "0" * 64, wheel, wheel_hash) + ["--execute"]

    assert main(args, model_factory=lambda **kwargs: calls.append(kwargs)) == 2

    output = capsys.readouterr()
    assert calls == []
    assert not (tmp_path / "result").exists()
    assert "BUNDLE_HASH_MISMATCH" in output.err
    assert "private" not in output.err


def test_cli_closes_model_construction_errors_without_leaking_secrets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle, bundle_hash = _bundle(tmp_path)
    wheel = tmp_path / "runtime.whl"
    wheel_hash = _build_test_wheel(wheel)
    key = tmp_path / "api.key"
    key.write_text("top-secret-key")

    def fail_model(**kwargs):
        raise RuntimeError(f"provider rejected {kwargs['api_key']} at /private/path")

    args = _cli_args(tmp_path, bundle, bundle_hash, wheel, wheel_hash)
    args[-1] = str(key)
    assert main(
        args + ["--execute"],
        model_factory=fail_model,
    ) == 2

    output = capsys.readouterr()
    assert "top-secret" not in output.err
    assert "private/path" not in output.err
    assert (tmp_path / "result").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("mutation", ["payload", "suffix", "candidate", "model"])
def test_cli_rejects_altered_trusted_inputs_before_model(
    tmp_path: Path, mutation: str, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_path, _ = _bundle(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    if mutation == "payload":
        bundle["stages"][0]["payload"]["fps"] = 3.0
    elif mutation == "suffix":
        bundle["stages"][0]["payload"]["schema_context"]["evidence_summary"] = None
        bundle["stages"][0]["original_job_payload_sha256"] = _digest(
            bundle["stages"][0]["payload"]
        )
    elif mutation == "candidate":
        bundle["stages"][1]["payload"]["schema_context"]["candidates"] = []
        bundle["stages"][1]["original_job_payload_sha256"] = _digest(
            bundle["stages"][1]["payload"]
        )
    else:
        bundle["stages"][1]["model_name"] = "another-model"
    bundle_path.write_text(_canonical(bundle))
    bundle_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    wheel = tmp_path / "runtime.whl"
    wheel_hash = _build_test_wheel(wheel)
    calls = []

    assert main(
        _cli_args(tmp_path, bundle_path, bundle_hash, wheel, wheel_hash) + ["--execute"],
        model_factory=lambda **kwargs: calls.append(kwargs),
    ) == 2

    assert calls == []
    assert not (tmp_path / "result").exists()
    assert "private" not in capsys.readouterr().err


def test_cli_rejects_wheel_package_drift_before_model(tmp_path: Path) -> None:
    bundle, bundle_hash = _bundle(tmp_path)
    wheel = tmp_path / "runtime.whl"
    _build_test_wheel(wheel, alter_domain=True)
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    calls = []

    assert main(
        _cli_args(tmp_path, bundle, bundle_hash, wheel, wheel_hash),
        model_factory=lambda **kwargs: calls.append(kwargs),
    ) == 2
    assert calls == []


def test_scene_alignment_rejects_tampered_generation_choices():
    from scripts.reverify_semantic_stages import _validate_scene_alignment
    summary = _summary()
    values = _scene_variables(summary)
    context = prepare_scene_choices(summary, values["SEGMENTS_JSON"], duration=2.0).context()
    _validate_scene_alignment(values, summary.prompt_record(), context)
    values["SCENE_SPATIAL_PROVENANCE_OPTIONS_JSON"]["options"] = [{"option_id": "forged"}]
    with pytest.raises(OperatorError, match="SCENE_PROMPT_CONTEXT_MISMATCH"):
        _validate_scene_alignment(values, summary.prompt_record(), context)


def test_scene_operator_persists_distinct_dto_and_public_projection(tmp_path):
    from test_scene_choices import fixture
    from las_repro.pipelines.scene_choices import SceneInputPackage, canonical
    draft, context = fixture()
    output = tmp_path / 'private'; output.mkdir(mode=0o700)
    stage = _scene_stage(tmp_path)
    summary = CvEvidenceSummary.model_validate(context['evidence_summary'])
    stage['payload'].update(schema_name='SceneSemanticsChoices', schema_context=context,
        span={'start': 0.0, 'end': context['duration']},
        prompt=PromptRenderer().scene_semantics(context['segments'],
            video_duration=context['duration'], evidence_summary=summary,
            scene_input=SceneInputPackage(canonical(context))))
    stage['original_template'] = _current_template('scene_semantics')
    model = _Model([{}, draft])
    report = run_stage(stage, output_dir=output, model=model)
    assert report['model_contract_valid'] is True
    assert report['public_projection_valid'] is True
    assert report['validated_dto_sha256'] != report['projected_scene_sha256']
    dto = json.loads((output / 'full_0001.scene_semantics.validated.private.json').read_text())
    scene = json.loads((output / 'full_0001.scene_semantics.projected-public-scene.private.json').read_text())
    assert dto == draft
    assert scene['locations'][0]['repair_history'] == ['initial', 'repair']
    assert model.requests[0].response_contract == model.requests[1].response_contract


def test_scene_operator_reports_response_contract_identity_and_invalid_status(tmp_path):
    output = tmp_path / 'private'; output.mkdir(mode=0o700)
    model = _Model([{}, {}])
    report = run_stage(_scene_stage(tmp_path), output_dir=output, model=model)
    assert report['model_contract_valid'] is False
    assert report['public_projection_valid'] is None
    for attempt, request in zip(report['attempts'], model.requests):
        assert attempt['response_format'] == request.response_contract.cache_identity()


@pytest.mark.parametrize('repair_type', ['unknown', 'hold'])
def test_scene_operator_repairs_event_enum_from_immutable_context(tmp_path, repair_type):
    from test_scene_choices import fixture
    from las_repro.pipelines.scene_choices import SceneInputPackage, canonical
    draft, context = fixture()
    draft['semantic_events'][0].update(event_type='hold', description='hand holds item visibly')
    repaired = copy.deepcopy(draft)
    repaired['semantic_events'][0]['event_type'] = repair_type
    stage = _scene_stage(tmp_path)
    summary = CvEvidenceSummary.model_validate(context['evidence_summary'])
    stage['payload'].update(schema_name='SceneSemanticsChoices', schema_context=context,
        span={'start': 0.0, 'end': context['duration']},
        prompt=PromptRenderer().scene_semantics(context['segments'],
            video_duration=context['duration'], evidence_summary=summary,
            scene_input=SceneInputPackage(canonical(context))))
    stage['original_template'] = _current_template('scene_semantics')
    original_stage = copy.deepcopy(stage)
    output = tmp_path / 'private'; output.mkdir(mode=0o700)
    model = _Model([draft, repaired])
    report = run_stage(stage, output_dir=output, model=model)
    code = 'SCENE_SEMANTICS_CHOICES_EVENT_TYPE_ENUM_VALUE'
    assert report['repair_issue_codes'] == [code]
    assert report['model_contract_valid'] is (repair_type == 'unknown')
    assert stage == original_stage
    assert len(model.requests) == 2
    initial, repair = model.requests
    assert initial.response_contract == repair.response_contract
    assert initial.span == repair.span
    assert initial.prompt.replace('null\n\nClosed choice repair',
        json.dumps({'issue_codes': [code]}, separators=(',', ':')) + '\n\nClosed choice repair') == repair.prompt
    assert '0907-scene-choice-refs-v7' in repair.prompt
    meanings = dict(line[2:].split(': ', 1) for line in repair.prompt.splitlines()
                    if line.startswith('- SCENE_SEMANTICS_CHOICES_') and ': ' in line)
    assert 'use unknown if no allowed type fits' in meanings[code]
    assert 'supported visual description' in meanings[code]
    assert 'another action taxonomy' in meanings[code]
    assert 'contacting' not in meanings['SCENE_SEMANTICS_CHOICES_ENUM_VALUE']
    expected_vocab = {
        'EVENT_TYPE': ['move', 'transport', 'grasp', 'reach', 'release', 'lift', 'place',
                       'approach', 'contact', 'push', 'pull', 'rotate', 'stop',
                       'autonomous_motion', 'state_change', 'occlusion_enter', 'occluded', 'occlusion_exit', 'unknown'],
        'ACTOR': ['left_hand', 'right_hand', 'both_hands', 'left_gripper', 'right_gripper', 'both_grippers', 'robot_arm', 'unknown'],
        'OUTCOME_STATUS': ['success', 'failure', 'partial', 'unknown'],
        'RELATION_DIRECTION': ['forward', 'reverse'],
        'RELATION_PREDICATE': ['left_of', 'right_of', 'above', 'below', 'inside', 'on', 'overlapping', 'near', 'occluding', 'unknown'],
    }
    for suffix, vocabulary in expected_vocab.items():
        instruction = meanings['SCENE_SEMANTICS_CHOICES_' + suffix + '_ENUM_VALUE']
        assert all(re.search(r'\b' + word + r'\b', instruction) for word in vocabulary)
    if repair_type == 'unknown':
        public = json.loads((output / 'full_0001.scene_semantics.projected-public-scene.private.json').read_text())
        assert public['objects'] == repaired['objects']
        assert public['semantic_events'] == repaired['semantic_events']
    else:
        assert report['final_issue_codes'] == [code]
        assert not (output / 'full_0001.scene_semantics.validated.private.json').exists()
