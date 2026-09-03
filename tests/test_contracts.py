from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from las_repro.config import Settings
from las_repro.contracts import PollRequest, PollResponse, SubmitRequest, SubmitResponse
from las_repro.domain import InferenceJob, InferenceStatus, TaskRecord, TaskStatus


def test_submit_accepts_local_model_and_discards_ark_secrets():
    """Dropping a compatibility secret must fail if it reaches persisted data."""
    request = SubmitRequest.model_validate(
        {
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
                "model_name": "qwen3-vl-8b-instruct",
                "ark_api_key": "must-not-survive",
                "ark_endpoint_id": "must-not-survive",
                "use_responses_api": True,
            },
        }
    )

    assert request.sanitized_data() == {
        "video_url": "/allowed/demo.mp4",
        "task_template": "general_video_captioning",
        "model_name": "qwen3-vl-8b-instruct",
    }
    assert request.compatibility_warnings() == [
        "ark_api_key was ignored; inference is fully local",
        "ark_endpoint_id was ignored; inference is fully local",
        "use_responses_api was ignored; local VideoSession caching is used",
    ]


def test_submit_rejects_unknown_operator():
    """A typo in the operator identifier must not create a task."""
    with pytest.raises(ValidationError):
        SubmitRequest.model_validate(
            {
                "operator_id": "other",
                "operator_version": "v1",
                "data": {
                    "video_url": "/allowed/demo.mp4",
                    "task_template": "general_video_captioning",
                },
            }
        )


@pytest.mark.parametrize("model_name", ["/etc/passwd", "../model", "org/model"])
def test_submit_rejects_paths_and_remote_ids_as_model_aliases(model_name):
    """Treating model_name as a path or remote ID must fail at the contract boundary."""
    with pytest.raises(ValidationError):
        SubmitRequest.model_validate(
            {
                "operator_id": "las_video_understanding",
                "operator_version": "v1",
                "data": {
                    "video_url": "/allowed/demo.mp4",
                    "task_template": "general_video_captioning",
                    "model_name": model_name,
                },
            }
        )


@pytest.mark.parametrize(
    "operator_id", ["las_long_video_understand", "las_video_understanding"]
)
def test_submit_accepts_each_supported_operator_id(operator_id):
    """Removing either supported LAS identifier must reject a compatible client."""
    request = SubmitRequest.model_validate(
        {
            "operator_id": operator_id,
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
            },
        }
    )

    assert request.operator_id == operator_id


def test_query_only_submit_uses_general_video_captioning_template():
    """Requiring an explicit template must not reject the query-mode quickstart."""
    request = SubmitRequest.model_validate(
        {
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": {
                "video_url": "https://example.test/demo.mp4",
                "query": "Describe the visible actions in order.",
            },
        }
    )

    assert request.data.effective_template() == "general_video_captioning"
    assert request.sanitized_data() == {
        "video_url": "https://example.test/demo.mp4",
        "query": "Describe the visible actions in order.",
        "model_name": "qwen3-vl-8b-instruct",
        "task_template": "general_video_captioning",
    }


@pytest.mark.parametrize("query", [None, "", " \t "])
def test_submit_rejects_blank_query_without_template(query):
    """Dropping the template-or-query guard must admit an unrouteable task."""
    with pytest.raises(ValidationError):
        SubmitRequest.model_validate(
            {
                "operator_id": "las_video_understanding",
                "operator_version": "v1",
                "data": {
                    "video_url": "https://example.test/demo.mp4",
                    "query": query,
                },
            }
        )


@pytest.mark.parametrize("fps", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_submit_rejects_non_positive_or_non_finite_fps(fps):
    """Removing FPS bounds must admit unusable local sampling requests."""
    with pytest.raises(ValidationError):
        SubmitRequest.model_validate(
            {
                "operator_id": "las_video_understanding",
                "operator_version": "v1",
                "data": {
                    "video_url": "https://example.test/demo.mp4",
                    "query": "Describe the video.",
                    "fps": fps,
                },
            }
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (0, None),
        (None, 1),
        (-1, 1),
        (1, 1),
        (2, 1),
        (float("nan"), 1),
        (0, float("inf")),
    ],
)
def test_submit_rejects_invalid_or_incomplete_clip_bounds(start, end):
    """Dropping paired ordered finite bounds must admit an invalid clip."""
    with pytest.raises(ValidationError):
        SubmitRequest.model_validate(
            {
                "operator_id": "las_video_understanding",
                "operator_version": "v1",
                "data": {
                    "video_url": "https://example.test/demo.mp4",
                    "query": "Describe the video.",
                    "start": start,
                    "end": end,
                },
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("media_resolution", "ultra"),
        ("reasoning_effort", "maximum"),
        ("clip_context", "automatic"),
    ],
)
def test_submit_rejects_undocumented_tuning_values(field, value):
    """Replacing a tuning literal with an open string must admit unknown behavior."""
    with pytest.raises(ValidationError):
        SubmitRequest.model_validate(
            {
                "operator_id": "las_video_understanding",
                "operator_version": "v1",
                "data": {
                    "video_url": "https://example.test/demo.mp4",
                    "query": "Describe the video.",
                    field: value,
                },
            }
        )


def test_requests_reject_top_level_typos_but_preserve_data_tuning_fields():
    """Only documented request data may be extended with local tuning options."""
    request = SubmitRequest.model_validate(
        {
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
                "sample_fps": 2.0,
            },
        }
    )

    assert request.sanitized_data()["sample_fps"] == 2.0
    with pytest.raises(ValidationError):
        SubmitRequest.model_validate(
            {
                "operator_id": "las_long_video_understand",
                "operator_version": "v1",
                "data": {
                    "video_url": "/allowed/demo.mp4",
                    "task_template": "general_video_captioning",
                },
                "taskd_id": "typo",
            }
        )
    with pytest.raises(ValidationError):
        PollRequest.model_validate(
            {
                "operator_id": "las_long_video_understand",
                "operator_version": "v1",
                "task_id": "task-1",
                "extra": True,
            }
        )


def test_supplied_cloud_compatibility_fields_are_sanitized_without_mutation():
    """Cloud-only compatibility inputs must never leak through sanitation."""
    data = {
        "video_url": "/allowed/demo.mp4",
        "task_template": "general_video_captioning",
        "ark_api_key": "secret",
        "ark_endpoint_id": "endpoint",
        "use_responses_api": False,
        "previous_response_ids": ["response-1"],
        "expire_in": 42,
    }
    request = SubmitRequest.model_validate(
        {
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "data": data,
        }
    )

    assert request.sanitized_data() == {
        "video_url": "/allowed/demo.mp4",
        "task_template": "general_video_captioning",
        "model_name": "qwen3-vl-8b-instruct",
    }
    assert request.compatibility_warnings() == [
        "ark_api_key was ignored; inference is fully local",
        "ark_endpoint_id was ignored; inference is fully local",
        "use_responses_api was ignored; local VideoSession caching is used",
        "previous_response_ids was ignored; local VideoSession caching is used",
        "expire_in was ignored; local task retention is configured by the service",
    ]
    assert data["ark_api_key"] == "secret"
    assert data["previous_response_ids"] == ["response-1"]


@pytest.mark.parametrize(
    ("field", "warning"),
    [
        ("ark_api_key", "ark_api_key was ignored; inference is fully local"),
        (
            "ark_endpoint_id",
            "ark_endpoint_id was ignored; inference is fully local",
        ),
        (
            "use_responses_api",
            "use_responses_api was ignored; local VideoSession caching is used",
        ),
        (
            "previous_response_ids",
            "previous_response_ids was ignored; local VideoSession caching is used",
        ),
        (
            "expire_in",
            "expire_in was ignored; local task retention is configured by the service",
        ),
    ],
)
def test_explicit_null_cloud_field_is_discarded_with_warning(field, warning):
    """Checking parsed values must not confuse explicit null with omission."""
    request = SubmitRequest.model_validate(
        {
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
                field: None,
            },
        }
    )

    assert request.compatibility_warnings() == [warning]
    assert field not in request.sanitized_data()


def test_omitted_cloud_fields_do_not_emit_discard_warnings():
    """Warning on absent compatibility fields would misreport the request."""
    request = SubmitRequest.model_validate(
        {
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
            },
        }
    )

    assert request.compatibility_warnings() == []


def test_settings_defaults_and_environment_overrides(monkeypatch, tmp_path):
    """Service settings must be configurable without accepting plaintext TOS secrets."""
    monkeypatch.setenv("LAS_DATABASE_PATH", str(tmp_path / "tasks.sqlite3"))
    monkeypatch.setenv("LAS_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("LAS_ALLOWED_MEDIA_ROOTS", f'{tmp_path / "media"},{tmp_path / "other"}')
    monkeypatch.setenv("LAS_MODEL_REGISTRY", '{"custom": "/models/custom"}')
    monkeypatch.setenv("LAS_GPU_DEVICES", "4,5")
    monkeypatch.setenv("LAS_TOS_ACCESS_KEY", "access-secret")

    settings = Settings.from_env()

    assert settings.database_path == tmp_path / "tasks.sqlite3"
    assert settings.work_root == tmp_path / "work"
    assert settings.allowed_media_roots == (tmp_path / "media", tmp_path / "other")
    assert settings.model_registry == {"custom": Path("/models/custom")}
    assert settings.gpu_devices == (4, 5)
    assert settings.segment_seconds == 30.0
    assert settings.segment_overlap_seconds == 2.0
    assert settings.max_fine_segment_seconds == 1.0
    assert settings.lease_seconds == 300
    assert settings.tos_access_key.get_secret_value() == "access-secret"


def test_domain_records_are_serializable_values_with_lifecycle_statuses():
    """Store records need stable lifecycle enums and independent JSON values."""
    task = TaskRecord(
        task_id="task-1",
        operator_id="las_long_video_understand",
        operator_version="v1",
        payload={"video_url": "/allowed/demo.mp4"},
        status=TaskStatus.PENDING,
    )
    job = InferenceJob(
        job_id="job-1",
        task_id=task.task_id,
        stage="general_segment",
        ordinal=0,
        payload={"segment": 0},
        status=InferenceStatus.PENDING,
    )

    assert asdict(task)["status"] is TaskStatus.PENDING
    assert asdict(job)["status"] is InferenceStatus.PENDING
    assert TaskStatus.COMPLETED.value == "COMPLETED"
    assert InferenceStatus.FAILED.value == "FAILED"


def test_submit_and_poll_responses_preserve_las_metadata_and_completed_data():
    """API handlers must be able to return LAS metadata with optional task data."""
    submitted = SubmitResponse.model_validate(
        {
            "metadata": {
                "task_id": "task-1",
                "task_status": "PENDING",
                "warnings": ["ark_api_key was ignored; inference is fully local"],
            }
        }
    )
    completed = PollResponse.model_validate(
        {
            "metadata": {
                "task_id": "task-1",
                "task_status": "COMPLETED",
                "progress": {"stage": "pass_a", "completed": 1, "total": 1},
            },
            "data": {"summary": "done"},
        }
    )

    assert submitted.metadata.business_code == "0"
    assert submitted.metadata.error_msg == ""
    assert completed.metadata.task_status is TaskStatus.COMPLETED
    assert completed.data == {"summary": "done"}
