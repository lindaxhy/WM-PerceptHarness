"""Tests for the synchronous in-memory job execution layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from las_repro.domain import InferenceJobSpec, InferenceStatus
from las_repro.execution import (
    ExecutionError,
    InferenceJobFailed,
    JobWaitTimeout,
    SyncJobStore,
    wait_for_jobs,
)
from las_repro.metrics import validate_inference_job_metrics
from las_repro.models.base import ModelRequest


class _ScriptedModel:
    """Return queued structured outputs and record requests."""

    def __init__(self, *outputs: Any) -> None:
        self.outputs = list(outputs)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> Any:
        self.requests.append(request)
        if not self.outputs:
            raise AssertionError("scripted model exhausted")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def _spec(video: Path, *, stage: str = "general_segment", ordinal: int = 0,
          **payload_overrides: Any) -> InferenceJobSpec:
    payload = {
        "video_path": str(video),
        "span": {"start": 0.0, "end": 1.0},
        "fps": 2.0,
        "prompt": "describe",
        "schema_name": "general_segment",
        "video_session_id": None,
        **payload_overrides,
    }
    return InferenceJobSpec(stage=stage, ordinal=ordinal, payload=payload)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"video")
    return path.resolve()


def _segment(description: str = "a visible action") -> dict[str, Any]:
    return {
        "segments": [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "scene": ["room"],
                "subjects": ["person"],
                "actions": ["moves"],
                "visible_text": [],
                "uncertainty": [],
                "description": description,
                "warnings": [],
            }
        ],
        "warnings": [],
    }


def test_jobs_execute_eagerly_and_results_are_terminal(video: Path) -> None:
    model = _ScriptedModel(_segment())
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct")

    [job] = store.create_inference_jobs("task-1", [_spec(video)])

    assert job.status is InferenceStatus.COMPLETED
    assert job.completed_by == "sync"
    assert job.result and "segments" in job.result
    assert job.metrics and job.metrics["inference_seconds"] >= 0.0
    validate_inference_job_metrics(job.metrics)
    assert len(model.requests) == 1


def test_duplicate_stage_ordinal_returns_existing_record(video: Path) -> None:
    model = _ScriptedModel(_segment())
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct")

    [first] = store.create_inference_jobs("task-1", [_spec(video)])
    [second] = store.create_inference_jobs("task-1", [_spec(video)])

    assert second is first
    assert len(model.requests) == 1


def test_model_exception_marks_job_failed_and_wait_raises(video: Path) -> None:
    model = _ScriptedModel(RuntimeError("private backend detail"))
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct")

    [job] = store.create_inference_jobs("task-1", [_spec(video)])

    assert job.status is InferenceStatus.FAILED
    with pytest.raises(InferenceJobFailed):
        wait_for_jobs(store, "task-1", [job.job_id], 1.0)


def test_invalid_video_path_fails_the_job_not_the_process(tmp_path: Path) -> None:
    store = SyncJobStore(
        _ScriptedModel(), default_model_alias="qwen3-vl-8b-instruct"
    )
    [job] = store.create_inference_jobs(
        "task-1", [_spec(tmp_path / "missing.mp4")]
    )
    assert job.status is InferenceStatus.FAILED


def test_cv_stage_without_executor_fails_cleanly(video: Path) -> None:
    store = SyncJobStore(
        _ScriptedModel(), default_model_alias="qwen3-vl-8b-instruct"
    )
    [job] = store.create_inference_jobs(
        "task-1",
        [InferenceJobSpec(stage="cv_evidence", ordinal=0, payload={},
                          model_name="sam3.1")],
    )
    assert job.status is InferenceStatus.FAILED
    assert "CV executor" in (job.error or "")


def test_wait_for_jobs_returns_results_in_requested_order(video: Path) -> None:
    model = _ScriptedModel(_segment("first"), _segment("second"))
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct")

    jobs = store.create_inference_jobs(
        "task-1", [_spec(video, ordinal=0), _spec(video, ordinal=1)]
    )
    results = wait_for_jobs(
        store, "task-1", [jobs[1].job_id, jobs[0].job_id], 1.0
    )
    assert results[0]["segments"][0]["description"] == "second"
    assert results[1]["segments"][0]["description"] == "first"


def test_wait_for_jobs_rejects_unknown_and_duplicate_ids(video: Path) -> None:
    model = _ScriptedModel(_segment())
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct")
    [job] = store.create_inference_jobs("task-1", [_spec(video)])

    with pytest.raises(KeyError):
        wait_for_jobs(store, "task-1", [job.job_id, "unknown"], 1.0)
    with pytest.raises(ValueError):
        wait_for_jobs(store, "task-1", [job.job_id, job.job_id], 1.0)
    with pytest.raises(ValueError):
        wait_for_jobs(store, "task-1", [job.job_id], float("inf"))


def test_nonmapping_model_output_uses_schema_failure_envelope(video: Path) -> None:
    model = _ScriptedModel("just text, not a mapping")
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct")

    [job] = store.create_inference_jobs("task-1", [_spec(video)])

    assert job.status is InferenceStatus.COMPLETED
    assert job.result == {
        "_schema_validation": {
            "schema_name": "general_segment",
            "status": "invalid",
            "issue_codes": ["GENERAL_SEGMENT_SCHEMA_INVALID"],
        }
    }
    assert "just text" not in json.dumps(job.result)


def test_metrics_validation_rejects_unknown_and_malformed_keys() -> None:
    validate_inference_job_metrics(None)
    validate_inference_job_metrics({"inference_seconds": 0.5, "input_tokens": 3})
    with pytest.raises(ValueError):
        validate_inference_job_metrics({"surprise": 1})
    with pytest.raises(ValueError):
        validate_inference_job_metrics({"inference_seconds": -0.5})
    with pytest.raises(ValueError):
        validate_inference_job_metrics({"input_tokens": -1})
    with pytest.raises(ValueError):
        validate_inference_job_metrics({"cache_hit": "yes"})
    with pytest.raises(ValueError):
        validate_inference_job_metrics({"semantic_cache_key": "not-a-digest"})
    with pytest.raises(TypeError):
        validate_inference_job_metrics(["not", "a", "mapping"])
