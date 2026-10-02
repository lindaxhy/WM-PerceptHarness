"""Tests for the synchronous in-memory job execution layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from percept_harness.domain import InferenceJobSpec, InferenceStatus
from percept_harness.execution import (
    ExecutionError,
    InferenceJobFailed,
    JobWaitTimeout,
    SyncJobStore,
    wait_for_jobs,
)
from percept_harness.metrics import validate_inference_job_metrics
from percept_harness.models.base import ModelRequest
from percept_harness.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS


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


class _CachingModel(_ScriptedModel):
    """Scripted model that opts into the semantic result cache."""

    supports_semantic_result_cache = True

    def semantic_cache_identity(self, request: ModelRequest) -> dict[str, Any]:
        return {"endpoint": "https://provider.example/v1", "model": request.model_name}


def test_job_error_text_reaches_the_wait_exception(video: Path) -> None:
    model = _ScriptedModel(RuntimeError("provider rejected the request"))
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct")

    [job] = store.create_inference_jobs("task-1", [_spec(video)])

    with pytest.raises(InferenceJobFailed) as info:
        wait_for_jobs(store, "task-1", [job.job_id], 1.0)
    assert info.value.job_error == "provider rejected the request"
    assert "provider rejected the request" in str(info.value)


def test_semantic_cache_replays_identical_requests(video: Path, tmp_path: Path) -> None:
    from percept_harness.execution import SemanticResultCache

    cache = SemanticResultCache(tmp_path / "semantic-cache")
    model = _CachingModel(_segment())
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct",
                         semantic_cache=cache)
    [first] = store.create_inference_jobs("task-1", [_spec(video)])
    assert first.status is InferenceStatus.COMPLETED
    assert first.metrics["semantic_cache_published"] is True
    validate_inference_job_metrics(first.metrics)

    replay_model = _CachingModel()  # no scripted outputs: a real call would fail
    replay_store = SyncJobStore(replay_model, default_model_alias="qwen3-vl-8b-instruct",
                                semantic_cache=cache)
    [second] = replay_store.create_inference_jobs("task-2", [_spec(video)])
    assert second.status is InferenceStatus.COMPLETED
    assert second.metrics["semantic_cache_hit"] is True
    assert second.result == first.result
    validate_inference_job_metrics(second.metrics)
    assert len(replay_model.requests) == 0


def test_semantic_cache_misses_on_changed_prompt_or_bytes(video: Path, tmp_path: Path) -> None:
    from percept_harness.execution import SemanticResultCache

    cache = SemanticResultCache(tmp_path / "semantic-cache")
    store = SyncJobStore(_CachingModel(_segment()),
                         default_model_alias="qwen3-vl-8b-instruct", semantic_cache=cache)
    store.create_inference_jobs("task-1", [_spec(video)])

    other_prompt = _CachingModel(_segment("another visible action"))
    store2 = SyncJobStore(other_prompt, default_model_alias="qwen3-vl-8b-instruct",
                          semantic_cache=cache)
    [job] = store2.create_inference_jobs("task-2", [_spec(video, prompt="different")])
    assert len(other_prompt.requests) == 1  # miss: prompt changed

    video.write_bytes(b"changed bytes")
    other_bytes = _CachingModel(_segment())
    store3 = SyncJobStore(other_bytes, default_model_alias="qwen3-vl-8b-instruct",
                          semantic_cache=cache)
    store3.create_inference_jobs("task-3", [_spec(video)])
    assert len(other_bytes.requests) == 1  # miss: video bytes changed


def test_semantic_cache_is_ignored_for_models_without_support(video: Path, tmp_path: Path) -> None:
    from percept_harness.execution import SemanticResultCache

    cache = SemanticResultCache(tmp_path / "semantic-cache")
    model = _ScriptedModel(_segment(), _segment())
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct",
                         semantic_cache=cache)
    store.create_inference_jobs("task-1", [_spec(video)])
    store2 = SyncJobStore(_ScriptedModel(_segment()),
                          default_model_alias="qwen3-vl-8b-instruct", semantic_cache=cache)
    [job] = store2.create_inference_jobs("task-2", [_spec(video)])
    assert job.metrics.get("semantic_cache_hit") is None


def test_corrupt_cache_entry_is_a_miss(video: Path, tmp_path: Path) -> None:
    from percept_harness.execution import SemanticResultCache

    root = tmp_path / "semantic-cache"
    cache = SemanticResultCache(root)
    model = _CachingModel(_segment())
    store = SyncJobStore(model, default_model_alias="qwen3-vl-8b-instruct",
                         semantic_cache=cache)
    [job] = store.create_inference_jobs("task-1", [_spec(video)])
    key = job.metrics["semantic_cache_key"]
    entry = root / key[:2] / f"{key}.json"
    entry.write_text("{corrupt", encoding="utf-8")

    fresh = _CachingModel(_segment())
    store2 = SyncJobStore(fresh, default_model_alias="qwen3-vl-8b-instruct",
                          semantic_cache=SemanticResultCache(root))
    [again] = store2.create_inference_jobs("task-2", [_spec(video)])
    assert again.status is InferenceStatus.COMPLETED
    assert len(fresh.requests) == 1  # corrupt entry ignored, model re-ran


def _coarse_spec(video: Path) -> InferenceJobSpec:
    """A spec whose schema has a real validator (unlike general_segment)."""
    return _spec(video, stage="coarse_plan", schema_name="CoarsePlan",
                 schema_context={"duration": 1.0})


_VALID_COARSE_PLAN = {
    "task_description": "lift the block",
    "entity_candidates": [
        {"name": "right hand", "aliases": ["hand"], "role": "actor"},
        {"name": "block", "aliases": ["cube"], "role": "manipulated_object"}],
    "actions": [{"action_index": 0, "start": 0.0, "end": 1.0,
                 "description": "hand lifts the block",
                 "event_type": "reach_and_grasp"}]}


def _seed_cache_key(video: Path, cache) -> str:
    """The exact key the store uses for ``_coarse_spec`` (real schema context)."""
    [job] = SyncJobStore(_CachingModel(_VALID_COARSE_PLAN),
                         default_model_alias="qwen3-vl-8b-instruct",
                         semantic_cache=cache).create_inference_jobs(
        "seed", [_coarse_spec(video)])
    return job.metrics["semantic_cache_key"]


def test_invalid_output_is_not_cached_and_a_retry_reaches_the_model(
    video: Path, tmp_path: Path
) -> None:
    """An invalid-output envelope is a verdict, not a result: retries must retry."""
    from percept_harness.execution import SemanticResultCache

    root = tmp_path / "semantic-cache"
    store = SyncJobStore(_CachingModel({"unexpected": "shape"}),
                         default_model_alias="qwen3-vl-8b-instruct",
                         semantic_cache=SemanticResultCache(root))
    [job] = store.create_inference_jobs("task-1", [_coarse_spec(video)])

    assert DEFAULT_OUTPUT_SCHEMAS.failure_codes("CoarsePlan", job.result) is not None
    assert "semantic_cache_published" not in job.metrics
    assert list(root.rglob("*.json")) == []

    retry_model = _CachingModel({"unexpected": "shape"})
    retry = SyncJobStore(retry_model, default_model_alias="qwen3-vl-8b-instruct",
                         semantic_cache=SemanticResultCache(root))
    retry.create_inference_jobs("task-2", [_coarse_spec(video)])
    assert len(retry_model.requests) == 1


def test_cached_invalid_envelope_written_by_older_code_is_a_miss(
    video: Path, tmp_path: Path
) -> None:
    """Entries published before this fix must not be replayed."""
    from percept_harness.execution import SemanticResultCache

    root = tmp_path / "semantic-cache"
    cache = SemanticResultCache(root)
    key = _seed_cache_key(video, cache)
    assert cache.publish(key, {  # the shape the old code stored
        "_schema_validation": {"schema_name": "CoarsePlan", "status": "invalid",
                               "issue_codes": ["COARSE_PLAN_EXTRA_FIELD"]}})

    retry_model = _CachingModel({"unexpected": "shape"})
    SyncJobStore(retry_model, default_model_alias="qwen3-vl-8b-instruct",
                 semantic_cache=SemanticResultCache(root)).create_inference_jobs(
        "task-2", [_coarse_spec(video)])
    assert len(retry_model.requests) == 1


def test_valid_result_is_cached_and_replayed_without_the_model(
    video: Path, tmp_path: Path
) -> None:
    """The raw model output is cached, then replayed through the same path."""
    from percept_harness.execution import SemanticResultCache

    root = tmp_path / "semantic-cache"
    valid = {"task_description": "lift the block",
             "entity_candidates": [
                 {"name": "right hand", "aliases": ["hand"], "role": "actor"},
                 {"name": "block", "aliases": ["cube"], "role": "manipulated_object"},
             ],
             "actions": [{"action_index": 0, "start": 0.0, "end": 1.0,
                          "description": "hand lifts the block",
                          "event_type": "reach_and_grasp"}]}
    store = SyncJobStore(_CachingModel(valid), default_model_alias="qwen3-vl-8b-instruct",
                         semantic_cache=SemanticResultCache(root))
    [first] = store.create_inference_jobs("task-1", [_coarse_spec(video)])
    assert DEFAULT_OUTPUT_SCHEMAS.failure_codes("CoarsePlan", first.result) is None
    assert first.metrics["semantic_cache_published"] is True

    replay_model = _CachingModel()
    [second] = SyncJobStore(
        replay_model, default_model_alias="qwen3-vl-8b-instruct",
        semantic_cache=SemanticResultCache(root),
    ).create_inference_jobs("task-2", [_coarse_spec(video)])
    assert second.metrics["semantic_cache_hit"] is True
    assert second.result == first.result
    assert len(replay_model.requests) == 0


def test_cached_normalized_envelope_from_older_code_is_a_miss(
    video: Path, tmp_path: Path
) -> None:
    """A sanitized envelope is not raw output; replaying it must not be attempted."""
    from percept_harness.execution import SemanticResultCache

    root = tmp_path / "semantic-cache"
    cache = SemanticResultCache(root)
    key = _seed_cache_key(video, cache)
    assert cache.publish(key, {  # a normalized envelope, as the old code stored
        "_schema_validation": {"schema_name": "CoarsePlan", "status": "normalized",
                               "issue_codes": ["COARSE_PLAN_TOPOLOGY_NORMALIZED"],
                               "normalized_field_count": 1},
        "data": {"task_description": "x", "entity_candidates": [], "actions": []}})

    retry_model = _CachingModel({"unexpected": "shape"})
    [job] = SyncJobStore(
        retry_model, default_model_alias="qwen3-vl-8b-instruct",
        semantic_cache=SemanticResultCache(root),
    ).create_inference_jobs("task-2", [_coarse_spec(video)])
    assert len(retry_model.requests) == 1  # reached the model, not replayed
    assert "semantic_cache_hit" not in job.metrics


def test_cache_stores_raw_model_output_not_the_sanitized_shape(
    video: Path, tmp_path: Path
) -> None:
    """The stored payload must equal what the model returned."""
    import json as _json
    from percept_harness.execution import SemanticResultCache

    root = tmp_path / "semantic-cache"
    valid = {"task_description": "lift the block",
             "entity_candidates": [
                 {"name": "right hand", "aliases": ["hand"], "role": "actor"},
                 {"name": "block", "aliases": ["cube"], "role": "manipulated_object"}],
             "actions": [{"action_index": 0, "start": 0.0, "end": 1.0,
                          "description": "hand lifts the block",
                          "event_type": "reach_and_grasp"}]}
    SyncJobStore(_CachingModel(valid), default_model_alias="qwen3-vl-8b-instruct",
                 semantic_cache=SemanticResultCache(root)).create_inference_jobs(
        "task-1", [_coarse_spec(video)])
    entries = list(root.rglob("*.json"))
    assert len(entries) == 1
    assert _json.loads(entries[0].read_text())["result"] == valid
