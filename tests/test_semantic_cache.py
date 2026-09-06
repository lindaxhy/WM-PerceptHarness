"""Durable semantic replay exercises real ARK parsing, validators and SQLite."""
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest

from las_repro.domain import InferenceJobSpec, InferenceStatus
from las_repro.media import FrameRef
from las_repro.models.ark import ArkVideoModel
from las_repro.models.fake import FakeVideoModel
from las_repro.workers import GPUWorker
from las_repro.store import SQLiteTaskStore


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@pytest.fixture
def store(tmp_path):
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store.initialize()
    return store


def job(store, tmp_path, *, data=b"video", overrides=None, stage="embodied_pass_a", ordinal=0):
    task = store.create_task({"video_url": "fixture", "task_template": "embodied", "model_name": "doubao-pro"})
    path = tmp_path / (task.task_id + ".mp4")
    path.write_bytes(data)
    payload = dict(video_path=str(path), start=0.0, end=1.0, fps=2.0,
                   prompt="describe", schema_name="CoarsePlan", schema_context={"duration": 1.0},
                   video_session_id=task.task_id)
    payload.update(overrides or {})
    return store.create_inference_jobs(task.task_id, [InferenceJobSpec(stage, ordinal, payload, model_name="doubao-pro")])[0]


class Provider:
    def __init__(self, response=None, before=None):
        self.calls = 0
        self.response = response
        self.before = before
        self.request = None

    def extract(self, path, span, fps, output):
        self.path = path
        image = output / "frame.jpg"
        image.write_bytes(b"jpeg")
        return [FrameRef(path=image, timestamp=span.start)]

    def transport(self, request):
        self.calls += 1
        if self.before:
            self.before(self)
        payload = json.loads(request.content)
        self.request = payload
        result = self.response
        if result is None:
            result = {"task_description": "move cup", "entity_candidates": [],
                      "actions": [{"action_index": 0, "event_type": "transport", "start": 0.0, "end": 1.0, "description": "move cup"}]}
        return httpx.Response(200, json={"status": "completed", "model": payload["model"],
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(result)}]}],
            "usage": {"input_tokens": 17, "output_tokens": 9}})

    def model(self, **kwargs):
        return ArkVideoModel(api_key="test-placeholder", model_registry={"doubao-pro": "resolved-v1"},
                             transport=httpx.MockTransport(self.transport), frame_extractor=self.extract, **kwargs)


def run(store, model, *, enabled=True, worker="ark", now=None):
    worker = GPUWorker(store, model, worker, "remote:ark", model_name="doubao-pro",
                       semantic_cache_enabled=enabled)
    assert worker.run_once(now=now)


def test_cross_task_replay_binds_bytes_and_completes_fresh_job(store, tmp_path):
    provider = Provider()
    with_model = provider.model()
    first = job(store, tmp_path)
    run(store, with_model)
    with_model.close()
    reopened = SQLiteTaskStore(store.database_path)
    reopened.initialize()
    second = job(reopened, tmp_path)
    run(reopened, provider.model())
    a, b = [store.get_inference_job(j.job_id) for j in (first, second)]
    assert a.status is b.status is InferenceStatus.COMPLETED
    assert provider.calls == 1
    assert a.result == b.result
    assert a.metrics["semantic_cache_hit"] is False
    assert a.metrics["semantic_cache_published"] is True
    assert b.metrics["semantic_cache_hit"] is True
    assert b.metrics["semantic_cache_published"] is False
    assert b.metrics["input_tokens"] == b.metrics["output_tokens"] == b.metrics["inference_seconds"] == 0
    assert b.metrics["semantic_cache_key"] == a.metrics["semantic_cache_key"]
    assert b.metrics["semantic_cache_result_sha256"] == hashlib.sha256(canonical(b.result).encode()).hexdigest()
    with sqlite3.connect(store.database_path) as db:
        identity = json.loads(db.execute("SELECT identity_json FROM semantic_results").fetchone()[0])
    assert identity["video_sha256"] == hashlib.sha256(b"video").hexdigest()
    assert str(tmp_path) not in canonical(identity)


@pytest.mark.parametrize("overrides,data,setting", [
    ({"prompt": "changed"}, b"video", None),
    ({"fps": 1.0}, b"video", None),
    ({"end": .9}, b"video", None),
    ({"schema_context": {"duration": 2.0}}, b"video", None),
    ({"media_resolution": "medium"}, b"video", None),
    ({"reasoning_effort": "high"}, b"video", None),
    ({"clip_context": "high"}, b"video", None),
    ({}, b"different", None),
    ({}, b"video", "max_frames"),
    ({}, b"video", "max_request_bytes"),
    ({}, b"video", "max_output_chars"),
    ({}, b"video", "model"),
])
def test_semantic_inputs_miss(store, tmp_path, overrides, data, setting):
    provider = Provider()
    model = provider.model()
    job(store, tmp_path)
    run(store, model)
    if setting == "model":
        model._registry["doubao-pro"] = "resolved-v2"
    elif setting:
        setattr(model, setting, getattr(model, setting) + 1)
    second = job(store, tmp_path, data=data, overrides=overrides)
    run(store, model)
    assert provider.calls == 2
    assert store.get_inference_job(second.job_id).metrics["semantic_cache_hit"] is False


def test_safe_failure_and_repair_replay_without_raw_content(store, tmp_path):
    provider = Provider(response={"raw": "private invalid response"})
    model = provider.model()
    results = []
    for repeat in range(2):
        first = job(store, tmp_path)
        run(store, model)
        failed = store.get_inference_job(first.job_id)
        codes = failed.result["_schema_validation"]["issue_codes"]
        provider.response = None
        repair = job(store, tmp_path, overrides={"prompt": "repair " + canonical(codes)}, ordinal=1)
        run(store, model)
        results.append((failed.result, store.get_inference_job(repair.job_id).result))
    assert provider.calls == 2
    assert results[0] == results[1]
    assert "private invalid" not in canonical(results)


def test_disable_and_generic_models_do_not_replay(store, tmp_path):
    provider = Provider()
    model = provider.model()
    for _ in range(2):
        j = job(store, tmp_path)
        run(store, model, enabled=False)
        assert "semantic_cache_hit" not in store.get_inference_job(j.job_id).metrics
    assert provider.calls == 2
    j = job(store, tmp_path)
    run(store, FakeVideoModel())
    assert "semantic_cache_hit" not in store.get_inference_job(j.job_id).metrics


def test_transport_failure_never_cached(store, tmp_path):
    def fail(_):
        raise httpx.ConnectError("private transport detail")
    provider = Provider(before=fail)
    model = provider.model()
    first = job(store, tmp_path)
    run(store, model)
    assert store.get_inference_job(first.job_id).status is InferenceStatus.FAILED
    provider.before = None
    second = job(store, tmp_path)
    run(store, model)
    assert provider.calls == 2
    assert store.get_inference_job(second.job_id).metrics["semantic_cache_hit"] is False


def test_changed_video_during_generation_suppresses_publication(store, tmp_path):
    provider = Provider(before=lambda p: p.path.write_bytes(b"changed"))
    model = provider.model()
    first = job(store, tmp_path)
    run(store, model)
    assert store.get_inference_job(first.job_id).metrics["semantic_cache_published"] is False
    provider.before = None
    job(store, tmp_path)
    run(store, model)
    assert provider.calls == 2


@pytest.mark.parametrize("damage", ["json", "digest", "identity", "schema", "oversize", "unsafe"])
def test_corrupt_cache_is_read_only_miss_then_replaced(store, tmp_path, damage):
    provider = Provider()
    model = provider.model()
    job(store, tmp_path)
    run(store, model)
    with sqlite3.connect(store.database_path) as db:
        if damage == "json": db.execute("UPDATE semantic_results SET result_json = '{'")
        if damage == "digest": db.execute("UPDATE semantic_results SET result_sha256 = ?", ("0" * 64,))
        if damage == "identity": db.execute("UPDATE semantic_results SET identity_json = '{}' ")
        if damage == "schema": db.execute("UPDATE semantic_results SET schema_version = -1")
        if damage in ("oversize", "unsafe"):
            raw = canonical({"raw": "x" * (8 * 1024 * 1024 if damage == "oversize" else 1)})
            db.execute("UPDATE semantic_results SET result_json=?, result_sha256=?", (raw, hashlib.sha256(raw.encode()).hexdigest()))
    second = job(store, tmp_path)
    run(store, model)
    assert provider.calls == 2
    assert store.get_inference_job(second.job_id).metrics["semantic_cache_published"] is True
    job(store, tmp_path)
    run(store, model)
    assert provider.calls == 2


def test_stale_lease_cannot_publish(store, tmp_path):
    first = job(store, tmp_path)
    def steal(_):
        store.expire_inference_job_lease(first.job_id, "ark", attempt=1, now=100.)
        assert store.claim_inference_job("winner", model_name="doubao-pro", lease_seconds=300., now=101.)
    provider = Provider(before=steal)
    run(store, provider.model(), now=100.)
    assert store.get_inference_job(first.job_id).worker_id == "winner"
    with sqlite3.connect(store.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM semantic_results").fetchone()[0] == 0


def test_divergent_simultaneous_misses_keep_first_valid_winner_and_own_metrics(store, tmp_path):
    barrier = Barrier(2)
    providers = [Provider(response={"raw": "invalid"}, before=lambda _: barrier.wait(timeout=5)),
                 Provider(before=lambda _: barrier.wait(timeout=5))]
    jobs = [job(store, tmp_path) for _ in range(2)]
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, store, p.model(), worker=f"ark-{i}") for i, p in enumerate(providers)]
        for future in futures: future.result(timeout=10)
    completed = [store.get_inference_job(j.job_id) for j in jobs]
    assert all(j.status is InferenceStatus.COMPLETED for j in completed)
    assert sum(j.metrics["semantic_cache_published"] for j in completed) == 1
    assert completed[0].result != completed[1].result
    for j in completed:
        assert j.metrics["semantic_cache_hit"] is False
        assert j.metrics["input_tokens"] == 17
        assert j.metrics["semantic_cache_result_sha256"] == hashlib.sha256(canonical(j.result).encode()).hexdigest()
    winner = next(j for j in completed if j.metrics["semantic_cache_published"])
    third = job(store, tmp_path)
    run(store, providers[0].model())
    assert store.get_inference_job(third.job_id).result == winner.result
    assert sum(p.calls for p in providers) == 2


def test_cache_write_failure_rolls_back_and_retries_completion(store, tmp_path):
    with sqlite3.connect(store.database_path) as db:
        db.execute("CREATE TRIGGER reject_cache BEFORE INSERT ON semantic_results BEGIN SELECT RAISE(ABORT, 'injected private error'); END")
    provider = Provider()
    j = job(store, tmp_path)
    run(store, provider.model())
    completed = store.get_inference_job(j.job_id)
    assert completed.status is InferenceStatus.COMPLETED
    assert completed.metrics["semantic_cache_published"] is False
    assert "private error" not in canonical(completed.result)
    with sqlite3.connect(store.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM semantic_results").fetchone()[0] == 0


def test_semantic_metrics_only_accept_exact_digests_and_booleans():
    from las_repro.store import validate_inference_job_metrics
    validate_inference_job_metrics({"semantic_cache_key": "a" * 64, "semantic_cache_result_sha256": "b" * 64,
                                   "semantic_cache_hit": True, "semantic_cache_published": False})
    for value in ("A" * 64, "g" * 64, "a" * 63, 1, None, "/private/path"):
        with pytest.raises(ValueError): validate_inference_job_metrics({"semantic_cache_key": value})
    for value in (1, "true", None):
        with pytest.raises(ValueError): validate_inference_job_metrics({"semantic_cache_published": value})


@pytest.mark.parametrize("schema", ["BoundaryPlan", "EnrichmentResult"])
def test_normalized_repair_envelope_replays_and_context_change_misses(store, tmp_path, schema):
    from test_embodied_validators import (_repairable_boundary_output, _boundary_fallback_context,
                                          _fallback_enrichment_result, _fallback_context)
    if schema == "BoundaryPlan":
        coarse, response = _repairable_boundary_output()
        context = _boundary_fallback_context(coarse, True)
        stage, flag = "embodied_pass_b", "allow_topology_fallback"
    else:
        response = _fallback_enrichment_result()
        context = _fallback_context(True)
        stage, flag = "embodied_enrichment", "allow_enum_unknown_fallback"
    provider = Provider(response=response)
    model = provider.model()
    outputs = []
    for _ in range(2):
        j = job(store, tmp_path, stage=stage, overrides={"schema_name": schema, "schema_context": context})
        run(store, model)
        outputs.append(store.get_inference_job(j.job_id).result)
    assert outputs[0]["_schema_validation"]["status"] == "normalized"
    assert outputs[0] == outputs[1]
    assert provider.calls == 1
    j = job(store, tmp_path, stage=stage, overrides={"schema_name": schema, "schema_context": {**context, flag: False}})
    run(store, model)
    assert provider.calls == 2
    assert store.get_inference_job(j.job_id).result["_schema_validation"]["status"] == "invalid"


@pytest.mark.parametrize("contract", ["adapter", "frames", "validator", "stage_tokens", "schema"])
def test_contract_changes_invalidate_replay(store, tmp_path, monkeypatch, contract):
    from las_repro import semantic_cache
    from las_repro.models.ark import ARK_STAGE_MAX_OUTPUT_TOKENS
    provider = Provider()
    model = provider.model()
    job(store, tmp_path)
    run(store, model)
    overrides = {}
    if contract == "adapter": model.adapter_contract_version = "new-adapter"
    if contract == "frames": model.frame_extraction_contract_version = "new-extractor"
    if contract == "validator": monkeypatch.setattr(semantic_cache, "VALIDATOR_CONTRACT_VERSION", "new-validator")
    if contract == "stage_tokens": monkeypatch.setitem(ARK_STAGE_MAX_OUTPUT_TOKENS, "embodied_pass_a", 999)
    if contract == "schema": overrides = {"schema_name": "UnregisteredSchema"}
    job(store, tmp_path, overrides=overrides)
    run(store, model)
    assert provider.calls == 2


def test_timeout_is_excluded_and_ignored_hints_are_labeled(store, tmp_path):
    provider = Provider()
    model = provider.model()
    overrides = {"media_resolution": "medium", "reasoning_effort": "high", "clip_context": "low"}
    job(store, tmp_path, overrides=overrides)
    run(store, model)
    model.timeout_seconds += 1
    job(store, tmp_path, overrides=overrides)
    run(store, model)
    assert provider.calls == 1
    with sqlite3.connect(store.database_path) as db:
        request = json.loads(db.execute("SELECT identity_json FROM semantic_results").fetchone()[0])["request"]
    assert request["thinking"] == {"type": "disabled"}
    assert request["image_pixel_limit"] == {"min_pixels": 4096, "max_pixels": 131072}
    assert request["recorded_hints"] == {"reasoning_effort": "high", "clip_context": "low"}


def test_cache_failure_retry_does_not_swallow_new_lease_owner(store, tmp_path, monkeypatch):
    from las_repro.store import SemanticCachePublicationError
    original = store.complete_inference_job
    first = job(store, tmp_path)
    def complete(job_id, result, **kwargs):
        if kwargs.get("semantic_publication"):
            store.expire_inference_job_lease(job_id, "ark", attempt=1, now=100.)
            store.claim_inference_job("winner", model_name="doubao-pro", lease_seconds=300., now=101.)
            raise SemanticCachePublicationError("safe failure")
        return original(job_id, result, **kwargs)
    monkeypatch.setattr(store, "complete_inference_job", complete)
    run(store, Provider().model(), now=100.)
    current = store.get_inference_job(first.job_id)
    assert current.status is InferenceStatus.RUNNING
    assert current.worker_id == "winner"
    with sqlite3.connect(store.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM semantic_results").fetchone()[0] == 0


def test_cache_lookup_does_not_delete_corruption(store, tmp_path):
    from las_repro.semantic_cache import make_identity, safe_result
    from las_repro.workers import _model_request
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
    provider = Provider()
    model = provider.model()
    j = job(store, tmp_path)
    run(store, model)
    request = _model_request(j)
    identity = make_identity(model, request, j.payload["schema_context"])
    with sqlite3.connect(store.database_path) as db:
        db.execute("UPDATE semantic_results SET result_json = '{'")
    assert store.lookup_semantic_result(identity, lambda v: safe_result(v, request.schema_name, j.payload["schema_context"], DEFAULT_OUTPUT_SCHEMAS)) is None
    with sqlite3.connect(store.database_path) as db:
        assert db.execute("SELECT result_json FROM semantic_results").fetchone()[0] == "{"


def test_eviction_bounds_entry_count_and_aggregate_bytes(store, tmp_path):
    from las_repro.semantic_cache import make_identity
    from las_repro.workers import _model_request
    provider = Provider()
    model = provider.model()
    for index in range(257):
        j = job(store, tmp_path, overrides={"prompt": str(index)})
        run(store, model, now=100. + index)
    assert provider.calls == 257
    with sqlite3.connect(store.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM semantic_results").fetchone()[0] == 256
        assert db.execute("SELECT MIN(created_at) FROM semantic_results").fetchone()[0] == 101.
    # Valid schema output near the per-entry limit; publication uses the same fenced store path.
    large = {"task_description": "x" * (7 * 1024 * 1024), "entity_candidates": [], "actions": []}
    for index in range(10):
        j = job(store, tmp_path, overrides={"prompt": "large" + str(index)})
        claim = store.claim_inference_job("bulk", model_name="doubao-pro", lease_seconds=300., now=1000. + index)
        identity = make_identity(model, _model_request(j), j.payload["schema_context"])
        store.complete_inference_job(j.job_id, large, worker_id="bulk", attempt=claim.attempt,
                                     now=1000. + index, semantic_publication=(identity, lambda value: value == large))
    with sqlite3.connect(store.database_path) as db:
        count, total = db.execute("SELECT COUNT(*), SUM(length(CAST(identity_json AS BLOB)) + length(CAST(result_json AS BLOB))) FROM semantic_results").fetchone()
        assert count == 9
        assert total <= 64 * 1024 * 1024
        assert db.execute("SELECT MIN(created_at) FROM semantic_results").fetchone()[0] == 1001.


def test_two_pipeline_tasks_replay_repair_and_preserve_real_cv_hit(tmp_path, monkeypatch):
    from test_embodied_pipeline import _ActionHarness
    from las_repro.cv.artifacts import CvArtifactStore
    from las_repro.cv.base import FakeCvEvidenceProvider
    from las_repro.cv.contracts import FrameTimeline, FrameTimestamp
    from las_repro.cv.worker import CVEvidenceWorker
    from las_repro.domain import TaskStatus
    from las_repro.pipelines import embodied as pipeline_module
    from las_repro.pipelines.embodied import EmbodiedActionPipeline
    from las_repro.pipelines.hybrid_result import validate_hybrid_result
    from las_repro.workers import wait_for_jobs

    fake = FakeVideoModel(failure_script={"embodied_enrichment": [{}]})
    model_calls = []
    class ArkFixture(ArkVideoModel):
        def generate(self, request):
            model_calls.append(request)
            self._metrics = {"input_tokens": 17, "output_tokens": 9}
            return fake.generate(request)
    model = ArkFixture(api_key="test-placeholder", model_registry={"qwen3-vl-8b-instruct": "fixture-ark"},
                       transport=httpx.MockTransport(lambda _: pytest.fail("no transport expected")))
    harness = _ActionHarness(tmp_path, model)
    harness.worker = GPUWorker(harness.store, model, "ark", "remote:ark", semantic_cache_enabled=True)
    harness.settings = harness.settings.model_copy(update={"cv_provider": "fake", "cv_cache_root": tmp_path / "cv-cache", "cv_timeout_seconds": 9.})
    timeline = FrameTimeline(frames=tuple(FrameTimestamp(frame_index=i, timestamp_seconds=i / 2) for i in range(4)))
    monkeypatch.setattr(pipeline_module, "probe_frame_timeline", lambda _: timeline)
    cv_calls = []
    class CvFixture(FakeCvEvidenceProvider):
        def analyze(self, request, staging_dir):
            cv_calls.append(request)
            return super().analyze(request, staging_dir)
    provider = CvFixture()
    with CvArtifactStore(harness.settings.cv_cache_root) as cache:
        cv_worker = CVEvidenceWorker(harness.store, provider, cache, "sam")
        def wait(store, task_id, job_ids, timeout):
            if store.get_inference_job(job_ids[0]).stage == "cv_evidence": cv_worker.run_once()
            else:
                while harness.worker.run_once(): pass
            return wait_for_jobs(store, task_id, job_ids, 0.)
        harness.pipeline = EmbodiedActionPipeline(probe=harness.probe, wait_jobs=wait, wait_timeout=.75)
        first = harness.run()
        calls = len(model_calls)
        harness.video_path = harness.allowed / "same-bytes-different-path.mp4"
        harness.video_path.write_bytes(b"deterministic silent visual fixture")
        second = harness.run()
    assert first.status is second.status is TaskStatus.COMPLETED
    assert len(model_calls) == calls
    assert len(cv_calls) == 1
    assert first.result["cv_evidence"]["cache_hit"] is False
    assert second.result["cv_evidence"]["cache_hit"] is True
    for key in ("artifact_key", "manifest_sha256"):
        assert first.result["cv_evidence"][key] == second.result["cv_evidence"][key]
    assert first.result["performance"]["repair_count"] == second.result["performance"]["repair_count"] == 1
    first_jobs = harness.store.list_inference_jobs(first.task_id)
    second_jobs = harness.store.list_inference_jobs(second.task_id)
    for a, b in zip(first_jobs, second_jobs):
        if a.stage == "cv_evidence":
            assert a.payload["entities"] == b.payload["entities"]
            assert a.payload["video_path"] != b.payload["video_path"]
            continue
        assert a.result == b.result
        assert a.payload["prompt"] == b.payload["prompt"]
        assert b.metrics["semantic_cache_hit"] is True
        assert b.metrics["inference_seconds"] == b.metrics["input_tokens"] == b.metrics["output_tokens"] == 0
    rows = [r for r in second.result["performance"]["stages"] if r.get("model_stage") not in (None, "cv_evidence")]
    assert all(r["provider_metrics"]["semantic_cache_hit"] for r in rows)
    assert all(r["cache_hit"] is False for r in rows)
    validate_hybrid_result(second.result)
    import copy
    invalid = copy.deepcopy(second.result)
    row = next(r for r in invalid["performance"]["stages"] if r.get("model_stage") == "embodied_pass_a")
    row["provider_metrics"]["semantic_cache_key"] = "A" * 64
    with pytest.raises(ValueError):
        validate_hybrid_result(invalid)


def test_video_change_on_release_is_checked_before_publication(store, tmp_path):
    provider = Provider()
    model = provider.model()
    original = model.release_request
    def release(request):
        request.video_path.write_bytes(b"replaced after generate")
        original(request)
    model.release_request = release
    first = job(store, tmp_path)
    run(store, model)
    assert store.get_inference_job(first.job_id).metrics["semantic_cache_published"] is False


def test_semantic_cache_settings_reject_ambiguous_switches():
    from las_repro.config import Settings
    for value in (1, 0, "yes", "no", "1", "0", [], {}):
        with pytest.raises(ValueError): Settings(ark_semantic_cache_enabled=value)


def test_job_completion_failure_rolls_back_cache_publication(store, tmp_path):
    with sqlite3.connect(store.database_path) as db:
        db.execute("CREATE TRIGGER reject_completion BEFORE UPDATE ON inference_jobs WHEN NEW.status = 'COMPLETED' BEGIN SELECT RAISE(ABORT, 'injected completion failure'); END")
    first = job(store, tmp_path)
    run(store, Provider().model())
    assert store.get_inference_job(first.job_id).status is InferenceStatus.FAILED
    with sqlite3.connect(store.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM semantic_results").fetchone()[0] == 0


def test_cache_lookup_io_failure_does_not_fail_good_generation(store, tmp_path, monkeypatch):
    def unavailable(*args):
        raise sqlite3.OperationalError("private cache read error")
    monkeypatch.setattr(store, "_semantic_row", unavailable)
    first = job(store, tmp_path)
    run(store, Provider().model())
    completed = store.get_inference_job(first.job_id)
    assert completed.status is InferenceStatus.COMPLETED
    assert completed.metrics["semantic_cache_published"] is False
    assert completed.metrics["semantic_cache_hit"] is False


def test_normalized_cache_corruption_cannot_be_sanitized_into_failure_hit(store, tmp_path):
    from test_embodied_validators import _fallback_enrichment_result, _fallback_context
    provider = Provider(response=_fallback_enrichment_result())
    model = provider.model()
    overrides = {"schema_name": "EnrichmentResult", "schema_context": _fallback_context(True)}
    first = job(store, tmp_path, stage="embodied_enrichment", overrides=overrides)
    run(store, model)
    with sqlite3.connect(store.database_path) as db:
        value = json.loads(db.execute("SELECT result_json FROM semantic_results").fetchone()[0])
        value["data"]["segments"][0]["confidence"] = 100.
        raw = canonical(value)
        db.execute("UPDATE semantic_results SET result_json=?, result_sha256=?", (raw, hashlib.sha256(raw.encode()).hexdigest()))
    second = job(store, tmp_path, stage="embodied_enrichment", overrides=overrides)
    run(store, model)
    assert provider.calls == 2
    result = store.get_inference_job(second.job_id)
    assert result.metrics["semantic_cache_hit"] is False
    assert result.result == store.get_inference_job(first.job_id).result


def test_schema_context_failure_still_releases_model_request(store, tmp_path):
    provider = Provider()
    model = provider.model()
    released = []
    model.release_request = released.append
    first = job(store, tmp_path, overrides={"schema_context": []})
    run(store, model)
    assert store.get_inference_job(first.job_id).status is InferenceStatus.FAILED
    assert len(released) == 1


def test_per_entry_limit_bounds_combined_canonical_identity_and_result(store, tmp_path):
    from dataclasses import replace
    from las_repro.semantic_cache import make_identity
    from las_repro.workers import _model_request
    j = job(store, tmp_path)
    model = Provider().model()
    identity = make_identity(model, _model_request(j), j.payload["schema_context"])
    data = json.loads(identity.json)
    data["schema_context"] = {"bounded_context": "x" * (5 * 1024 * 1024)}
    encoded = canonical(data)
    identity = replace(identity, json=encoded, key=hashlib.sha256(encoded.encode()).hexdigest())
    claim = store.claim_inference_job("bulk", model_name="doubao-pro", lease_seconds=300.)
    result = {"description": "x" * (4 * 1024 * 1024)}
    completed = store.complete_inference_job(j.job_id, result, worker_id="bulk", attempt=claim.attempt,
        semantic_publication=(identity, lambda value: value == result))
    assert completed.status is InferenceStatus.COMPLETED
    assert completed.metrics["semantic_cache_published"] is False
    with sqlite3.connect(store.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM semantic_results").fetchone()[0] == 0
