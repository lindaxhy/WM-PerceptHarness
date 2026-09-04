from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
from typing import Any

import pytest

from las_repro.cv.artifacts import CvArtifactHandle, CvArtifactStore, cv_cache_key
from las_repro.cv.base import CvOutOfMemoryError, FakeCvEvidenceProvider
from las_repro.cv.contracts import (
    CvEvidenceArtifact,
    CvEvidenceRequest,
    EntityPrompt,
    EntityRole,
    EvidenceThresholds,
    FrameTimeline,
    FrameTimestamp,
    SamplingPolicy,
)
from las_repro.cv.worker import CVEvidenceWorker, cv_request_from_job
from las_repro.domain import InferenceJob, InferenceJobSpec, InferenceStatus
from las_repro.models.fake import FakeVideoModel
from las_repro.store import SQLiteTaskStore
from las_repro.workers import GPUWorker


@pytest.fixture
def store(tmp_path: Path) -> SQLiteTaskStore:
    value = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    value.initialize()
    return value


@pytest.fixture
def cache(tmp_path: Path) -> CvArtifactStore:
    value = CvArtifactStore(tmp_path / "cv-cache")
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def cv_request(tmp_path: Path) -> CvEvidenceRequest:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"deterministic video fixture")
    return CvEvidenceRequest(
        schema_version="cv_request_v1",
        provider="fake",
        model_identity="fake-sam31-v1",
        video_path=video_path,
        video_sha256="a" * 64,
        duration_seconds=2.0,
        frame_count=20,
        checkpoint_sha256="b" * 64,
        timeline=FrameTimeline(
            frames=(
                FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
                FrameTimestamp(frame_index=5, timestamp_seconds=0.5),
                FrameTimestamp(frame_index=10, timestamp_seconds=1.0),
            )
        ),
        entities=(
            EntityPrompt(
                entity_id="right_hand",
                canonical_label="right hand",
                aliases=("hand",),
                role=EntityRole.ACTOR,
            ),
            EntityPrompt(
                entity_id="cup",
                canonical_label="cup",
                aliases=("mug",),
                role=EntityRole.MANIPULATED_OBJECT,
            ),
        ),
        sampling=SamplingPolicy(
            short_video_seconds=30.0,
            scan_fps=8.0,
            max_fps=30.0,
            refinement_radius_seconds=1.0,
        ),
        thresholds=EvidenceThresholds(
            min_confidence=0.5,
            min_area_fraction=0.01,
            occlusion_visibility_drop=0.5,
        ),
    )


def create_job(
    store: SQLiteTaskStore,
    request: CvEvidenceRequest,
    *,
    stage: str = "cv_evidence",
    model_name: str = "sam3.1",
    payload: dict[str, Any] | None = None,
    now: float = 10.0,
) -> InferenceJob:
    task = store.create_task(
        {
            "video_url": str(request.video_path),
            "task_template": "embodied_video_captioning",
            "model_name": "qwen3-vl-8b-instruct",
        }
    )
    [job] = store.create_inference_jobs(
        task.task_id,
        [
            InferenceJobSpec(
                stage=stage,
                ordinal=0,
                payload=(
                    request.model_dump(mode="json") if payload is None else payload
                ),
                model_name=model_name,
            )
        ],
        now=now,
    )
    return job


def assert_cached_handle(
    cache: CvArtifactStore,
    request: CvEvidenceRequest,
    result: dict[str, Any],
) -> CvArtifactHandle:
    handle = CvArtifactHandle(
        key=result["artifact_key"],
        manifest_sha256=result["manifest_sha256"],
    )
    assert handle == cache.lookup(cv_cache_key(request))
    assert cache.load(handle).status.value == result["status"]
    return handle


def test_cv_worker_claims_only_sam_jobs_and_returns_manifest_handle(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """A successful CV job must publish only a small validated artifact handle."""
    cv_job = create_job(store, cv_request)
    worker = CVEvidenceWorker(
        store,
        FakeCvEvidenceProvider(),
        cache,
        "fake-sam",
        lease_seconds=30,
        monotonic=lambda: 4.0,
    )

    assert worker.run_once(now=20.0)

    done = store.get_inference_job(cv_job.job_id)
    assert done is not None
    assert done.status is InferenceStatus.COMPLETED
    assert done.completed_by == "fake-sam"
    assert done.result is not None
    assert set(done.result) == {
        "artifact_key",
        "manifest_sha256",
        "cache_hit",
        "status",
    }
    assert done.result["cache_hit"] is False
    assert done.metrics == {
        "cache_hit": False,
        "oom_retry": False,
        "processed_frames": 3,
        "execution_chunk_frames": 1,
        "entity_prompts": 2,
        "track_count": 2,
        "peak_allocated_bytes": 0,
        "inference_seconds": 0.0,
    }
    assert_cached_handle(cache, cv_request, done.result)


def test_fake_provider_publishes_valid_empty_entity_evidence(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """A valid empty inventory must not leave undeclared staging content."""
    empty_request = cv_request.model_copy(update={"entities": ()})
    job = create_job(store, empty_request)

    assert CVEvidenceWorker(
        store, FakeCvEvidenceProvider(), cache, "fake-sam"
    ).run_once(now=20.0)

    done = store.get_inference_job(job.job_id)
    assert done is not None and done.result is not None
    assert done.status is InferenceStatus.COMPLETED
    assert done.metrics is not None
    assert done.metrics["entity_prompts"] == 0
    assert done.metrics["track_count"] == 0
    handle = assert_cached_handle(cache, empty_request, done.result)
    artifact = cache.load(handle)
    assert artifact.entities == ()
    assert artifact.tracks == ()
    assert artifact.files == ()


def test_cv_worker_persists_only_provider_reported_sampling_and_chunk_metrics(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    class SampledMetricsProvider(FakeCvEvidenceProvider):
        def request_metrics(self) -> dict[str, int]:
            metrics = super().request_metrics()
            metrics.update(
                {
                    "processed_frames": 1,
                    "execution_chunk_frames": 7,
                    "private_provider_counter": 999,
                }
            )
            return metrics

    job = create_job(store, cv_request)
    worker = CVEvidenceWorker(
        store, SampledMetricsProvider(), cache, "sampled-metrics"
    )

    assert worker.run_once(now=20.0)

    done = store.get_inference_job(job.job_id)
    assert done is not None and done.metrics is not None
    assert done.metrics["processed_frames"] == 1
    assert done.metrics["execution_chunk_frames"] == 7
    assert "private_provider_counter" not in done.metrics


def test_second_identical_job_is_cache_hit_without_provider_invocation(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """Looking up after inference would needlessly execute the provider twice."""

    class CountingProvider(FakeCvEvidenceProvider):
        def __init__(self) -> None:
            super().__init__()
            self.invocations = 0

        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            self.invocations += 1
            return super().analyze(request, staging_dir)

    first_job = create_job(store, cv_request, now=10.0)
    second_job = create_job(store, cv_request, now=11.0)
    provider = CountingProvider()
    worker = CVEvidenceWorker(store, provider, cache, "fake-sam")

    assert worker.run_once(now=20.0)
    assert worker.run_once(now=21.0)

    first = store.get_inference_job(first_job.job_id)
    second = store.get_inference_job(second_job.job_id)
    assert first is not None and first.result is not None
    assert second is not None and second.result is not None
    assert provider.invocations == 1
    assert first.result["cache_hit"] is False
    assert second.result["cache_hit"] is True
    assert first.result["artifact_key"] == second.result["artifact_key"]
    assert first.result["manifest_sha256"] == second.result["manifest_sha256"]
    assert second.metrics == {
        "cache_hit": True,
        "oom_retry": False,
        "processed_frames": 0,
        "execution_chunk_frames": 0,
        "entity_prompts": 2,
        "track_count": 2,
        "peak_allocated_bytes": 0,
        "inference_seconds": 0.0,
    }


def test_qwen_and_cv_workers_cannot_cross_claim_model_queues(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """Dropping either model filter could execute a job in the wrong runtime."""
    cv_job = create_job(store, cv_request)
    qwen_worker = GPUWorker(store, FakeVideoModel(), "qwen", "cuda:0")
    assert qwen_worker.run_once(now=20.0) is False

    cv_worker = CVEvidenceWorker(store, FakeCvEvidenceProvider(), cache, "sam")
    assert cv_worker.run_once(now=20.0) is True
    completed_cv = store.get_inference_job(cv_job.job_id)
    assert completed_cv is not None
    assert completed_cv.status is InferenceStatus.COMPLETED

    qwen_job = create_job(
        store,
        cv_request,
        stage="general_segment",
        model_name="qwen3-vl-8b-instruct",
        payload={
            "video_path": str(cv_request.video_path),
            "start": 0.0,
            "end": 1.0,
            "fps": 2.0,
            "prompt": "describe",
            "schema_name": "general_segment",
        },
    )
    assert cv_worker.run_once(now=21.0) is False
    pending_qwen = store.get_inference_job(qwen_job.job_id)
    assert pending_qwen is not None
    assert pending_qwen.status is InferenceStatus.PENDING


def test_cv_request_parser_requires_exact_route_and_strict_payload(
    cv_request: CvEvidenceRequest,
) -> None:
    """Permissive parsing could let a wrong stage or boolean count reach a provider."""
    base = InferenceJob(
        job_id="job",
        task_id="task",
        stage="cv_evidence",
        ordinal=0,
        model_name="sam3.1",
        payload=cv_request.model_dump(mode="json"),
    )
    assert cv_request_from_job(base) == cv_request
    with pytest.raises(ValueError, match="CV job routing is invalid"):
        cv_request_from_job(
            InferenceJob(**{**base.__dict__, "stage": "general_segment"})
        )
    with pytest.raises(ValueError, match="CV job routing is invalid"):
        cv_request_from_job(InferenceJob(**{**base.__dict__, "model_name": "qwen"}))
    malformed = base.payload | {"frame_count": True}
    with pytest.raises(ValueError):
        cv_request_from_job(InferenceJob(**{**base.__dict__, "payload": malformed}))


def test_cv_worker_retries_one_oom_with_smaller_chunk_and_same_request(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """An OOM retry may change batching, never evidence identity or sampling."""

    class OomOnceProvider(FakeCvEvidenceProvider):
        def __init__(self) -> None:
            super().__init__(execution_chunk_frames=8)
            self.seen: list[tuple[int, CvEvidenceRequest, Path]] = []

        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            self.seen.append((self.execution_chunk_frames, request, staging_dir))
            if len(self.seen) == 1:
                (staging_dir / "partial.secret").write_text("discard me")
                raise CvOutOfMemoryError("raw CUDA allocator detail")
            return super().analyze(request, staging_dir)

    job = create_job(store, cv_request)
    provider = OomOnceProvider()
    times = iter((10.0, 11.5))
    worker = CVEvidenceWorker(
        store,
        provider,
        cache,
        "sam",
        monotonic=lambda: next(times),
    )

    assert worker.run_once(now=20.0)

    done = store.get_inference_job(job.job_id)
    assert done is not None and done.result is not None
    assert [chunk for chunk, _, _ in provider.seen] == [8, 4]
    assert provider.seen[0][1] is provider.seen[1][1]
    assert provider.seen[0][1].model_dump(
        mode="json"
    ) == cv_request.model_dump(mode="json")
    assert provider.seen[0][2] != provider.seen[1][2]
    assert all(not staging.exists() for _, _, staging in provider.seen)
    assert done.metrics == {
        "cache_hit": False,
        "oom_retry": True,
        "processed_frames": 3,
        "execution_chunk_frames": 4,
        "entity_prompts": 2,
        "track_count": 2,
        "peak_allocated_bytes": 0,
        "inference_seconds": 1.5,
    }
    assert_cached_handle(cache, cv_request, done.result)


def test_second_oom_fails_safely_and_never_populates_cache(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """Retrying more than once or publishing a partial stage breaks bounded recovery."""

    class AlwaysOomProvider(FakeCvEvidenceProvider):
        def __init__(self) -> None:
            super().__init__(execution_chunk_frames=3)
            self.chunks: list[int] = []
            self.staging: list[Path] = []

        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            self.chunks.append(self.execution_chunk_frames)
            self.staging.append(staging_dir)
            (staging_dir / "private-path.txt").write_text(str(request.video_path))
            raise CvOutOfMemoryError(f"OOM while reading {request.video_path}")

    job = create_job(store, cv_request)
    provider = AlwaysOomProvider()

    assert CVEvidenceWorker(store, provider, cache, "sam").run_once(now=20.0)

    failed = store.get_inference_job(job.job_id)
    assert failed is not None
    assert failed.status is InferenceStatus.FAILED
    assert failed.error == "CV evidence inference failed"
    assert failed.result is None
    assert failed.metrics is None
    assert provider.chunks == [3, 1]
    assert all(not staging.exists() for staging in provider.staging)
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_stale_cv_result_is_neither_published_nor_completed(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """A recovered lease owner must fence the old provider before cache publication."""
    job = create_job(store, cv_request)

    class RecoverDuringAnalysis(FakeCvEvidenceProvider):
        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            artifact = super().analyze(request, staging_dir)
            recovered = store.claim_inference_job(
                "sam-new", model_name="sam3.1", lease_seconds=10.0, now=101.1
            )
            assert recovered is not None
            store.complete_inference_job(
                recovered.job_id,
                {"winner": "new-owner"},
                worker_id="sam-new",
                attempt=recovered.attempt,
                now=101.2,
            )
            return artifact

    stale = CVEvidenceWorker(
        store,
        RecoverDuringAnalysis(),
        cache,
        "sam-old",
        lease_seconds=1.0,
    )

    assert stale.run_once(now=100.0)

    completed = store.get_inference_job(job.job_id)
    assert completed is not None
    assert completed.result == {"winner": "new-owner"}
    assert completed.completed_by == "sam-new"
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_lease_reclaimed_at_artifact_commit_cannot_install_stale_cache(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lease ownership must be held across the atomic cache installation."""
    job = create_job(store, cv_request)
    original_validate_entry = cache._validate_entry
    reclaimed_after_preparation = False

    def reclaim_after_preparation(entry: Path, expected_key: str):
        nonlocal reclaimed_after_preparation
        validated = original_validate_entry(entry, expected_key)
        if entry.name.startswith(".publish-") and not reclaimed_after_preparation:
            reclaimed_after_preparation = True
            recovered = store.claim_inference_job(
                "sam-new",
                model_name="sam3.1",
                lease_seconds=10.0,
                now=101.1,
            )
            assert recovered is not None
            store.complete_inference_job(
                recovered.job_id,
                {"winner": "new-owner"},
                worker_id="sam-new",
                attempt=recovered.attempt,
                now=101.2,
            )
        return validated

    monkeypatch.setattr(cache, "_validate_entry", reclaim_after_preparation)
    stale = CVEvidenceWorker(
        store,
        FakeCvEvidenceProvider(),
        cache,
        "sam-old",
        lease_seconds=1.0,
    )

    assert stale.run_once(now=100.0)

    completed = store.get_inference_job(job.job_id)
    assert reclaimed_after_preparation is True
    assert completed is not None
    assert completed.result == {"winner": "new-owner"}
    assert completed.completed_by == "sam-new"
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_cv_worker_interrupt_expires_lease_and_removes_staging(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """A BaseException must leave the claimed generation immediately recoverable."""
    job = create_job(store, cv_request)
    staging_paths: list[Path] = []

    class InterruptedProvider(FakeCvEvidenceProvider):
        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            staging_paths.append(staging_dir)
            (staging_dir / "partial").write_text("private")
            raise KeyboardInterrupt

    worker = CVEvidenceWorker(store, InterruptedProvider(), cache, "sam-old")

    with pytest.raises(KeyboardInterrupt):
        worker.run_once(now=100.0)

    interrupted = store.get_inference_job(job.job_id)
    assert interrupted is not None
    assert interrupted.status is InferenceStatus.RUNNING
    assert interrupted.worker_id is None
    assert interrupted.lease_until == 100.0
    assert all(not path.exists() for path in staging_paths)
    recovered = store.claim_inference_job(
        "sam-new", model_name="sam3.1", lease_seconds=10.0, now=100.0
    )
    assert recovered is not None
    assert recovered.attempt == 2


@pytest.mark.parametrize("failure_kind", ["provider", "malformed_artifact"])
def test_cv_worker_sanitizes_provider_and_publication_failures(
    failure_kind: str,
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """Provider text, prompts, and host paths must not cross the durable boundary."""
    job = create_job(store, cv_request)
    staging_paths: list[Path] = []

    class UnsafeProvider(FakeCvEvidenceProvider):
        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            staging_paths.append(staging_dir)
            if failure_kind == "provider":
                raise RuntimeError(
                    f"token=secret prompt={request.entities[0].canonical_label} "
                    f"path={request.video_path}"
                )
            artifact = super().analyze(request, staging_dir)
            return artifact.model_copy(update={"model_identity": "wrong-model"})

    assert CVEvidenceWorker(store, UnsafeProvider(), cache, "sam").run_once(now=20.0)

    failed = store.get_inference_job(job.job_id)
    assert failed is not None
    assert failed.status is InferenceStatus.FAILED
    assert failed.error == "CV evidence inference failed"
    assert "secret" not in failed.error
    assert "right hand" not in failed.error
    assert str(cv_request.video_path) not in failed.error
    assert all(not path.exists() for path in staging_paths)
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_provider_metrics_failure_cannot_publish_a_future_cache_hit(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """All provider output must be accepted before durable cache publication."""
    job = create_job(store, cv_request)
    staging_paths: list[Path] = []

    class UnsafeMetricsProvider(FakeCvEvidenceProvider):
        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            staging_paths.append(staging_dir)
            return super().analyze(request, staging_dir)

        def request_metrics(self) -> dict[str, int]:
            raise RuntimeError(
                f"allocator secret for {cv_request.entities[0].canonical_label} "
                f"at {cv_request.video_path}"
            )

    assert CVEvidenceWorker(
        store, UnsafeMetricsProvider(), cache, "sam"
    ).run_once(now=20.0)

    failed = store.get_inference_job(job.job_id)
    assert failed is not None
    assert failed.status is InferenceStatus.FAILED
    assert failed.error == "CV evidence inference failed"
    assert all(not path.exists() for path in staging_paths)
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_decreasing_inference_clock_cannot_publish_cache_entry(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """A negative elapsed metric must fail before durable artifact installation."""
    job = create_job(store, cv_request)
    times = iter((10.0, 9.0))
    worker = CVEvidenceWorker(
        store,
        FakeCvEvidenceProvider(),
        cache,
        "sam",
        monotonic=lambda: next(times),
    )

    assert worker.run_once(now=20.0)

    failed = store.get_inference_job(job.job_id)
    assert failed is not None
    assert failed.status is InferenceStatus.FAILED
    assert failed.error == "CV evidence inference failed"
    assert failed.result is None
    assert failed.metrics is None
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_overflowing_inference_elapsed_cannot_publish_cache_entry(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """Finite readings whose subtraction overflows must fail before publication."""
    job = create_job(store, cv_request)
    limit = sys.float_info.max
    times = iter((-limit, limit))
    worker = CVEvidenceWorker(
        store,
        FakeCvEvidenceProvider(),
        cache,
        "sam",
        monotonic=lambda: next(times),
    )

    assert worker.run_once(now=20.0)

    failed = store.get_inference_job(job.job_id)
    assert failed is not None
    assert failed.status is InferenceStatus.FAILED
    assert failed.error == "CV evidence inference failed"
    assert failed.result is None
    assert failed.metrics is None
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_cv_worker_rejects_malformed_payload_before_provider_and_keeps_cache_open(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """Strict request failure must skip provider work without taking cache ownership."""

    class NeverCalledProvider(FakeCvEvidenceProvider):
        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            raise AssertionError("provider must not receive malformed payload")

    malformed = cv_request.model_dump(mode="json")
    malformed["frame_count"] = True
    job = create_job(store, cv_request, payload=malformed)

    worker = CVEvidenceWorker(store, NeverCalledProvider(), cache, "sam")
    assert worker.run_once(now=20.0)

    failed = store.get_inference_job(job.job_id)
    assert failed is not None
    assert failed.error == "CV evidence inference failed"
    assert cache.lookup(cv_cache_key(cv_request)) is None


def test_cv_worker_renews_lease_while_provider_is_blocked(
    store: SQLiteTaskStore,
    cv_request: CvEvidenceRequest,
    cache: CvArtifactStore,
) -> None:
    """A long CV call must retain ownership through the existing heartbeat pattern."""
    job = create_job(store, cv_request)
    started = threading.Event()
    finish = threading.Event()

    class BlockingProvider(FakeCvEvidenceProvider):
        def analyze(
            self, request: CvEvidenceRequest, staging_dir: Path
        ) -> CvEvidenceArtifact:
            started.set()
            assert finish.wait(timeout=2.0)
            return super().analyze(request, staging_dir)

    worker = CVEvidenceWorker(
        store,
        BlockingProvider(),
        cache,
        "sam-old",
        lease_seconds=0.08,
        heartbeat_interval=0.01,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.run_once)
        assert started.wait(timeout=1.0)
        time.sleep(0.16)
        assert (
            store.claim_inference_job(
                "sam-new", model_name="sam3.1", lease_seconds=1.0
            )
            is None
        )
        finish.set()
        assert future.result(timeout=2.0) is True

    completed = store.get_inference_job(job.job_id)
    assert completed is not None
    assert completed.status is InferenceStatus.COMPLETED
