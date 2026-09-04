"""Leased execution and content-addressed publication for CV evidence jobs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from numbers import Real
import time
from typing import Any

from ..domain import InferenceJob
from ..store import (
    InvalidTransition,
    SQLiteTaskStore,
    WorkerMismatch,
    validate_inference_job_metrics,
)
from ..workers import _LeaseKeeper
from .artifacts import CvArtifactHandle, CvArtifactStore, cv_cache_key
from .base import CvEvidenceProvider, CvOutOfMemoryError
from .contracts import CvEvidenceArtifact, CvEvidenceRequest


_CV_MODEL_NAME = "sam3.1"
_FAILURE_MESSAGE = "CV evidence inference failed"


def cv_request_from_job(job: InferenceJob) -> CvEvidenceRequest:
    """Parse one exact, correctly routed CV request payload."""
    if job.stage != "cv_evidence" or job.model_name != _CV_MODEL_NAME:
        raise ValueError("CV job routing is invalid")
    return CvEvidenceRequest.model_validate(job.payload, strict=True)


class CVEvidenceWorker:
    """Claim and execute at most one isolated CV evidence job."""

    def __init__(
        self,
        store: SQLiteTaskStore,
        provider: CvEvidenceProvider,
        artifact_store: CvArtifactStore,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        heartbeat_interval: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.provider = provider
        self.artifact_store = artifact_store
        self.worker_id = _nonblank(worker_id, "worker_id")
        self.lease_seconds = _positive_finite(lease_seconds, "lease_seconds")
        default_interval = self.lease_seconds / 3.0
        self.heartbeat_interval = _positive_finite(
            default_interval if heartbeat_interval is None else heartbeat_interval,
            "heartbeat_interval",
        )
        if self.heartbeat_interval >= self.lease_seconds:
            raise ValueError("heartbeat_interval must be shorter than lease_seconds")
        self._monotonic = monotonic

    def run_once(self, *, now: float | None = None) -> bool:
        """Process at most one eligible CV job, returning whether one was claimed."""
        job: InferenceJob | None = None
        claim_returned = False

        def register_claim(claimed: InferenceJob) -> None:
            nonlocal job
            job = claimed

        try:
            claimed = self.store.claim_inference_job(
                self.worker_id,
                model_name=_CV_MODEL_NAME,
                lease_seconds=self.lease_seconds,
                now=now,
                on_claim=register_claim,
            )
            claim_returned = True
            if claimed is None:
                return False
            job = claimed
            with _LeaseKeeper(
                lambda heartbeat_now: self.store.heartbeat_inference_job(
                    job.job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                    attempt=job.attempt,
                    now=heartbeat_now,
                ),
                interval=self.heartbeat_interval,
                fixed_now=now,
            ):
                request = cv_request_from_job(job)
                handle, artifact, metrics = self._resolve_artifact(
                    request, job, now=now
                )
            self.store.complete_inference_job(
                job.job_id,
                {
                    "artifact_key": handle.key,
                    "manifest_sha256": handle.manifest_sha256,
                    "cache_hit": metrics["cache_hit"],
                    "status": artifact.status.value,
                },
                worker_id=self.worker_id,
                attempt=job.attempt,
                now=now,
                metrics=metrics,
            )
        except (InvalidTransition, WorkerMismatch):
            if not claim_returned:
                raise
            # A recovered owner won the lease race. The stale result is dropped.
            pass
        except Exception:
            if not claim_returned:
                if job is not None:
                    self._expire_after_interrupted_claim(job, now=now)
                raise
            try:
                self.store.fail_inference_job(
                    job.job_id,
                    _FAILURE_MESSAGE,
                    worker_id=self.worker_id,
                    attempt=job.attempt,
                    now=now,
                )
            except (InvalidTransition, WorkerMismatch):
                # Recovery may have completed while the failed call was unwinding.
                pass
        except BaseException:
            if job is not None:
                self._expire_after_interrupted_claim(job, now=now)
            raise
        return True

    def _resolve_artifact(
        self,
        request: CvEvidenceRequest,
        job: InferenceJob,
        *,
        now: float | None,
    ) -> tuple[CvArtifactHandle, CvEvidenceArtifact, dict[str, Any]]:
        key = cv_cache_key(request)
        cached = self.artifact_store.lookup(key)
        if cached is not None:
            artifact = self.artifact_store.load(cached)
            metrics = _cv_metrics(
                artifact,
                cache_hit=True,
                oom_retry=False,
                processed_frames=0,
                execution_chunk_frames=0,
                peak_allocated_bytes=0,
                inference_seconds=0.0,
            )
            validate_inference_job_metrics(metrics)
            return (
                cached,
                artifact,
                metrics,
            )

        started = _finite_clock(self._monotonic())
        oom_retry = False
        for attempt_index in range(2):
            try:
                with self.artifact_store.staging(key) as staging:
                    artifact = self.provider.analyze(request, staging)
                    inference_seconds = _finite_clock(self._monotonic()) - started
                    provider_metrics = _provider_execution_metrics(
                        self.provider
                    )
                    metrics = _cv_metrics(
                        artifact,
                        cache_hit=False,
                        oom_retry=oom_retry,
                        processed_frames=provider_metrics["processed_frames"],
                        execution_chunk_frames=provider_metrics[
                            "execution_chunk_frames"
                        ],
                        peak_allocated_bytes=provider_metrics[
                            "peak_allocated_bytes"
                        ],
                        inference_seconds=inference_seconds,
                    )
                    validate_inference_job_metrics(metrics)
                    self.store.heartbeat_inference_job(
                        job.job_id,
                        self.worker_id,
                        lease_seconds=self.lease_seconds,
                        attempt=job.attempt,
                        now=now,
                    )
                    handle = self.artifact_store.publish(
                        request,
                        staging,
                        artifact,
                        commit_guard=lambda: self.store.inference_job_lease_guard(
                            job.job_id,
                            self.worker_id,
                            attempt=job.attempt,
                        ),
                    )
                break
            except CvOutOfMemoryError:
                if attempt_index != 0:
                    raise
                oom_retry = True
                _halve_execution_chunk(self.provider)
        else:  # pragma: no cover - range and break make this unreachable.
            raise RuntimeError
        return (
            handle,
            artifact,
            metrics,
        )

    def _expire_after_interrupted_claim(
        self, job: InferenceJob, *, now: float | None
    ) -> None:
        try:
            self.store.expire_inference_job_lease(
                job.job_id,
                self.worker_id,
                attempt=job.attempt,
                now=now,
            )
        except (InvalidTransition, WorkerMismatch):
            pass


def _halve_execution_chunk(provider: CvEvidenceProvider) -> None:
    current = getattr(provider, "execution_chunk_frames", None)
    if isinstance(current, bool) or not isinstance(current, int) or current <= 0:
        raise ValueError("provider execution chunk must be a positive integer")
    smaller = max(1, current // 2)
    setter = getattr(provider, "set_execution_chunk_frames", None)
    if callable(setter):
        setter(smaller)
    else:
        try:
            setattr(provider, "execution_chunk_frames", smaller)
        except (AttributeError, TypeError):
            raise ValueError("provider execution chunk cannot be changed") from None


def _provider_execution_metrics(provider: CvEvidenceProvider) -> dict[str, int]:
    request_metrics = getattr(provider, "request_metrics", None)
    reported = request_metrics() if callable(request_metrics) else None
    if reported is None:
        raise ValueError("provider request metrics are required")
    if not isinstance(reported, Mapping):
        raise TypeError("provider request_metrics must return a mapping or None")
    values: dict[str, int] = {}
    for name in (
        "processed_frames",
        "execution_chunk_frames",
        "peak_allocated_bytes",
    ):
        value = reported.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("provider execution metric must be a non-negative integer")
        values[name] = value
    if values["execution_chunk_frames"] <= 0:
        raise ValueError("provider execution chunk must be a positive integer")
    return values


def _cv_metrics(
    artifact: CvEvidenceArtifact,
    *,
    cache_hit: bool,
    oom_retry: bool,
    processed_frames: int,
    execution_chunk_frames: int,
    peak_allocated_bytes: int,
    inference_seconds: float,
) -> dict[str, Any]:
    return {
        "cache_hit": cache_hit,
        "oom_retry": oom_retry,
        "processed_frames": processed_frames,
        "execution_chunk_frames": execution_chunk_frames,
        "entity_prompts": len(artifact.entities),
        "track_count": len(artifact.tracks),
        "peak_allocated_bytes": peak_allocated_bytes,
        "inference_seconds": inference_seconds,
    }


def _nonblank(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _positive_finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _finite_clock(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError("monotonic clock must return a finite real number")
    return float(value)
