"""Synchronous, store-free execution for CV evidence jobs."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .artifacts import CvArtifactStore, cv_cache_key
from .base import CvEvidenceProvider, CvOutOfMemoryError
from .contracts import CvEvidenceArtifact, CvEvidenceRequest


class SyncCvExecutor:
    """Run one CV evidence request in-process against the artifact cache."""

    def __init__(
        self,
        provider: CvEvidenceProvider,
        artifact_store: CvArtifactStore,
    ) -> None:
        self.provider = provider
        self.artifact_store = artifact_store

    def __call__(
        self, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request = CvEvidenceRequest.model_validate(payload, strict=True)
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
            return _published_result(cached.key, cached.manifest_sha256, artifact, True), metrics

        started = time.monotonic()
        oom_retry = False
        for attempt_index in range(2):
            try:
                with self.artifact_store.staging(key) as staging:
                    artifact = self.provider.analyze(request, staging)
                    inference_seconds = time.monotonic() - started
                    provider_metrics = _provider_execution_metrics(self.provider)
                    metrics = _cv_metrics(
                        artifact,
                        cache_hit=False,
                        oom_retry=oom_retry,
                        processed_frames=provider_metrics["processed_frames"],
                        execution_chunk_frames=provider_metrics["execution_chunk_frames"],
                        peak_allocated_bytes=provider_metrics["peak_allocated_bytes"],
                        inference_seconds=inference_seconds,
                    )
                    handle = self.artifact_store.publish(request, staging, artifact)
                return _published_result(handle.key, handle.manifest_sha256, artifact, False), metrics
            except CvOutOfMemoryError:
                if attempt_index == 1:
                    raise
                oom_retry = True
                _halve_execution_chunk(self.provider)
        raise RuntimeError("CV evidence execution did not produce an artifact")


def _published_result(
    key: str,
    manifest_sha256: str,
    artifact: CvEvidenceArtifact,
    cache_hit: bool,
) -> dict[str, Any]:
    return {
        "artifact_key": key,
        "manifest_sha256": manifest_sha256,
        "cache_hit": cache_hit,
        "status": artifact.status.value,
    }


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
    for name in ("processed_frames", "execution_chunk_frames", "peak_allocated_bytes"):
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
