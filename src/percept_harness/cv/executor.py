"""Synchronous, in-process execution of CV evidence jobs through the artifact cache."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .artifacts import CvArtifactStore, cv_cache_key
from .base import CvOutOfMemoryError
from .contracts import CvEvidenceArtifact, CvEvidenceRequest

if TYPE_CHECKING:
    from .sam31 import Sam31EvidenceProvider


class SyncCvExecutor:
    """Run one CV evidence request in-process against the artifact cache."""

    def __init__(
        self,
        provider: Sam31EvidenceProvider,
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
            return _published_result(cached.key, artifact, True), metrics

        started = time.monotonic()
        for attempt_index in range(2):
            try:
                with self.artifact_store.staging(key) as staging:
                    artifact = self.provider.analyze(request, staging)
                    inference_seconds = time.monotonic() - started
                    provider_metrics = self.provider.request_metrics()
                    metrics = _cv_metrics(
                        artifact,
                        cache_hit=False,
                        oom_retry=bool(attempt_index),
                        processed_frames=provider_metrics["processed_frames"],
                        execution_chunk_frames=provider_metrics["execution_chunk_frames"],
                        peak_allocated_bytes=provider_metrics["peak_allocated_bytes"],
                        inference_seconds=inference_seconds,
                    )
                    handle = self.artifact_store.publish(request, staging, artifact)
                return _published_result(handle.key, artifact, False), metrics
            except CvOutOfMemoryError:
                if attempt_index == 1:
                    raise
                self.provider.execution_chunk_frames = max(
                    1, self.provider.execution_chunk_frames // 2
                )


def _published_result(
    key: str,
    artifact: CvEvidenceArtifact,
    cache_hit: bool,
) -> dict[str, Any]:
    return {
        "artifact_key": key,
        "cache_hit": cache_hit,
        "status": artifact.status.value,
    }


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
