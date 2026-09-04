"""Provider-independent contracts and normalization for CV evidence."""

from typing import Any

from .base import (
    CvEvidenceProvider,
    CvOutOfMemoryError,
    CvProviderError,
    FakeCvEvidenceProvider,
)
from .contracts import (
    ArtifactFile,
    CvEvidenceArtifact,
    CvEvidenceRequest,
    CvTrack,
    EntityPrompt,
    EntityRole,
    EvidenceStatus,
    EvidenceThresholds,
    FrameTimeline,
    FrameTimestamp,
    SamplingPolicy,
    TrackObservation,
)
from .entities import EntityCandidate, NormalizedEntities, normalize_entities

__all__ = [
    "ArtifactFile",
    "CvEvidenceArtifact",
    "CvEvidenceProvider",
    "CvEvidenceRequest",
    "CVEvidenceWorker",
    "CvOutOfMemoryError",
    "CvProviderError",
    "CvTrack",
    "EntityCandidate",
    "EntityPrompt",
    "EntityRole",
    "EvidenceStatus",
    "EvidenceThresholds",
    "FrameTimeline",
    "FrameTimestamp",
    "FakeCvEvidenceProvider",
    "NormalizedEntities",
    "SamplingPolicy",
    "TrackObservation",
    "cv_request_from_job",
    "normalize_entities",
]


def __getattr__(name: str) -> Any:
    """Load worker exports lazily so schema-only imports stay cycle-free."""
    if name in {"CVEvidenceWorker", "cv_request_from_job"}:
        from .worker import CVEvidenceWorker, cv_request_from_job

        return {
            "CVEvidenceWorker": CVEvidenceWorker,
            "cv_request_from_job": cv_request_from_job,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
