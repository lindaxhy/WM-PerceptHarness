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
    OverlayRecord,
    SamplingPolicy,
    TrackObservation,
)
from .entities import EntityCandidate, NormalizedEntities, normalize_entities

__all__ = [
    "ArtifactFile",
    "CvEvidenceArtifact",
    "CvEvidenceProvider",
    "CvEvidenceRequest",
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
    "OverlayRecord",
    "SamplingPolicy",
    "Sam31EvidenceProvider",
    "SyncCvExecutor",
    "TrackObservation",
    "normalize_entities",
]


def __getattr__(name: str) -> Any:
    """Load heavy exports lazily so schema-only imports stay cycle-free."""
    if name == "Sam31EvidenceProvider":
        from .sam31 import Sam31EvidenceProvider

        return Sam31EvidenceProvider
    if name == "SyncCvExecutor":
        from .executor import SyncCvExecutor

        return SyncCvExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
