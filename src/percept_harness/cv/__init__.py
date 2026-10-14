"""Provider-independent contracts and lazy adapters for CV evidence."""

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
    EvidenceStatus,
    EvidenceThresholds,
    FrameTimeline,
    FrameTimestamp,
    OverlayRecord,
    SamplingPolicy,
    TrackObservation,
)

__all__ = [
    "ArtifactFile",
    "CvEvidenceArtifact",
    "CvEvidenceProvider",
    "CvEvidenceRequest",
    "CvOutOfMemoryError",
    "CvProviderError",
    "CvTrack",
    "EntityPrompt",
    "EvidenceStatus",
    "EvidenceThresholds",
    "FrameTimeline",
    "FrameTimestamp",
    "FakeCvEvidenceProvider",
    "OverlayRecord",
    "SamplingPolicy",
    "Sam31EvidenceProvider",
    "SyncCvExecutor",
    "TrackObservation",
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
