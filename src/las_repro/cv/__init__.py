"""Provider-independent contracts and normalization for CV evidence."""

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
    "CvEvidenceRequest",
    "CvTrack",
    "EntityCandidate",
    "EntityPrompt",
    "EntityRole",
    "EvidenceStatus",
    "EvidenceThresholds",
    "FrameTimeline",
    "FrameTimestamp",
    "NormalizedEntities",
    "SamplingPolicy",
    "TrackObservation",
    "normalize_entities",
]
