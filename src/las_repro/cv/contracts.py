"""Frozen, provider-independent contracts for computer-vision evidence."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator


Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
ObjectId = Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")]
TrackId = Annotated[StrictStr, Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")]
Timestamp = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
PositiveTimestamp = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]


class StrictModel(BaseModel):
    """A JSON-compatible process boundary that rejects drift and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def freeze_json_sequences(cls, value: Any) -> Any:
        """Accept JSON arrays while retaining tuple-only values after validation."""
        if isinstance(value, list):
            return tuple(cls.freeze_json_sequences(item) for item in value)
        if isinstance(value, dict):
            return {
                key: cls.freeze_json_sequences(item) for key, item in value.items()
            }
        return value


class EntityRole(StrEnum):
    ACTOR = "actor"
    MANIPULATED_OBJECT = "manipulated_object"
    CONTAINER = "container"
    OCCLUDER = "occluder"
    SURFACE = "surface"
    OTHER = "other"


class EvidenceStatus(StrEnum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


class EntityPrompt(StrictModel):
    entity_id: ObjectId
    canonical_label: StrictStr
    aliases: tuple[StrictStr, ...]
    role: EntityRole

    @field_validator("role", mode="before")
    @classmethod
    def parse_role(cls, value: EntityRole | str) -> EntityRole:
        return EntityRole(value)

    @field_validator("canonical_label")
    @classmethod
    def require_canonical_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("canonical_label must not be blank")
        return value


class FrameTimestamp(StrictModel):
    frame_index: NonnegativeInt
    timestamp_seconds: Timestamp


class FrameTimeline(StrictModel):
    frames: tuple[FrameTimestamp, ...]

    @model_validator(mode="after")
    def require_strictly_increasing_frames(self) -> FrameTimeline:
        if not self.frames:
            raise ValueError("timeline must contain at least one frame")
        for previous, current in zip(self.frames, self.frames[1:]):
            if previous.frame_index >= current.frame_index:
                raise ValueError("frame indices must be strictly increasing")
            if previous.timestamp_seconds >= current.timestamp_seconds:
                raise ValueError("frame timestamps must be strictly increasing")
        return self


class SamplingPolicy(StrictModel):
    short_video_seconds: PositiveTimestamp
    scan_fps: PositiveTimestamp
    max_fps: PositiveTimestamp
    refinement_radius_seconds: PositiveTimestamp

    @model_validator(mode="after")
    def require_scan_rate_not_to_exceed_maximum(self) -> SamplingPolicy:
        if self.scan_fps > self.max_fps:
            raise ValueError("scan_fps must not exceed max_fps")
        return self


class EvidenceThresholds(StrictModel):
    min_confidence: Confidence
    min_area_fraction: Fraction
    occlusion_visibility_drop: Fraction


class CvEvidenceRequest(StrictModel):
    schema_version: Literal["cv_request_v1"]
    provider: Literal["fake", "sam31"]
    model_identity: StrictStr
    video_path: Path
    video_sha256: Sha256
    duration_seconds: PositiveTimestamp
    frame_count: PositiveInt
    checkpoint_sha256: Sha256
    timeline: FrameTimeline
    entities: tuple[EntityPrompt, ...]
    sampling: SamplingPolicy
    thresholds: EvidenceThresholds

    @field_validator("video_path", mode="before")
    @classmethod
    def parse_json_video_path(cls, value: Path | str) -> Path:
        return Path(value) if isinstance(value, str) else value

    @field_validator("model_identity")
    @classmethod
    def require_model_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_identity must not be blank")
        return value

    @model_validator(mode="after")
    def validate_source_alignment(self) -> CvEvidenceRequest:
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique")
        for frame in self.timeline.frames:
            if frame.frame_index >= self.frame_count:
                raise ValueError("timeline frame index must be below frame_count")
            if frame.timestamp_seconds > self.duration_seconds:
                raise ValueError("timeline timestamp must not exceed duration_seconds")
        return self


def _validate_relative_posix_path(value: str) -> str:
    if not value or "\\" in value or value.startswith("/"):
        raise ValueError("artifact path must be a relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact path must not contain traversal")
    return value


class TrackObservation(StrictModel):
    frame_index: NonnegativeInt
    timestamp_seconds: Timestamp
    bbox_xyxy: tuple[Fraction, Fraction, Fraction, Fraction]
    mask_ref: StrictStr | None
    visible: StrictBool
    confidence: Confidence
    area_fraction: Fraction
    center_xy: tuple[Fraction, Fraction]

    @field_validator("mask_ref")
    @classmethod
    def validate_mask_reference(cls, value: str | None) -> str | None:
        return _validate_relative_posix_path(value) if value is not None else None

    @model_validator(mode="after")
    def require_ordered_bounding_box(self) -> TrackObservation:
        left, top, right, bottom = self.bbox_xyxy
        if left >= right or top >= bottom:
            raise ValueError("bbox_xyxy must have ordered positive-width bounds")
        return self


class CvTrack(StrictModel):
    track_id: TrackId
    entity_id: ObjectId
    observations: tuple[TrackObservation, ...]
    status: EvidenceStatus = EvidenceStatus.AVAILABLE

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: EvidenceStatus | str) -> EvidenceStatus:
        return EvidenceStatus(value)

    @model_validator(mode="after")
    def require_ordered_observations(self) -> CvTrack:
        for previous, current in zip(self.observations, self.observations[1:]):
            if previous.frame_index >= current.frame_index:
                raise ValueError("track observation frame indices must be strictly increasing")
            if previous.timestamp_seconds >= current.timestamp_seconds:
                raise ValueError("track observation timestamps must be strictly increasing")
        return self


class ArtifactFile(StrictModel):
    path: StrictStr
    sha256: Sha256
    size_bytes: NonnegativeInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_posix_path(value)


class CvEvidenceArtifact(StrictModel):
    schema_version: Literal["cv_evidence_v1"]
    status: EvidenceStatus
    provider: Literal["fake", "sam31"]
    model_identity: StrictStr
    video_sha256: Sha256
    checkpoint_sha256: Sha256
    entities: tuple[EntityPrompt, ...]
    tracks: tuple[CvTrack, ...]
    files: tuple[ArtifactFile, ...]
    warnings: tuple[StrictStr, ...] = ()

    @field_validator("status", mode="before")
    @classmethod
    def parse_artifact_status(cls, value: EvidenceStatus | str) -> EvidenceStatus:
        return EvidenceStatus(value)

    @field_validator("model_identity")
    @classmethod
    def require_artifact_model_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_identity must not be blank")
        return value

    @model_validator(mode="after")
    def require_closed_references(self) -> CvEvidenceArtifact:
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("artifact entity IDs must be unique")
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique")
        file_paths = [file.path for file in self.files]
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("artifact file paths must be unique")
        known_entities = set(entity_ids)
        known_files = set(file_paths)
        for track in self.tracks:
            if track.entity_id not in known_entities:
                raise ValueError("track entity_id must reference an artifact entity")
            for observation in track.observations:
                if observation.mask_ref is not None and observation.mask_ref not in known_files:
                    raise ValueError("mask_ref must reference an artifact file")
        return self
