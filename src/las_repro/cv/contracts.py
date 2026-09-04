"""Frozen, provider-independent contracts for computer-vision evidence."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)


_MAX_IDENTIFIER_CHARS = 128
_MAX_MODEL_IDENTITY_CHARS = 256
_MAX_ENTITY_LABEL_CHARS = 256
_MAX_ENTITY_ALIASES = 256
_MAX_ALIAS_CHARS = 128
_MAX_TIMELINE_FRAMES = 100_000
_MAX_PROCESSED_FRAMES = 10_000
_MAX_ARTIFACT_ENTITIES = 64
_MAX_ARTIFACT_TRACKS = 256
_MAX_TRACK_OBSERVATIONS = 10_000
_MAX_TOTAL_ARTIFACT_OBSERVATIONS = 64_000
_MAX_ARTIFACT_FILES = 1_024
_MAX_ARTIFACT_WARNINGS = 64
_MAX_WARNING_CHARS = 256
_MAX_ARTIFACT_PATH_CHARS = 512
_MAX_OVERLAY_RECORDS = 24


Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
ObjectId = Annotated[
    StrictStr,
    Field(
        max_length=_MAX_IDENTIFIER_CHARS,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    ),
]
TrackId = ObjectId
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
    canonical_label: Annotated[
        StrictStr, Field(max_length=_MAX_ENTITY_LABEL_CHARS)
    ]
    aliases: Annotated[
        tuple[Annotated[StrictStr, Field(max_length=_MAX_ALIAS_CHARS)], ...],
        Field(max_length=_MAX_ENTITY_ALIASES),
    ]
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
    frames: Annotated[
        tuple[FrameTimestamp, ...], Field(max_length=_MAX_TIMELINE_FRAMES)
    ]

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
    model_identity: Annotated[
        StrictStr, Field(max_length=_MAX_MODEL_IDENTITY_CHARS)
    ]
    video_path: Path
    video_sha256: Sha256
    duration_seconds: PositiveTimestamp
    frame_count: PositiveInt
    checkpoint_sha256: Sha256
    timeline: FrameTimeline
    entities: Annotated[
        tuple[EntityPrompt, ...], Field(max_length=_MAX_ARTIFACT_ENTITIES)
    ]
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
    if (
        not value
        or len(value) > _MAX_ARTIFACT_PATH_CHARS
        or "\\" in value
        or value.startswith("/")
    ):
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
    observations: Annotated[
        tuple[TrackObservation, ...], Field(max_length=_MAX_TRACK_OBSERVATIONS)
    ]
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
    path: Annotated[StrictStr, Field(max_length=_MAX_ARTIFACT_PATH_CHARS)]
    sha256: Sha256
    size_bytes: NonnegativeInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_posix_path(value)


class OverlayRecord(StrictModel):
    """Structured provenance for one prompt-safe rendered overlay."""

    path: Annotated[StrictStr, Field(max_length=_MAX_ARTIFACT_PATH_CHARS)]
    track_id: TrackId
    frame_index: NonnegativeInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        value = _validate_relative_posix_path(value)
        if not value.startswith("overlays/") or not value.endswith(".png"):
            raise ValueError("overlay path must be an overlays/*.png artifact")
        return value


class CvEvidenceArtifact(StrictModel):
    schema_version: Literal["cv_evidence_v1"]
    status: EvidenceStatus
    provider: Literal["fake", "sam31"]
    model_identity: Annotated[
        StrictStr, Field(max_length=_MAX_MODEL_IDENTITY_CHARS)
    ]
    video_sha256: Sha256
    checkpoint_sha256: Sha256
    processed_timeline: FrameTimeline | None
    entities: Annotated[
        tuple[EntityPrompt, ...], Field(max_length=_MAX_ARTIFACT_ENTITIES)
    ]
    tracks: Annotated[
        tuple[CvTrack, ...], Field(max_length=_MAX_ARTIFACT_TRACKS)
    ]
    files: Annotated[
        tuple[ArtifactFile, ...], Field(max_length=_MAX_ARTIFACT_FILES)
    ]
    overlay_records: Annotated[
        tuple[OverlayRecord, ...], Field(max_length=_MAX_OVERLAY_RECORDS)
    ]
    warnings: Annotated[
        tuple[Annotated[StrictStr, Field(max_length=_MAX_WARNING_CHARS)], ...],
        Field(max_length=_MAX_ARTIFACT_WARNINGS),
    ] = ()

    @model_validator(mode="before")
    @classmethod
    def enforce_raw_structural_bounds(cls, value: Any) -> Any:
        """Reject oversized nested containers before validating their items."""
        if not isinstance(value, Mapping):
            return value

        def bounded_sequence(name: str, maximum: int) -> tuple[Any, ...] | list[Any]:
            sequence = value.get(name, ())
            if isinstance(sequence, (tuple, list)) and len(sequence) > maximum:
                raise ValueError(f"artifact {name} exceeds its structural bound")
            return sequence if isinstance(sequence, (tuple, list)) else []

        entities = bounded_sequence("entities", _MAX_ARTIFACT_ENTITIES)
        for entity in entities:
            aliases = (
                entity.aliases
                if isinstance(entity, EntityPrompt)
                else entity.get("aliases", ())
                if isinstance(entity, Mapping)
                else ()
            )
            if isinstance(aliases, (tuple, list)) and (
                len(aliases) > _MAX_ENTITY_ALIASES
            ):
                raise ValueError("artifact entity aliases exceed their structural bound")

        tracks = bounded_sequence("tracks", _MAX_ARTIFACT_TRACKS)
        total_observations = 0
        for track in tracks:
            observations = (
                track.observations
                if isinstance(track, CvTrack)
                else track.get("observations", ())
                if isinstance(track, Mapping)
                else ()
            )
            if isinstance(observations, (tuple, list)):
                if len(observations) > _MAX_TRACK_OBSERVATIONS:
                    raise ValueError(
                        "artifact track observations exceed their structural bound"
                    )
                total_observations += len(observations)
                if total_observations > _MAX_TOTAL_ARTIFACT_OBSERVATIONS:
                    raise ValueError("artifact has too many total observations")

        bounded_sequence("files", _MAX_ARTIFACT_FILES)
        bounded_sequence("overlay_records", _MAX_OVERLAY_RECORDS)
        bounded_sequence("warnings", _MAX_ARTIFACT_WARNINGS)
        processed_timeline = value.get("processed_timeline")
        processed_frames = (
            processed_timeline.frames
            if isinstance(processed_timeline, FrameTimeline)
            else processed_timeline.get("frames", ())
            if isinstance(processed_timeline, Mapping)
            else ()
        )
        if isinstance(processed_frames, (tuple, list)) and (
            len(processed_frames) > _MAX_PROCESSED_FRAMES
        ):
            raise ValueError("processed timeline exceeds its artifact bound")
        return value

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
        if self.status is EvidenceStatus.AVAILABLE and self.processed_timeline is None:
            raise ValueError("available artifact requires a processed timeline")
        if self.processed_timeline is not None and (
            len(self.processed_timeline.frames) > _MAX_PROCESSED_FRAMES
        ):
            raise ValueError("processed timeline exceeds its artifact bound")
        if self.status is EvidenceStatus.DISABLED and (
            self.processed_timeline is not None
            or self.tracks
            or self.files
            or self.overlay_records
        ):
            raise ValueError("disabled artifact cannot contain processed evidence")
        if self.processed_timeline is None and any(
            track.observations for track in self.tracks
        ):
            raise ValueError("observations require a processed timeline")
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
        processed_clock = (
            {
                frame.frame_index: frame.timestamp_seconds
                for frame in self.processed_timeline.frames
            }
            if self.processed_timeline is not None
            else {}
        )
        total_observations = 0
        for track in self.tracks:
            if track.entity_id not in known_entities:
                raise ValueError("track entity_id must reference an artifact entity")
            total_observations += len(track.observations)
            if total_observations > _MAX_TOTAL_ARTIFACT_OBSERVATIONS:
                raise ValueError("artifact has too many total observations")
            for observation in track.observations:
                if processed_clock.get(observation.frame_index) != (
                    observation.timestamp_seconds
                ):
                    raise ValueError(
                        "observation must close to the processed timeline"
                    )
                if observation.mask_ref is not None and observation.mask_ref not in known_files:
                    raise ValueError("mask_ref must reference an artifact file")
        tracks_by_id = {track.track_id: track for track in self.tracks}
        overlay_paths = [record.path for record in self.overlay_records]
        if len(overlay_paths) != len(set(overlay_paths)):
            raise ValueError("overlay record paths must be unique")
        declared_overlay_paths = {
            artifact_file.path
            for artifact_file in self.files
            if artifact_file.path.startswith("overlays/")
        }
        if set(overlay_paths) != declared_overlay_paths:
            raise ValueError("overlay records must exactly cover overlay files")
        for record in self.overlay_records:
            track = tracks_by_id.get(record.track_id)
            if track is None:
                raise ValueError("overlay track_id must reference an artifact track")
            if not any(
                observation.frame_index == record.frame_index
                and observation.visible
                for observation in track.observations
            ):
                raise ValueError("overlay frame must reference a visible observation")
        return self
