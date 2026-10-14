"""Frozen, provider-independent contracts for computer-vision evidence."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)


# Shared limits used by schema validation and the SAM/cache runtime.
MAX_OVERLAYS = 24
MAX_PROCESSED_FRAMES = 10_000
MAX_OBSERVATIONS = 64_000


Label = Annotated[StrictStr, Field(min_length=1, max_length=256, pattern=r"\S")]
CacheIdentity = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
ObjectId = Annotated[
    StrictStr,
    Field(
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    ),
]
TrackId = ObjectId
Timestamp = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
PositiveTimestamp = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, le=2**63 - 1, strict=True)]
NonnegativeInt = Annotated[
    int, Field(ge=0, le=2**63 - 1, strict=True)
]


class StrictModel(BaseModel):
    """A JSON-compatible process boundary that rejects drift and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def freeze_json_sequences(cls, value: Any) -> Any:
        """Accept JSON arrays for immutable tuple fields; Pydantic validates items."""
        if not isinstance(value, dict):
            return value
        value = value.copy()
        for name, field in cls.model_fields.items():
            sequence = value.get(name)
            if get_origin(field.annotation) is tuple and isinstance(sequence, (list, tuple)):
                maximum = next((item.max_length for item in field.metadata
                                if getattr(item, "max_length", None) is not None), None)
                if maximum is not None and len(sequence) > maximum:
                    raise ValueError(f"{name} must have at most {maximum} items")
                value[name] = tuple(sequence)
        return value


class EvidenceStatus(StrEnum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


class EntityPrompt(StrictModel):
    entity_id: ObjectId
    canonical_label: Label


class FrameTimestamp(StrictModel):
    frame_index: NonnegativeInt
    timestamp_seconds: Timestamp


class FrameTimeline(StrictModel):
    frames: Annotated[
        tuple[FrameTimestamp, ...], Field(max_length=100_000)
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
    max_fps: PositiveTimestamp


class EvidenceThresholds(StrictModel):
    min_confidence: Confidence


class SpatialPrompt(StrictModel):
    """One entity's reference box on the original decoder-order timeline."""

    entity_id: ObjectId
    frame_index: NonnegativeInt
    bbox_xyxy: tuple[Fraction, Fraction, Fraction, Fraction]

    @model_validator(mode="after")
    def require_ordered_box(self) -> SpatialPrompt:
        left, top, right, bottom = self.bbox_xyxy
        if left >= right or top >= bottom:
            raise ValueError("bbox_xyxy must have ordered positive-width bounds")
        return self


class CvEvidenceRequest(StrictModel):
    schema_version: Literal["cv_request_v1"]
    provider: Literal["sam31"]
    model_identity: Label
    video_path: Path
    video_identity: CacheIdentity
    duration_seconds: PositiveTimestamp
    frame_count: PositiveInt
    checkpoint_identity: CacheIdentity
    timeline: FrameTimeline
    entities: Annotated[
        tuple[EntityPrompt, ...], Field(min_length=1, max_length=1)
    ]
    sampling: SamplingPolicy
    thresholds: EvidenceThresholds
    mode: Literal["probe", "track"] = "track"
    spatial_prompts: Annotated[
        tuple[SpatialPrompt, ...], Field(min_length=1, max_length=1)
    ]

    @field_validator("video_path", mode="before")
    @classmethod
    def parse_json_video_path(
        cls, value: Path | str, info: ValidationInfo
    ) -> Path | str:
        if info.mode == "json":
            return value
        return Path(value) if isinstance(value, str) else value


    @model_validator(mode="after")
    def validate_source_alignment(self) -> CvEvidenceRequest:
        if self.spatial_prompts[0].entity_id != self.entities[0].entity_id:
            raise ValueError("spatial prompt must reference the target entity")
        known_frames = {frame.frame_index for frame in self.timeline.frames}
        if any(prompt.frame_index not in known_frames for prompt in self.spatial_prompts):
            raise ValueError("spatial prompt frame must belong to the request timeline")
        for frame in self.timeline.frames:
            if frame.frame_index >= self.frame_count:
                raise ValueError("timeline frame index must be below frame_count")
            if frame.timestamp_seconds > self.duration_seconds:
                raise ValueError("timeline timestamp must not exceed duration_seconds")
        return self


def _validate_relative_posix_path(value: str) -> str:
    if (
        not value
        or len(value) > 512
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
        if value is None:
            return None
        value = _validate_relative_posix_path(value)
        if value.startswith("overlays/"):
            raise ValueError("mask_ref cannot reference a rendered overlay")
        return value

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
        tuple[TrackObservation, ...], Field(max_length=MAX_PROCESSED_FRAMES)
    ]
    status: EvidenceStatus = Field(default=EvidenceStatus.AVAILABLE, strict=False)


    @model_validator(mode="after")
    def require_ordered_observations(self) -> CvTrack:
        if self.status is EvidenceStatus.AVAILABLE and not self.observations:
            raise ValueError("available track requires at least one observation")
        if self.status is not EvidenceStatus.AVAILABLE and self.observations:
            raise ValueError("nonavailable track cannot contain observations")
        for previous, current in zip(self.observations, self.observations[1:]):
            if previous.frame_index >= current.frame_index:
                raise ValueError("track observation frame indices must be strictly increasing")
            if previous.timestamp_seconds >= current.timestamp_seconds:
                raise ValueError("track observation timestamps must be strictly increasing")
        return self


class ArtifactFile(StrictModel):
    path: Annotated[StrictStr, Field(max_length=512)]
    size_bytes: NonnegativeInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_posix_path(value)


class OverlayRecord(StrictModel):
    """Structured provenance for one prompt-safe rendered overlay."""

    path: Annotated[StrictStr, Field(max_length=512)]
    track_id: TrackId
    frame_index: NonnegativeInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        value = _validate_relative_posix_path(value)
        if (
            len(value.split("/")) != 2
            or not value.startswith("overlays/")
            or not value.endswith(".png")
        ):
            raise ValueError("overlay path must be an overlays/*.png artifact")
        return value


class CvEvidenceArtifact(StrictModel):
    schema_version: Literal["cv_evidence_v1"]
    status: EvidenceStatus = Field(strict=False)
    provider: Literal["sam31"]
    model_identity: Label
    video_identity: CacheIdentity
    checkpoint_identity: CacheIdentity
    processed_timeline: FrameTimeline | None
    entities: Annotated[
        tuple[EntityPrompt, ...], Field(min_length=1, max_length=1)
    ]
    tracks: Annotated[
        tuple[CvTrack, ...], Field(max_length=256)
    ]
    files: tuple[ArtifactFile, ...]
    overlay_records: Annotated[
        tuple[OverlayRecord, ...], Field(max_length=MAX_OVERLAYS)
    ]


    @model_validator(mode="before")
    @classmethod
    def bound_observations(cls, value: Any) -> Any:
        """Cap the nested workload before constructing observation models."""
        if isinstance(value, dict) and isinstance(value.get("tracks"), (list, tuple)):
            total = 0
            for track in value["tracks"]:
                observations = track.get("observations", ()) if isinstance(track, dict) else getattr(track, "observations", ())
                if isinstance(observations, (list, tuple)):
                    total += len(observations)
                    if total > MAX_OBSERVATIONS:
                        raise ValueError("artifact has too many total observations")
        return value

    @model_validator(mode="after")
    def require_closed_references(self) -> CvEvidenceArtifact:
        if self.status is EvidenceStatus.AVAILABLE and self.processed_timeline is None:
            raise ValueError("available artifact requires a processed timeline")
        if self.processed_timeline is not None and (
            len(self.processed_timeline.frames) > MAX_PROCESSED_FRAMES
        ):
            raise ValueError("processed timeline exceeds its artifact bound")
        if self.status is not EvidenceStatus.AVAILABLE and (
            self.processed_timeline is not None
            or self.tracks
            or self.files
            or self.overlay_records
        ):
            raise ValueError("nonavailable artifact cannot contain processed evidence")
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
            if total_observations > MAX_OBSERVATIONS:
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
        mask_paths = {
            observation.mask_ref
            for track in self.tracks
            for observation in track.observations
            if observation.mask_ref is not None
        }
        if mask_paths & declared_overlay_paths:
            raise ValueError("mask and overlay artifact paths must be disjoint")
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
