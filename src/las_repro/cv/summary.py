"""Bounded, provider-independent CV summaries for semantic prompt stages.

This module intentionally consumes only validated scalar evidence.  Artifact
payloads, including segmentation archives, are never opened here.  Spatial
overlap is therefore named and computed as a bounding-box proxy rather than as
mask IoU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import heapq
from io import StringIO
import json
import math
import re
from typing import Annotated, Any, Iterator, Literal

from pydantic import Field, StrictBool, StrictStr, field_validator, model_validator

from .contracts import (
    ArtifactFile,
    Confidence,
    CvEvidenceArtifact,
    CvTrack,
    EntityRole,
    EntityPrompt,
    EvidenceStatus,
    EvidenceThresholds,
    Fraction,
    FrameTimeline,
    FrameTimestamp,
    NonnegativeInt,
    ObjectId,
    OverlayRecord,
    StrictModel,
    Timestamp,
    TrackId,
    TrackObservation,
)


CandidateId = Annotated[
    StrictStr, Field(pattern=r"^occ_[0-9a-f]{12}_[0-9]{4}$")
]
SummaryId = Annotated[StrictStr, Field(pattern=r"^cvs_[0-9a-f]{64}$")]

_DEFAULT_MAX_TRACKS = 64
_DEFAULT_MAX_OBSERVATIONS_PER_TRACK = 64
_DEFAULT_MAX_RELATIONS = 512
_DEFAULT_MAX_OVERLAYS = 24
_MAX_PROMPT_CHARS = 200_000
_MAX_TRACK_LIMIT = 256
_MAX_OBSERVATION_LIMIT = 256
_MAX_RELATION_LIMIT = 8_192
_MAX_OVERLAY_LIMIT = 24
_MAX_ENTITY_SUMMARIES = 64
_MAX_IDENTIFIER_CHARS = 128
_MAX_LABEL_CHARS = 256
_MAX_ALIAS_CHARS = 128
_MAX_ALIASES_PER_ENTITY = 16
_MAX_CANDIDATE_PROMPT_CHARS = _MAX_PROMPT_CHARS
_MAX_BUNDLE_CANDIDATES = 256
_MAX_CANDIDATE_OCCLUDERS = 16
_MAX_CANDIDATE_SUPPORTING_FRAMES = 64
_MAX_TOTAL_CANDIDATE_COMPONENTS = 32_768
_MAX_CANONICAL_INT = 2**63 - 1
_MIN_BUNDLE_ENVELOPE_CHARS = 512
_MAX_VALIDATION_METADATA_CHARS = 256
_ABS_MAX_ARTIFACT_ENTITIES = 64
_ABS_MAX_ARTIFACT_TRACKS = 256
_ABS_MAX_ALIASES_PER_INPUT_ENTITY = 256
_ABS_MAX_OBSERVATIONS_PER_INPUT_TRACK = 10_000
_ABS_MAX_TOTAL_INPUT_OBSERVATIONS = 64_000
_ABS_MAX_ARTIFACT_FILES = 1_024
_ABS_MAX_ARTIFACT_WARNINGS = 64
_ABS_MAX_TIMELINE_FRAMES = 100_000
_ABS_MAX_PROCESSED_FRAMES = 10_000
_ABS_MAX_INPUT_STRING_CHARS = 512
_ABS_MAX_SUMMARY_WARNINGS = 64
_MAX_VISIBILITY_RUNS_PER_TRACK = 256
_MAX_RELATION_PAIR_SCANS = 262_144
_EDGE_PROXIMITY_THRESHOLD = 0.05
_MANDATORY_PRIORITY_TEXT = (
    "first,last,state_changes,min_area_context,max_area,lowest_confidence_context"
)
_SAFE_OVERLAY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SummaryEntity(StrictModel):
    """Bounded entity text retained for later trusted-data prompt sections."""

    entity_id: ObjectId
    canonical_label: Annotated[StrictStr, Field(max_length=_MAX_LABEL_CHARS)]
    aliases: Annotated[
        tuple[Annotated[StrictStr, Field(max_length=_MAX_ALIAS_CHARS)], ...],
        Field(max_length=_MAX_ALIASES_PER_ENTITY),
    ]
    role: EntityRole

    @field_validator("role", mode="before")
    @classmethod
    def parse_role(cls, value: EntityRole | str) -> EntityRole:
        return EntityRole(value)

    @field_validator("canonical_label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if not value.strip() or len(value) > _MAX_LABEL_CHARS:
            raise ValueError("canonical_label is outside the summary bound")
        return value

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) > _MAX_ALIASES_PER_ENTITY:
            raise ValueError("too many summary aliases")
        if any(not value.strip() or len(value) > _MAX_ALIAS_CHARS for value in values):
            raise ValueError("alias is outside the summary bound")
        return values


class SummaryObservation(StrictModel):
    """One selected scalar observation, with no artifact payload reference."""

    source_ordinal: NonnegativeInt
    frame_index: NonnegativeInt
    timestamp_seconds: Timestamp
    bbox_xyxy: tuple[Fraction, Fraction, Fraction, Fraction]
    visible: StrictBool
    confidence: Confidence
    area_fraction: Fraction
    center_xy: tuple[Fraction, Fraction]
    edge_proximity: Fraction

    @model_validator(mode="after")
    def validate_geometry(self) -> SummaryObservation:
        left, top, right, bottom = self.bbox_xyxy
        if left >= right or top >= bottom:
            raise ValueError("bbox_xyxy must have positive ordered bounds")
        expected_edge = min(left, top, 1.0 - right, 1.0 - bottom)
        if not math.isclose(
            self.edge_proximity, expected_edge, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("edge_proximity must match bbox geometry")
        return self


class VisibilityGap(StrictModel):
    """An observed-clock interval where a previously visible track is missing."""

    last_visible_frame: NonnegativeInt
    last_visible_time: Timestamp
    first_missing_frame: NonnegativeInt
    first_missing_time: Timestamp
    last_missing_frame: NonnegativeInt
    last_missing_time: Timestamp
    first_revisible_frame: NonnegativeInt | None
    first_revisible_time: Timestamp | None
    minimum_confidence: Confidence | None
    edge_departure: StrictBool

    @model_validator(mode="after")
    def validate_boundary_order(self) -> VisibilityGap:
        if not (
            self.last_visible_frame < self.first_missing_frame
            <= self.last_missing_frame
        ):
            raise ValueError("visibility gap frames are not ordered")
        if not (
            self.last_visible_time < self.first_missing_time
            <= self.last_missing_time
        ):
            raise ValueError("visibility gap times are not ordered")
        if (self.first_revisible_frame is None) != (
            self.first_revisible_time is None
        ):
            raise ValueError("revisibility frame and time must be paired")
        if self.first_revisible_frame is not None:
            if self.first_revisible_frame <= self.last_missing_frame:
                raise ValueError("revisibility frame must follow the gap")
            if self.first_revisible_time is None or (
                self.first_revisible_time <= self.last_missing_time
            ):
                raise ValueError("revisibility time must follow the gap")
        return self


class VisibilityRun(StrictModel):
    """A compact witness for one constant visibility state on processed frames."""

    state: Literal["visible", "missing"]
    start_frame: NonnegativeInt
    start_time: Timestamp
    end_frame: NonnegativeInt
    end_time: Timestamp
    minimum_confidence: Confidence | None

    @model_validator(mode="after")
    def validate_run(self) -> VisibilityRun:
        if self.start_frame > self.end_frame or self.start_time > self.end_time:
            raise ValueError("visibility run boundaries must be ordered")
        if self.state == "visible" and self.minimum_confidence is not None:
            raise ValueError("visible run cannot carry missing confidence")
        return self


class SummaryTrack(StrictModel):
    """A bounded track plus lifecycle gaps derived before downsampling."""

    track_id: TrackId
    entity_id: ObjectId
    status: EvidenceStatus
    source_observation_count: NonnegativeInt
    observations: Annotated[
        tuple[SummaryObservation, ...], Field(max_length=_MAX_OBSERVATION_LIMIT)
    ]
    visibility_runs: Annotated[
        tuple[VisibilityRun, ...], Field(max_length=_MAX_VISIBILITY_RUNS_PER_TRACK)
    ]
    visibility_lifecycle_complete: StrictBool
    missing_intervals: Annotated[
        tuple[VisibilityGap, ...], Field(max_length=_MAX_OBSERVATION_LIMIT)
    ]
    candidate_search_complete: StrictBool

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: EvidenceStatus | str) -> EvidenceStatus:
        return EvidenceStatus(value)

    @model_validator(mode="after")
    def validate_selected_order(self) -> SummaryTrack:
        if len(self.observations) > self.source_observation_count:
            raise ValueError("selected observations exceed source count")
        if self.candidate_search_complete and (
            len(self.observations) != self.source_observation_count
        ):
            raise ValueError("complete candidate search cannot omit observations")
        if self.status is not EvidenceStatus.AVAILABLE and (
            self.observations
            or self.visibility_runs
            or self.visibility_lifecycle_complete
            or self.missing_intervals
            or self.candidate_search_complete
        ):
            raise ValueError("nonavailable tracks cannot retain candidate evidence")
        for previous, current in zip(self.observations, self.observations[1:]):
            if previous.source_ordinal >= current.source_ordinal:
                raise ValueError("source ordinals must increase")
            if previous.frame_index >= current.frame_index:
                raise ValueError("selected frame indices must increase")
            if previous.timestamp_seconds >= current.timestamp_seconds:
                raise ValueError("selected timestamps must increase")
        for previous, current in zip(
            self.missing_intervals, self.missing_intervals[1:]
        ):
            previous_end = (
                previous.first_revisible_frame
                if previous.first_revisible_frame is not None
                else previous.last_missing_frame
            )
            if previous_end >= current.first_missing_frame:
                raise ValueError("missing intervals must be disjoint and ordered")
        for previous, current in zip(self.visibility_runs, self.visibility_runs[1:]):
            if previous.state == current.state:
                raise ValueError("adjacent visibility runs must change state")
            if previous.end_frame >= current.start_frame or (
                previous.end_time >= current.start_time
            ):
                raise ValueError("visibility runs must be disjoint and ordered")
        expected_gaps = _gaps_from_visibility_runs(
            self.visibility_runs,
            self.observations,
            lifecycle_complete=self.visibility_lifecycle_complete,
        )
        if self.missing_intervals != expected_gaps:
            raise ValueError("missing intervals must match the visibility lifecycle")
        return self


class SpatialRelation(StrictModel):
    """A same-frame, canonical unordered-pair bounding-box relation proxy."""

    frame_index: NonnegativeInt
    timestamp_seconds: Timestamp
    subject_track_id: TrackId
    object_track_id: TrackId
    bbox_iou: Fraction
    subject_bbox_covered_fraction: Fraction
    object_bbox_covered_fraction: Fraction
    center_distance_fraction: Fraction
    area_similarity: Fraction
    subject_inside_object: StrictBool
    object_inside_subject: StrictBool

    @model_validator(mode="after")
    def reject_self_relation(self) -> SpatialRelation:
        if self.subject_track_id >= self.object_track_id:
            raise ValueError("a spatial relation needs two tracks")
        return self


class SummaryOverlay(StrictModel):
    """Prompt-safe overlay path with explicit track/frame provenance."""

    path: Annotated[StrictStr, Field(max_length=512)]
    track_id: TrackId
    frame_index: NonnegativeInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_overlay_ref(value)


class SummaryWarning(StrictModel):
    """A closed warning code with a code-specific scalar payload grammar."""

    code: Literal[
        "ALIASES_TRUNCATED",
        "ARTIFACT_WARNINGS_OMITTED",
        "ENTITIES_WITHOUT_AVAILABLE_TRACKS",
        "ENTITIES_TRUNCATED",
        "ENTITY_TEXT_TRUNCATED",
        "LIFECYCLE_GAPS_TRUNCATED",
        "MANDATORY_LANDMARKS_TRUNCATED",
        "OBSERVATIONS_TRUNCATED",
        "OVERLAYS_TRUNCATED",
        "PROMPT_DATA_TRUNCATED",
        "RELATIONS_TRUNCATED",
        "TRACKS_TRUNCATED",
        "UNAVAILABLE_EVIDENCE_OMITTED",
        "VISIBILITY_RUNS_TRUNCATED",
    ]
    kept: NonnegativeInt | None = None
    omitted: NonnegativeInt | None = None
    count: NonnegativeInt | None = None
    tracks: NonnegativeInt | None = None
    priority: Literal[
        "first,last,state_changes,min_area_context,max_area,lowest_confidence_context"
    ] | None = None
    budget_enforced: StrictBool | None = None

    @model_validator(mode="after")
    def validate_payload_grammar(self) -> SummaryWarning:
        populated = {
            name
            for name in (
                "kept",
                "omitted",
                "count",
                "tracks",
                "priority",
                "budget_enforced",
            )
            if getattr(self, name) is not None
        }
        required = {
            "ENTITIES_TRUNCATED": {"kept", "omitted"},
            "ALIASES_TRUNCATED": {"omitted"},
            "ENTITY_TEXT_TRUNCATED": {"count"},
            "TRACKS_TRUNCATED": {"kept", "omitted"},
            "OBSERVATIONS_TRUNCATED": {"kept", "omitted"},
            "MANDATORY_LANDMARKS_TRUNCATED": {"tracks", "priority"},
            "LIFECYCLE_GAPS_TRUNCATED": {"kept", "omitted"},
            "RELATIONS_TRUNCATED": {"kept", "omitted"},
            "OVERLAYS_TRUNCATED": {"kept", "omitted"},
            "ARTIFACT_WARNINGS_OMITTED": {"count"},
            "ENTITIES_WITHOUT_AVAILABLE_TRACKS": {"count"},
            "UNAVAILABLE_EVIDENCE_OMITTED": {"count"},
            "VISIBILITY_RUNS_TRUNCATED": {"kept", "omitted"},
            "PROMPT_DATA_TRUNCATED": {"budget_enforced"},
        }[self.code]
        if populated != required:
            raise ValueError("warning payload does not match its closed code grammar")
        if self.omitted is not None and self.omitted == 0:
            raise ValueError("truncation warning must omit at least one item")
        if self.count is not None and self.count == 0:
            raise ValueError("count warning must describe at least one item")
        if self.tracks is not None and self.tracks == 0:
            raise ValueError("track warning must describe at least one track")
        if self.code == "PROMPT_DATA_TRUNCATED" and self.budget_enforced is not True:
            raise ValueError("prompt truncation warning must confirm budget enforcement")
        return self


class CvEvidenceSummary(StrictModel):
    """Frozen evidence summary whose prompt projection has a hard char cap."""

    schema_version: Literal["cv_summary_v1"]
    summary_id: SummaryId
    status: EvidenceStatus
    candidate_search_complete: StrictBool
    observed_clock: Annotated[
        tuple[FrameTimestamp, ...], Field(max_length=_ABS_MAX_PROCESSED_FRAMES)
    ]
    entities: Annotated[
        tuple[SummaryEntity, ...], Field(max_length=_MAX_ENTITY_SUMMARIES)
    ]
    tracks: Annotated[
        tuple[SummaryTrack, ...], Field(max_length=_MAX_TRACK_LIMIT)
    ]
    relations: Annotated[
        tuple[SpatialRelation, ...], Field(max_length=_MAX_RELATION_LIMIT)
    ]
    relations_complete: StrictBool
    overlay_refs: Annotated[
        tuple[Annotated[StrictStr, Field(max_length=512)], ...],
        Field(max_length=_MAX_OVERLAY_LIMIT),
    ]
    overlays: Annotated[
        tuple[SummaryOverlay, ...], Field(max_length=_MAX_OVERLAY_LIMIT)
    ]
    overlays_complete: StrictBool
    warnings: Annotated[
        tuple[SummaryWarning, ...], Field(max_length=_ABS_MAX_SUMMARY_WARNINGS)
    ]
    prompt_char_limit: Annotated[
        int, Field(gt=0, le=_MAX_PROMPT_CHARS, strict=True)
    ] = _MAX_PROMPT_CHARS

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: EvidenceStatus | str) -> EvidenceStatus:
        return EvidenceStatus(value)

    @field_validator("overlay_refs")
    @classmethod
    def validate_overlay_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) > _MAX_OVERLAY_LIMIT:
            raise ValueError("too many overlay references")
        for value in values:
            _validate_overlay_ref(value)
        if len(values) != len(set(values)):
            raise ValueError("overlay references must be unique")
        return values

    @field_validator("observed_clock")
    @classmethod
    def validate_observed_clock(
        cls, values: tuple[FrameTimestamp, ...]
    ) -> tuple[FrameTimestamp, ...]:
        if len(values) > _ABS_MAX_PROCESSED_FRAMES:
            raise ValueError("summary observed clock exceeds its bound")
        for previous, current in zip(values, values[1:]):
            if previous.frame_index >= current.frame_index or (
                previous.timestamp_seconds >= current.timestamp_seconds
            ):
                raise ValueError("summary observed clock must increase")
        return values

    @field_validator("warnings")
    @classmethod
    def validate_warnings(
        cls, values: tuple[SummaryWarning, ...]
    ) -> tuple[SummaryWarning, ...]:
        codes = [warning.code for warning in values]
        if len(codes) != len(set(codes)):
            raise ValueError("summary warning codes must be unique")
        return values

    @model_validator(mode="after")
    def validate_closed_references(self) -> CvEvidenceSummary:
        if self.candidate_search_complete and (
            self.status is not EvidenceStatus.AVAILABLE
            or any(not track.candidate_search_complete for track in self.tracks)
            or any(
                warning.code
                in {
                    "TRACKS_TRUNCATED",
                    "OBSERVATIONS_TRUNCATED",
                    "LIFECYCLE_GAPS_TRUNCATED",
                    "MANDATORY_LANDMARKS_TRUNCATED",
                    "UNAVAILABLE_EVIDENCE_OMITTED",
                    "VISIBILITY_RUNS_TRUNCATED",
                    "ENTITIES_WITHOUT_AVAILABLE_TRACKS",
                    "ARTIFACT_WARNINGS_OMITTED",
                }
                for warning in self.warnings
            )
        ):
            raise ValueError("summary candidate-search completeness is inconsistent")
        if self.status is not EvidenceStatus.AVAILABLE and (
            self.tracks
            or self.relations
            or self.overlay_refs
            or self.overlays
            or self.observed_clock
        ):
            raise ValueError("nonavailable summaries cannot retain visual evidence")
        entity_ids = [entity.entity_id for entity in self.entities]
        track_ids = [track.track_id for track in self.tracks]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("summary entity IDs must be unique")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("summary track IDs must be unique")
        known_entities = set(entity_ids)
        known_tracks = set(track_ids)
        if any(track.entity_id not in known_entities for track in self.tracks):
            raise ValueError("summary tracks must reference retained entities")
        frame_times = {
            item.frame_index: item.timestamp_seconds for item in self.observed_clock
        }
        for track in self.tracks:
            for observation in track.observations:
                if frame_times.get(observation.frame_index) != (
                    observation.timestamp_seconds
                ):
                    raise ValueError(
                        "observation is not closed to the authoritative observed clock"
                    )
            for run in track.visibility_runs:
                if frame_times.get(run.start_frame) != run.start_time or (
                    frame_times.get(run.end_frame) != run.end_time
                ):
                    raise ValueError(
                        "visibility lifecycle is not closed to the observed clock"
                    )
            for observation in track.observations:
                matching_runs = [
                    run
                    for run in track.visibility_runs
                    if run.start_frame <= observation.frame_index <= run.end_frame
                ]
                if len(matching_runs) != 1 or (
                    (matching_runs[0].state == "visible")
                    != observation.visible
                ):
                    raise ValueError(
                        "retained observation contradicts the visibility lifecycle"
                    )
            visible_boundaries = {
                (item.frame_index, item.timestamp_seconds)
                for item in track.observations
                if item.visible
            }
            for gap in track.missing_intervals:
                gap_boundaries = (
                    (gap.last_visible_frame, gap.last_visible_time),
                    (gap.first_missing_frame, gap.first_missing_time),
                    (gap.last_missing_frame, gap.last_missing_time),
                    (
                        gap.first_revisible_frame,
                        gap.first_revisible_time,
                    ),
                )
                for frame_index, timestamp in gap_boundaries:
                    if frame_index is None or timestamp is None:
                        continue
                    if frame_times.get(frame_index) != timestamp:
                        raise ValueError(
                            "gap is not closed to the authoritative observed clock"
                        )
                if (
                    gap.last_visible_frame,
                    gap.last_visible_time,
                ) not in visible_boundaries:
                    raise ValueError(
                        "gap last-visible boundary needs a retained track observation"
                    )
                if gap.first_revisible_frame is not None and (
                    gap.first_revisible_frame,
                    gap.first_revisible_time,
                ) not in visible_boundaries:
                    raise ValueError(
                        "gap revisibility boundary needs a retained track observation"
                    )
                if any(
                    item.visible
                    and gap.first_missing_frame
                    <= item.frame_index
                    <= gap.last_missing_frame
                    for item in track.observations
                ):
                    raise ValueError(
                        "gap cannot mark a retained visible observation as missing"
                    )
        observations = {
            (track.track_id, item.frame_index, item.timestamp_seconds)
            for track in self.tracks
            for item in track.observations
            if item.visible
        }
        relation_keys: set[tuple[int, str, str]] = set()
        observations_by_key = {
            (track.track_id, item.frame_index, item.timestamp_seconds): item
            for track in self.tracks
            for item in track.observations
            if item.visible
        }
        for relation in self.relations:
            if (
                relation.subject_track_id not in known_tracks
                or relation.object_track_id not in known_tracks
            ):
                raise ValueError("relation track reference is not closed")
            if (
                relation.subject_track_id,
                relation.frame_index,
                relation.timestamp_seconds,
            ) not in observations or (
                relation.object_track_id,
                relation.frame_index,
                relation.timestamp_seconds,
            ) not in observations:
                raise ValueError("relation geometry must be aligned to one frame")
            key = (
                relation.frame_index,
                relation.subject_track_id,
                relation.object_track_id,
            )
            if key in relation_keys:
                raise ValueError("summary relations must be unique")
            relation_keys.add(key)
            expected = _spatial_relation(
                relation.subject_track_id,
                observations_by_key[
                    (
                        relation.subject_track_id,
                        relation.frame_index,
                        relation.timestamp_seconds,
                    )
                ],
                relation.object_track_id,
                observations_by_key[
                    (
                        relation.object_track_id,
                        relation.frame_index,
                        relation.timestamp_seconds,
                    )
                ],
            )
            if relation != expected:
                raise ValueError("relation geometry does not match retained boxes")
        relation_warning = any(
            warning.code == "RELATIONS_TRUNCATED" for warning in self.warnings
        )
        visible_counts_by_frame: dict[int, int] = {}
        for _, frame_index, _ in observations:
            visible_counts_by_frame[frame_index] = (
                visible_counts_by_frame.get(frame_index, 0) + 1
            )
        possible_relation_count = sum(
            count * (count - 1) // 2
            for count in visible_counts_by_frame.values()
        )
        if self.relations_complete != (
            len(self.relations) == possible_relation_count
        ):
            raise ValueError(
                "relation completeness must match retained same-frame geometry"
            )
        if self.relations_complete == relation_warning:
            raise ValueError("relation completeness must match truncation warning")
        relation_warning_record = next(
            (
                warning
                for warning in self.warnings
                if warning.code == "RELATIONS_TRUNCATED"
            ),
            None,
        )
        if relation_warning_record is not None and (
            relation_warning_record.kept != len(self.relations)
            or relation_warning_record.omitted
            != possible_relation_count - len(self.relations)
        ):
            raise ValueError("relation warning must match retained geometry")
        overlay_warning = any(
            warning.code == "OVERLAYS_TRUNCATED" for warning in self.warnings
        )
        if self.overlays_complete == overlay_warning:
            raise ValueError("overlay completeness must match truncation warning")
        if tuple(item.path for item in self.overlays) != self.overlay_refs:
            raise ValueError("overlay reference projection must match provenance")
        visible_frames = {
            (track.track_id, observation.frame_index)
            for track in self.tracks
            for observation in track.observations
            if observation.visible
        }
        if any(
            (overlay.track_id, overlay.frame_index) not in visible_frames
            for overlay in self.overlays
        ):
            raise ValueError("overlay provenance must close to retained visibility")
        if self.summary_id != _summary_identity(self):
            raise ValueError("summary identity does not match canonical content")
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return a fresh allowlisted JSON record, rejecting an oversized model."""
        _preflight_summary(self)
        validated = _revalidate_summary(self)
        return _materialize_bounded_record(
            _summary_projection(validated),
            maximum=validated.prompt_char_limit,
            message="CV summary prompt record exceeds its character cap",
        )


class OccluderProvenance(StrictModel):
    """One track-specific source of positive aligned overlap support."""

    entity_id: ObjectId
    track_id: TrackId
    supporting_frames: Annotated[
        tuple[NonnegativeInt, ...],
        Field(max_length=_MAX_CANDIDATE_SUPPORTING_FRAMES),
    ]

    @field_validator("supporting_frames")
    @classmethod
    def validate_supporting_frames(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("occluder provenance needs at least one supporting frame")
        if tuple(sorted(set(values))) != values:
            raise ValueError("supporting frames must be unique and ordered")
        return values


class OcclusionCandidate(StrictModel):
    """An evidence-supported transition awaiting semantic adjudication."""

    candidate_id: CandidateId
    target_entity_id: ObjectId
    target_track_id: TrackId
    possible_occluders: Annotated[
        tuple[OccluderProvenance, ...], Field(max_length=_MAX_CANDIDATE_OCCLUDERS)
    ]
    possible_occluder_entity_ids: Annotated[
        tuple[ObjectId, ...], Field(max_length=_MAX_CANDIDATE_OCCLUDERS)
    ]
    allowed_start_times: Annotated[
        tuple[Timestamp, ...], Field(min_length=1, max_length=8)
    ]
    allowed_end_times: Annotated[
        tuple[Timestamp, ...], Field(min_length=1, max_length=8)
    ]
    last_visible_frame: NonnegativeInt
    first_revisible_frame: NonnegativeInt | None
    edge_departure: StrictBool
    low_confidence: StrictBool
    overlay_refs: Annotated[
        tuple[Annotated[StrictStr, Field(max_length=512)], ...],
        Field(max_length=_MAX_OVERLAY_LIMIT),
    ]
    observation_support_complete: StrictBool
    relation_support_complete: StrictBool
    overlay_support_complete: StrictBool
    support_complete: StrictBool
    source_search_complete: StrictBool

    @field_validator("overlay_refs")
    @classmethod
    def validate_overlay_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) > _MAX_OVERLAY_LIMIT:
            raise ValueError("too many candidate overlay references")
        for value in values:
            _validate_overlay_ref(value)
        if len(values) != len(set(values)):
            raise ValueError("candidate overlay references must be unique")
        return values

    @model_validator(mode="after")
    def validate_candidate_boundaries(self) -> OcclusionCandidate:
        if not self.allowed_start_times or not self.allowed_end_times:
            raise ValueError("candidate boundaries must use observed timestamps")
        if tuple(sorted(set(self.allowed_start_times))) != self.allowed_start_times:
            raise ValueError("candidate start times must be unique and ordered")
        if tuple(sorted(set(self.allowed_end_times))) != self.allowed_end_times:
            raise ValueError("candidate end times must be unique and ordered")
        if max(self.allowed_start_times) >= min(self.allowed_end_times):
            raise ValueError("candidate boundaries must guarantee positive duration")
        if (
            self.first_revisible_frame is not None
            and self.first_revisible_frame <= self.last_visible_frame
        ):
            raise ValueError("candidate revisibility frame must follow last visibility")
        if tuple(sorted(set(self.possible_occluder_entity_ids))) != (
            self.possible_occluder_entity_ids
        ):
            raise ValueError("possible occluders must be unique and ordered")
        if tuple(
            sorted(
                self.possible_occluders,
                key=lambda item: (item.entity_id, item.track_id),
            )
        ) != self.possible_occluders:
            raise ValueError("possible occluder provenance must be ordered")
        if len({item.track_id for item in self.possible_occluders}) != len(
            self.possible_occluders
        ):
            raise ValueError("possible occluder track IDs must be unique")
        if any(item.track_id == self.target_track_id for item in self.possible_occluders):
            raise ValueError("target track cannot occlude itself")
        projected_entities = tuple(
            sorted({item.entity_id for item in self.possible_occluders})
        )
        if projected_entities != self.possible_occluder_entity_ids:
            raise ValueError("occluder entity projection must match track provenance")
        if self.support_complete != (
            self.observation_support_complete
            and self.relation_support_complete
            and self.overlay_support_complete
        ):
            raise ValueError("candidate support completeness fields disagree")
        ordinal = int(self.candidate_id.rsplit("_", 1)[1])
        if ordinal <= 0 or self.candidate_id != _candidate_identity(self, ordinal):
            raise ValueError("candidate identity does not match canonical content")
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return fresh allowlisted values without exposing model internals."""
        _preflight_candidate(self)
        validated = _revalidate_candidate(self)
        return _materialize_bounded_record(
            _candidate_projection(validated),
            maximum=_MAX_CANDIDATE_PROMPT_CHARS,
            message="occlusion candidate prompt record exceeds its cap",
        )


def _candidate_prompt_record(candidate: OcclusionCandidate) -> dict[str, Any]:
    """Materialize one already-bounded private compatibility projection."""
    return _materialize_record(_candidate_projection(candidate))


def _candidate_identity(candidate: OcclusionCandidate, ordinal: int) -> str:
    projection = _candidate_projection(
        candidate,
        include_candidate_id=False,
        ordinal=ordinal,
    )
    return f"occ_{_candidate_hash_prefix(projection)}_{ordinal:04d}"


class CvPromptBundle(StrictModel):
    """The only aggregate prompt record for one summary and its candidates."""

    schema_version: Literal["cv_prompt_bundle_v1"]
    summary: CvEvidenceSummary
    thresholds: EvidenceThresholds
    candidates: Annotated[
        tuple[OcclusionCandidate, ...], Field(max_length=_MAX_BUNDLE_CANDIDATES)
    ]
    source_search_complete: StrictBool
    candidates_complete: StrictBool
    truncation_codes: Annotated[
        tuple[
            Literal[
                "SUMMARY_CANDIDATE_SEARCH_INCOMPLETE",
                "CANDIDATE_SOURCE_TRUNCATED",
                "CANDIDATE_COUNT_TRUNCATED",
                "CANDIDATE_PROMPT_TRUNCATED",
            ],
            ...,
        ],
        Field(max_length=4),
    ]
    prompt_char_limit: Annotated[
        int, Field(gt=0, le=_MAX_PROMPT_CHARS, strict=True)
    ] = _MAX_PROMPT_CHARS
    candidate_limit: Annotated[
        int, Field(gt=0, le=_MAX_BUNDLE_CANDIDATES, strict=True)
    ] = _MAX_BUNDLE_CANDIDATES

    @model_validator(mode="before")
    @classmethod
    def enforce_raw_candidate_aggregate(cls, value: Any) -> Any:
        """Reject multiplicative candidate shapes before nested validation."""
        if isinstance(value, cls) or type(value) is not dict:
            return value
        candidates = value.get("candidates")
        if type(candidates) not in {list, tuple}:
            if isinstance(candidates, (list, tuple)):
                raise ValueError("bundle candidates use a custom sequence")
            return value
        if len(candidates) > _MAX_BUNDLE_CANDIDATES:
            raise ValueError("bundle candidates exceed their structural bound")
        aggregate_components = 0
        for candidate in candidates:
            aggregate_components += _candidate_component_count(candidate)
            if aggregate_components > _MAX_TOTAL_CANDIDATE_COMPONENTS:
                raise ValueError("aggregate candidate components exceed their bound")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> CvPromptBundle:
        if len(self.candidates) > _MAX_BUNDLE_CANDIDATES:
            raise ValueError("too many candidates in aggregate prompt bundle")
        if len({item.candidate_id for item in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("aggregate candidate IDs must be unique")
        if self.truncation_codes != tuple(dict.fromkeys(self.truncation_codes)):
            raise ValueError("aggregate truncation codes must be unique and ordered")
        set_truncation_codes = {
            "CANDIDATE_SOURCE_TRUNCATED",
            "CANDIDATE_COUNT_TRUNCATED",
            "CANDIDATE_PROMPT_TRUNCATED",
        }
        if self.candidates_complete != (
            not any(code in set_truncation_codes for code in self.truncation_codes)
        ):
            raise ValueError("candidate set completeness must match set truncation")
        if (
            "SUMMARY_CANDIDATE_SEARCH_INCOMPLETE" in self.truncation_codes
        ) == self.summary.candidate_search_complete:
            raise ValueError("bundle summary-search completeness is inconsistent")
        if self.source_search_complete != self.summary.candidate_search_complete:
            raise ValueError("bundle source-search completeness is inconsistent")
        track_entities = {
            track.track_id: track.entity_id for track in self.summary.tracks
        }
        tracks_by_id = {track.track_id: track for track in self.summary.tracks}
        observed_times = {
            item.timestamp_seconds for item in self.summary.observed_clock
        }
        summary_overlays = set(self.summary.overlay_refs)
        positive_relation_keys = {
            (
                frozenset(
                    (relation.subject_track_id, relation.object_track_id)
                ),
                relation.frame_index,
            )
            for relation in self.summary.relations
            if max(
                relation.bbox_iou,
                relation.subject_bbox_covered_fraction,
                relation.object_bbox_covered_fraction,
            )
            > 0.0
        }
        for candidate in self.candidates:
            if (
                track_entities.get(candidate.target_track_id)
                != candidate.target_entity_id
            ):
                raise ValueError("candidate target provenance is not closed to summary")
            if any(
                track_entities.get(item.track_id) != item.entity_id
                for item in candidate.possible_occluders
            ):
                raise ValueError("candidate occluder provenance is not closed to summary")
            if any(
                (
                    frozenset((candidate.target_track_id, item.track_id)),
                    frame_index,
                )
                not in positive_relation_keys
                for item in candidate.possible_occluders
                for frame_index in item.supporting_frames
            ):
                raise ValueError("candidate occluder lacks positive relation support")
            if any(
                timestamp not in observed_times
                for timestamp in (
                    *candidate.allowed_start_times,
                    *candidate.allowed_end_times,
                )
            ):
                raise ValueError("candidate times are not closed to the observed clock")
            if any(
                reference not in summary_overlays
                for reference in candidate.overlay_refs
            ):
                raise ValueError("candidate overlay is not closed to summary")
            target_track = tracks_by_id[candidate.target_track_id]
            if candidate.observation_support_complete != (
                target_track.candidate_search_complete
            ):
                raise ValueError("candidate observation completeness disagrees with summary")
            if (
                candidate.relation_support_complete
                and not self.summary.relations_complete
            ):
                raise ValueError("candidate relation completeness exceeds summary")
            if (
                candidate.source_search_complete
                != self.summary.candidate_search_complete
            ):
                raise ValueError("candidate source-search completeness disagrees")
        expected = _expected_bundle_components(
            self.summary,
            self.thresholds,
            candidate_limit=self.candidate_limit,
            prompt_char_limit=self.prompt_char_limit,
        )
        if (
            self.candidates != expected.candidates
            or self.source_search_complete != expected.source_search_complete
            or self.candidates_complete != expected.candidates_complete
            or self.truncation_codes != expected.truncation_codes
        ):
            raise ValueError("bundle must contain the canonical candidate set")
        if (
            _canonical_char_count(
                _bundle_projection(self), maximum=self.prompt_char_limit
            )
            > self.prompt_char_limit
        ):
            raise ValueError("aggregate CV prompt record exceeds its character cap")
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return one fresh, allowlisted, aggregate record for a model job."""
        _preflight_bundle(self)
        validated = _revalidate_bundle(self)
        return _materialize_bounded_record(
            _bundle_projection(validated),
            maximum=validated.prompt_char_limit,
            message="aggregate CV prompt record exceeds its character cap",
        )


@dataclass(frozen=True, slots=True)
class _AssemblyMetadata:
    total_entities: int
    kept_entities: int
    omitted_aliases: int
    truncated_texts: int
    total_tracks: int
    kept_tracks: int
    total_observations: int
    kept_observations: int
    mandatory_conflicts: int
    total_gaps: int
    kept_gaps: int
    total_visibility_runs: int
    kept_visibility_runs: int
    total_relations: int
    kept_relations: int
    total_overlays: int
    kept_overlays: int
    artifact_warning_count: int
    uncovered_entity_count: int
    unavailable_evidence_count: int


@dataclass(frozen=True, slots=True)
class _CandidateDraft:
    target_track_id: str
    target_entity_id: str
    allowed_start_times: tuple[float, ...]
    allowed_end_times: tuple[float, ...]
    last_visible_frame: int
    first_revisible_frame: int | None
    edge_departure: bool
    low_confidence: bool
    evidence_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ExpectedBundleComponents:
    candidates: tuple[OcclusionCandidate, ...]
    source_search_complete: bool
    candidates_complete: bool
    truncation_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CanonicalObject:
    """A lazy JSON object whose values may reference bounded model tuples."""

    fields: tuple[tuple[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _CanonicalArray:
    """A replayable lazy JSON array with an optional per-item projection."""

    values: tuple[Any, ...]
    projector: Any = None


def summarize_cv_evidence(
    artifact: CvEvidenceArtifact,
    *,
    timeline: FrameTimeline | None = None,
    max_tracks: int = _DEFAULT_MAX_TRACKS,
    max_observations_per_track: int = _DEFAULT_MAX_OBSERVATIONS_PER_TRACK,
    max_relations: int = _DEFAULT_MAX_RELATIONS,
    max_overlays: int = _DEFAULT_MAX_OVERLAYS,
    max_prompt_chars: int = _MAX_PROMPT_CHARS,
) -> CvEvidenceSummary:
    """Validate and reduce one artifact to finite, deterministic prompt evidence.

    Mandatory observation priority is first, last, chronological visibility
    change boundaries, minimum area, maximum area, then lowest confidence.  If
    the observation cap is smaller than that distinct set, later priorities are
    dropped and an explicit warning is retained.
    """
    _validate_limit("max_tracks", max_tracks, maximum=_MAX_TRACK_LIMIT)
    _validate_limit(
        "max_observations_per_track",
        max_observations_per_track,
        maximum=_MAX_OBSERVATION_LIMIT,
    )
    _validate_limit("max_relations", max_relations, maximum=_MAX_RELATION_LIMIT)
    _validate_limit("max_overlays", max_overlays, maximum=_MAX_OVERLAY_LIMIT)
    _validate_limit("max_prompt_chars", max_prompt_chars, maximum=_MAX_PROMPT_CHARS)

    _preflight_artifact(artifact)
    if timeline is not None:
        _preflight_timeline(timeline)
    validated_artifact = _revalidate_artifact(artifact)
    validated_timeline = (
        _revalidate_timeline(timeline) if timeline is not None else None
    )
    frame_clock = _validated_frame_clock(validated_artifact, validated_timeline)
    _validate_identifier_bounds(validated_artifact)

    track_cap = min(max_tracks, len(validated_artifact.tracks))
    observation_cap = max_observations_per_track
    relation_cap = max_relations
    overlay_cap = max_overlays
    alias_cap = _MAX_ALIASES_PER_ENTITY
    include_untracked_entities = True
    prompt_truncated = False

    def assemble() -> CvEvidenceSummary:
        return _assemble_summary(
            validated_artifact,
            frame_clock,
            track_cap=track_cap,
            observation_cap=observation_cap,
            relation_cap=relation_cap,
            overlay_cap=overlay_cap,
            alias_cap=alias_cap,
            include_untracked_entities=include_untracked_entities,
            prompt_char_limit=max_prompt_chars,
            prompt_truncated=prompt_truncated,
        )

    summary = assemble()
    if _summary_fits(summary, max_prompt_chars):
        summary.prompt_record()
        return summary

    prompt_truncated = True
    relation_cap = 0
    summary = assemble()
    if _summary_fits(summary, max_prompt_chars):
        summary.prompt_record()
        return summary

    overlay_cap = 0
    summary = assemble()
    if _summary_fits(summary, max_prompt_chars):
        summary.prompt_record()
        return summary

    alias_cap = 0
    include_untracked_entities = False
    summary = assemble()
    if _summary_fits(summary, max_prompt_chars):
        summary.prompt_record()
        return summary

    # Preserve the three-observation context needed to identify a local quality
    # dip before dropping whole, stably ordered tracks.  Both cap searches are
    # logarithmic, so hostile-but-valid inputs cannot trigger hundreds of full
    # summary rebuilds while the prompt budget is being reduced.
    original_track_cap = track_cap
    low = 3
    high = observation_cap - 1
    best_observation_summary: CvEvidenceSummary | None = None
    while low <= high:
        candidate_cap = (low + high) // 2
        observation_cap = candidate_cap
        candidate_summary = assemble()
        if _summary_fits(candidate_summary, max_prompt_chars):
            best_observation_summary = candidate_summary
            low = candidate_cap + 1
        else:
            high = candidate_cap - 1
    if best_observation_summary is not None:
        best_observation_summary.prompt_record()
        return best_observation_summary

    observation_cap = min(3, max_observations_per_track)
    low = 1
    high = original_track_cap - 1
    best_track_summary: CvEvidenceSummary | None = None
    while low <= high:
        candidate_cap = (low + high) // 2
        track_cap = candidate_cap
        candidate_summary = assemble()
        if _summary_fits(candidate_summary, max_prompt_chars):
            best_track_summary = candidate_summary
            low = candidate_cap + 1
        else:
            high = candidate_cap - 1
    if best_track_summary is not None:
        best_track_summary.prompt_record()
        return best_track_summary

    track_cap = min(1, original_track_cap)
    for reduced_observation_cap in range(min(observation_cap, 2), 0, -1):
        observation_cap = reduced_observation_cap
        summary = assemble()
        if _summary_fits(summary, max_prompt_chars):
            summary.prompt_record()
            return summary

    track_cap = 0
    summary = assemble()
    if _summary_fits(summary, max_prompt_chars):
        summary.prompt_record()
        return summary

    raise ValueError("CV evidence summary cannot fit the prompt character cap")


def build_occlusion_candidates(
    summary: CvEvidenceSummary,
    thresholds: EvidenceThresholds,
) -> tuple[OcclusionCandidate, ...]:
    """Propose visibility-change skeletons without semantic classification."""
    _preflight_summary(summary)
    _preflight_thresholds(thresholds)
    validated_summary = _revalidate_summary(summary)
    validated_thresholds = _revalidate_thresholds(thresholds)
    candidates, truncated = _canonical_candidates(
        validated_summary, validated_thresholds
    )
    if truncated:
        raise ValueError(
            "candidate set exceeds the public bound; use build_cv_prompt_bundle"
        )
    return candidates


def _canonical_candidates(
    summary: CvEvidenceSummary,
    thresholds: EvidenceThresholds,
) -> tuple[tuple[OcclusionCandidate, ...], bool]:
    if summary.status is not EvidenceStatus.AVAILABLE:
        return (), False
    tracks = tuple(
        sorted(summary.tracks, key=lambda item: item.track_id)
    )
    bounded_drafts = heapq.nsmallest(
        _MAX_BUNDLE_CANDIDATES + 1,
        _candidate_drafts(tracks, thresholds),
        key=_draft_sort_key,
    )
    set_truncated = len(bounded_drafts) > _MAX_BUNDLE_CANDIDATES
    ordered_drafts = tuple(bounded_drafts[:_MAX_BUNDLE_CANDIDATES])
    entities = {item.entity_id: item for item in summary.entities}
    track_entities = {item.track_id: item.entity_id for item in tracks}
    candidates: list[OcclusionCandidate] = []
    for ordinal, draft in enumerate(ordered_drafts, start=1):
        possible_occluders, occluder_support_complete = _possible_occluders(
            draft,
            summary.relations,
            entities,
            track_entities,
        )
        possible_occluder_entity_ids = tuple(
            sorted({item.entity_id for item in possible_occluders})
        )
        overlay_fields = _candidate_overlay_fields(
            summary.overlays,
            draft.target_track_id,
            possible_occluders,
            draft.evidence_frames,
            overlays_complete=summary.overlays_complete,
        )
        observation_support_complete = next(
            track.candidate_search_complete
            for track in tracks
            if track.track_id == draft.target_track_id
        )
        relation_support_complete = (
            summary.relations_complete and occluder_support_complete
        )
        support_complete = (
            observation_support_complete
            and relation_support_complete
            and bool(overlay_fields["overlay_support_complete"])
        )
        values: dict[str, Any] = dict(
            target_entity_id=draft.target_entity_id,
            target_track_id=draft.target_track_id,
            possible_occluders=possible_occluders,
            possible_occluder_entity_ids=possible_occluder_entity_ids,
            allowed_start_times=draft.allowed_start_times,
            allowed_end_times=draft.allowed_end_times,
            last_visible_frame=draft.last_visible_frame,
            first_revisible_frame=draft.first_revisible_frame,
            edge_departure=draft.edge_departure,
            low_confidence=draft.low_confidence,
            observation_support_complete=observation_support_complete,
            relation_support_complete=relation_support_complete,
            support_complete=support_complete,
            source_search_complete=summary.candidate_search_complete,
            **overlay_fields,
        )
        provisional = OcclusionCandidate.model_construct(
            candidate_id=f"occ_{'0' * 12}_{ordinal:04d}",
            **values,
        )
        candidate = OcclusionCandidate(
            candidate_id=_candidate_identity(provisional, ordinal),
            **values,
        )
        candidates.append(candidate)
    return tuple(candidates), set_truncated


def _candidate_drafts(
    tracks: tuple[SummaryTrack, ...],
    thresholds: EvidenceThresholds,
) -> Iterator[_CandidateDraft]:
    for track in tracks:
        for gap in sorted(
            track.missing_intervals,
            key=lambda item: (item.first_missing_frame, item.last_missing_frame),
        ):
            yield _CandidateDraft(
                target_track_id=track.track_id,
                target_entity_id=track.entity_id,
                allowed_start_times=(gap.last_visible_time,),
                allowed_end_times=(
                    gap.first_revisible_time
                    if gap.first_revisible_time is not None
                    else gap.last_missing_time,
                ),
                last_visible_frame=gap.last_visible_frame,
                first_revisible_frame=gap.first_revisible_frame,
                edge_departure=gap.edge_departure,
                low_confidence=(
                    gap.minimum_confidence is not None
                    and gap.minimum_confidence < thresholds.min_confidence
                ),
                evidence_frames=_ordered_frames(
                    gap.last_visible_frame,
                    gap.first_missing_frame,
                    gap.last_missing_frame,
                    gap.first_revisible_frame,
                ),
            )
        yield from _uncertain_observation_drafts(track, thresholds)


def build_cv_prompt_bundle(
    summary: CvEvidenceSummary,
    thresholds: EvidenceThresholds,
    *,
    max_candidates: int = _MAX_BUNDLE_CANDIDATES,
    max_prompt_chars: int | None = None,
) -> CvPromptBundle:
    """Build a deterministic, whole-prompt-capped semantic-stage input."""
    _validate_limit(
        "max_candidates", max_candidates, maximum=_MAX_BUNDLE_CANDIDATES
    )
    _preflight_summary(summary)
    _preflight_thresholds(thresholds)
    validated_summary = _revalidate_summary(summary)
    validated_thresholds = _revalidate_thresholds(thresholds)
    effective_prompt_limit = (
        validated_summary.prompt_char_limit
        if max_prompt_chars is None
        else max_prompt_chars
    )
    _validate_limit(
        "max_prompt_chars",
        effective_prompt_limit,
        maximum=min(_MAX_PROMPT_CHARS, validated_summary.prompt_char_limit),
    )
    expected = _expected_bundle_components(
        validated_summary,
        validated_thresholds,
        candidate_limit=max_candidates,
        prompt_char_limit=effective_prompt_limit,
    )
    return CvPromptBundle(
        schema_version="cv_prompt_bundle_v1",
        summary=validated_summary,
        thresholds=validated_thresholds,
        candidates=expected.candidates,
        source_search_complete=expected.source_search_complete,
        candidates_complete=expected.candidates_complete,
        truncation_codes=expected.truncation_codes,
        prompt_char_limit=effective_prompt_limit,
        candidate_limit=max_candidates,
    )


def _expected_bundle_components(
    summary: CvEvidenceSummary,
    thresholds: EvidenceThresholds,
    *,
    candidate_limit: int,
    prompt_char_limit: int,
) -> _ExpectedBundleComponents:
    candidates, generation_truncated = _canonical_candidates(summary, thresholds)
    count_truncated = len(candidates) > candidate_limit
    initially_kept = candidates[:candidate_limit]
    source_search_complete = summary.candidate_search_complete

    def codes(*, prompt_truncated: bool) -> tuple[str, ...]:
        return tuple(
            code
            for code, enabled in (
                (
                    "SUMMARY_CANDIDATE_SEARCH_INCOMPLETE",
                    not source_search_complete,
                ),
                ("CANDIDATE_SOURCE_TRUNCATED", generation_truncated),
                ("CANDIDATE_COUNT_TRUNCATED", count_truncated),
                ("CANDIDATE_PROMPT_TRUNCATED", prompt_truncated),
            )
            if enabled
        )

    initial_codes = codes(prompt_truncated=False)
    initial_complete = not (generation_truncated or count_truncated)
    candidate_char_counts = tuple(
        _canonical_char_count(_candidate_projection(candidate))
        for candidate in initially_kept
    )
    initial_total = _bundle_empty_candidates_char_count(
        summary,
        thresholds,
        source_search_complete=source_search_complete,
        candidates_complete=initial_complete,
        truncation_codes=initial_codes,
    )
    initial_total += sum(candidate_char_counts)
    initial_total += max(0, len(candidate_char_counts) - 1)
    if initial_total <= prompt_char_limit:
        return _ExpectedBundleComponents(
            candidates=initially_kept,
            source_search_complete=source_search_complete,
            candidates_complete=initial_complete,
            truncation_codes=initial_codes,
        )

    prompt_codes = codes(prompt_truncated=True)
    prompt_total = _bundle_empty_candidates_char_count(
        summary,
        thresholds,
        source_search_complete=source_search_complete,
        candidates_complete=False,
        truncation_codes=prompt_codes,
    )
    if prompt_total > prompt_char_limit:
        raise ValueError("CV summary leaves no room for an aggregate prompt bundle")
    kept_count = 0
    for candidate_chars in candidate_char_counts:
        added_chars = candidate_chars + (1 if kept_count else 0)
        if prompt_total + added_chars > prompt_char_limit:
            break
        prompt_total += added_chars
        kept_count += 1
    return _ExpectedBundleComponents(
        candidates=initially_kept[:kept_count],
        source_search_complete=source_search_complete,
        candidates_complete=False,
        truncation_codes=prompt_codes,
    )


def _structural_error(name: str) -> ValueError:
    return ValueError(f"{name} exceeds the CV structural input bound")


def _preflight_tuple(value: object, name: str, maximum: int) -> tuple[Any, ...]:
    if type(value) is not tuple or len(value) > maximum:
        raise _structural_error(name)
    return value


def _preflight_text(value: object, name: str) -> str:
    if type(value) is not str or len(value) > _ABS_MAX_INPUT_STRING_CHARS:
        raise _structural_error(name)
    return value


def _preflight_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_CANONICAL_INT,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or value > maximum
    ):
        raise _structural_error(name)
    return value


def _preflight_float(
    value: object,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value) or value < minimum:
        raise _structural_error(name)
    if maximum is not None and value > maximum:
        raise _structural_error(name)
    return value


def _preflight_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise _structural_error(name)
    return value


def _preflight_frame(frame: object, name: str) -> None:
    if not isinstance(frame, FrameTimestamp):
        raise _structural_error(f"{name} item type")
    _preflight_int(frame.frame_index, f"{name} frame_index")
    _preflight_float(frame.timestamp_seconds, f"{name} timestamp_seconds")


def _preflight_artifact(artifact: object) -> None:
    if not isinstance(artifact, CvEvidenceArtifact):
        raise _structural_error("artifact")
    for field_name in (
        "schema_version",
        "provider",
        "model_identity",
        "video_sha256",
        "checkpoint_sha256",
    ):
        _preflight_text(getattr(artifact, field_name, None), f"artifact.{field_name}")
    if not isinstance(getattr(artifact, "status", None), EvidenceStatus):
        raise _structural_error("artifact.status")
    processed_timeline = getattr(artifact, "processed_timeline", None)
    if processed_timeline is not None:
        if not isinstance(processed_timeline, FrameTimeline):
            raise _structural_error("artifact.processed_timeline")
        processed_frames = _preflight_tuple(
            processed_timeline.frames,
            "artifact.processed_timeline.frames",
            _ABS_MAX_PROCESSED_FRAMES,
        )
        for frame in processed_frames:
            _preflight_frame(frame, "artifact.processed_timeline")
    entities = _preflight_tuple(
        getattr(artifact, "entities", None),
        "artifact.entities",
        _ABS_MAX_ARTIFACT_ENTITIES,
    )
    for entity in entities:
        if not isinstance(entity, EntityPrompt):
            raise _structural_error("artifact.entities item type")
        _preflight_text(entity.entity_id, "artifact entity_id")
        _preflight_text(entity.canonical_label, "artifact canonical_label")
        if not isinstance(entity.role, EntityRole):
            raise _structural_error("artifact entity role")
        aliases = _preflight_tuple(
            entity.aliases,
            "artifact entity aliases",
            _ABS_MAX_ALIASES_PER_INPUT_ENTITY,
        )
        for alias in aliases:
            _preflight_text(alias, "artifact alias")

    tracks = _preflight_tuple(
        getattr(artifact, "tracks", None),
        "artifact.tracks",
        _ABS_MAX_ARTIFACT_TRACKS,
    )
    total_observations = 0
    for track in tracks:
        if not isinstance(track, CvTrack):
            raise _structural_error("artifact.tracks item type")
        _preflight_text(track.track_id, "artifact track_id")
        _preflight_text(track.entity_id, "artifact track entity_id")
        if not isinstance(track.status, EvidenceStatus):
            raise _structural_error("artifact track status")
        observations = _preflight_tuple(
            track.observations,
            "artifact track observations",
            _ABS_MAX_OBSERVATIONS_PER_INPUT_TRACK,
        )
        total_observations += len(observations)
        if total_observations > _ABS_MAX_TOTAL_INPUT_OBSERVATIONS:
            raise _structural_error("artifact total observations")
        for observation in observations:
            if not isinstance(observation, TrackObservation):
                raise _structural_error("artifact observations item type")
            _preflight_tuple(
                observation.bbox_xyxy, "artifact observation bbox", 4
            )
            _preflight_tuple(
                observation.center_xy, "artifact observation center", 2
            )
            _preflight_int(
                observation.frame_index, "artifact observation frame_index"
            )
            _preflight_float(
                observation.timestamp_seconds,
                "artifact observation timestamp_seconds",
            )
            for value in (*observation.bbox_xyxy, *observation.center_xy):
                _preflight_float(
                    value, "artifact observation geometry", maximum=1.0
                )
            _preflight_bool(observation.visible, "artifact observation visible")
            _preflight_float(
                observation.confidence,
                "artifact observation confidence",
                maximum=1.0,
            )
            _preflight_float(
                observation.area_fraction,
                "artifact observation area_fraction",
                maximum=1.0,
            )
            if observation.mask_ref is not None:
                _preflight_text(observation.mask_ref, "artifact mask_ref")

    files = _preflight_tuple(
        getattr(artifact, "files", None),
        "artifact.files",
        _ABS_MAX_ARTIFACT_FILES,
    )
    for artifact_file in files:
        if not isinstance(artifact_file, ArtifactFile):
            raise _structural_error("artifact.files item type")
        _preflight_text(artifact_file.path, "artifact file path")
        _preflight_text(artifact_file.sha256, "artifact file sha256")
        _preflight_int(artifact_file.size_bytes, "artifact file size_bytes")
    overlay_records = _preflight_tuple(
        getattr(artifact, "overlay_records", None),
        "artifact.overlay_records",
        _MAX_OVERLAY_LIMIT,
    )
    for record in overlay_records:
        if not isinstance(record, OverlayRecord):
            raise _structural_error("artifact.overlay_records item type")
        _preflight_text(record.path, "artifact overlay path")
        _preflight_text(record.track_id, "artifact overlay track_id")
        _preflight_int(record.frame_index, "artifact overlay frame_index")
    artifact_warnings = _preflight_tuple(
        getattr(artifact, "warnings", None),
        "artifact.warnings",
        _ABS_MAX_ARTIFACT_WARNINGS,
    )
    for warning in artifact_warnings:
        _preflight_text(warning, "artifact warning")


def _preflight_timeline(timeline: object) -> None:
    if not isinstance(timeline, FrameTimeline):
        raise _structural_error("timeline")
    frames = _preflight_tuple(
        getattr(timeline, "frames", None),
        "timeline.frames",
        _ABS_MAX_TIMELINE_FRAMES,
    )
    for frame in frames:
        _preflight_frame(frame, "timeline.frames")


def _preflight_summary(summary: object) -> None:
    if not isinstance(summary, CvEvidenceSummary):
        raise _structural_error("summary")
    _preflight_text(
        getattr(summary, "schema_version", None), "summary.schema_version"
    )
    _preflight_text(getattr(summary, "summary_id", None), "summary.summary_id")
    observed_clock = _preflight_tuple(
        getattr(summary, "observed_clock", None),
        "summary.observed_clock",
        _ABS_MAX_PROCESSED_FRAMES,
    )
    if any(not isinstance(item, FrameTimestamp) for item in observed_clock):
        raise _structural_error("summary.observed_clock item type")
    for frame in observed_clock:
        _preflight_frame(frame, "summary.observed_clock")
    if not isinstance(getattr(summary, "status", None), EvidenceStatus):
        raise _structural_error("summary.status")
    _preflight_bool(
        getattr(summary, "candidate_search_complete", None),
        "summary.candidate_search_complete",
    )
    entities = _preflight_tuple(
        getattr(summary, "entities", None),
        "summary.entities",
        _MAX_ENTITY_SUMMARIES,
    )
    for entity in entities:
        if not isinstance(entity, SummaryEntity):
            raise _structural_error("summary.entities item type")
        _preflight_text(entity.entity_id, "summary entity_id")
        _preflight_text(entity.canonical_label, "summary canonical_label")
        if not isinstance(entity.role, EntityRole):
            raise _structural_error("summary entity role")
        aliases = _preflight_tuple(
            entity.aliases, "summary entity aliases", _MAX_ALIASES_PER_ENTITY
        )
        for alias in aliases:
            _preflight_text(alias, "summary alias")

    tracks = _preflight_tuple(
        getattr(summary, "tracks", None), "summary.tracks", _MAX_TRACK_LIMIT
    )
    for track in tracks:
        if not isinstance(track, SummaryTrack):
            raise _structural_error("summary.tracks item type")
        _preflight_text(track.track_id, "summary track_id")
        _preflight_text(track.entity_id, "summary track entity_id")
        if not isinstance(track.status, EvidenceStatus):
            raise _structural_error("summary track status")
        _preflight_int(
            track.source_observation_count,
            "summary track source_observation_count",
        )
        _preflight_bool(
            track.visibility_lifecycle_complete,
            "summary track visibility_lifecycle_complete",
        )
        _preflight_bool(
            track.candidate_search_complete,
            "summary track candidate_search_complete",
        )
        observations = _preflight_tuple(
            track.observations,
            "summary track observations",
            _MAX_OBSERVATION_LIMIT,
        )
        for observation in observations:
            if not isinstance(observation, SummaryObservation):
                raise _structural_error("summary observations item type")
            _preflight_tuple(observation.bbox_xyxy, "summary observation bbox", 4)
            _preflight_tuple(observation.center_xy, "summary observation center", 2)
            _preflight_int(
                observation.source_ordinal, "summary observation source_ordinal"
            )
            _preflight_int(
                observation.frame_index, "summary observation frame_index"
            )
            _preflight_float(
                observation.timestamp_seconds,
                "summary observation timestamp_seconds",
            )
            for value in (
                *observation.bbox_xyxy,
                *observation.center_xy,
                observation.confidence,
                observation.area_fraction,
                observation.edge_proximity,
            ):
                _preflight_float(
                    value, "summary observation scalar", maximum=1.0
                )
            _preflight_bool(observation.visible, "summary observation visible")
        runs = _preflight_tuple(
            track.visibility_runs,
            "summary visibility runs",
            _MAX_VISIBILITY_RUNS_PER_TRACK,
        )
        for run in runs:
            if not isinstance(run, VisibilityRun):
                raise _structural_error("summary visibility run item type")
            _preflight_text(run.state, "summary visibility run state")
            _preflight_int(run.start_frame, "summary visibility run start_frame")
            _preflight_float(run.start_time, "summary visibility run start_time")
            _preflight_int(run.end_frame, "summary visibility run end_frame")
            _preflight_float(run.end_time, "summary visibility run end_time")
            if run.minimum_confidence is not None:
                _preflight_float(
                    run.minimum_confidence,
                    "summary visibility run minimum_confidence",
                    maximum=1.0,
                )
        gaps = _preflight_tuple(
            track.missing_intervals,
            "summary missing intervals",
            _MAX_OBSERVATION_LIMIT,
        )
        if any(not isinstance(item, VisibilityGap) for item in gaps):
            raise _structural_error("summary missing intervals item type")
        for gap in gaps:
            for field_name in (
                "last_visible_frame",
                "first_missing_frame",
                "last_missing_frame",
            ):
                _preflight_int(
                    getattr(gap, field_name), f"summary gap {field_name}"
                )
            for field_name in (
                "last_visible_time",
                "first_missing_time",
                "last_missing_time",
            ):
                _preflight_float(
                    getattr(gap, field_name), f"summary gap {field_name}"
                )
            if gap.first_revisible_frame is not None:
                _preflight_int(
                    gap.first_revisible_frame,
                    "summary gap first_revisible_frame",
                )
            if gap.first_revisible_time is not None:
                _preflight_float(
                    gap.first_revisible_time,
                    "summary gap first_revisible_time",
                )
            if gap.minimum_confidence is not None:
                _preflight_float(
                    gap.minimum_confidence,
                    "summary gap minimum_confidence",
                    maximum=1.0,
                )
            _preflight_bool(gap.edge_departure, "summary gap edge_departure")

    relations = _preflight_tuple(
        getattr(summary, "relations", None),
        "summary.relations",
        _MAX_RELATION_LIMIT,
    )
    for relation in relations:
        if not isinstance(relation, SpatialRelation):
            raise _structural_error("summary.relations item type")
        _preflight_text(
            relation.subject_track_id, "summary relation subject_track_id"
        )
        _preflight_text(
            relation.object_track_id, "summary relation object_track_id"
        )
        _preflight_int(relation.frame_index, "summary relation frame_index")
        _preflight_float(
            relation.timestamp_seconds, "summary relation timestamp_seconds"
        )
        for value in (
            relation.bbox_iou,
            relation.subject_bbox_covered_fraction,
            relation.object_bbox_covered_fraction,
            relation.center_distance_fraction,
            relation.area_similarity,
        ):
            _preflight_float(value, "summary relation scalar", maximum=1.0)
        _preflight_bool(
            relation.subject_inside_object,
            "summary relation subject_inside_object",
        )
        _preflight_bool(
            relation.object_inside_subject,
            "summary relation object_inside_subject",
        )
    overlays = _preflight_tuple(
        getattr(summary, "overlay_refs", None),
        "summary.overlay_refs",
        _MAX_OVERLAY_LIMIT,
    )
    for overlay in overlays:
        _preflight_text(overlay, "summary overlay reference")
    structured_overlays = _preflight_tuple(
        getattr(summary, "overlays", None),
        "summary.overlays",
        _MAX_OVERLAY_LIMIT,
    )
    for overlay in structured_overlays:
        if not isinstance(overlay, SummaryOverlay):
            raise _structural_error("summary.overlays item type")
        _preflight_text(overlay.path, "summary overlay path")
        _preflight_text(overlay.track_id, "summary overlay track_id")
        _preflight_int(overlay.frame_index, "summary overlay frame_index")
    summary_warnings = _preflight_tuple(
        getattr(summary, "warnings", None),
        "summary.warnings",
        _ABS_MAX_SUMMARY_WARNINGS,
    )
    for warning in summary_warnings:
        if not isinstance(warning, SummaryWarning):
            raise _structural_error("summary warning item type")
        _preflight_text(warning.code, "summary warning code")
        for field_name in ("kept", "omitted", "count", "tracks"):
            value = getattr(warning, field_name)
            if value is not None:
                _preflight_int(value, f"summary warning {field_name}")
        if warning.priority is not None:
            _preflight_text(warning.priority, "summary warning priority")
        if warning.budget_enforced is not None:
            _preflight_bool(
                warning.budget_enforced, "summary warning budget_enforced"
            )
    _preflight_bool(summary.relations_complete, "summary.relations_complete")
    _preflight_bool(summary.overlays_complete, "summary.overlays_complete")
    _preflight_int(summary.prompt_char_limit, "summary.prompt_char_limit", minimum=1)


def _preflight_candidate(candidate: object) -> None:
    if not isinstance(candidate, OcclusionCandidate):
        raise _structural_error("candidate item type")
    _preflight_text(candidate.candidate_id, "candidate_id")
    _preflight_text(candidate.target_entity_id, "candidate target_entity_id")
    _preflight_text(candidate.target_track_id, "candidate target_track_id")
    possible_occluders = _preflight_tuple(
        candidate.possible_occluders,
        "candidate possible_occluders",
        _MAX_CANDIDATE_OCCLUDERS,
    )
    for provenance in possible_occluders:
        if not isinstance(provenance, OccluderProvenance):
            raise _structural_error("candidate possible_occluders item type")
        _preflight_text(provenance.entity_id, "candidate occluder entity_id")
        _preflight_text(provenance.track_id, "candidate occluder track_id")
        supporting_frames = _preflight_tuple(
            provenance.supporting_frames,
            "candidate occluder supporting_frames",
            _MAX_CANDIDATE_SUPPORTING_FRAMES,
        )
        for frame_index in supporting_frames:
            _preflight_int(frame_index, "candidate occluder supporting frame")
    possible_entity_ids = _preflight_tuple(
        candidate.possible_occluder_entity_ids,
        "candidate possible_occluder_entity_ids",
        _MAX_CANDIDATE_OCCLUDERS,
    )
    for entity_id in possible_entity_ids:
        _preflight_text(entity_id, "candidate possible occluder entity_id")
    _preflight_tuple(candidate.allowed_start_times, "candidate start times", 8)
    _preflight_tuple(candidate.allowed_end_times, "candidate end times", 8)
    overlays = _preflight_tuple(
        candidate.overlay_refs,
        "candidate overlay_refs",
        _MAX_OVERLAY_LIMIT,
    )
    for overlay in overlays:
        _preflight_text(overlay, "candidate overlay reference")
    for value in (*candidate.allowed_start_times, *candidate.allowed_end_times):
        _preflight_float(value, "candidate boundary timestamp")
    _preflight_int(candidate.last_visible_frame, "candidate last_visible_frame")
    if candidate.first_revisible_frame is not None:
        _preflight_int(
            candidate.first_revisible_frame,
            "candidate first_revisible_frame",
        )
    for field_name in (
        "edge_departure",
        "low_confidence",
        "observation_support_complete",
        "relation_support_complete",
        "overlay_support_complete",
        "support_complete",
        "source_search_complete",
    ):
        _preflight_bool(getattr(candidate, field_name), f"candidate {field_name}")


def _preflight_thresholds(thresholds: object) -> None:
    if not isinstance(thresholds, EvidenceThresholds):
        raise _structural_error("thresholds")
    for field_name in (
        "min_confidence",
        "min_area_fraction",
        "occlusion_visibility_drop",
    ):
        _preflight_float(
            getattr(thresholds, field_name),
            f"thresholds.{field_name}",
            maximum=1.0,
        )


def _preflight_bundle(bundle: object) -> None:
    if not isinstance(bundle, CvPromptBundle):
        raise _structural_error("bundle")
    _preflight_text(bundle.schema_version, "bundle.schema_version")
    _preflight_summary(bundle.summary)
    _preflight_thresholds(bundle.thresholds)
    candidates = _preflight_tuple(
        bundle.candidates, "bundle.candidates", _MAX_BUNDLE_CANDIDATES
    )
    aggregate_components = 0
    for candidate in candidates:
        aggregate_components += _candidate_component_count(candidate)
        if aggregate_components > _MAX_TOTAL_CANDIDATE_COMPONENTS:
            raise _structural_error("aggregate candidate components")
    for candidate in candidates:
        _preflight_candidate(candidate)
    codes = _preflight_tuple(bundle.truncation_codes, "bundle.truncation_codes", 4)
    for code in codes:
        _preflight_text(code, "bundle truncation code")
    _preflight_bool(bundle.source_search_complete, "bundle.source_search_complete")
    _preflight_bool(bundle.candidates_complete, "bundle.candidates_complete")
    _preflight_int(bundle.prompt_char_limit, "bundle.prompt_char_limit", minimum=1)
    _preflight_int(bundle.candidate_limit, "bundle.candidate_limit", minimum=1)


def _candidate_component_count(candidate: object) -> int:
    """Count bounded nested shapes without traversing supporting frame values."""
    if isinstance(candidate, OcclusionCandidate):
        get_candidate = lambda name: getattr(candidate, name)
    elif type(candidate) is dict:
        if any(
            type(name) is not str or name not in OcclusionCandidate.model_fields
            for name in candidate
        ):
            raise _structural_error("candidate extra fields")
        get_candidate = lambda name: candidate.get(name)
    else:
        raise _structural_error("candidate item type")
    possible_occluders = _shape_sequence(
        get_candidate("possible_occluders"),
        "candidate possible_occluders",
        _MAX_CANDIDATE_OCCLUDERS,
    )
    count = len(possible_occluders)
    for provenance in possible_occluders:
        if isinstance(provenance, OccluderProvenance):
            supporting_frames = provenance.supporting_frames
        elif type(provenance) is dict:
            if any(
                type(name) is not str
                or name not in OccluderProvenance.model_fields
                for name in provenance
            ):
                raise _structural_error("candidate provenance extra fields")
            supporting_frames = provenance.get("supporting_frames")
        else:
            raise _structural_error("candidate possible_occluders item type")
        count += len(
            _shape_sequence(
                supporting_frames,
                "candidate occluder supporting_frames",
                _MAX_CANDIDATE_SUPPORTING_FRAMES,
            )
        )
    for name, values, maximum in (
        (
            "candidate possible_occluder_entity_ids",
            get_candidate("possible_occluder_entity_ids"),
            _MAX_CANDIDATE_OCCLUDERS,
        ),
        ("candidate start times", get_candidate("allowed_start_times"), 8),
        ("candidate end times", get_candidate("allowed_end_times"), 8),
        (
            "candidate overlay_refs",
            get_candidate("overlay_refs"),
            _MAX_OVERLAY_LIMIT,
        ),
    ):
        count += len(_shape_sequence(values, name, maximum))
    return count


def _shape_sequence(
    value: object, name: str, maximum: int
) -> tuple[Any, ...] | list[Any]:
    if type(value) not in {tuple, list} or len(value) > maximum:
        raise _structural_error(name)
    return value


def _revalidate_artifact(artifact: CvEvidenceArtifact) -> CvEvidenceArtifact:
    return CvEvidenceArtifact.model_validate(
        {
            "schema_version": artifact.schema_version,
            "status": artifact.status.value,
            "provider": artifact.provider,
            "model_identity": artifact.model_identity,
            "video_sha256": artifact.video_sha256,
            "checkpoint_sha256": artifact.checkpoint_sha256,
            "processed_timeline": (
                _timeline_validation_record(artifact.processed_timeline)
                if artifact.processed_timeline is not None
                else None
            ),
            "entities": tuple(
                {
                    "entity_id": entity.entity_id,
                    "canonical_label": entity.canonical_label,
                    "aliases": entity.aliases,
                    "role": entity.role.value,
                }
                for entity in artifact.entities
            ),
            "tracks": tuple(
                {
                    "track_id": track.track_id,
                    "entity_id": track.entity_id,
                    "status": track.status.value,
                    "observations": tuple(
                        {
                            "frame_index": item.frame_index,
                            "timestamp_seconds": item.timestamp_seconds,
                            "bbox_xyxy": item.bbox_xyxy,
                            "mask_ref": item.mask_ref,
                            "visible": item.visible,
                            "confidence": item.confidence,
                            "area_fraction": item.area_fraction,
                            "center_xy": item.center_xy,
                        }
                        for item in track.observations
                    ),
                }
                for track in artifact.tracks
            ),
            "files": tuple(
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in artifact.files
            ),
            "overlay_records": tuple(
                {
                    "path": item.path,
                    "track_id": item.track_id,
                    "frame_index": item.frame_index,
                }
                for item in artifact.overlay_records
            ),
            "warnings": artifact.warnings,
        },
        strict=True,
    )


def _revalidate_timeline(timeline: FrameTimeline) -> FrameTimeline:
    return FrameTimeline.model_validate(
        _timeline_validation_record(timeline), strict=True
    )


def _timeline_validation_record(timeline: FrameTimeline) -> dict[str, Any]:
    return {
        "frames": tuple(
            {
                "frame_index": item.frame_index,
                "timestamp_seconds": item.timestamp_seconds,
            }
            for item in timeline.frames
        )
    }


def _revalidate_summary(summary: CvEvidenceSummary) -> CvEvidenceSummary:
    return CvEvidenceSummary.model_validate_json(
        _bounded_validation_json(
            public_projection=_summary_projection(summary),
            validation_projection=_summary_projection(summary, validation=True),
            maximum=summary.prompt_char_limit,
            message="CV summary prompt record exceeds its character cap",
        ),
        strict=True,
    )


def _revalidate_candidate(candidate: OcclusionCandidate) -> OcclusionCandidate:
    projection = _candidate_projection(candidate)
    return OcclusionCandidate.model_validate_json(
        _bounded_validation_json(
            public_projection=projection,
            validation_projection=projection,
            maximum=_MAX_CANDIDATE_PROMPT_CHARS,
            message="occlusion candidate prompt record exceeds its cap",
        ),
        strict=True,
    )


def _revalidate_bundle(bundle: CvPromptBundle) -> CvPromptBundle:
    return CvPromptBundle.model_validate_json(
        _bounded_validation_json(
            public_projection=_bundle_projection(bundle),
            validation_projection=_bundle_projection(bundle, validation=True),
            maximum=bundle.prompt_char_limit,
            message="aggregate CV prompt record exceeds its character cap",
        ),
        strict=True,
    )


def _revalidate_thresholds(thresholds: EvidenceThresholds) -> EvidenceThresholds:
    return EvidenceThresholds.model_validate(
        {
            "min_confidence": thresholds.min_confidence,
            "min_area_fraction": thresholds.min_area_fraction,
            "occlusion_visibility_drop": thresholds.occlusion_visibility_drop,
        },
        strict=True,
    )


def _validate_limit(name: str, value: int, *, maximum: int) -> None:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{name} must be a positive integer no greater than {maximum}")


def _validate_identifier_bounds(artifact: CvEvidenceArtifact) -> None:
    identifiers = [entity.entity_id for entity in artifact.entities]
    identifiers.extend(track.track_id for track in artifact.tracks)
    if any(len(value) > _MAX_IDENTIFIER_CHARS for value in identifiers):
        raise ValueError("CV evidence identifier exceeds the summary bound")


def _validated_frame_clock(
    artifact: CvEvidenceArtifact,
    timeline: FrameTimeline | None,
) -> tuple[FrameTimestamp, ...]:
    processed = (
        artifact.processed_timeline.frames
        if artifact.processed_timeline is not None
        else ()
    )
    if artifact.status is EvidenceStatus.AVAILABLE and not processed:
        raise ValueError("available artifact has no processed timeline")
    if timeline is not None:
        declared = {
            item.frame_index: item.timestamp_seconds for item in timeline.frames
        }
        if any(
            declared.get(frame.frame_index) != frame.timestamp_seconds
            for frame in processed
        ):
            raise ValueError(
                "processed timeline must be an exact subset of the source timeline"
            )
    return processed if artifact.status is EvidenceStatus.AVAILABLE else ()


def _assemble_summary(
    artifact: CvEvidenceArtifact,
    frame_clock: tuple[FrameTimestamp, ...],
    *,
    track_cap: int,
    observation_cap: int,
    relation_cap: int,
    overlay_cap: int,
    alias_cap: int,
    include_untracked_entities: bool,
    prompt_char_limit: int,
    prompt_truncated: bool,
) -> CvEvidenceSummary:
    all_tracks = tuple(
        sorted(
            artifact.tracks,
            key=lambda item: (
                item.status is not EvidenceStatus.AVAILABLE,
                item.track_id,
            ),
        )
    )
    selected_tracks = (
        all_tracks[:track_cap]
        if artifact.status is EvidenceStatus.AVAILABLE
        else ()
    )
    summary_tracks: list[SummaryTrack] = []
    mandatory_conflicts = 0
    total_gaps = 0
    kept_gaps = 0
    total_visibility_runs = 0
    kept_visibility_runs = 0
    overlay_frames_by_track: dict[str, set[int]] = {}
    for record in artifact.overlay_records:
        overlay_frames_by_track.setdefault(record.track_id, set()).add(
            record.frame_index
        )
    for track in selected_tracks:
        if track.status is not EvidenceStatus.AVAILABLE:
            summary_tracks.append(
                SummaryTrack(
                    track_id=track.track_id,
                    entity_id=track.entity_id,
                    status=track.status,
                    source_observation_count=len(track.observations),
                    observations=(),
                    visibility_runs=(),
                    visibility_lifecycle_complete=False,
                    missing_intervals=(),
                    candidate_search_complete=False,
                )
            )
            continue
        retained_runs, source_run_count, track_gap_count = _derive_visibility_runs(
            track,
            frame_clock,
            cap=min(observation_cap, _MAX_VISIBILITY_RUNS_PER_TRACK),
        )
        lifecycle_complete = len(retained_runs) == source_run_count
        lifecycle_boundary_frames = {
            frame_index
            for run in retained_runs
            for frame_index in (run.start_frame, run.end_frame)
            if run.state == "visible"
        }
        lifecycle_boundary_frames.update(
            overlay_frames_by_track.get(track.track_id, ())
        )
        selected, mandatory_conflict = _reduce_observations(
            track.observations,
            observation_cap,
            priority_frames=lifecycle_boundary_frames,
            maximum_frame=(
                None
                if lifecycle_complete
                else (retained_runs[-1].end_frame if retained_runs else -1)
            ),
        )
        mandatory_conflicts += int(mandatory_conflict)
        summary_observations = tuple(
            _summary_observation(source_ordinal, observation)
            for source_ordinal, observation in selected
        )
        retained_gaps = _gaps_from_visibility_runs(
            retained_runs,
            summary_observations,
            lifecycle_complete=lifecycle_complete,
        )
        total_gaps += track_gap_count
        kept_gaps += len(retained_gaps)
        total_visibility_runs += source_run_count
        kept_visibility_runs += len(retained_runs)
        summary_tracks.append(
            SummaryTrack(
                track_id=track.track_id,
                entity_id=track.entity_id,
                status=track.status,
                source_observation_count=len(track.observations),
                observations=summary_observations,
                visibility_runs=retained_runs,
                visibility_lifecycle_complete=lifecycle_complete,
                missing_intervals=retained_gaps,
                candidate_search_complete=(
                    len(selected) == len(track.observations)
                    and lifecycle_complete
                    and len(retained_gaps) == track_gap_count
                ),
            )
        )
    tracks = tuple(summary_tracks)
    entities, omitted_aliases, truncated_texts, total_entity_count = (
        _summarize_entities(
            artifact,
            tracks,
            alias_cap=alias_cap,
            include_untracked=include_untracked_entities,
        )
    )
    relations, total_relations, relation_scan_complete = _relations(
        tracks, cap=relation_cap
    )
    all_overlays = tuple(
        SummaryOverlay(
            path=record.path,
            track_id=record.track_id,
            frame_index=record.frame_index,
        )
        for record in sorted(artifact.overlay_records, key=lambda item: item.path)
        if artifact.status is EvidenceStatus.AVAILABLE
    )
    retained_visible = {
        (track.track_id, observation.frame_index)
        for track in tracks
        for observation in track.observations
        if observation.visible
    }
    eligible_overlays = tuple(
        overlay
        for overlay in all_overlays
        if (overlay.track_id, overlay.frame_index) in retained_visible
    )
    overlays = eligible_overlays[:overlay_cap]
    uncovered_entity_count = 0
    if artifact.status is EvidenceStatus.AVAILABLE:
        available_entity_ids = {
            track.entity_id
            for track in all_tracks
            if track.status is EvidenceStatus.AVAILABLE
        }
        uncovered_entity_count = sum(
            entity.entity_id not in available_entity_ids
            for entity in artifact.entities
        )
    metadata = _AssemblyMetadata(
        total_entities=total_entity_count,
        kept_entities=len(entities),
        omitted_aliases=omitted_aliases,
        truncated_texts=truncated_texts,
        total_tracks=len(all_tracks),
        kept_tracks=len(tracks),
        total_observations=sum(len(track.observations) for track in selected_tracks),
        kept_observations=sum(len(track.observations) for track in tracks),
        mandatory_conflicts=mandatory_conflicts,
        total_gaps=total_gaps,
        kept_gaps=kept_gaps,
        total_visibility_runs=total_visibility_runs,
        kept_visibility_runs=kept_visibility_runs,
        total_relations=total_relations,
        kept_relations=len(relations),
        total_overlays=len(all_overlays),
        kept_overlays=len(overlays),
        artifact_warning_count=len(artifact.warnings),
        uncovered_entity_count=uncovered_entity_count,
        unavailable_evidence_count=(
            1
            if artifact.status is EvidenceStatus.UNAVAILABLE
            else 0
            if artifact.status is EvidenceStatus.DISABLED
            else sum(
                track.status is not EvidenceStatus.AVAILABLE
                for track in selected_tracks
            )
        ),
    )
    values: dict[str, Any] = dict(
        schema_version="cv_summary_v1",
        status=artifact.status,
        candidate_search_complete=(
            artifact.status is EvidenceStatus.AVAILABLE
            and len(selected_tracks) == len(all_tracks)
            and all(track.candidate_search_complete for track in tracks)
            and not artifact.warnings
            and uncovered_entity_count == 0
        ),
        observed_clock=_retained_observed_clock(frame_clock, tracks),
        entities=entities,
        tracks=tracks,
        relations=relations,
        relations_complete=(
            relation_scan_complete and total_relations == len(relations)
        ),
        overlay_refs=tuple(item.path for item in overlays),
        overlays=overlays,
        overlays_complete=len(all_overlays) == len(overlays),
        warnings=_summary_warnings(metadata, prompt_truncated=prompt_truncated),
        prompt_char_limit=prompt_char_limit,
    )
    provisional = CvEvidenceSummary.model_construct(
        summary_id=f"cvs_{'0' * 64}",
        **values,
    )
    return CvEvidenceSummary(
        summary_id=_summary_identity(provisional),
        **values,
    )


def _retained_observed_clock(
    frame_clock: tuple[FrameTimestamp, ...],
    tracks: tuple[SummaryTrack, ...],
) -> tuple[FrameTimestamp, ...]:
    required_frames: set[int] = set()
    for track in tracks:
        required_frames.update(item.frame_index for item in track.observations)
        for run in track.visibility_runs:
            required_frames.update((run.start_frame, run.end_frame))
        for gap in track.missing_intervals:
            required_frames.update(
                frame_index
                for frame_index in (
                    gap.last_visible_frame,
                    gap.first_missing_frame,
                    gap.last_missing_frame,
                    gap.first_revisible_frame,
                )
                if frame_index is not None
            )
    return tuple(
        item for item in frame_clock if item.frame_index in required_frames
    )


def _summarize_entities(
    artifact: CvEvidenceArtifact,
    tracks: tuple[SummaryTrack, ...],
    *,
    alias_cap: int,
    include_untracked: bool,
) -> tuple[tuple[SummaryEntity, ...], int, int, int]:
    by_id = {entity.entity_id: entity for entity in artifact.entities}
    referenced = sorted({track.entity_id for track in tracks})
    remaining = sorted(set(by_id) - set(referenced)) if include_untracked else []
    selected_ids = (referenced + remaining)[:_MAX_ENTITY_SUMMARIES]
    entities: list[SummaryEntity] = []
    omitted_aliases = 0
    truncated_texts = 0
    for entity_id in selected_ids:
        source = by_id[entity_id]
        label, label_truncated = _truncate_text(
            source.canonical_label, _MAX_LABEL_CHARS
        )
        truncated_texts += int(label_truncated)
        unique_aliases: dict[str, str] = {}
        for alias in source.aliases:
            key = alias.casefold()
            retained = unique_aliases.get(key)
            if retained is None or alias < retained:
                unique_aliases[key] = alias
        ordered_aliases = tuple(
            sorted(unique_aliases.values(), key=lambda value: (value.casefold(), value))
        )
        omitted_aliases += max(0, len(ordered_aliases) - alias_cap)
        aliases: list[str] = []
        for alias in ordered_aliases[:alias_cap]:
            bounded, was_truncated = _truncate_text(alias, _MAX_ALIAS_CHARS)
            truncated_texts += int(was_truncated)
            aliases.append(bounded)
        entities.append(
            SummaryEntity(
                entity_id=source.entity_id,
                canonical_label=label,
                aliases=tuple(aliases),
                role=source.role,
            )
        )
    return tuple(entities), omitted_aliases, truncated_texts, len(artifact.entities)


def _truncate_text(value: str, maximum: int) -> tuple[str, bool]:
    if len(value) <= maximum:
        return value, False
    return value[:maximum], True


def _reduce_observations(
    observations: tuple[TrackObservation, ...],
    cap: int,
    *,
    priority_frames: set[int] | None = None,
    maximum_frame: int | None = None,
) -> tuple[tuple[tuple[int, TrackObservation], ...], bool]:
    indexed_observations = tuple(
        (source_ordinal, observation)
        for source_ordinal, observation in enumerate(observations)
        if maximum_frame is None or observation.frame_index <= maximum_frame
    )
    if not indexed_observations:
        return (), False
    bounded_observations = tuple(item[1] for item in indexed_observations)
    priority: list[int] = []
    selected_indices: set[int] = set()
    mandatory_conflict = False

    def add(index: int) -> None:
        nonlocal mandatory_conflict
        if index in selected_indices:
            return
        if len(priority) < cap:
            priority.append(index)
            selected_indices.add(index)
        else:
            mandatory_conflict = True

    def add_context(index: int) -> None:
        for contextual_index in (index - 1, index, index + 1):
            if 0 <= contextual_index < len(bounded_observations):
                add(contextual_index)

    add(0)
    add(len(bounded_observations) - 1)
    positions_by_frame = {
        observation.frame_index: index
        for index, observation in enumerate(bounded_observations)
    }
    for frame_index in sorted(priority_frames or ()):
        position = positions_by_frame.get(frame_index)
        if position is not None:
            add(position)
    for index, (previous, current) in enumerate(
        zip(bounded_observations, bounded_observations[1:]), start=1
    ):
        if previous.visible != current.visible:
            add(index - 1)
            add(index)
    add_context(
        min(
            range(len(bounded_observations)),
            key=lambda index: (
                bounded_observations[index].area_fraction,
                bounded_observations[index].frame_index,
            ),
        )
    )
    add(
        min(
            range(len(bounded_observations)),
            key=lambda index: (
                -bounded_observations[index].area_fraction,
                bounded_observations[index].frame_index,
            ),
        )
    )
    add_context(
        min(
            range(len(bounded_observations)),
            key=lambda index: (
                bounded_observations[index].confidence,
                bounded_observations[index].frame_index,
            ),
        )
    )
    chosen = list(priority)
    remaining_slots = cap - len(chosen)
    if remaining_slots > 0 and len(selected_indices) < len(bounded_observations):
        for slot in range(1, remaining_slots + 1):
            ideal = slot * (len(bounded_observations) - 1) / (remaining_slots + 1)
            selected = _nearest_unselected_index(
                ideal, len(bounded_observations), selected_indices
            )
            if selected is None:
                break
            chosen.append(selected)
            selected_indices.add(selected)
    if len(chosen) < cap:
        for index in range(len(bounded_observations)):
            if index in selected_indices:
                continue
            chosen.append(index)
            selected_indices.add(index)
            if len(chosen) == cap:
                break
    return (
        tuple(indexed_observations[index] for index in sorted(chosen)),
        mandatory_conflict,
    )


def _nearest_unselected_index(
    ideal: float, count: int, selected: set[int]
) -> int | None:
    lower = math.floor(ideal)
    for radius in range(count + 1):
        candidates = {
            lower - radius,
            lower + 1 + radius,
        }
        available = [
            index
            for index in candidates
            if 0 <= index < count and index not in selected
        ]
        if available:
            return min(available, key=lambda index: (abs(index - ideal), index))
    return None


def _summary_observation(
    source_ordinal: int, observation: TrackObservation
) -> SummaryObservation:
    left, top, right, bottom = observation.bbox_xyxy
    edge_proximity = min(left, top, 1.0 - right, 1.0 - bottom)
    return SummaryObservation(
        source_ordinal=source_ordinal,
        frame_index=observation.frame_index,
        timestamp_seconds=observation.timestamp_seconds,
        bbox_xyxy=observation.bbox_xyxy,
        visible=observation.visible,
        confidence=observation.confidence,
        area_fraction=observation.area_fraction,
        center_xy=observation.center_xy,
        edge_proximity=edge_proximity,
    )


def _derive_visibility_runs(
    track: CvTrack,
    frame_clock: tuple[FrameTimestamp, ...],
    *,
    cap: int,
) -> tuple[tuple[VisibilityRun, ...], int, int]:
    if not frame_clock:
        return (), 0, 0
    observations = {item.frame_index: item for item in track.observations}
    retained: list[VisibilityRun] = []
    total_runs = 0
    gap_count = 0
    has_seen_visible = False
    state: Literal["visible", "missing"] | None = None
    start: FrameTimestamp | None = None
    end: FrameTimestamp | None = None
    minimum_confidence: float | None = None

    def finish() -> None:
        nonlocal total_runs, gap_count, has_seen_visible
        if state is None or start is None or end is None:
            return
        total_runs += 1
        if state == "missing" and has_seen_visible:
            gap_count += 1
        if state == "visible":
            has_seen_visible = True
        if len(retained) < cap:
            retained.append(
                VisibilityRun(
                    state=state,
                    start_frame=start.frame_index,
                    start_time=start.timestamp_seconds,
                    end_frame=end.frame_index,
                    end_time=end.timestamp_seconds,
                    minimum_confidence=(
                        minimum_confidence if state == "missing" else None
                    ),
                )
            )

    for frame in frame_clock:
        observation = observations.get(frame.frame_index)
        current_state: Literal["visible", "missing"] = (
            "visible"
            if observation is not None and observation.visible
            else "missing"
        )
        if current_state != state:
            finish()
            state = current_state
            start = frame
            minimum_confidence = None
        end = frame
        if current_state == "missing" and observation is not None:
            minimum_confidence = (
                observation.confidence
                if minimum_confidence is None
                else min(minimum_confidence, observation.confidence)
            )
    finish()
    if total_runs > len(retained) and retained:
        # A truncated suffix cannot truthfully turn its final missing run into a
        # permanent disappearance.  Drop that unresolved boundary witness.
        retained.pop()
    return tuple(retained), total_runs, gap_count


def _gaps_from_visibility_runs(
    runs: tuple[VisibilityRun, ...],
    observations: tuple[SummaryObservation, ...],
    *,
    lifecycle_complete: bool,
) -> tuple[VisibilityGap, ...]:
    observations_by_frame = {
        observation.frame_index: observation for observation in observations
    }
    gaps: list[VisibilityGap] = []
    for index, run in enumerate(runs):
        if run.state != "missing" or index == 0:
            continue
        previous = runs[index - 1]
        if previous.state != "visible":
            continue
        last_visible = observations_by_frame.get(previous.end_frame)
        if last_visible is None or not last_visible.visible:
            continue
        following = runs[index + 1] if index + 1 < len(runs) else None
        if following is None and not lifecycle_complete:
            continue
        first_revisible = (
            observations_by_frame.get(following.start_frame)
            if following is not None and following.state == "visible"
            else None
        )
        if following is not None and (
            first_revisible is None or not first_revisible.visible
        ):
            continue
        gaps.append(
            VisibilityGap(
                last_visible_frame=previous.end_frame,
                last_visible_time=previous.end_time,
                first_missing_frame=run.start_frame,
                first_missing_time=run.start_time,
                last_missing_frame=run.end_frame,
                last_missing_time=run.end_time,
                first_revisible_frame=(
                    following.start_frame if following is not None else None
                ),
                first_revisible_time=(
                    following.start_time if following is not None else None
                ),
                minimum_confidence=run.minimum_confidence,
                edge_departure=(
                    last_visible.edge_proximity <= _EDGE_PROXIMITY_THRESHOLD
                ),
            )
        )
    return tuple(gaps)


def _relations(
    tracks: tuple[SummaryTrack, ...], *, cap: int
) -> tuple[tuple[SpatialRelation, ...], int, bool]:
    by_frame: dict[int, list[tuple[str, SummaryObservation]]] = {}
    for track in tracks:
        if track.status is not EvidenceStatus.AVAILABLE:
            continue
        for observation in track.observations:
            if observation.visible:
                by_frame.setdefault(observation.frame_index, []).append(
                    (track.track_id, observation)
                )
    relations: list[SpatialRelation] = []
    aligned_by_frame = {
        frame_index: sorted(values, key=lambda item: item[0])
        for frame_index, values in by_frame.items()
    }
    total_relations = sum(
        len(values) * (len(values) - 1) // 2
        for values in aligned_by_frame.values()
    )
    if cap == 0:
        return (), total_relations, True
    pair_scans = 0
    scan_complete = True
    for retain_positive in (True, False):
        for frame_index in sorted(aligned_by_frame):
            aligned = aligned_by_frame[frame_index]
            for subject_index, (subject_track_id, subject) in enumerate(aligned):
                for object_index in range(subject_index + 1, len(aligned)):
                    object_track_id, object_observation = aligned[object_index]
                    pair_scans += 1
                    if pair_scans > _MAX_RELATION_PAIR_SCANS:
                        scan_complete = False
                        return tuple(relations), total_relations, scan_complete
                    positive = _boxes_overlap(
                        subject.bbox_xyxy, object_observation.bbox_xyxy
                    )
                    if positive != retain_positive:
                        continue
                    relations.append(
                        _spatial_relation(
                            subject_track_id,
                            subject,
                            object_track_id,
                            object_observation,
                        )
                    )
                    if len(relations) == cap:
                        return tuple(relations), total_relations, scan_complete
    return tuple(relations), total_relations, scan_complete


def _boxes_overlap(
    left_box: tuple[float, float, float, float],
    right_box: tuple[float, float, float, float],
) -> bool:
    left, top, right, bottom = left_box
    other_left, other_top, other_right, other_bottom = right_box
    return (
        min(right, other_right) > max(left, other_left)
        and min(bottom, other_bottom) > max(top, other_top)
    )


def _spatial_relation(
    subject_track_id: str,
    subject: SummaryObservation,
    object_track_id: str,
    object_observation: SummaryObservation,
) -> SpatialRelation:
    if subject.frame_index != object_observation.frame_index or (
        subject.timestamp_seconds != object_observation.timestamp_seconds
    ):
        raise ValueError("spatial relation inputs must use one frame")
    s_left, s_top, s_right, s_bottom = subject.bbox_xyxy
    o_left, o_top, o_right, o_bottom = object_observation.bbox_xyxy
    intersection_width = max(0.0, min(s_right, o_right) - max(s_left, o_left))
    intersection_height = max(0.0, min(s_bottom, o_bottom) - max(s_top, o_top))
    intersection = intersection_width * intersection_height
    subject_box_area = (s_right - s_left) * (s_bottom - s_top)
    object_box_area = (o_right - o_left) * (o_bottom - o_top)
    union = subject_box_area + object_box_area - intersection
    bbox_iou = intersection / union if union > 0 else 0.0
    subject_covered = intersection / subject_box_area
    object_covered = intersection / object_box_area
    dx = subject.center_xy[0] - object_observation.center_xy[0]
    dy = subject.center_xy[1] - object_observation.center_xy[1]
    center_distance = min(1.0, math.hypot(dx, dy) / math.sqrt(2.0))
    maximum_area = max(subject.area_fraction, object_observation.area_fraction)
    area_similarity = (
        min(subject.area_fraction, object_observation.area_fraction) / maximum_area
        if maximum_area > 0
        else 1.0
    )
    return SpatialRelation(
        frame_index=subject.frame_index,
        timestamp_seconds=subject.timestamp_seconds,
        subject_track_id=subject_track_id,
        object_track_id=object_track_id,
        bbox_iou=_unit_float(bbox_iou),
        subject_bbox_covered_fraction=_unit_float(subject_covered),
        object_bbox_covered_fraction=_unit_float(object_covered),
        center_distance_fraction=_unit_float(center_distance),
        area_similarity=_unit_float(area_similarity),
        subject_inside_object=(
            s_left >= o_left
            and s_top >= o_top
            and s_right <= o_right
            and s_bottom <= o_bottom
        ),
        object_inside_subject=(
            o_left >= s_left
            and o_top >= s_top
            and o_right <= s_right
            and o_bottom <= s_bottom
        ),
    )


def _unit_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("relation feature must be finite")
    return min(1.0, max(0.0, float(value)))


def _validate_overlay_ref(value: str) -> str:
    if (
        type(value) is not str
        or len(value) > 512
        or "\\" in value
        or value.startswith("/")
    ):
        raise ValueError("overlay reference must be bounded relative POSIX")
    parts = value.split("/")
    if (
        len(parts) != 2
        or parts[0] != "overlays"
        or any(
            part in {"", ".", ".."}
            or _SAFE_OVERLAY_COMPONENT.fullmatch(part) is None
            for part in parts
        )
        or not parts[-1].endswith(".png")
    ):
        raise ValueError("overlay reference is not allowlisted")
    return value


def _summary_warnings(
    metadata: _AssemblyMetadata, *, prompt_truncated: bool
) -> tuple[SummaryWarning, ...]:
    warnings: list[SummaryWarning] = []
    if metadata.total_entities > metadata.kept_entities:
        warnings.append(
            SummaryWarning(
                code="ENTITIES_TRUNCATED",
                kept=metadata.kept_entities,
                omitted=metadata.total_entities - metadata.kept_entities,
            )
        )
    if metadata.omitted_aliases:
        warnings.append(
            SummaryWarning(
                code="ALIASES_TRUNCATED", omitted=metadata.omitted_aliases
            )
        )
    if metadata.truncated_texts:
        warnings.append(
            SummaryWarning(
                code="ENTITY_TEXT_TRUNCATED", count=metadata.truncated_texts
            )
        )
    if metadata.total_tracks > metadata.kept_tracks:
        warnings.append(
            SummaryWarning(
                code="TRACKS_TRUNCATED",
                kept=metadata.kept_tracks,
                omitted=metadata.total_tracks - metadata.kept_tracks,
            )
        )
    if metadata.total_observations > metadata.kept_observations:
        warnings.append(
            SummaryWarning(
                code="OBSERVATIONS_TRUNCATED",
                kept=metadata.kept_observations,
                omitted=metadata.total_observations - metadata.kept_observations,
            )
        )
    if metadata.mandatory_conflicts:
        warnings.append(
            SummaryWarning(
                code="MANDATORY_LANDMARKS_TRUNCATED",
                tracks=metadata.mandatory_conflicts,
                priority=_MANDATORY_PRIORITY_TEXT,
            )
        )
    if metadata.total_gaps > metadata.kept_gaps:
        warnings.append(
            SummaryWarning(
                code="LIFECYCLE_GAPS_TRUNCATED",
                kept=metadata.kept_gaps,
                omitted=metadata.total_gaps - metadata.kept_gaps,
            )
        )
    if metadata.total_visibility_runs > metadata.kept_visibility_runs:
        warnings.append(
            SummaryWarning(
                code="VISIBILITY_RUNS_TRUNCATED",
                kept=metadata.kept_visibility_runs,
                omitted=(
                    metadata.total_visibility_runs
                    - metadata.kept_visibility_runs
                ),
            )
        )
    if metadata.total_relations > metadata.kept_relations:
        warnings.append(
            SummaryWarning(
                code="RELATIONS_TRUNCATED",
                kept=metadata.kept_relations,
                omitted=metadata.total_relations - metadata.kept_relations,
            )
        )
    if metadata.total_overlays > metadata.kept_overlays:
        warnings.append(
            SummaryWarning(
                code="OVERLAYS_TRUNCATED",
                kept=metadata.kept_overlays,
                omitted=metadata.total_overlays - metadata.kept_overlays,
            )
        )
    if metadata.artifact_warning_count:
        warnings.append(
            SummaryWarning(
                code="ARTIFACT_WARNINGS_OMITTED",
                count=metadata.artifact_warning_count,
            )
        )
    if metadata.uncovered_entity_count:
        warnings.append(
            SummaryWarning(
                code="ENTITIES_WITHOUT_AVAILABLE_TRACKS",
                count=metadata.uncovered_entity_count,
            )
        )
    if metadata.unavailable_evidence_count:
        warnings.append(
            SummaryWarning(
                code="UNAVAILABLE_EVIDENCE_OMITTED",
                count=metadata.unavailable_evidence_count,
            )
        )
    if prompt_truncated:
        warnings.append(
            SummaryWarning(code="PROMPT_DATA_TRUNCATED", budget_enforced=True)
        )
    return tuple(warnings)


def _canonical_array(
    values: tuple[Any, ...], projector: Any = None
) -> _CanonicalArray:
    return _CanonicalArray(values=values, projector=projector)


def _frame_projection(frame: FrameTimestamp) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("frame_index", frame.frame_index),
            ("timestamp_seconds", frame.timestamp_seconds),
        )
    )


def _entity_projection(entity: SummaryEntity) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("entity_id", entity.entity_id),
            ("canonical_label", entity.canonical_label),
            ("aliases", _canonical_array(entity.aliases)),
            ("role", entity.role.value),
        )
    )


def _observation_projection(observation: SummaryObservation) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("source_ordinal", observation.source_ordinal),
            ("frame_index", observation.frame_index),
            ("timestamp_seconds", observation.timestamp_seconds),
            ("bbox_xyxy", _canonical_array(observation.bbox_xyxy)),
            ("visible", observation.visible),
            ("confidence", observation.confidence),
            ("area_fraction", observation.area_fraction),
            ("center_xy", _canonical_array(observation.center_xy)),
            ("edge_proximity", observation.edge_proximity),
        )
    )


def _visibility_run_projection(run: VisibilityRun) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("state", run.state),
            ("start_frame", run.start_frame),
            ("start_time", run.start_time),
            ("end_frame", run.end_frame),
            ("end_time", run.end_time),
            ("minimum_confidence", run.minimum_confidence),
        )
    )


def _visibility_gap_projection(gap: VisibilityGap) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("last_visible_frame", gap.last_visible_frame),
            ("last_visible_time", gap.last_visible_time),
            ("first_missing_frame", gap.first_missing_frame),
            ("first_missing_time", gap.first_missing_time),
            ("last_missing_frame", gap.last_missing_frame),
            ("last_missing_time", gap.last_missing_time),
            ("first_revisible_frame", gap.first_revisible_frame),
            ("first_revisible_time", gap.first_revisible_time),
            ("minimum_confidence", gap.minimum_confidence),
            ("edge_departure", gap.edge_departure),
        )
    )


def _track_projection(track: SummaryTrack) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("track_id", track.track_id),
            ("entity_id", track.entity_id),
            ("status", track.status.value),
            ("source_observation_count", track.source_observation_count),
            ("candidate_search_complete", track.candidate_search_complete),
            (
                "visibility_lifecycle_complete",
                track.visibility_lifecycle_complete,
            ),
            (
                "observations",
                _canonical_array(track.observations, _observation_projection),
            ),
            (
                "visibility_runs",
                _canonical_array(track.visibility_runs, _visibility_run_projection),
            ),
            (
                "missing_intervals",
                _canonical_array(
                    track.missing_intervals,
                    _visibility_gap_projection,
                ),
            ),
        )
    )


def _relation_projection(relation: SpatialRelation) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("frame_index", relation.frame_index),
            ("timestamp_seconds", relation.timestamp_seconds),
            ("subject_track_id", relation.subject_track_id),
            ("object_track_id", relation.object_track_id),
            ("bbox_iou", relation.bbox_iou),
            (
                "subject_bbox_covered_fraction",
                relation.subject_bbox_covered_fraction,
            ),
            (
                "object_bbox_covered_fraction",
                relation.object_bbox_covered_fraction,
            ),
            ("center_distance_fraction", relation.center_distance_fraction),
            ("area_similarity", relation.area_similarity),
            ("subject_inside_object", relation.subject_inside_object),
            ("object_inside_subject", relation.object_inside_subject),
        )
    )


def _overlay_projection(overlay: SummaryOverlay) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("path", overlay.path),
            ("track_id", overlay.track_id),
            ("frame_index", overlay.frame_index),
        )
    )


def _warning_projection(warning: SummaryWarning) -> _CanonicalObject:
    fields: list[tuple[str, Any]] = [("code", warning.code)]
    for field_name in (
        "kept",
        "omitted",
        "count",
        "tracks",
        "priority",
        "budget_enforced",
    ):
        value = getattr(warning, field_name)
        if value is not None:
            fields.append((field_name, value))
    return _CanonicalObject(tuple(fields))


def _summary_projection(
    summary: CvEvidenceSummary,
    *,
    validation: bool = False,
    include_summary_id: bool = True,
) -> _CanonicalObject:
    """Build the shared lazy summary field spec for both trust projections."""
    fields: list[tuple[str, Any]] = [
        ("schema_version", summary.schema_version),
        ("status", summary.status.value),
        ("candidate_search_complete", summary.candidate_search_complete),
        (
            "observed_clock",
            _canonical_array(summary.observed_clock, _frame_projection),
        ),
        ("entities", _canonical_array(summary.entities, _entity_projection)),
        ("tracks", _canonical_array(summary.tracks, _track_projection)),
        ("relations", _canonical_array(summary.relations, _relation_projection)),
        ("relations_complete", summary.relations_complete),
        ("overlay_refs", _canonical_array(summary.overlay_refs)),
        ("overlays", _canonical_array(summary.overlays, _overlay_projection)),
        ("overlays_complete", summary.overlays_complete),
        ("warnings", _canonical_array(summary.warnings, _warning_projection)),
    ]
    if include_summary_id:
        fields.append(("summary_id", summary.summary_id))
    if validation:
        fields.append(("prompt_char_limit", summary.prompt_char_limit))
    return _CanonicalObject(tuple(fields))


def _provenance_projection(provenance: OccluderProvenance) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("entity_id", provenance.entity_id),
            ("track_id", provenance.track_id),
            (
                "supporting_frames",
                _canonical_array(provenance.supporting_frames),
            ),
        )
    )


def _candidate_projection(
    candidate: OcclusionCandidate,
    *,
    include_candidate_id: bool = True,
    ordinal: int | None = None,
) -> _CanonicalObject:
    fields: list[tuple[str, Any]] = [
        ("target_entity_id", candidate.target_entity_id),
        ("target_track_id", candidate.target_track_id),
        (
            "possible_occluders",
            _canonical_array(
                candidate.possible_occluders,
                _provenance_projection,
            ),
        ),
        (
            "possible_occluder_entity_ids",
            _canonical_array(candidate.possible_occluder_entity_ids),
        ),
        ("allowed_start_times", _canonical_array(candidate.allowed_start_times)),
        ("allowed_end_times", _canonical_array(candidate.allowed_end_times)),
        ("last_visible_frame", candidate.last_visible_frame),
        ("first_revisible_frame", candidate.first_revisible_frame),
        ("edge_departure", candidate.edge_departure),
        ("low_confidence", candidate.low_confidence),
        ("overlay_refs", _canonical_array(candidate.overlay_refs)),
        (
            "observation_support_complete",
            candidate.observation_support_complete,
        ),
        ("relation_support_complete", candidate.relation_support_complete),
        ("overlay_support_complete", candidate.overlay_support_complete),
        ("support_complete", candidate.support_complete),
        ("source_search_complete", candidate.source_search_complete),
    ]
    if include_candidate_id:
        fields.append(("candidate_id", candidate.candidate_id))
    if ordinal is not None:
        fields.append(("ordinal", ordinal))
    return _CanonicalObject(tuple(fields))


def _threshold_projection(thresholds: EvidenceThresholds) -> _CanonicalObject:
    return _CanonicalObject(
        (
            ("min_confidence", thresholds.min_confidence),
            ("min_area_fraction", thresholds.min_area_fraction),
            (
                "occlusion_visibility_drop",
                thresholds.occlusion_visibility_drop,
            ),
        )
    )


def _bundle_projection_values(
    summary: CvEvidenceSummary,
    thresholds: EvidenceThresholds,
    candidates: tuple[OcclusionCandidate, ...],
    *,
    source_search_complete: bool,
    candidates_complete: bool,
    truncation_codes: tuple[str, ...],
    validation: bool = False,
    prompt_char_limit: int | None = None,
    candidate_limit: int | None = None,
) -> _CanonicalObject:
    """Build one shared bundle spec, adding private validation fields on demand."""
    fields: list[tuple[str, Any]] = [
        ("schema_version", "cv_prompt_bundle_v1"),
        ("summary", _summary_projection(summary, validation=validation)),
        ("thresholds", _threshold_projection(thresholds)),
        ("candidates", _canonical_array(candidates, _candidate_projection)),
        ("source_search_complete", source_search_complete),
        ("candidates_complete", candidates_complete),
        ("truncation_codes", _canonical_array(truncation_codes)),
    ]
    if validation:
        if prompt_char_limit is None or candidate_limit is None:
            raise ValueError("bundle validation projection requires private limits")
        fields.extend(
            (
                ("prompt_char_limit", prompt_char_limit),
                ("candidate_limit", candidate_limit),
            )
        )
    return _CanonicalObject(tuple(fields))


def _bundle_projection(
    bundle: CvPromptBundle, *, validation: bool = False
) -> _CanonicalObject:
    return _bundle_projection_values(
        bundle.summary,
        bundle.thresholds,
        bundle.candidates,
        source_search_complete=bundle.source_search_complete,
        candidates_complete=bundle.candidates_complete,
        truncation_codes=bundle.truncation_codes,
        validation=validation,
        prompt_char_limit=bundle.prompt_char_limit,
        candidate_limit=bundle.candidate_limit,
    )


def _iter_canonical_json(value: Any) -> Iterator[str]:
    """Yield canonical JSON directly from lazy specs without nested containers."""
    if isinstance(value, _CanonicalObject):
        yield "{"
        for index, (name, field_value) in enumerate(
            sorted(value.fields, key=lambda item: item[0])
        ):
            if index:
                yield ","
            yield json.dumps(name, ensure_ascii=False)
            yield ":"
            yield from _iter_canonical_json(field_value)
        yield "}"
        return
    if isinstance(value, _CanonicalArray):
        yield "["
        for index, item in enumerate(value.values):
            if index:
                yield ","
            projected = item if value.projector is None else value.projector(item)
            yield from _iter_canonical_json(projected)
        yield "]"
        return
    if type(value) is dict:
        yield "{"
        for index, name in enumerate(sorted(value)):
            if type(name) is not str:
                raise ValueError("canonical JSON object keys must be plain strings")
            if index:
                yield ","
            yield json.dumps(name, ensure_ascii=False)
            yield ":"
            yield from _iter_canonical_json(value[name])
        yield "}"
        return
    if type(value) in {list, tuple}:
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from _iter_canonical_json(item)
        yield "]"
        return
    if value is None or type(value) in {str, int, float, bool}:
        yield json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return
    raise ValueError("canonical JSON contains a non-plain scalar")


def _materialize_value(value: Any) -> Any:
    if isinstance(value, _CanonicalObject):
        return {
            name: _materialize_value(field_value)
            for name, field_value in sorted(value.fields, key=lambda item: item[0])
        }
    if isinstance(value, _CanonicalArray):
        return [
            _materialize_value(
                item if value.projector is None else value.projector(item)
            )
            for item in value.values
        ]
    if type(value) is dict:
        return {
            name: _materialize_value(value[name])
            for name in sorted(value)
        }
    if type(value) in {list, tuple}:
        return [_materialize_value(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError("canonical JSON contains a non-plain scalar")


def _materialize_record(projection: Any) -> dict[str, Any]:
    record = _materialize_value(projection)
    if type(record) is not dict:
        raise ValueError("canonical prompt projection must be an object")
    return record


def _materialize_bounded_record(
    projection: Any, *, maximum: int, message: str
) -> dict[str, Any]:
    if _canonical_char_count(projection, maximum=maximum) > maximum:
        raise ValueError(message)
    return _materialize_record(projection)


def _bounded_validation_json(
    *,
    public_projection: Any,
    validation_projection: Any,
    maximum: int,
    message: str,
) -> str:
    """Serialize validation JSON only after the public projection is in budget."""
    if _canonical_char_count(public_projection, maximum=maximum) > maximum:
        raise ValueError(message)
    validation_maximum = maximum + _MAX_VALIDATION_METADATA_CHARS
    total = 0
    buffer = StringIO()
    for chunk in _iter_canonical_json(validation_projection):
        total += len(chunk)
        if total > validation_maximum:
            raise ValueError("private prompt validation metadata exceeds its bound")
        buffer.write(chunk)
    return buffer.getvalue()


def _summary_prompt_record(summary: CvEvidenceSummary) -> dict[str, Any]:
    """Materialize one already-bounded private compatibility projection."""
    return _materialize_record(_summary_projection(summary))


def _summary_validation_record(summary: CvEvidenceSummary) -> dict[str, Any]:
    """Materialize the private projection only for compatibility diagnostics."""
    return _materialize_record(_summary_projection(summary, validation=True))


def _summary_identity(summary: CvEvidenceSummary) -> str:
    projection = _summary_projection(
        summary,
        validation=True,
        include_summary_id=False,
    )
    return f"cvs_{_canonical_sha256(projection)}"


def _canonical_sha256(record: Any) -> str:
    digest = hashlib.sha256()
    for chunk in _iter_canonical_json(record):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _warning_prompt_record(warning: SummaryWarning) -> dict[str, Any]:
    return _materialize_record(_warning_projection(warning))


def _bundle_prompt_record_values(
    summary: CvEvidenceSummary,
    thresholds: EvidenceThresholds,
    candidates: tuple[OcclusionCandidate, ...],
    *,
    source_search_complete: bool,
    candidates_complete: bool,
    truncation_codes: tuple[str, ...],
) -> dict[str, Any]:
    """Materialize a bounded compatibility projection, never used for sizing."""
    return _materialize_record(
        _bundle_projection_values(
            summary,
            thresholds,
            candidates,
            source_search_complete=source_search_complete,
            candidates_complete=candidates_complete,
            truncation_codes=truncation_codes,
        )
    )


def _bundle_prompt_record(bundle: CvPromptBundle) -> dict[str, Any]:
    """Materialize one already-bounded private compatibility projection."""
    return _materialize_record(_bundle_projection(bundle))


def _bundle_empty_candidates_char_count(
    summary: CvEvidenceSummary,
    thresholds: EvidenceThresholds,
    *,
    source_search_complete: bool,
    candidates_complete: bool,
    truncation_codes: tuple[str, ...],
) -> int:
    return _canonical_char_count(
        _bundle_projection_values(
            summary,
            thresholds,
            (),
            source_search_complete=source_search_complete,
            candidates_complete=candidates_complete,
            truncation_codes=truncation_codes,
        )
    )


def _canonical_char_count(record: Any, *, maximum: int | None = None) -> int:
    total = 0
    for chunk in _iter_canonical_json(record):
        total += len(chunk)
        if maximum is not None and total > maximum:
            return total
    return total


def _summary_fits(summary: CvEvidenceSummary, cap: int) -> bool:
    reserved_cap = max(1, cap - min(_MIN_BUNDLE_ENVELOPE_CHARS, cap // 4))
    return (
        _canonical_char_count(
            _summary_projection(summary, validation=True),
            maximum=reserved_cap,
        )
        <= reserved_cap
        and _canonical_char_count(
            _summary_projection(summary),
            maximum=reserved_cap,
        )
        <= reserved_cap
    )


def _uncertain_observation_drafts(
    track: SummaryTrack,
    thresholds: EvidenceThresholds,
) -> tuple[_CandidateDraft, ...]:
    observations = track.observations
    if len(observations) < 3:
        return ()
    uncertain: list[tuple[bool, bool]] = [(False, False)] * len(observations)
    for index in range(1, len(observations) - 1):
        previous = observations[index - 1]
        current = observations[index]
        following = observations[index + 1]
        if not (previous.visible and current.visible and following.visible):
            continue
        if not (
            previous.source_ordinal + 1 == current.source_ordinal
            and current.source_ordinal + 1 == following.source_ordinal
        ):
            continue
        low_confidence = current.confidence < thresholds.min_confidence
        reference_area = max(previous.area_fraction, following.area_fraction)
        area_reduction = current.area_fraction < thresholds.min_area_fraction or (
            reference_area > 0
            and current.area_fraction
            <= reference_area * (1.0 - thresholds.occlusion_visibility_drop)
        )
        uncertain[index] = (low_confidence or area_reduction, low_confidence)

    drafts: list[_CandidateDraft] = []
    index = 1
    while index < len(observations) - 1:
        if not uncertain[index][0]:
            index += 1
            continue
        start = index
        end = index
        while (
            end + 1 < len(observations) - 1
            and uncertain[end + 1][0]
            and observations[end].source_ordinal + 1
            == observations[end + 1].source_ordinal
        ):
            end += 1
        previous = observations[start - 1]
        following = observations[end + 1]
        if (
            previous.visible
            and following.visible
            and previous.source_ordinal + 1 == observations[start].source_ordinal
            and observations[end].source_ordinal + 1 == following.source_ordinal
        ):
            drafts.append(
                _CandidateDraft(
                    target_track_id=track.track_id,
                    target_entity_id=track.entity_id,
                    allowed_start_times=(previous.timestamp_seconds,),
                    allowed_end_times=(following.timestamp_seconds,),
                    last_visible_frame=previous.frame_index,
                    first_revisible_frame=following.frame_index,
                    edge_departure=(
                        previous.edge_proximity <= _EDGE_PROXIMITY_THRESHOLD
                    ),
                    low_confidence=any(
                        uncertain[offset][1]
                        for offset in range(start, end + 1)
                    ),
                    evidence_frames=_ordered_frames(
                        previous.frame_index,
                        *(
                            observations[offset].frame_index
                            for offset in range(start, end + 1)
                        ),
                        following.frame_index,
                    ),
                )
            )
        index = end + 1
    return tuple(drafts)


def _possible_occluders(
    draft: _CandidateDraft,
    relations: tuple[SpatialRelation, ...],
    entities: Mapping[str, SummaryEntity],
    track_entities: Mapping[str, str],
) -> tuple[tuple[OccluderProvenance, ...], bool]:
    frames = set(draft.evidence_frames)
    possible: dict[tuple[str, str], set[int]] = {}
    for relation in relations:
        if (
            draft.target_track_id
            not in (relation.subject_track_id, relation.object_track_id)
            or relation.frame_index not in frames
            or max(
                relation.bbox_iou,
                relation.subject_bbox_covered_fraction,
                relation.object_bbox_covered_fraction,
            )
            <= 0.0
        ):
            continue
        other_track_id = (
            relation.object_track_id
            if relation.subject_track_id == draft.target_track_id
            else relation.subject_track_id
        )
        entity_id = track_entities.get(other_track_id)
        entity = entities.get(entity_id) if entity_id is not None else None
        if entity is not None and other_track_id != draft.target_track_id:
            possible.setdefault(
                (entity.entity_id, other_track_id), set()
            ).add(relation.frame_index)
    ordered = tuple(sorted(possible.items()))
    complete = len(ordered) <= _MAX_CANDIDATE_OCCLUDERS and all(
        len(supporting_frames) <= _MAX_CANDIDATE_SUPPORTING_FRAMES
        for _, supporting_frames in ordered
    )
    selected = ordered[:_MAX_CANDIDATE_OCCLUDERS]
    return tuple(
        OccluderProvenance(
            entity_id=entity_id,
            track_id=track_id,
            supporting_frames=tuple(sorted(supporting_frames))[
                :_MAX_CANDIDATE_SUPPORTING_FRAMES
            ],
        )
        for (entity_id, track_id), supporting_frames in selected
    ), complete


def _draft_sort_key(draft: _CandidateDraft) -> tuple[Any, ...]:
    return (
        draft.target_entity_id,
        draft.last_visible_frame,
        (
            draft.first_revisible_frame
            if draft.first_revisible_frame is not None
            else 2**63 - 1
        ),
        draft.allowed_start_times,
        draft.allowed_end_times,
        draft.target_track_id,
    )


def _candidate_hash_prefix(payload: Any) -> str:
    return _canonical_sha256(payload)[:12]


def _candidate_overlay_fields(
    overlays: tuple[SummaryOverlay, ...],
    target_track_id: str,
    possible_occluders: tuple[OccluderProvenance, ...],
    evidence_frames: tuple[int, ...],
    *,
    overlays_complete: bool,
) -> dict[str, Any]:
    involved_track_ids = (
        target_track_id,
        *(item.track_id for item in possible_occluders),
    )
    evidence_frame_set = set(evidence_frames)
    track_rank = {
        track_id: ordinal for ordinal, track_id in enumerate(involved_track_ids)
    }
    selected = tuple(
        overlay.path
        for overlay in sorted(
            (
                item
                for item in overlays
                if item.track_id in involved_track_ids
                and item.frame_index in evidence_frame_set
            ),
            key=lambda item: (
                track_rank[item.track_id],
                item.frame_index,
                item.path,
            ),
        )
    )
    return {
        "overlay_refs": selected,
        "overlay_support_complete": overlays_complete,
    }


def _ordered_frames(*values: int | None) -> tuple[int, ...]:
    return tuple(sorted({value for value in values if value is not None}))


__all__ = [
    "CandidateId",
    "CvEvidenceSummary",
    "CvPromptBundle",
    "OccluderProvenance",
    "OcclusionCandidate",
    "SpatialRelation",
    "SummaryId",
    "SummaryEntity",
    "SummaryObservation",
    "SummaryOverlay",
    "SummaryTrack",
    "SummaryWarning",
    "VisibilityGap",
    "VisibilityRun",
    "build_occlusion_candidates",
    "build_cv_prompt_bundle",
    "summarize_cv_evidence",
]
