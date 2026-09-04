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
    StrictModel,
    Timestamp,
    TrackId,
    TrackObservation,
)


CandidateId = Annotated[
    StrictStr, Field(pattern=r"^occ_[0-9a-f]{12}_[0-9]{4}$")
]

_DEFAULT_MAX_TRACKS = 64
_DEFAULT_MAX_OBSERVATIONS_PER_TRACK = 64
_DEFAULT_MAX_RELATIONS = 512
_DEFAULT_MAX_OVERLAYS = 24
_MAX_PROMPT_CHARS = 200_000
_MAX_TRACK_LIMIT = 256
_MAX_OBSERVATION_LIMIT = 256
_MAX_RELATION_LIMIT = 8_192
_MAX_OVERLAY_LIMIT = 24
_MAX_ENTITY_SUMMARIES = 256
_MAX_IDENTIFIER_CHARS = 128
_MAX_LABEL_CHARS = 256
_MAX_ALIAS_CHARS = 128
_MAX_ALIASES_PER_ENTITY = 16
_MAX_CANDIDATE_PROMPT_CHARS = _MAX_PROMPT_CHARS
_MAX_BUNDLE_CANDIDATES = 256
_ABS_MAX_ARTIFACT_ENTITIES = 512
_ABS_MAX_ARTIFACT_TRACKS = 512
_ABS_MAX_ALIASES_PER_INPUT_ENTITY = 4_096
_ABS_MAX_OBSERVATIONS_PER_INPUT_TRACK = 10_000
_ABS_MAX_TOTAL_INPUT_OBSERVATIONS = 1_000_000
_ABS_MAX_ARTIFACT_FILES = 20_000
_ABS_MAX_ARTIFACT_WARNINGS = 1_024
_ABS_MAX_TIMELINE_FRAMES = 100_000
_ABS_MAX_INPUT_STRING_CHARS = 4_096
_ABS_MAX_SUMMARY_WARNINGS = 64
_ABS_MAX_CANDIDATE_INPUTS = 4_096
_EDGE_PROXIMITY_THRESHOLD = 0.05
_MANDATORY_PRIORITY_TEXT = (
    "first,last,state_changes,min_area_context,max_area,lowest_confidence_context"
)
_SAFE_OVERLAY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SummaryEntity(StrictModel):
    """Bounded entity text retained for later trusted-data prompt sections."""

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


class SummaryTrack(StrictModel):
    """A bounded track plus lifecycle gaps derived before downsampling."""

    track_id: TrackId
    entity_id: ObjectId
    status: EvidenceStatus
    source_observation_count: NonnegativeInt
    observations: tuple[SummaryObservation, ...]
    missing_intervals: tuple[VisibilityGap, ...]
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
        return self


class SpatialRelation(StrictModel):
    """A same-frame, directed bounding-box relation proxy."""

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
        if self.subject_track_id == self.object_track_id:
            raise ValueError("a spatial relation needs two tracks")
        return self


class SummaryWarning(StrictModel):
    """A closed warning code with a code-specific scalar payload grammar."""

    code: Literal[
        "ALIASES_TRUNCATED",
        "ARTIFACT_WARNINGS_OMITTED",
        "ENTITIES_TRUNCATED",
        "ENTITY_TEXT_TRUNCATED",
        "LIFECYCLE_GAPS_TRUNCATED",
        "MANDATORY_LANDMARKS_TRUNCATED",
        "OBSERVATIONS_TRUNCATED",
        "OVERLAY_BINDINGS_INCOMPLETE",
        "OVERLAYS_TRUNCATED",
        "PROMPT_DATA_TRUNCATED",
        "RELATIONS_TRUNCATED",
        "TRACKS_TRUNCATED",
        "UNAVAILABLE_EVIDENCE_OMITTED",
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
            "OVERLAY_BINDINGS_INCOMPLETE": {"count"},
            "ARTIFACT_WARNINGS_OMITTED": {"count"},
            "UNAVAILABLE_EVIDENCE_OMITTED": {"count"},
            "PROMPT_DATA_TRUNCATED": {"budget_enforced"},
        }[self.code]
        if populated != required:
            raise ValueError("warning payload does not match its closed code grammar")
        if self.code == "PROMPT_DATA_TRUNCATED" and self.budget_enforced is not True:
            raise ValueError("prompt truncation warning must confirm budget enforcement")
        return self


class CvEvidenceSummary(StrictModel):
    """Frozen evidence summary whose prompt projection has a hard char cap."""

    schema_version: Literal["cv_summary_v1"]
    status: EvidenceStatus
    candidate_search_complete: StrictBool
    observed_clock: tuple[FrameTimestamp, ...]
    entities: tuple[SummaryEntity, ...]
    tracks: tuple[SummaryTrack, ...]
    relations: tuple[SpatialRelation, ...]
    relations_complete: StrictBool
    overlay_refs: tuple[StrictStr, ...]
    overlays_complete: StrictBool
    warnings: tuple[SummaryWarning, ...]
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
        if len(values) > _ABS_MAX_TIMELINE_FRAMES:
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
        if len(values) != len(set(values)):
            raise ValueError("summary warnings must be unique")
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
                }
                for warning in self.warnings
            )
        ):
            raise ValueError("summary candidate-search completeness is inconsistent")
        if self.status is not EvidenceStatus.AVAILABLE and (
            self.tracks
            or self.relations
            or self.overlay_refs
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
        relation_warning = any(
            warning.code == "RELATIONS_TRUNCATED" for warning in self.warnings
        )
        if self.relations_complete == relation_warning:
            raise ValueError("relation completeness must match truncation warning")
        overlay_warning = any(
            warning.code == "OVERLAYS_TRUNCATED" for warning in self.warnings
        )
        if self.overlays_complete == overlay_warning:
            raise ValueError("overlay completeness must match truncation warning")
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return a fresh allowlisted JSON record, rejecting an oversized model."""
        record = _summary_prompt_record(self)
        if _canonical_char_count(record) > self.prompt_char_limit:
            raise ValueError("CV summary prompt record exceeds its character cap")
        return record


class OccluderProvenance(StrictModel):
    """One track-specific source of positive aligned overlap support."""

    entity_id: ObjectId
    track_id: TrackId
    supporting_frames: tuple[NonnegativeInt, ...]

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
    possible_occluders: tuple[OccluderProvenance, ...]
    possible_occluder_entity_ids: tuple[ObjectId, ...]
    allowed_start_times: tuple[Timestamp, ...]
    allowed_end_times: tuple[Timestamp, ...]
    last_visible_frame: NonnegativeInt
    first_revisible_frame: NonnegativeInt | None
    edge_departure: StrictBool
    low_confidence: StrictBool
    overlay_refs: tuple[StrictStr, ...]
    observation_support_complete: StrictBool
    relation_support_complete: StrictBool
    overlay_support_complete: StrictBool
    support_complete: StrictBool
    candidate_set_complete: StrictBool

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
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return fresh allowlisted values without exposing model internals."""
        record: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "target_entity_id": self.target_entity_id,
            "target_track_id": self.target_track_id,
            "possible_occluders": [
                {
                    "entity_id": item.entity_id,
                    "track_id": item.track_id,
                    "supporting_frames": list(item.supporting_frames),
                }
                for item in self.possible_occluders
            ],
            "possible_occluder_entity_ids": list(
                self.possible_occluder_entity_ids
            ),
            "allowed_start_times": list(self.allowed_start_times),
            "allowed_end_times": list(self.allowed_end_times),
            "last_visible_frame": self.last_visible_frame,
            "first_revisible_frame": self.first_revisible_frame,
            "edge_departure": self.edge_departure,
            "low_confidence": self.low_confidence,
            "overlay_refs": list(self.overlay_refs),
            "observation_support_complete": self.observation_support_complete,
            "relation_support_complete": self.relation_support_complete,
            "overlay_support_complete": self.overlay_support_complete,
            "support_complete": self.support_complete,
            "candidate_set_complete": self.candidate_set_complete,
        }
        if _canonical_char_count(record) > _MAX_CANDIDATE_PROMPT_CHARS:
            raise ValueError("occlusion candidate prompt record exceeds its cap")
        return record


class CvPromptBundle(StrictModel):
    """The only aggregate prompt record for one summary and its candidates."""

    schema_version: Literal["cv_prompt_bundle_v1"]
    summary: CvEvidenceSummary
    candidates: tuple[OcclusionCandidate, ...]
    candidates_complete: StrictBool
    truncation_codes: tuple[
        Literal[
            "SUMMARY_CANDIDATE_SEARCH_INCOMPLETE",
            "CANDIDATE_SOURCE_TRUNCATED",
            "CANDIDATE_COUNT_TRUNCATED",
            "CANDIDATE_PROMPT_TRUNCATED",
        ], ...
    ]
    prompt_char_limit: Annotated[
        int, Field(gt=0, le=_MAX_PROMPT_CHARS, strict=True)
    ] = _MAX_PROMPT_CHARS

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
        if self.candidates_complete != (not self.truncation_codes):
            raise ValueError("candidate completeness must match truncation codes")
        if (
            "SUMMARY_CANDIDATE_SEARCH_INCOMPLETE" in self.truncation_codes
        ) == self.summary.candidate_search_complete:
            raise ValueError("bundle summary-search completeness is inconsistent")
        if any(not item.candidate_set_complete for item in self.candidates) and (
            "CANDIDATE_SOURCE_TRUNCATED" not in self.truncation_codes
        ):
            raise ValueError("bundle omits candidate source-truncation signal")
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
                relation.subject_track_id,
                relation.object_track_id,
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
            if track_entities.get(candidate.target_track_id) != candidate.target_entity_id:
                raise ValueError("candidate target provenance is not closed to summary")
            if any(
                track_entities.get(item.track_id) != item.entity_id
                for item in candidate.possible_occluders
            ):
                raise ValueError("candidate occluder provenance is not closed to summary")
            if any(
                (
                    candidate.target_track_id,
                    item.track_id,
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
            if any(reference not in summary_overlays for reference in candidate.overlay_refs):
                raise ValueError("candidate overlay is not closed to summary")
            involved_tracks = {
                candidate.target_track_id,
                *(item.track_id for item in candidate.possible_occluders),
            }
            if any(
                (resolved := _resolve_overlay_track(reference, track_entities)) is None
                or resolved[0] not in involved_tracks
                for reference in candidate.overlay_refs
            ):
                raise ValueError("candidate overlay provenance is not closed to summary")
            target_track = tracks_by_id[candidate.target_track_id]
            if candidate.observation_support_complete != (
                target_track.candidate_search_complete
            ):
                raise ValueError("candidate observation completeness disagrees with summary")
            if candidate.relation_support_complete != self.summary.relations_complete:
                raise ValueError("candidate relation completeness disagrees with summary")
        if _canonical_char_count(_bundle_prompt_record(self)) > self.prompt_char_limit:
            raise ValueError("aggregate CV prompt record exceeds its character cap")
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return one fresh, allowlisted, aggregate record for a model job."""
        record = _bundle_prompt_record(self)
        if _canonical_char_count(record) > self.prompt_char_limit:
            raise ValueError("aggregate CV prompt record exceeds its character cap")
        return record


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
    total_relations: int
    kept_relations: int
    total_overlays: int
    kept_overlays: int
    artifact_warning_count: int
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
    # dip before dropping whole, stably ordered tracks.  Only an exceptionally
    # small budget is allowed to reduce the final retained track below three.
    for reduced_observation_cap in range(observation_cap - 1, 2, -1):
        observation_cap = reduced_observation_cap
        summary = assemble()
        if _summary_fits(summary, max_prompt_chars):
            summary.prompt_record()
            return summary

    for reduced_track_cap in range(track_cap - 1, 0, -1):
        track_cap = reduced_track_cap
        summary = assemble()
        if _summary_fits(summary, max_prompt_chars):
            summary.prompt_record()
            return summary

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
    validated_summary = _revalidate_summary(summary)
    validated_summary.prompt_record()
    if validated_summary.status is not EvidenceStatus.AVAILABLE:
        return ()
    validated_thresholds = _revalidate_thresholds(thresholds)
    tracks = tuple(
        sorted(validated_summary.tracks, key=lambda item: item.track_id)
    )
    bounded_drafts = heapq.nsmallest(
        _MAX_BUNDLE_CANDIDATES + 1,
        _candidate_drafts(tracks, validated_thresholds),
        key=_draft_sort_key,
    )
    candidate_set_complete = len(bounded_drafts) <= _MAX_BUNDLE_CANDIDATES
    ordered_drafts = tuple(bounded_drafts[:_MAX_BUNDLE_CANDIDATES])
    entities = {item.entity_id: item for item in validated_summary.entities}
    track_entities = {item.track_id: item.entity_id for item in tracks}
    candidates: list[OcclusionCandidate] = []
    for ordinal, draft in enumerate(ordered_drafts, start=1):
        possible_occluders = _possible_occluders(
            draft,
            validated_summary.relations,
            entities,
            track_entities,
        )
        possible_occluder_entity_ids = tuple(
            sorted({item.entity_id for item in possible_occluders})
        )
        payload = _draft_hash_payload(draft, possible_occluders)
        prefix = _candidate_hash_prefix(payload)
        if re.fullmatch(r"[0-9a-f]{12}", prefix) is None:
            raise ValueError("candidate hash prefix is invalid")
        overlay_fields = _candidate_overlay_fields(
            validated_summary.overlay_refs,
            draft.target_track_id,
            possible_occluders,
            draft.evidence_frames,
            track_entities,
            overlays_complete=validated_summary.overlays_complete,
        )
        observation_support_complete = next(
            track.candidate_search_complete
            for track in tracks
            if track.track_id == draft.target_track_id
        )
        support_complete = (
            observation_support_complete
            and validated_summary.relations_complete
            and bool(overlay_fields["overlay_support_complete"])
        )
        candidate = OcclusionCandidate(
            candidate_id=f"occ_{prefix}_{ordinal:04d}",
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
            relation_support_complete=validated_summary.relations_complete,
            support_complete=support_complete,
            candidate_set_complete=candidate_set_complete,
            **overlay_fields,
        )
        candidate.prompt_record()
        candidates.append(candidate)
    return tuple(candidates)


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
    candidates: tuple[OcclusionCandidate, ...],
    *,
    max_candidates: int = _MAX_BUNDLE_CANDIDATES,
    max_prompt_chars: int | None = None,
) -> CvPromptBundle:
    """Build a deterministic, whole-prompt-capped semantic-stage input."""
    _validate_limit(
        "max_candidates", max_candidates, maximum=_MAX_BUNDLE_CANDIDATES
    )
    _preflight_summary(summary)
    _preflight_candidates(candidates)
    validated_summary = _revalidate_summary(summary)
    validated_summary.prompt_record()
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
    validated_candidates = tuple(
        _revalidate_candidate(item) for item in candidates
    )
    ordered = tuple(sorted(validated_candidates, key=_candidate_sort_key))
    if len({item.candidate_id for item in ordered}) != len(ordered):
        raise ValueError("candidate IDs must be unique before prompt bundling")
    kept = list(ordered[:max_candidates])
    summary_search_incomplete = not validated_summary.candidate_search_complete
    source_truncated = any(not item.candidate_set_complete for item in ordered)
    count_truncated = len(ordered) > len(kept)
    prompt_truncated = False
    while True:
        truncation_codes = tuple(
            code
            for code, enabled in (
                (
                    "SUMMARY_CANDIDATE_SEARCH_INCOMPLETE",
                    summary_search_incomplete,
                ),
                ("CANDIDATE_SOURCE_TRUNCATED", source_truncated),
                ("CANDIDATE_COUNT_TRUNCATED", count_truncated),
                ("CANDIDATE_PROMPT_TRUNCATED", prompt_truncated),
            )
            if enabled
        )
        record = _bundle_prompt_record_values(
            validated_summary,
            tuple(kept),
            candidates_complete=not truncation_codes,
            truncation_codes=truncation_codes,
        )
        if _canonical_char_count(record) <= effective_prompt_limit:
            return CvPromptBundle(
                schema_version="cv_prompt_bundle_v1",
                summary=validated_summary,
                candidates=tuple(kept),
                candidates_complete=not truncation_codes,
                truncation_codes=truncation_codes,
                prompt_char_limit=effective_prompt_limit,
            )
        if not kept:
            raise ValueError("CV summary leaves no room for an aggregate prompt bundle")
        kept.pop()
        prompt_truncated = True


def _structural_error(name: str) -> ValueError:
    return ValueError(f"{name} exceeds the CV structural input bound")


def _preflight_tuple(value: object, name: str, maximum: int) -> tuple[Any, ...]:
    if not isinstance(value, tuple) or len(value) > maximum:
        raise _structural_error(name)
    return value


def _preflight_text(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) > _ABS_MAX_INPUT_STRING_CHARS:
        raise _structural_error(name)
    return value


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
        if not isinstance(frame, FrameTimestamp):
            raise _structural_error("timeline.frames item type")


def _preflight_summary(summary: object) -> None:
    if not isinstance(summary, CvEvidenceSummary):
        raise _structural_error("summary")
    _preflight_text(
        getattr(summary, "schema_version", None), "summary.schema_version"
    )
    observed_clock = _preflight_tuple(
        getattr(summary, "observed_clock", None),
        "summary.observed_clock",
        _ABS_MAX_TIMELINE_FRAMES,
    )
    if any(not isinstance(item, FrameTimestamp) for item in observed_clock):
        raise _structural_error("summary.observed_clock item type")
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
        gaps = _preflight_tuple(
            track.missing_intervals,
            "summary missing intervals",
            _MAX_OBSERVATION_LIMIT,
        )
        if any(not isinstance(item, VisibilityGap) for item in gaps):
            raise _structural_error("summary missing intervals item type")

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
    overlays = _preflight_tuple(
        getattr(summary, "overlay_refs", None),
        "summary.overlay_refs",
        _MAX_OVERLAY_LIMIT,
    )
    for overlay in overlays:
        _preflight_text(overlay, "summary overlay reference")
    summary_warnings = _preflight_tuple(
        getattr(summary, "warnings", None),
        "summary.warnings",
        _ABS_MAX_SUMMARY_WARNINGS,
    )
    for warning in summary_warnings:
        if not isinstance(warning, SummaryWarning):
            raise _structural_error("summary warning item type")
        _preflight_text(warning.code, "summary warning code")


def _preflight_candidates(candidates: object) -> None:
    values = _preflight_tuple(
        candidates, "candidates", _ABS_MAX_CANDIDATE_INPUTS
    )
    for candidate in values:
        if not isinstance(candidate, OcclusionCandidate):
            raise _structural_error("candidates item type")
        _preflight_text(candidate.candidate_id, "candidate_id")
        _preflight_text(candidate.target_entity_id, "candidate target_entity_id")
        _preflight_text(candidate.target_track_id, "candidate target_track_id")
        possible_occluders = _preflight_tuple(
            candidate.possible_occluders,
            "candidate possible_occluders",
            _MAX_TRACK_LIMIT,
        )
        for provenance in possible_occluders:
            if not isinstance(provenance, OccluderProvenance):
                raise _structural_error("candidate possible_occluders item type")
            _preflight_text(provenance.entity_id, "candidate occluder entity_id")
            _preflight_text(provenance.track_id, "candidate occluder track_id")
            _preflight_tuple(
                provenance.supporting_frames,
                "candidate occluder supporting_frames",
                _MAX_OBSERVATION_LIMIT,
            )
        possible_entity_ids = _preflight_tuple(
            candidate.possible_occluder_entity_ids,
            "candidate possible_occluder_entity_ids",
            _MAX_ENTITY_SUMMARIES,
        )
        for entity_id in possible_entity_ids:
            _preflight_text(entity_id, "candidate possible occluder entity_id")
        _preflight_tuple(
            candidate.allowed_start_times, "candidate start times", 8
        )
        _preflight_tuple(candidate.allowed_end_times, "candidate end times", 8)
        overlays = _preflight_tuple(
            candidate.overlay_refs,
            "candidate overlay_refs",
            _MAX_OVERLAY_LIMIT,
        )
        for overlay in overlays:
            _preflight_text(overlay, "candidate overlay reference")


def _revalidate_artifact(artifact: CvEvidenceArtifact) -> CvEvidenceArtifact:
    payload = (
        artifact.model_dump(mode="python", warnings=False)
        if isinstance(artifact, CvEvidenceArtifact)
        else artifact
    )
    return CvEvidenceArtifact.model_validate(payload, strict=True)


def _revalidate_timeline(timeline: FrameTimeline) -> FrameTimeline:
    payload = (
        timeline.model_dump(mode="python", warnings=False)
        if isinstance(timeline, FrameTimeline)
        else timeline
    )
    return FrameTimeline.model_validate(payload, strict=True)


def _revalidate_summary(summary: CvEvidenceSummary) -> CvEvidenceSummary:
    payload = (
        summary.model_dump(mode="python", warnings=False)
        if isinstance(summary, CvEvidenceSummary)
        else summary
    )
    return CvEvidenceSummary.model_validate(payload, strict=True)


def _revalidate_candidate(candidate: OcclusionCandidate) -> OcclusionCandidate:
    payload = (
        candidate.model_dump(mode="python", warnings=False)
        if isinstance(candidate, OcclusionCandidate)
        else candidate
    )
    return OcclusionCandidate.model_validate(payload, strict=True)


def _revalidate_thresholds(thresholds: EvidenceThresholds) -> EvidenceThresholds:
    payload = (
        thresholds.model_dump(mode="python", warnings=False)
        if isinstance(thresholds, EvidenceThresholds)
        else thresholds
    )
    return EvidenceThresholds.model_validate(payload, strict=True)


def _validate_limit(name: str, value: int, *, maximum: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= maximum
    ):
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
    observed: dict[int, float] = {}
    evidence_tracks = (
        artifact.tracks
        if artifact.status is EvidenceStatus.AVAILABLE
        else ()
    )
    for track in evidence_tracks:
        if track.status is not EvidenceStatus.AVAILABLE:
            continue
        for observation in track.observations:
            prior = observed.setdefault(
                observation.frame_index, observation.timestamp_seconds
            )
            if prior != observation.timestamp_seconds:
                raise ValueError("frame timeline has conflicting timestamps")

    if timeline is not None:
        declared = {
            item.frame_index: item.timestamp_seconds for item in timeline.frames
        }
        if any(
            frame_index not in declared or declared[frame_index] != timestamp
            for frame_index, timestamp in observed.items()
        ):
            raise ValueError("frame timeline does not align with observations")
        return timeline.frames

    ordered = tuple(
        FrameTimestamp(frame_index=frame_index, timestamp_seconds=timestamp)
        for frame_index, timestamp in sorted(observed.items())
    )
    for previous, current in zip(ordered, ordered[1:]):
        if previous.timestamp_seconds >= current.timestamp_seconds:
            raise ValueError("frame timeline must increase globally")
    return ordered


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
    for track in selected_tracks:
        if track.status is not EvidenceStatus.AVAILABLE:
            summary_tracks.append(
                SummaryTrack(
                    track_id=track.track_id,
                    entity_id=track.entity_id,
                    status=track.status,
                    source_observation_count=len(track.observations),
                    observations=(),
                    missing_intervals=(),
                    candidate_search_complete=False,
                )
            )
            continue
        selected, mandatory_conflict = _reduce_observations(
            track.observations, observation_cap
        )
        mandatory_conflicts += int(mandatory_conflict)
        derived_gaps, track_gap_count = _derive_missing_intervals(
            track, frame_clock, cap=observation_cap
        )
        retained_visible_frames = {
            observation.frame_index
            for _, observation in selected
            if observation.visible
        }
        retained_gaps = tuple(
            gap
            for gap in derived_gaps
            if gap.last_visible_frame in retained_visible_frames
            and (
                gap.first_revisible_frame is None
                or gap.first_revisible_frame in retained_visible_frames
            )
        )
        total_gaps += track_gap_count
        kept_gaps += len(retained_gaps)
        summary_tracks.append(
            SummaryTrack(
                track_id=track.track_id,
                entity_id=track.entity_id,
                status=track.status,
                source_observation_count=len(track.observations),
                observations=tuple(
                    _summary_observation(source_ordinal, observation)
                    for source_ordinal, observation in selected
                ),
                missing_intervals=retained_gaps,
                candidate_search_complete=(
                    len(selected) == len(track.observations)
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
    relations, total_relations = _relations(tracks, cap=relation_cap)
    all_overlays = _overlay_refs(artifact)
    overlays = (
        all_overlays[:overlay_cap]
        if artifact.status is EvidenceStatus.AVAILABLE
        else ()
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
        total_relations=total_relations,
        kept_relations=len(relations),
        total_overlays=len(all_overlays),
        kept_overlays=len(overlays),
        artifact_warning_count=len(artifact.warnings),
        unavailable_evidence_count=(
            len(all_tracks)
            if artifact.status is not EvidenceStatus.AVAILABLE
            else sum(
                track.status is not EvidenceStatus.AVAILABLE
                for track in selected_tracks
            )
        ),
    )
    return CvEvidenceSummary(
        schema_version="cv_summary_v1",
        status=artifact.status,
        candidate_search_complete=(
            artifact.status is EvidenceStatus.AVAILABLE
            and len(selected_tracks) == len(all_tracks)
            and all(track.candidate_search_complete for track in tracks)
        ),
        observed_clock=_retained_observed_clock(frame_clock, tracks),
        entities=entities,
        tracks=tracks,
        relations=relations,
        relations_complete=total_relations == len(relations),
        overlay_refs=overlays,
        overlays_complete=len(all_overlays) == len(overlays),
        warnings=_summary_warnings(metadata, prompt_truncated=prompt_truncated),
        prompt_char_limit=prompt_char_limit,
    )


def _retained_observed_clock(
    frame_clock: tuple[FrameTimestamp, ...],
    tracks: tuple[SummaryTrack, ...],
) -> tuple[FrameTimestamp, ...]:
    required_frames: set[int] = set()
    for track in tracks:
        required_frames.update(item.frame_index for item in track.observations)
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
    observations: tuple[TrackObservation, ...], cap: int
) -> tuple[tuple[tuple[int, TrackObservation], ...], bool]:
    if not observations:
        return (), False
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
            if 0 <= contextual_index < len(observations):
                add(contextual_index)

    add(0)
    add(len(observations) - 1)
    for index, (previous, current) in enumerate(
        zip(observations, observations[1:]), start=1
    ):
        if previous.visible != current.visible:
            add(index - 1)
            add(index)
    add_context(
        min(
            range(len(observations)),
            key=lambda index: (
                observations[index].area_fraction,
                observations[index].frame_index,
            ),
        )
    )
    add(
        min(
            range(len(observations)),
            key=lambda index: (
                -observations[index].area_fraction,
                observations[index].frame_index,
            ),
        )
    )
    add_context(
        min(
            range(len(observations)),
            key=lambda index: (
                observations[index].confidence,
                observations[index].frame_index,
            ),
        )
    )
    chosen = list(priority)
    remaining_slots = cap - len(chosen)
    if remaining_slots > 0 and len(selected_indices) < len(observations):
        for slot in range(1, remaining_slots + 1):
            ideal = slot * (len(observations) - 1) / (remaining_slots + 1)
            selected = _nearest_unselected_index(
                ideal, len(observations), selected_indices
            )
            if selected is None:
                break
            chosen.append(selected)
            selected_indices.add(selected)
    if len(chosen) < cap:
        for index in range(len(observations)):
            if index in selected_indices:
                continue
            chosen.append(index)
            selected_indices.add(index)
            if len(chosen) == cap:
                break
    return (
        tuple((index, observations[index]) for index in sorted(chosen)),
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


def _derive_missing_intervals(
    track: CvTrack,
    frame_clock: tuple[FrameTimestamp, ...],
    *,
    cap: int,
) -> tuple[tuple[VisibilityGap, ...], int]:
    if not frame_clock:
        return (), 0
    observations = {item.frame_index: item for item in track.observations}
    gaps: list[VisibilityGap] = []
    gap_count = 0
    last_visible: TrackObservation | None = None
    offset = 0
    while offset < len(frame_clock):
        frame = frame_clock[offset]
        observation = observations.get(frame.frame_index)
        if observation is not None and observation.visible:
            last_visible = observation
            offset += 1
            continue
        if last_visible is None:
            offset += 1
            continue
        first_missing = frame
        last_missing = frame
        scan = offset
        revisible: TrackObservation | None = None
        missing_confidences: list[float] = []
        while scan < len(frame_clock):
            current_frame = frame_clock[scan]
            current = observations.get(current_frame.frame_index)
            if current is not None and current.visible:
                revisible = current
                break
            if current is not None:
                missing_confidences.append(current.confidence)
            last_missing = current_frame
            scan += 1
        gap_count += 1
        if len(gaps) < cap:
            gaps.append(
                VisibilityGap(
                    last_visible_frame=last_visible.frame_index,
                    last_visible_time=last_visible.timestamp_seconds,
                    first_missing_frame=first_missing.frame_index,
                    first_missing_time=first_missing.timestamp_seconds,
                    last_missing_frame=last_missing.frame_index,
                    last_missing_time=last_missing.timestamp_seconds,
                    first_revisible_frame=(
                        revisible.frame_index if revisible is not None else None
                    ),
                    first_revisible_time=(
                        revisible.timestamp_seconds
                        if revisible is not None
                        else None
                    ),
                    minimum_confidence=(
                        min(missing_confidences) if missing_confidences else None
                    ),
                    edge_departure=_edge_departure(last_visible),
                )
            )
        if revisible is None:
            break
        last_visible = revisible
        offset = scan + 1
    return tuple(gaps), gap_count


def _relations(
    tracks: tuple[SummaryTrack, ...], *, cap: int
) -> tuple[tuple[SpatialRelation, ...], int]:
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
        len(values) * (len(values) - 1) for values in aligned_by_frame.values()
    )
    if cap == 0:
        return (), total_relations
    for retain_positive in (True, False):
        for frame_index in sorted(aligned_by_frame):
            aligned = aligned_by_frame[frame_index]
            for subject_track_id, subject in aligned:
                for object_track_id, object_observation in aligned:
                    if subject_track_id == object_track_id:
                        continue
                    relation = _spatial_relation(
                        subject_track_id,
                        subject,
                        object_track_id,
                        object_observation,
                    )
                    positive = max(
                        relation.bbox_iou,
                        relation.subject_bbox_covered_fraction,
                        relation.object_bbox_covered_fraction,
                    ) > 0.0
                    if positive != retain_positive:
                        continue
                    relations.append(relation)
                    if len(relations) == cap:
                        return tuple(relations), total_relations
    return tuple(relations), total_relations


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


def _overlay_refs(artifact: CvEvidenceArtifact) -> tuple[str, ...]:
    references: list[str] = []
    for artifact_file in artifact.files:
        if not artifact_file.path.startswith("overlays/"):
            continue
        _validate_overlay_ref(artifact_file.path)
        references.append(artifact_file.path)
    return tuple(sorted(references))


def _validate_overlay_ref(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 512
        or "\\" in value
        or value.startswith("/")
    ):
        raise ValueError("overlay reference must be bounded relative POSIX")
    parts = value.split("/")
    if (
        len(parts) < 2
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


def _summary_prompt_record(summary: CvEvidenceSummary) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "status": summary.status.value,
        "candidate_search_complete": summary.candidate_search_complete,
        "observed_clock": [
            {
                "frame_index": item.frame_index,
                "timestamp_seconds": item.timestamp_seconds,
            }
            for item in summary.observed_clock
        ],
        "entities": [
            {
                "entity_id": entity.entity_id,
                "canonical_label": entity.canonical_label,
                "aliases": list(entity.aliases),
                "role": entity.role.value,
            }
            for entity in summary.entities
        ],
        "tracks": [
            {
                "track_id": track.track_id,
                "entity_id": track.entity_id,
                "status": track.status.value,
                "source_observation_count": track.source_observation_count,
                "candidate_search_complete": track.candidate_search_complete,
                "observations": [
                    {
                        "source_ordinal": observation.source_ordinal,
                        "frame_index": observation.frame_index,
                        "timestamp_seconds": observation.timestamp_seconds,
                        "bbox_xyxy": list(observation.bbox_xyxy),
                        "visible": observation.visible,
                        "confidence": observation.confidence,
                        "area_fraction": observation.area_fraction,
                        "center_xy": list(observation.center_xy),
                        "edge_proximity": observation.edge_proximity,
                    }
                    for observation in track.observations
                ],
                "missing_intervals": [
                    {
                        "last_visible_frame": gap.last_visible_frame,
                        "last_visible_time": gap.last_visible_time,
                        "first_missing_frame": gap.first_missing_frame,
                        "first_missing_time": gap.first_missing_time,
                        "last_missing_frame": gap.last_missing_frame,
                        "last_missing_time": gap.last_missing_time,
                        "first_revisible_frame": gap.first_revisible_frame,
                        "first_revisible_time": gap.first_revisible_time,
                        "minimum_confidence": gap.minimum_confidence,
                        "edge_departure": gap.edge_departure,
                    }
                    for gap in track.missing_intervals
                ],
            }
            for track in summary.tracks
        ],
        "relations": [
            {
                "frame_index": relation.frame_index,
                "timestamp_seconds": relation.timestamp_seconds,
                "subject_track_id": relation.subject_track_id,
                "object_track_id": relation.object_track_id,
                "bbox_iou": relation.bbox_iou,
                "subject_bbox_covered_fraction": (
                    relation.subject_bbox_covered_fraction
                ),
                "object_bbox_covered_fraction": (
                    relation.object_bbox_covered_fraction
                ),
                "center_distance_fraction": relation.center_distance_fraction,
                "area_similarity": relation.area_similarity,
                "subject_inside_object": relation.subject_inside_object,
                "object_inside_subject": relation.object_inside_subject,
            }
            for relation in summary.relations
        ],
        "relations_complete": summary.relations_complete,
        "overlay_refs": list(summary.overlay_refs),
        "overlays_complete": summary.overlays_complete,
        "warnings": [_warning_prompt_record(item) for item in summary.warnings],
    }


def _warning_prompt_record(warning: SummaryWarning) -> dict[str, Any]:
    record: dict[str, Any] = {"code": warning.code}
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
            record[field_name] = value
    return record


def _bundle_prompt_record_values(
    summary: CvEvidenceSummary,
    candidates: tuple[OcclusionCandidate, ...],
    *,
    candidates_complete: bool,
    truncation_codes: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "cv_prompt_bundle_v1",
        "summary": summary.prompt_record(),
        "candidates": [candidate.prompt_record() for candidate in candidates],
        "candidates_complete": candidates_complete,
        "truncation_codes": list(truncation_codes),
    }


def _bundle_prompt_record(bundle: CvPromptBundle) -> dict[str, Any]:
    return _bundle_prompt_record_values(
        bundle.summary,
        bundle.candidates,
        candidates_complete=bundle.candidates_complete,
        truncation_codes=bundle.truncation_codes,
    )


def _canonical_char_count(record: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _summary_fits(summary: CvEvidenceSummary, cap: int) -> bool:
    return (
        len(summary.model_dump_json()) <= cap
        and _canonical_char_count(_summary_prompt_record(summary)) <= cap
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
) -> tuple[OccluderProvenance, ...]:
    frames = set(draft.evidence_frames)
    possible: dict[tuple[str, str], set[int]] = {}
    for relation in relations:
        if (
            relation.subject_track_id != draft.target_track_id
            or relation.frame_index not in frames
            or max(
                relation.bbox_iou,
                relation.subject_bbox_covered_fraction,
                relation.object_bbox_covered_fraction,
            )
            <= 0.0
        ):
            continue
        entity_id = track_entities.get(relation.object_track_id)
        entity = entities.get(entity_id) if entity_id is not None else None
        if entity is not None and relation.object_track_id != draft.target_track_id:
            possible.setdefault(
                (entity.entity_id, relation.object_track_id), set()
            ).add(relation.frame_index)
    return tuple(
        OccluderProvenance(
            entity_id=entity_id,
            track_id=track_id,
            supporting_frames=tuple(sorted(supporting_frames)),
        )
        for (entity_id, track_id), supporting_frames in sorted(possible.items())
    )


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


def _candidate_sort_key(candidate: OcclusionCandidate) -> tuple[Any, ...]:
    return (
        candidate.target_entity_id,
        candidate.target_track_id,
        candidate.last_visible_frame,
        (
            candidate.first_revisible_frame
            if candidate.first_revisible_frame is not None
            else 2**63 - 1
        ),
        candidate.allowed_start_times,
        candidate.allowed_end_times,
        candidate.candidate_id,
    )


def _draft_hash_payload(
    draft: _CandidateDraft,
    possible_occluders: tuple[OccluderProvenance, ...],
) -> dict[str, Any]:
    return {
        "target_track_id": draft.target_track_id,
        "target_entity_id": draft.target_entity_id,
        "possible_occluders": [
            {
                "entity_id": item.entity_id,
                "track_id": item.track_id,
                "supporting_frames": list(item.supporting_frames),
            }
            for item in possible_occluders
        ],
        "allowed_start_times": list(draft.allowed_start_times),
        "allowed_end_times": list(draft.allowed_end_times),
        "last_visible_frame": draft.last_visible_frame,
        "first_revisible_frame": draft.first_revisible_frame,
        "edge_departure": draft.edge_departure,
        "low_confidence": draft.low_confidence,
    }


def _candidate_hash_prefix(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def _candidate_overlay_fields(
    overlay_refs: tuple[str, ...],
    target_track_id: str,
    possible_occluders: tuple[OccluderProvenance, ...],
    evidence_frames: tuple[int, ...],
    track_entities: Mapping[str, str],
    *,
    overlays_complete: bool,
) -> dict[str, Any]:
    selected: list[str] = []
    involved_track_ids = (
        target_track_id,
        *(item.track_id for item in possible_occluders),
    )
    involved_entities = {
        track_entities[track_id] for track_id in involved_track_ids
    }
    evidence_frame_set = set(evidence_frames)
    complete = overlays_complete
    resolved_selected: list[tuple[str, int, str]] = []
    for reference in overlay_refs:
        resolved = _resolve_overlay_track(reference, track_entities)
        if resolved is None:
            if _overlay_may_describe(
                reference, involved_entities, evidence_frame_set
            ):
                complete = False
            continue
        track_id, frame_index = resolved
        if track_id in involved_track_ids and frame_index in evidence_frame_set:
            resolved_selected.append((track_id, frame_index, reference))
    track_rank = {
        track_id: ordinal for ordinal, track_id in enumerate(involved_track_ids)
    }
    selected.extend(
        reference
        for track_id, _, reference in sorted(
            resolved_selected,
            key=lambda item: (track_rank[item[0]], item[1], item[2]),
        )
    )
    return {
        "overlay_refs": tuple(selected),
        "overlay_support_complete": complete,
    }


def _resolve_overlay_track(
    reference: str, track_entities: Mapping[str, str]
) -> tuple[str, int] | None:
    name = reference.rsplit("/", 1)[-1]
    match = re.fullmatch(r"(.+)-([0-9]{8})\.png", name)
    if match is None:
        return None
    stem, frame_text = match.groups()
    matches: list[str] = []
    for track_id, entity_id in track_entities.items():
        prefix = f"{entity_id}_"
        if not track_id.startswith(prefix):
            continue
        object_id = track_id[len(prefix) :]
        if object_id.isdigit() and stem == f"{entity_id}-{object_id}":
            matches.append(track_id)
    if len(matches) != 1:
        return None
    return matches[0], int(frame_text)


def _overlay_may_describe(
    reference: str,
    entity_ids: set[str],
    evidence_frames: set[int],
) -> bool:
    name = reference.rsplit("/", 1)[-1]
    match = re.fullmatch(r"(.+)-([0-9]{8})\.png", name)
    if match is None or int(match.group(2)) not in evidence_frames:
        return False
    return any(match.group(1).startswith(f"{entity_id}-") for entity_id in entity_ids)


def _ordered_frames(*values: int | None) -> tuple[int, ...]:
    return tuple(sorted({value for value in values if value is not None}))


def _edge_departure(observation: TrackObservation) -> bool:
    left, top, right, bottom = observation.bbox_xyxy
    return min(left, top, 1.0 - right, 1.0 - bottom) <= _EDGE_PROXIMITY_THRESHOLD


__all__ = [
    "CandidateId",
    "CvEvidenceSummary",
    "CvPromptBundle",
    "OccluderProvenance",
    "OcclusionCandidate",
    "SpatialRelation",
    "SummaryEntity",
    "SummaryObservation",
    "SummaryTrack",
    "SummaryWarning",
    "VisibilityGap",
    "build_occlusion_candidates",
    "build_cv_prompt_bundle",
    "summarize_cv_evidence",
]
