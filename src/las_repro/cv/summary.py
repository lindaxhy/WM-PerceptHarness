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
import json
import math
import re
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictStr, field_validator, model_validator

from .contracts import (
    Confidence,
    CvEvidenceArtifact,
    CvTrack,
    EntityRole,
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
_EDGE_PROXIMITY_THRESHOLD = 0.05
_MANDATORY_PRIORITY_TEXT = (
    "first,last,state_changes,min_area,max_area,lowest_confidence"
)
_WARNING_CODES = frozenset(
    {
        "ALIASES_TRUNCATED",
        "ARTIFACT_WARNINGS_OMITTED",
        "ENTITIES_TRUNCATED",
        "ENTITY_TEXT_TRUNCATED",
        "LIFECYCLE_GAPS_TRUNCATED",
        "MANDATORY_LANDMARKS_TRUNCATED",
        "OBSERVATIONS_TRUNCATED",
        "OVERLAYS_TRUNCATED",
        "PROMPT_DATA_TRUNCATED",
        "RELATIONS_TRUNCATED",
        "TRACKS_TRUNCATED",
    }
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

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: EvidenceStatus | str) -> EvidenceStatus:
        return EvidenceStatus(value)

    @model_validator(mode="after")
    def validate_selected_order(self) -> SummaryTrack:
        if len(self.observations) > self.source_observation_count:
            raise ValueError("selected observations exceed source count")
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


class CvEvidenceSummary(StrictModel):
    """Frozen evidence summary whose prompt projection has a hard char cap."""

    schema_version: Literal["cv_summary_v1"]
    status: EvidenceStatus
    entities: tuple[SummaryEntity, ...]
    tracks: tuple[SummaryTrack, ...]
    relations: tuple[SpatialRelation, ...]
    overlay_refs: tuple[StrictStr, ...]
    warnings: tuple[StrictStr, ...]
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

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("summary warnings must be unique")
        for value in values:
            if len(value) > 512 or not value.isascii() or "/" in value or "\\" in value:
                raise ValueError("summary warning is not safely bounded")
            code = value.split(":", 1)[0]
            if code not in _WARNING_CODES:
                raise ValueError("unknown summary warning code")
        return values

    @model_validator(mode="after")
    def validate_closed_references(self) -> CvEvidenceSummary:
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
        frame_times: dict[int, float] = {}
        for track in self.tracks:
            for observation in track.observations:
                prior = frame_times.setdefault(
                    observation.frame_index, observation.timestamp_seconds
                )
                if prior != observation.timestamp_seconds:
                    raise ValueError("frame timeline has conflicting timestamps")
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
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return a fresh allowlisted JSON record, rejecting an oversized model."""
        record = _summary_prompt_record(self)
        if _canonical_char_count(record) > self.prompt_char_limit:
            raise ValueError("CV summary prompt record exceeds its character cap")
        return record


class OcclusionCandidate(StrictModel):
    """An evidence-supported transition awaiting semantic adjudication."""

    candidate_id: CandidateId
    target_entity_id: ObjectId
    possible_occluder_entity_ids: tuple[ObjectId, ...]
    allowed_start_times: tuple[Timestamp, ...]
    allowed_end_times: tuple[Timestamp, ...]
    last_visible_frame: NonnegativeInt
    first_revisible_frame: NonnegativeInt | None
    edge_departure: StrictBool
    low_confidence: StrictBool
    overlay_refs: tuple[StrictStr, ...]

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
        if max(self.allowed_start_times) > min(self.allowed_end_times):
            raise ValueError("candidate start times must not follow end times")
        if self.target_entity_id in self.possible_occluder_entity_ids:
            raise ValueError("target cannot be its own possible occluder")
        if tuple(sorted(set(self.possible_occluder_entity_ids))) != (
            self.possible_occluder_entity_ids
        ):
            raise ValueError("possible occluders must be unique and ordered")
        return self

    def prompt_record(self) -> dict[str, Any]:
        """Return fresh allowlisted values without exposing model internals."""
        record: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "target_entity_id": self.target_entity_id,
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
        }
        if _canonical_char_count(record) > _MAX_CANDIDATE_PROMPT_CHARS:
            raise ValueError("occlusion candidate prompt record exceeds its cap")
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

    for reduced_observation_cap in range(observation_cap - 1, 0, -1):
        observation_cap = reduced_observation_cap
        summary = assemble()
        if _summary_fits(summary, max_prompt_chars):
            summary.prompt_record()
            return summary

    for reduced_track_cap in range(track_cap - 1, -1, -1):
        track_cap = reduced_track_cap
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
    validated_summary = _revalidate_summary(summary)
    validated_summary.prompt_record()
    validated_thresholds = _revalidate_thresholds(thresholds)
    tracks = tuple(
        sorted(validated_summary.tracks, key=lambda item: item.track_id)
    )
    drafts: list[_CandidateDraft] = []
    for track in tracks:
        for gap in sorted(
            track.missing_intervals,
            key=lambda item: (item.first_missing_frame, item.last_missing_frame),
        ):
            drafts.append(
                _CandidateDraft(
                    target_track_id=track.track_id,
                    target_entity_id=track.entity_id,
                    allowed_start_times=_ordered_times(
                        gap.last_visible_time, gap.first_missing_time
                    ),
                    allowed_end_times=_ordered_times(
                        gap.last_missing_time,
                        gap.first_revisible_time,
                    ),
                    last_visible_frame=gap.last_visible_frame,
                    first_revisible_frame=gap.first_revisible_frame,
                    edge_departure=gap.edge_departure,
                    low_confidence=(
                        gap.minimum_confidence is not None
                        and gap.minimum_confidence
                        < validated_thresholds.min_confidence
                    ),
                    evidence_frames=_ordered_frames(
                        gap.last_visible_frame,
                        gap.first_missing_frame,
                        gap.last_missing_frame,
                        gap.first_revisible_frame,
                    ),
                )
            )
        drafts.extend(_uncertain_observation_drafts(track, validated_thresholds))

    ordered_drafts = tuple(sorted(drafts, key=_draft_sort_key))
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
        payload = _draft_hash_payload(draft, possible_occluders)
        prefix = _candidate_hash_prefix(payload)
        if re.fullmatch(r"[0-9a-f]{12}", prefix) is None:
            raise ValueError("candidate hash prefix is invalid")
        candidate = OcclusionCandidate(
            candidate_id=f"occ_{prefix}_{ordinal:04d}",
            target_entity_id=draft.target_entity_id,
            possible_occluder_entity_ids=possible_occluders,
            allowed_start_times=draft.allowed_start_times,
            allowed_end_times=draft.allowed_end_times,
            last_visible_frame=draft.last_visible_frame,
            first_revisible_frame=draft.first_revisible_frame,
            edge_departure=draft.edge_departure,
            low_confidence=draft.low_confidence,
            overlay_refs=_candidate_overlays(
                validated_summary.overlay_refs,
                draft.target_entity_id,
                possible_occluders,
                draft.evidence_frames,
            ),
        )
        candidate.prompt_record()
        candidates.append(candidate)
    return tuple(candidates)


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
    for track in artifact.tracks:
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
    all_tracks = tuple(sorted(artifact.tracks, key=lambda item: item.track_id))
    selected_tracks = all_tracks[:track_cap]
    summary_tracks: list[SummaryTrack] = []
    mandatory_conflicts = 0
    total_gaps = 0
    kept_gaps = 0
    for track in selected_tracks:
        selected, mandatory_conflict = _reduce_observations(
            track.observations, observation_cap
        )
        mandatory_conflicts += int(mandatory_conflict)
        retained_gaps, track_gap_count = _derive_missing_intervals(
            track, frame_clock, cap=observation_cap
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
    overlays = all_overlays[:overlay_cap]
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
    )
    return CvEvidenceSummary(
        schema_version="cv_summary_v1",
        status=artifact.status,
        entities=entities,
        tracks=tracks,
        relations=relations,
        overlay_refs=overlays,
        warnings=_summary_warnings(metadata, prompt_truncated=prompt_truncated),
        prompt_char_limit=prompt_char_limit,
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

    add(0)
    add(len(observations) - 1)
    for index, (previous, current) in enumerate(
        zip(observations, observations[1:]), start=1
    ):
        if previous.visible != current.visible:
            add(index - 1)
            add(index)
    add(
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
    add(
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
    for frame_index in sorted(aligned_by_frame):
        aligned = aligned_by_frame[frame_index]
        for subject_track_id, subject in aligned:
            for object_track_id, object_observation in aligned:
                if subject_track_id == object_track_id:
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
) -> tuple[str, ...]:
    warnings: list[str] = []
    if metadata.total_entities > metadata.kept_entities:
        warnings.append(
            "ENTITIES_TRUNCATED:"
            f"kept={metadata.kept_entities},"
            f"omitted={metadata.total_entities - metadata.kept_entities}"
        )
    if metadata.omitted_aliases:
        warnings.append(
            f"ALIASES_TRUNCATED:omitted={metadata.omitted_aliases}"
        )
    if metadata.truncated_texts:
        warnings.append(
            f"ENTITY_TEXT_TRUNCATED:count={metadata.truncated_texts}"
        )
    if metadata.total_tracks > metadata.kept_tracks:
        warnings.append(
            "TRACKS_TRUNCATED:"
            f"kept={metadata.kept_tracks},"
            f"omitted={metadata.total_tracks - metadata.kept_tracks}"
        )
    if metadata.total_observations > metadata.kept_observations:
        warnings.append(
            "OBSERVATIONS_TRUNCATED:"
            f"kept={metadata.kept_observations},"
            f"omitted={metadata.total_observations - metadata.kept_observations}"
        )
    if metadata.mandatory_conflicts:
        warnings.append(
            "MANDATORY_LANDMARKS_TRUNCATED:"
            f"tracks={metadata.mandatory_conflicts},"
            f"priority={_MANDATORY_PRIORITY_TEXT}"
        )
    if metadata.total_gaps > metadata.kept_gaps:
        warnings.append(
            "LIFECYCLE_GAPS_TRUNCATED:"
            f"kept={metadata.kept_gaps},"
            f"omitted={metadata.total_gaps - metadata.kept_gaps}"
        )
    if metadata.total_relations > metadata.kept_relations:
        warnings.append(
            "RELATIONS_TRUNCATED:"
            f"kept={metadata.kept_relations},"
            f"omitted={metadata.total_relations - metadata.kept_relations}"
        )
    if metadata.total_overlays > metadata.kept_overlays:
        warnings.append(
            "OVERLAYS_TRUNCATED:"
            f"kept={metadata.kept_overlays},"
            f"omitted={metadata.total_overlays - metadata.kept_overlays}"
        )
    if metadata.artifact_warning_count:
        warnings.append(
            "ARTIFACT_WARNINGS_OMITTED:"
            f"count={metadata.artifact_warning_count}"
        )
    if prompt_truncated:
        warnings.append("PROMPT_DATA_TRUNCATED:budget_enforced=1")
    return tuple(warnings)


def _summary_prompt_record(summary: CvEvidenceSummary) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "status": summary.status.value,
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
        "overlay_refs": list(summary.overlay_refs),
        "warnings": list(summary.warnings),
    }


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
                    allowed_start_times=_ordered_times(
                        previous.timestamp_seconds,
                        observations[start].timestamp_seconds,
                    ),
                    allowed_end_times=_ordered_times(
                        observations[end].timestamp_seconds,
                        following.timestamp_seconds,
                    ),
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
) -> tuple[str, ...]:
    frames = set(draft.evidence_frames)
    possible: set[str] = set()
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
        if (
            entity is not None
            and entity.entity_id != draft.target_entity_id
        ):
            possible.add(entity.entity_id)
    return tuple(sorted(possible))


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


def _draft_hash_payload(
    draft: _CandidateDraft, possible_occluders: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "target_track_id": draft.target_track_id,
        "target_entity_id": draft.target_entity_id,
        "possible_occluder_entity_ids": list(possible_occluders),
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


def _candidate_overlays(
    overlay_refs: tuple[str, ...],
    target_entity_id: str,
    possible_occluders: tuple[str, ...],
    evidence_frames: tuple[int, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for entity_id in (target_entity_id, *possible_occluders):
        for frame_index in evidence_frames:
            for reference in overlay_refs:
                if reference in selected:
                    continue
                if _overlay_matches(reference, entity_id, frame_index):
                    selected.append(reference)
    return tuple(selected)


def _overlay_matches(reference: str, entity_id: str, frame_index: int) -> bool:
    name = reference.rsplit("/", 1)[-1]
    if not name.endswith(f"-{frame_index:08d}.png"):
        return False
    return name.startswith(f"{entity_id}-")


def _ordered_times(*values: float | None) -> tuple[float, ...]:
    return tuple(sorted({value for value in values if value is not None}))


def _ordered_frames(*values: int | None) -> tuple[int, ...]:
    return tuple(sorted({value for value in values if value is not None}))


def _edge_departure(observation: TrackObservation) -> bool:
    left, top, right, bottom = observation.bbox_xyxy
    return min(left, top, 1.0 - right, 1.0 - bottom) <= _EDGE_PROXIMITY_THRESHOLD


__all__ = [
    "CandidateId",
    "CvEvidenceSummary",
    "OcclusionCandidate",
    "SpatialRelation",
    "SummaryEntity",
    "SummaryObservation",
    "SummaryTrack",
    "VisibilityGap",
    "build_occlusion_candidates",
    "summarize_cv_evidence",
]
