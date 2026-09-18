"""Strict, evidence-constrained semantic adjudication of occlusion candidates."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any

from pydantic import Field, StrictStr, field_validator

from ..cv.contracts import CvTrack, StrictModel
from ..cv.summary import OcclusionCandidate, _preflight_candidate, _revalidate_candidate
from .validators import TemporalIssue, TemporalValidationError


Timestamp = Annotated[float, Field(ge=0, strict=True, allow_inf_nan=False)]
Confidence = Annotated[
    float, Field(ge=0, le=1, strict=True, allow_inf_nan=False)
]
Identifier = Annotated[
    StrictStr,
    Field(max_length=128, pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"),
]


class OcclusionClassification(StrEnum):
    OCCLUSION = "occlusion"
    OUT_OF_FRAME = "out_of_frame"
    DETECTOR_LOSS = "detector_loss"
    UNKNOWN = "unknown"


class OcclusionEventType(StrEnum):
    ENTER = "occlusion_enter"
    OCCLUDED = "occluded"
    EXIT = "occlusion_exit"


class OcclusionInterval(StrictModel):
    event_type: OcclusionEventType
    start: Timestamp
    end: Timestamp
    # One sentence describing only this phase's visible change. Optional at
    # the schema layer so older outputs still parse; validate_occlusion_decisions
    # requires it on every event of an occlusion decision.
    description: Annotated[StrictStr, Field(max_length=1024)] | None = None

    @field_validator("event_type", mode="before")
    @classmethod
    def parse_event_type(cls, value: Any) -> OcclusionEventType:
        if type(value) is OcclusionEventType:
            return value
        if type(value) is not str:
            raise ValueError("event_type must be a plain string")
        return OcclusionEventType(value)


class OcclusionDecision(StrictModel):
    candidate_id: Identifier
    classification: OcclusionClassification
    target_entity_id: Identifier
    occluder_entity_id: Identifier
    events: Annotated[tuple[OcclusionInterval, ...], Field(max_length=24)]
    visual_evidence: Annotated[StrictStr, Field(max_length=1024)]
    confidence: Confidence

    @field_validator("classification", mode="before")
    @classmethod
    def parse_classification(cls, value: Any) -> OcclusionClassification:
        if type(value) is OcclusionClassification:
            return value
        if type(value) is not str:
            raise ValueError("classification must be a plain string")
        return OcclusionClassification(value)

    @field_validator("visual_evidence")
    @classmethod
    def reject_blank_evidence(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("visual evidence must not be blank")
        return value


class OcclusionDecisionSet(StrictModel):
    decisions: Annotated[tuple[OcclusionDecision, ...], Field(max_length=256)]


_PROHIBITED_EVIDENCE_STRUCTURE = re.compile(r"[{}\[\]\\/]")
_RAW_MASK_SUFFIXES = (".mask", ".npy", ".npz")
_RAW_MASK_FILENAME = re.compile(
    r"(?i)(?<![a-z0-9_.-])[a-z0-9_.-]+\.(?:mask|npy|npz)(?![a-z0-9_])"
)
_BINARY_MASK_ROW = re.compile(r"\s*[01](?:\s*,\s*[01])+\s*")


def _has_prohibited_evidence_content(value: str) -> bool:
    if (
        _PROHIBITED_EVIDENCE_STRUCTURE.search(value)
        or _RAW_MASK_FILENAME.search(value)
        or value.casefold().rstrip().endswith(_RAW_MASK_SUFFIXES)
        or any(
            ord(character) < 32 and character not in "\t\n\r"
            for character in value
        )
    ):
        return True
    lines = value.splitlines()
    return "mask" in value.casefold() and sum(
        _BINARY_MASK_ROW.fullmatch(line) is not None for line in lines
    ) >= 2


def _validated_candidates(candidates):
    if type(candidates) is not tuple or len(candidates) > 256:
        raise ValueError("occlusion candidates exceed their structural bound")
    for candidate in candidates:
        _preflight_candidate(candidate)
    return tuple(_revalidate_candidate(c) for c in candidates)


def validate_occlusion_decisions(
    result: OcclusionDecisionSet,
    candidates: tuple[OcclusionCandidate, ...],
    *,
    duration: float,
) -> None:
    """Close every model decision to one trusted candidate skeleton."""
    if type(result) is not OcclusionDecisionSet:
        raise TypeError("result must be an OcclusionDecisionSet")
    candidates = _validated_candidates(candidates)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("duration must be finite and positive")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")

    issues: list[TemporalIssue] = []
    if len(result.decisions) != len(candidates):
        issues.append(
            TemporalIssue(
                "OCCLUSION_DECISION_CARDINALITY",
                ("decisions",),
                "there must be exactly one decision per trusted candidate",
            )
        )
    for index, (decision, candidate) in enumerate(
        zip(result.decisions, candidates, strict=False)
    ):
        path = ("decisions", index)
        if decision.candidate_id != candidate.candidate_id:
            issues.append(TemporalIssue("OCCLUSION_CANDIDATE_ORDER", path + ("candidate_id",), "decision candidate IDs must preserve trusted order"))
        if decision.target_entity_id != candidate.target_entity_id:
            issues.append(TemporalIssue("OCCLUSION_TARGET_MISMATCH", path + ("target_entity_id",), "decision target must match its candidate"))
        if decision.occluder_entity_id != "unknown" and decision.occluder_entity_id not in candidate.possible_occluder_entity_ids:
            issues.append(TemporalIssue("OCCLUSION_OCCLUDER_NOT_PROPOSED", path + ("occluder_entity_id",), "occluder must be proposed by trusted evidence or unknown"))
        if _has_prohibited_evidence_content(decision.visual_evidence):
            issues.append(
                TemporalIssue(
                    "OCCLUSION_EVIDENCE_PROHIBITED_CONTENT",
                    path + ("visual_evidence",),
                    "visual evidence must be plain text without serialized data or paths",
                )
            )
        if decision.classification is not OcclusionClassification.OCCLUSION:
            if decision.events:
                issues.append(TemporalIssue("NON_OCCLUSION_HAS_EVENTS", path + ("events",), "non-occlusion decisions cannot emit positive events"))
            continue
        if not decision.events:
            issues.append(TemporalIssue("OCCLUSION_EVENTS_EMPTY", path + ("events",), "an occlusion decision needs at least one positive event"))
        previous_by_type: dict[OcclusionEventType, OcclusionInterval] = {}
        previous_key: tuple[float, float, str] | None = None
        for event_index, event in enumerate(decision.events):
            event_path = path + ("events", event_index)
            key = (event.start, event.end, event.event_type.value)
            if event.description is None or not event.description.strip():
                issues.append(TemporalIssue("OCCLUSION_EVENT_DESCRIPTION_MISSING", event_path + ("description",),
                                            "each occlusion event needs its own phase-specific description"))
            elif _has_prohibited_evidence_content(event.description):
                issues.append(
                    TemporalIssue(
                        "OCCLUSION_EVIDENCE_PROHIBITED_CONTENT",
                        event_path + ("description",),
                        "event descriptions must be plain text without serialized data or paths",
                    )
                )
            if previous_key is not None and key < previous_key:
                issues.append(TemporalIssue("OCCLUSION_EVENTS_NOT_ORDERED", event_path, "events must be ordered by start, end, and type"))
            if not any((event.event_type.value, event.start, event.end) ==
                       (option.event_type, option.start, option.end)
                       for option in candidate.allowed_event_intervals):
                issues.append(TemporalIssue("OCCLUSION_INTERVAL_NOT_ALLOWED", event_path,
                                            "event must match one complete typed candidate interval"))
            if not event.start < event.end:
                issues.append(TemporalIssue("OCCLUSION_EVENT_NONPOSITIVE_DURATION", event_path, "event intervals must have positive duration"))
            if event.end > duration:
                issues.append(TemporalIssue("OCCLUSION_EVENT_OUTSIDE_VIDEO", event_path, "event intervals must remain within the video"))
            prior = previous_by_type.get(event.event_type)
            if prior is not None and event.start < prior.end:
                issues.append(TemporalIssue("OCCLUSION_EVENT_OVERLAP", event_path, "same-type event intervals cannot overlap"))
            previous_by_type[event.event_type] = event
            previous_key = key
    if issues:
        raise TemporalValidationError(issues)


def project_occlusion_events(
    result: OcclusionDecisionSet,
    candidates: tuple[OcclusionCandidate, ...],
    tracks: tuple[CvTrack, ...],
    segments: Sequence[Mapping[str, Any]],
    *,
    repair_history: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Project only positive decisions into deterministic scoring records."""
    if repair_history not in {("initial",), ("initial", "repair"), ("initial", "repair", "repair")}:
        raise ValueError("repair_history is not a closed adjudication history")
    if type(tracks) is not tuple or any(type(track) is not CvTrack for track in tracks):
        raise TypeError("tracks must be a tuple of CvTrack values")
    if isinstance(segments, (str, bytes, bytearray)) or not isinstance(segments, Sequence):
        raise TypeError("segments must be a sequence of mappings")
    candidates = _validated_candidates(candidates)
    validation_duration = max(
        (option.end for candidate in candidates for option in candidate.allowed_event_intervals),
        default=1.0,
    )
    validate_occlusion_decisions(result, candidates, duration=validation_duration)
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    projected: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for decision in result.decisions:
        if decision.classification is not OcclusionClassification.OCCLUSION:
            continue
        candidate = candidates_by_id.get(decision.candidate_id)
        if candidate is None:
            raise ValueError("decision references an unknown candidate")
        # Candidate provenance remains authoritative even when a degraded raw
        # track is omitted from ``tracks``; never turn that omission into a
        # semantic claim.  Materializing the tuple above still validates the
        # caller's provider-independent track boundary.
        source_tracks = [candidate.target_track_id]
        source_tracks.extend(
            item.track_id for item in candidate.possible_occluders
        )
        source_tracks = list(dict.fromkeys(source_tracks))
        keyframe_ids = [PurePosixPath(reference).stem for reference in candidate.overlay_refs]
        for event in decision.events:
            segment_indices: list[int] = []
            for segment in segments:
                if not isinstance(segment, Mapping):
                    raise TypeError("segments must contain mappings")
                index = segment.get("segment_index")
                start = segment.get("start")
                end = segment.get("end")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError("segment_index must be a non-negative integer")
                if not _finite_number(start) or not _finite_number(end) or not float(start) < float(end):
                    raise ValueError("segment bounds must be finite and positive")
                if float(start) < event.end and float(end) > event.start:
                    segment_indices.append(index)
            record = {
                "event_index": 0,
                "start": event.start,
                "end": event.end,
                "event_type": event.event_type.value,
                "target_entity_id": decision.target_entity_id,
                "occluder_entity_id": decision.occluder_entity_id,
                "description": event.description or decision.visual_evidence,
                "confidence": decision.confidence,
                "branch": "occlusion",
                "model_stage": "occlusion_semantics",
                "source_candidate_id": decision.candidate_id,
                "source_segment_indices": segment_indices,
                "source_track_ids": source_tracks,
                "source_keyframe_ids": keyframe_ids,
                "evidence_mode": "hybrid",
                "repair_history": list(repair_history),
                "review_status": "unreviewed",
            }
            projected.append(((event.start, event.end, decision.candidate_id, event.event_type.value), record))
    projected.sort(key=lambda item: item[0])
    for index, (_, record) in enumerate(projected):
        record["event_index"] = index
    return [record for _, record in projected]


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


__all__ = [
    "OcclusionClassification",
    "OcclusionDecision",
    "OcclusionDecisionSet",
    "OcclusionEventType",
    "OcclusionInterval",
    "project_occlusion_events",
    "validate_occlusion_decisions",
]
