"""Canonical branches and exact legacy projections.

Without external context validation authenticates structure and internal
references only. Artifact-aware callers must supply a digest-verified bounded
summary, authoritative PTS, and the actual configured occlusion candidates.
Public artifact hashes alone do not authenticate external provenance.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from ..cv.summary import CvEvidenceSummary
from ..store import validate_inference_job_metrics
from .occlusion import (
    OcclusionDecisionSet,
    _has_prohibited_evidence_content,
    project_occlusion_events,
    validate_occlusion_decisions,
)
from .scene_semantics import (
    SceneSemanticEvent,
    SceneSemantics,
    unavailable_scene_semantics,
    validate_scene_semantics,
)
from .semantic_events import build_semantic_events

ACTION_ADDITIONS = frozenset(
    {
        "branch",
        "model_stage",
        "evidence_mode",
        "source_track_ids",
        "source_keyframe_ids",
        "repair_history",
        "review_status",
    }
)
SCENE_ADDITIONS = ACTION_ADDITIONS | {"source_segment_indices"}
SCENE_KEYS = frozenset(
    {
        "objects",
        "initial_state",
        "final_state",
        "locations",
        "relations",
        "outcome",
        "semantic_events",
    }
)
HYBRID_KEYS = frozenset({"annotation_branches", "cv_evidence", "performance"})
NORMALIZABLE_FIELDS = ("actor", "actor_state", "skill", "visual_motion_state")
AUDIT_WARNING_CODES = frozenset(
    {
        "ENTITY_ALIASES_TRUNCATED",
        "CV_ENTITY_LIMIT_APPLIED",
        "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN",
        "BOUNDARY_TOPOLOGY_NORMALIZED",
    }
)
_STATUSES = {"available", "unavailable", "disabled"}
_STAGES = {
    "media_decode",
    "pass_a",
    "sam31",
    "action_enrichment",
    "occlusion",
    "scene_facts",
    "merge",
}


def legacy_action_projection(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key not in ACTION_ADDITIONS
    }


def legacy_scene_event_projection(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key not in SCENE_ADDITIONS
    }


def _name(value: str) -> str:
    return " ".join(
        value.casefold().replace("_", " ").replace("gripper", "hand").split()
    )


def evidence_sources(
    event: Mapping[str, Any], names: Sequence[str], summary: CvEvidenceSummary | None
) -> tuple[list[str], list[str]]:
    """Match semantic names to overlapping retained, validated observations."""
    if summary is None:
        return [], []
    wanted = {_name(name) for name in names if _name(name) not in {"unknown", "none"}}
    entities = {
        entity.entity_id
        for entity in summary.entities
        if wanted
        & {
            _name(entity.canonical_label),
            _name(entity.entity_id),
            *(_name(alias) for alias in entity.aliases),
        }
    }
    tracks = sorted(
        track.track_id
        for track in summary.tracks
        if track.entity_id in entities
        and track.status == "available"
        and any(
            event["start"] <= obs.timestamp_seconds < event["end"]
            for obs in track.observations
        )
    )
    clock = {
        frame.frame_index: frame.timestamp_seconds for frame in summary.observed_clock
    }
    frames = sorted(
        {
            PurePosixPath(overlay.path).stem
            for overlay in summary.overlays
            if overlay.track_id in tracks
            and event["start"] <= clock[overlay.frame_index] < event["end"]
        }
    )
    return tracks, frames


def _provenance(event, segments, names, summary, branch, stage, history):
    tracks, frames = evidence_sources(event, names, summary)
    return {
        "branch": branch,
        "model_stage": stage,
        "source_segment_indices": [
            s["segment_index"]
            for s in segments
            if s["start"] < event["end"] and s["end"] > event["start"]
        ],
        "source_track_ids": tracks,
        "source_keyframe_ids": frames,
        "evidence_mode": "hybrid" if tracks else "vlm_only",
        "repair_history": list(history),
        "review_status": "not_required",
    }


def build_hybrid_result(
    *,
    task_description,
    segments,
    scene,
    scene_status,
    cv_evidence,
    warnings,
    performance,
    duration=None,
    evidence_summary=None,
    action_history=("initial",),
    scene_history=("initial",),
    occlusion=None,
):
    if evidence_summary is not None:
        evidence_summary.prompt_record()
    actions = []
    for event in build_semantic_events(segments):
        event.update(
            _provenance(
                event,
                segments,
                [event["actor"], event["target"]],
                evidence_summary,
                "action",
                "embodied_enrichment",
                action_history,
            )
        )
        actions.append(event)
    scene = copy.deepcopy(scene)
    names = {obj["object_id"]: obj["name"] for obj in scene["objects"]}
    events = []
    for original in scene["semantic_events"]:
        event = copy.deepcopy(original)
        event.update(
            _provenance(
                event,
                segments,
                [event["actor"], names.get(event["target_object_id"], "unknown")],
                evidence_summary,
                "scene",
                "scene_semantics",
                scene_history,
            )
        )
        events.append(event)
    for key in ("locations", "relations"):
        for row in scene[key]:
            row["repair_history"] = list(scene_history)
    legacy_scene = [legacy_scene_event_projection(event) for event in events]
    if duration is None:
        duration = segments[-1]["end"] if segments else 0.0
    result = {
        "task_description": task_description,
        "duration": float(duration),
        "segments": copy.deepcopy(segments),
        **scene,
        "semantic_events": copy.deepcopy(legacy_scene),
        "grouped_semantic_events": [legacy_action_projection(e) for e in actions],
        "annotation_branches": {
            "action_events": actions,
            "occlusion": copy.deepcopy(
                occlusion
                or {"status": cv_evidence["status"], "decisions": [], "events": []}
            ),
            "scene_facts": {
                "status": scene_status,
                **{
                    k: copy.deepcopy(scene[k]) for k in SCENE_KEYS - {"semantic_events"}
                },
                "events": events,
            },
            "legacy_scene_events": copy.deepcopy(legacy_scene),
        },
        "cv_evidence": copy.deepcopy(cv_evidence),
        "performance": copy.deepcopy(performance),
    }
    if warnings:
        result["warnings"] = copy.deepcopy(warnings)
    return result


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(
        left, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("hybrid result fields are invalid")


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("hybrid result numeric value is invalid")


def _ids(values, *, integer=False, keyframe=False):
    if type(values) is not list or len(values) != len(set(values)):
        raise ValueError("hybrid provenance IDs are invalid")
    for value in values:
        if integer:
            if type(value) is not int or value < 0:
                raise ValueError("hybrid segment ID is invalid")
        elif (
            type(value) is not str
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,498}"
                if keyframe
                else r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*",
                value,
            )
            or (not keyframe and len(value) > 128)
        ):
            raise ValueError("hybrid evidence ID is invalid")


class ProvenanceValidationError(ValueError):
    """Closed categories only; callers must still allowlist public repair codes."""

    def __init__(self, issue_codes: Sequence[str]) -> None:
        self.issue_codes = tuple(dict.fromkeys(issue_codes))
        super().__init__("event provenance is invalid")


def validate_event_provenance(
    event, *, segments, summary, frame_pts, spatial=False, expected_names=None
):
    try:
        for key in ("source_track_ids", "source_keyframe_ids"):
            _ids(event[key], keyframe=key == "source_keyframe_ids")
        _ids(event["source_segment_indices"], integer=True)
    except (ValueError, TypeError, KeyError):
        raise ProvenanceValidationError(("PROVENANCE_INVALID",)) from None
    issues: list[str] = []
    expected = [
        s["segment_index"]
        for s in segments
        if s["start"] < event["end"] and s["end"] > event["start"]
    ]
    if event["source_segment_indices"] != expected:
        issues.append("SOURCE_SEGMENTS_INVALID")
    if event["repair_history"] not in (["initial"], ["initial", "repair"], ["initial", "repair", "repair"]):
        issues.append("PROVENANCE_INVALID")
    if event["evidence_mode"] != (
        "hybrid" if event["source_track_ids"] else "vlm_only"
    ):
        issues.append("PROVENANCE_INVALID")
    if event["source_keyframe_ids"] and not event["source_track_ids"]:
        issues.append("KEYFRAMES_INVALID")
    if summary is None:
        if issues:
            raise ProvenanceValidationError(issues)
        return
    tracks = {track.track_id: track for track in summary.tracks}
    if not set(event["source_track_ids"]) <= tracks.keys():
        issues.append("TRACKS_INVALID")
    overlays = {
        PurePosixPath(overlay.path).stem: overlay for overlay in summary.overlays
    }
    if not set(event["source_keyframe_ids"]) <= overlays.keys():
        issues.append("KEYFRAMES_INVALID")
    for key in event["source_keyframe_ids"]:
        if key in overlays and overlays[key].track_id not in event["source_track_ids"]:
            issues.append("KEYFRAMES_INVALID")
    if spatial:
        clock = {frame.timestamp_seconds for frame in summary.observed_clock}
        if event["start"] not in clock or event["end"] not in clock:
            issues.append("TIME_NOT_OBSERVED")
    if expected_names is not None:
        expected_tracks, expected_frames = evidence_sources(
            event, expected_names, summary
        )
        if spatial:
            if not set(event["source_track_ids"]) <= set(expected_tracks):
                issues.append("TRACKS_INVALID")
            if not set(event["source_keyframe_ids"]) <= set(expected_frames):
                issues.append("KEYFRAMES_INVALID")
        elif (event["source_track_ids"], event["source_keyframe_ids"]) != (
            expected_tracks,
            expected_frames,
        ):
            issues.append("PROVENANCE_INVALID")
    if issues:
        raise ProvenanceValidationError(issues)


def validate_hybrid_result(
    result, *, evidence_summary=None, frame_pts=None, occlusion_candidates=None
):
    """Check public structure; authenticate external references when supplied."""
    if type(result) is not dict:
        raise ValueError("hybrid result must be an object")
    _exact(
        result,
        {"task_description", "duration", "segments", "grouped_semantic_events"}
        | SCENE_KEYS
        | HYBRID_KEYS
        | ({"warnings"} if "warnings" in result else set()),
    )
    json.dumps(result, allow_nan=False)
    branches = result["annotation_branches"]
    _exact(
        branches, {"action_events", "occlusion", "scene_facts", "legacy_scene_events"}
    )
    cv = result["cv_evidence"]
    status = cv.get("status")
    if status not in _STATUSES:
        raise ValueError("CV status is invalid")
    _exact(
        cv,
        {"status", "artifact_key", "manifest_sha256", "cache_hit"}
        if status == "available"
        else {"status"},
    )
    if status == "available" and (
        any(
            type(cv[k]) is not str or not re.fullmatch("[0-9a-f]{64}", cv[k])
            for k in ("artifact_key", "manifest_sha256")
        )
        or type(cv["cache_hit"]) is not bool
    ):
        raise ValueError("CV artifact identity is invalid")
    if evidence_summary is not None:
        if (
            type(evidence_summary) is not CvEvidenceSummary
            or status != "available"
            or frame_pts is None
        ):
            raise ValueError("trusted evidence context is invalid")
        evidence_summary.prompt_record()
        if evidence_summary.status != "available":
            raise ValueError("trusted summary is unavailable")
        if list(frame_pts) != sorted(set(frame_pts)):
            raise ValueError("source PTS must be ordered and unique")
        for timestamp in frame_pts:
            _number(timestamp)
        if not {f.timestamp_seconds for f in evidence_summary.observed_clock} <= set(
            frame_pts
        ):
            raise ValueError("summary contains foreign PTS")
    elif frame_pts is not None or occlusion_candidates is not None:
        raise ValueError("partial trusted evidence context")
    segments = result["segments"]
    expected_actions = build_semantic_events(segments)
    if not segments:
        raise ValueError("hybrid segments are empty")
    duration = result["duration"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("hybrid duration must be a number")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("hybrid duration must be finite and positive")
    if segments[0]["start"] < 0 or segments[-1]["end"] > duration:
        raise ValueError("segments must stay within the hybrid duration")
    actions = branches["action_events"]
    if type(actions) is not list or len(actions) != len(expected_actions):
        raise ValueError("canonical action count is invalid")
    for event, expected in zip(actions, expected_actions):
        _exact(event, set(expected) | ACTION_ADDITIONS)
        if (
            not _same_json(legacy_action_projection(event), expected)
            or event["branch"] != "action"
            or event["model_stage"] != "embodied_enrichment"
            or event["review_status"] != "not_required"
        ):
            raise ValueError("canonical action is invalid")
        validate_event_provenance(
            event,
            segments=segments,
            summary=evidence_summary,
            frame_pts=frame_pts,
            expected_names=[event["actor"], event["target"]],
        )
    if not _same_json(result["grouped_semantic_events"], expected_actions):
        raise ValueError("legacy action projection differs")
    scene = branches["scene_facts"]
    _exact(scene, (SCENE_KEYS - {"semantic_events"}) | {"status", "events"})
    if scene["status"] not in _STATUSES:
        raise ValueError("scene status is invalid")
    for key in SCENE_KEYS - {"semantic_events"}:
        if result[key] != scene[key]:
            raise ValueError("scene projection differs")
    parsed_scene = SceneSemantics.model_validate(
        {key: result[key] for key in SCENE_KEYS}
    )
    validate_scene_semantics(
        parsed_scene, duration, spatial_evidence_available=status == "available"
    )
    names = {obj["object_id"]: obj["name"] for obj in scene["objects"]}
    legacy = []
    for event in scene["events"]:
        projected = legacy_scene_event_projection(event)
        _exact(event, set(SceneSemanticEvent.model_fields) | SCENE_ADDITIONS)
        if (
            event["branch"] != "scene"
            or event["model_stage"] != "scene_semantics"
            or event["review_status"] != "not_required"
        ):
            raise ValueError("canonical scene provenance is invalid")
        validate_event_provenance(
            event,
            segments=segments,
            summary=evidence_summary,
            frame_pts=frame_pts,
            expected_names=[
                event["actor"],
                names.get(event["target_object_id"], "unknown"),
            ],
        )
        legacy.append(projected)
    if not _same_json(legacy, result["semantic_events"]) or not _same_json(
        legacy, branches["legacy_scene_events"]
    ):
        raise ValueError("legacy scene projection differs")
    for key in ("locations", "relations"):
        for row in scene[key]:
            objects = (
                [row["object_id"]]
                if key == "locations"
                else [row["subject_object_id"], row["object_object_id"]]
            )
            validate_event_provenance(
                row,
                segments=segments,
                summary=evidence_summary,
                frame_pts=frame_pts,
                spatial=True,
                expected_names=[names[obj] for obj in objects],
            )
    occlusion = branches["occlusion"]
    _exact(occlusion, {"status", "decisions", "events"})
    if occlusion["status"] not in _STATUSES:
        raise ValueError("occlusion status is invalid")
    decisions = OcclusionDecisionSet.model_validate(
        {"decisions": occlusion["decisions"]}
    )
    _validate_public_occlusion(decisions, occlusion["events"], duration)
    for index, event in enumerate(occlusion["events"]):
        _exact(
            event,
            {
                "event_index",
                "start",
                "end",
                "event_type",
                "target_entity_id",
                "occluder_entity_id",
                "description",
                "confidence",
                "source_candidate_id",
            }
            | SCENE_ADDITIONS,
        )
        if (
            event["event_index"] != index
            or event["branch"] != "occlusion"
            or event["model_stage"] != "occlusion_semantics"
            or event["review_status"] != "unreviewed"
            or event["evidence_mode"] != "hybrid"
        ):
            raise ValueError("occlusion event provenance is invalid")
        validate_event_provenance(
            event, segments=segments, summary=evidence_summary, frame_pts=frame_pts
        )
    if (
        evidence_summary is not None
        and occlusion["status"] == "available"
        and (decisions.decisions or occlusion_candidates is not None)
    ):
        if occlusion_candidates is None:
            raise ValueError("trusted occlusion candidates are required")
        validate_occlusion_decisions(decisions, occlusion_candidates, duration=duration)
        histories = {tuple(e["repair_history"]) for e in occlusion["events"]}
        if len(histories) > 1 or occlusion["events"] != project_occlusion_events(
            decisions,
            occlusion_candidates,
            (),
            segments,
            repair_history=next(iter(histories), ("initial",)),
        ):
            raise ValueError("occlusion projection differs")
    warnings = result.get("warnings", [])
    if type(warnings) is not list or any(
        type(w) is not dict or type(w.get("code")) is not str for w in warnings
    ):
        raise ValueError("hybrid warnings are invalid")
    codes = [w["code"] for w in warnings]
    warning_fields = {
        "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN": {"code", "fields", "count"},
        "BOUNDARY_TOPOLOGY_NORMALIZED": {"code", "issue_codes", "count"},
        "CV_ENTITY_LIMIT_APPLIED": {"code", "omitted_count", "limit", "message"},
        "ENTITY_ALIASES_TRUNCATED": {"code", "omitted_count"},
        "CV_EVIDENCE_UNAVAILABLE": {"code"},
        "OCCLUSION_UNAVAILABLE": {"code"},
        "SCENE_SEMANTICS_UNAVAILABLE": {"code"},
    }
    for warning in warnings:
        if warning["code"] not in warning_fields:
            raise ValueError("unknown hybrid warning")
        _exact(warning, warning_fields[warning["code"]])
        if warning["code"] in AUDIT_WARNING_CODES:
            validate_audit_warning(
                warning,
                segment_count=len(segments),
                unknown_counts={
                    field: sum(segment.get(field) == "unknown" for segment in segments)
                    for field in NORMALIZABLE_FIELDS
                },
            )
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate warning codes")
    for branch_status, code in (
        (status, "CV_EVIDENCE_UNAVAILABLE"),
        (scene["status"], "SCENE_SEMANTICS_UNAVAILABLE"),
        (occlusion["status"], "OCCLUSION_UNAVAILABLE"),
    ):
        if (branch_status == "unavailable") != (code in codes):
            raise ValueError("warning and status disagree")
    if scene["status"] != "available" and any(
        scene[k]
        for k in (
            "objects",
            "initial_state",
            "final_state",
            "locations",
            "relations",
            "events",
        )
    ):
        raise ValueError("unavailable scene contains claims")
    if scene["status"] != "available" and not _same_json(
        scene["outcome"], unavailable_scene_semantics()["outcome"]
    ):
        raise ValueError("unavailable scene outcome is not conservative")
    if occlusion["status"] != "available" and (
        occlusion["decisions"] or occlusion["events"]
    ):
        raise ValueError("unavailable occlusion contains claims")
    if status != "available":
        if occlusion["status"] != status or scene["locations"] or scene["relations"]:
            raise ValueError("unavailable CV contains spatial claims")
        if any(
            e["source_track_ids"] or e["source_keyframe_ids"]
            for e in actions + scene["events"]
        ):
            raise ValueError("unavailable CV contains provenance")
    _validate_performance(result["performance"])
    stage_counts = {}
    for row in result["performance"]["stages"]:
        if "model_stage" in row:
            stage_counts[row["model_stage"]] = (
                stage_counts.get(row["model_stage"], 0) + 1
            )
    for event in (
        actions
        + scene["events"]
        + scene["locations"]
        + scene["relations"]
        + occlusion["events"]
    ):
        history = ["initial"] + ["repair"] * max(
            0, stage_counts.get(event["model_stage"], 0) - 1
        )
        if event["repair_history"] != history:
            raise ValueError("event history differs from producing stage")
    if result["performance"]["degradation_count"] != sum(
        value == "unavailable"
        for value in (status, scene["status"], occlusion["status"])
    ):
        raise ValueError("degradation count differs from statuses")
    _reject_artifact_text(result)


def validate_audit_warning(
    warning: Mapping[str, Any],
    *,
    segment_count: int = 0,
    unknown_counts: Mapping[str, int] | None = None,
) -> None:
    """Shared semantic audit checks for public hybrid results and legacy export."""
    warning = dict(warning)
    code = warning.get("code")
    if code == "ENTITY_ALIASES_TRUNCATED":
        _exact(warning, {"code", "omitted_count"})
        count = warning["omitted_count"]
        if type(count) is not int or not 1 <= count <= 64 * 256:
            raise ValueError("alias omission count is invalid")
    elif code == "CV_ENTITY_LIMIT_APPLIED":
        _exact(warning, {"code", "omitted_count", "limit", "message"})
        count, limit = warning["omitted_count"], warning["limit"]
        if (
            type(count) is not int
            or not 1 <= count <= 63
            or type(limit) is not int
            or not 1 <= limit <= 16
            or count + limit > 64
        ):
            raise ValueError("entity omission count is invalid")
        noun = "candidate" if count == 1 else "candidates"
        if warning["message"] != f"{count} entity {noun} omitted by limit {limit}":
            raise ValueError("entity omission message disagrees with count")
    elif code == "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN":
        _exact(warning, {"code", "fields", "count"})
        fields, count = warning["fields"], warning["count"]
        if (
            type(fields) is not list
            or not fields
            or any(type(field) is not str for field in fields)
            or fields != [field for field in NORMALIZABLE_FIELDS if field in fields]
        ):
            raise ValueError("normalization fields are invalid")
        if (
            type(count) is not int
            or not len(fields) <= count <= len(fields) * segment_count
        ):
            raise ValueError("normalization count is invalid")
        counts = unknown_counts or {}
        if any(counts.get(field, 0) == 0 for field in fields) or count > sum(
            counts.get(field, 0) for field in fields
        ):
            raise ValueError("normalization count lacks unknown output fields")
    elif code == "BOUNDARY_TOPOLOGY_NORMALIZED":
        _exact(warning, {"code", "issue_codes", "count"})
        issues = warning["issue_codes"]
        allowed = (
            "SEGMENT_TOO_LONG",
            "SEGMENT_BOUNDARY_NOT_ADJACENT",
            "SEGMENT_DESCRIPTION_INVALID",
        )
        if (
            type(issues) is not list
            or not issues
            or issues != [item for item in allowed if item in issues]
        ):
            raise ValueError("boundary normalization codes are invalid")
        if type(warning["count"]) is not int or warning["count"] != segment_count:
            raise ValueError("boundary normalization count differs from segments")
    else:
        raise ValueError("unknown audit warning")


def _reject_artifact_text(value):
    if isinstance(value, dict):
        for child in value.values():
            _reject_artifact_text(child)
    elif isinstance(value, list):
        for child in value:
            _reject_artifact_text(child)
    elif isinstance(value, str) and _has_prohibited_evidence_content(value):
        raise ValueError("hybrid result contains prohibited artifact content")


def _validate_performance(performance):
    _exact(
        performance, {"stages", "total_seconds", "repair_count", "degradation_count"}
    )
    _number(performance["total_seconds"])
    for key in ("repair_count", "degradation_count"):
        if type(performance[key]) is not int or performance[key] < 0:
            raise ValueError("performance count is invalid")
    if type(performance["stages"]) is not list:
        raise ValueError("performance stages are invalid")
    counts = {}
    mapping = {
        "embodied_pass_a": "pass_a",
        "embodied_pass_b": "action_enrichment",
        "embodied_enrichment": "action_enrichment",
        "cv_evidence": "sam31",
        "scene_semantics": "scene_facts",
        "occlusion_semantics": "occlusion",
    }
    for stage in performance["stages"]:
        if stage.get("stage") not in _STAGES:
            raise ValueError("performance stage is invalid")
        if "elapsed_seconds" in stage:
            _exact(stage, {"stage", "elapsed_seconds"})
            _number(stage["elapsed_seconds"])
        else:
            _exact(
                stage,
                {
                    "stage",
                    "model_stage",
                    "queue_seconds",
                    "wall_seconds",
                    "inference_seconds",
                    "attempt_count",
                    "worker_id",
                    "cache_hit",
                    "provider_metrics",
                },
            )
            if mapping.get(stage["model_stage"]) != stage["stage"]:
                raise ValueError("performance model stage is invalid")
            counts[stage["model_stage"]] = counts.get(stage["model_stage"], 0) + 1
            for key in ("queue_seconds", "wall_seconds", "inference_seconds"):
                if stage[key] is not None:
                    _number(stage[key])
            if type(stage["attempt_count"]) is not int or stage["attempt_count"] < 0:
                raise ValueError("attempt count is invalid")
            if stage["worker_id"] is not None and (
                type(stage["worker_id"]) is not str
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", stage["worker_id"])
            ):
                raise ValueError("worker identity is invalid")
            if type(stage["cache_hit"]) is not bool:
                raise ValueError("cache hit is invalid")
            validate_inference_job_metrics(stage["provider_metrics"])
    _MAX_STAGE_JOBS = {
        "cv_evidence": 1,
        "scene_semantics": 3,
        "occlusion_semantics": 3,
    }
    if any(
        count > _MAX_STAGE_JOBS.get(name, 2) for name, count in counts.items()
    ) or performance["repair_count"] != sum(count - 1 for count in counts.values()):
        raise ValueError("repair count differs from immutable job rows")


def _validate_public_occlusion(decisions, events, duration):
    if type(events) is not list:
        raise ValueError("occlusion events must be a list")
    expected = []
    seen = set()
    for decision in decisions.decisions:
        if (
            not re.fullmatch(r"occ_[0-9a-f]{12}_[0-9]{4}", decision.candidate_id)
            or decision.candidate_id in seen
        ):
            raise ValueError("occlusion candidate identity is invalid")
        seen.add(decision.candidate_id)
        if bool(decision.events) != (decision.classification == "occlusion"):
            raise ValueError("occlusion decision and events disagree")
        previous = None
        ends = {}
        for interval in decision.events:
            key = (interval.start, interval.end, interval.event_type.value)
            if (
                not 0 <= interval.start < interval.end <= duration
                or (previous is not None and key < previous)
                or interval.start < ends.get(interval.event_type, 0)
            ):
                raise ValueError("occlusion intervals are invalid")
            previous = key
            ends[interval.event_type] = interval.end
            expected.append(
                {
                    "source_candidate_id": decision.candidate_id,
                    "target_entity_id": decision.target_entity_id,
                    "occluder_entity_id": decision.occluder_entity_id,
                    "start": interval.start,
                    "end": interval.end,
                    "event_type": interval.event_type.value,
                    "description": decision.visual_evidence,
                    "confidence": decision.confidence,
                }
            )
    expected.sort(
        key=lambda row: (
            row["start"],
            row["end"],
            row["source_candidate_id"],
            row["event_type"],
        )
    )
    if len(events) != len(expected):
        raise ValueError("occlusion event count differs from decisions")
    for event, projection in zip(events, expected):
        if any(event.get(key) != value for key, value in projection.items()):
            raise ValueError("occlusion event differs from its decision")


def build_performance(
    jobs,
    *,
    media_seconds,
    merge_seconds,
    total_seconds,
    degradation_count,
    cv_seconds=0.0,
    occlusion_seconds=0.0,
):
    """One row per immutable job; boundary preparation belongs to action enrichment."""
    mapping = {
        "embodied_pass_a": "pass_a",
        "embodied_pass_b": "action_enrichment",
        "embodied_enrichment": "action_enrichment",
        "cv_evidence": "sam31",
        "scene_semantics": "scene_facts",
        "occlusion_semantics": "occlusion",
    }
    rows = [{"stage": "media_decode", "elapsed_seconds": media_seconds}]
    selected = sorted(
        (job for job in jobs if job.stage in mapping),
        key=lambda job: (job.created_at, job.stage, job.ordinal),
    )
    for job in selected:
        metrics = dict(job.metrics or {})
        validate_inference_job_metrics(metrics)
        rows.append(
            {
                "stage": mapping[job.stage],
                "model_stage": job.stage,
                "queue_seconds": None
                if job.started_at is None
                else job.started_at - job.created_at,
                "wall_seconds": None
                if job.finished_at is None
                else job.finished_at - job.created_at,
                "inference_seconds": metrics.pop("inference_seconds", None),
                "attempt_count": job.attempt,
                "worker_id": job.completed_by or job.worker_id,
                "cache_hit": metrics.pop("cache_hit", False),
                "provider_metrics": metrics,
            }
        )
    for name, elapsed in (("sam31", cv_seconds), ("occlusion", occlusion_seconds)):
        if not any(row["stage"] == name for row in rows):
            rows.append({"stage": name, "elapsed_seconds": elapsed})
    rows.append({"stage": "merge", "elapsed_seconds": merge_seconds})
    return {
        "stages": rows,
        "total_seconds": total_seconds,
        "repair_count": sum(job.ordinal > 0 for job in selected),
        "degradation_count": degradation_count,
    }
