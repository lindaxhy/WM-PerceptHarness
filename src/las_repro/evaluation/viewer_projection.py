"""Closed, display-only projection of validated canonical hybrid annotations.

The standalone function checks structure and internal references, not external
CV truth. An exporter must first authenticate source media, run metadata and
artifact-derived context. Optional overlay references are supplied only after
registered PNG bytes have been digest-checked by the artifact store.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

from ..pipelines.hybrid_result import validate_hybrid_result
from ..pipelines.occlusion import _has_prohibited_evidence_content
from .las_alignment import canonical_json, digest

_FINE_FIELDS = frozenset(
    {
        "segment_index",
        "action_index",
        "start",
        "end",
        "description",
        "actor",
        "actor_state",
        "skill",
        "target",
        "visual_motion_state",
        "event_type",
        "confidence",
    }
)
_OVERLAY_FIELDS = frozenset(
    {
        "keyframe_id",
        "track_id",
        "frame_index",
        "timestamp_seconds",
        "path",
        "sha256",
        "size_bytes",
    }
)


def _sha(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid viewer digest")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid viewer number")
    return value


def _safe_text(value, *, limit=4096):
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > limit
        or _has_prohibited_evidence_content(value)
    ):
        raise ValueError("invalid viewer text")
    return value


def _review_claims(review, *, sample_id, result_sha, events):
    if review is None:
        return {}
    if type(review) is not dict or set(review) != {
        "schema_version",
        "sample_id",
        "prediction_sha256",
        "reviewer",
        "reviewer_kind",
        "claims",
    }:
        raise ValueError("invalid viewer review")
    if (
        review["schema_version"] != "las_occlusion_review_v1"
        or review["sample_id"] != sample_id
        or review["prediction_sha256"] != result_sha
        or review["reviewer_kind"] != "human"
        or type(review["claims"]) is not list
    ):
        raise ValueError("viewer review identity mismatch")
    _safe_text(review["reviewer"], limit=128)
    expected = {f"occlusion_{event['event_index']}": event for event in events}
    claims = {}
    for claim in review["claims"]:
        if type(claim) is not dict or set(claim) != {
            "event_id",
            "correct",
            "target_entity_id",
            "occluder_entity_id",
            "event_type",
            "start",
            "end",
            "visual_reason",
        }:
            raise ValueError("invalid viewer review claim")
        event_id = _safe_text(claim["event_id"], limit=128)
        if (
            event_id not in expected
            or event_id in claims
            or type(claim["correct"]) is not bool
        ):
            raise ValueError("foreign or duplicate viewer review claim")
        event = expected[event_id]
        for name in ("target_entity_id", "occluder_entity_id", "event_type"):
            if claim[name] != event[name]:
                raise ValueError("viewer review event mismatch")
        if any(_number(claim[name]) != event[name] for name in ("start", "end")):
            raise ValueError("viewer review timing mismatch")
        _safe_text(claim["visual_reason"], limit=1024)
        claims[event_id] = claim
    if set(claims) != set(expected):
        raise ValueError("incomplete viewer review")
    return claims


def _overlays(references, events, duration):
    if references is None:
        return []
    if type(references) is not list or len(references) > 24:
        raise ValueError("invalid viewer overlays")
    seen = set()
    total = 0
    for row in references:
        if type(row) is not dict or set(row) != _OVERLAY_FIELDS:
            raise ValueError("invalid viewer overlay fields")
        key = _safe_text(row["keyframe_id"], limit=499)
        _safe_text(row["track_id"], limit=128)
        if key in seen:
            raise ValueError("duplicate viewer overlay")
        seen.add(key)
        sha = _sha(row["sha256"])
        path = row["path"]
        if (
            type(path) is not str
            or len(path) > 512
            or not re.fullmatch(r"(?:[A-Za-z0-9_-]+/)+[0-9a-f]{64}\.png", path)
            or path.rsplit("/", 1)[-1] != f"{sha}.png"
        ):
            raise ValueError("unsafe viewer overlay path")
        if type(row["frame_index"]) is not int or row["frame_index"] < 0:
            raise ValueError("invalid viewer overlay frame")
        if (
            type(row["size_bytes"]) is not int
            or not 0 < row["size_bytes"] <= 64 * 1024 * 1024
        ):
            raise ValueError("invalid viewer overlay size")
        total += row["size_bytes"]
        timestamp = _number(row["timestamp_seconds"])
        if not 0 <= timestamp < duration or total > 64 * 1024 * 1024:
            raise ValueError("viewer overlay exceeds bounds")
        consumers = [event for event in events if key in event["source_keyframe_ids"]]
        if not consumers or any(
            row["track_id"] not in event["source_track_ids"] for event in consumers
        ):
            raise ValueError("foreign viewer overlay")
    return sorted(copy.deepcopy(references), key=lambda row: row["keyframe_id"])


def project_hybrid_viewer_data(
    sample_id: str,
    duration: float,
    result: Mapping[str, Any],
    *,
    source_sha256: str,
    review: Mapping[str, Any] | None = None,
    include_fine_segments: bool = False,
    source_result_sha256: str | None = None,
    model_identity: str | None = None,
    overlay_references: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project safe display fields without mutating the original annotation.

    ``source_sha256`` identifies the video. ``source_result_sha256`` optionally
    identifies original JSON bytes (which may have historical whitespace).
    Reviews bind those exact bytes; a separate canonical digest is always kept.
    Omitting overlays is useful for review preparation; supplied overlays must
    belong to retained event references, but this function does not read files.
    """
    try:
        if type(sample_id) is not str or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", sample_id
        ):
            raise ValueError("invalid viewer sample")
        if _number(duration) <= 0 or type(include_fine_segments) is not bool:
            raise ValueError("invalid viewer options")
        _sha(source_sha256)
        if model_identity is not None:
            _safe_text(model_identity, limit=256)
        validate_hybrid_result(result)
        segments = result["segments"]
        if segments[0]["start"] != 0 or segments[-1]["end"] != duration:
            raise ValueError("viewer duration does not match annotation")
        for segment in segments:
            if not set(segment) <= _FINE_FIELDS:
                raise ValueError("unknown viewer fine field")
            for key, value in segment.items():
                if key in {"segment_index", "action_index"}:
                    if type(value) is not int or value < 0:
                        raise ValueError("invalid viewer fine index")
                elif key not in {"start", "end", "confidence"}:
                    _safe_text(value)
        canonical_result_sha = digest(result)
        original_sha = (
            canonical_result_sha
            if source_result_sha256 is None
            else _sha(source_result_sha256)
        )
        branches = result["annotation_branches"]
        occlusion = branches["occlusion"]
        claims = _review_claims(
            review,
            sample_id=sample_id,
            result_sha=original_sha,
            events=occlusion["events"],
        )

        def events(rows, prefix):
            output = []
            for row in rows:
                event = copy.deepcopy(row)
                event["id"] = f"{prefix}_{row['event_index']}"
                event["review"] = None
                claim = claims.get(event["id"])
                if claim is not None:
                    event["review_status"] = (
                        "supported" if claim["correct"] else "unsupported"
                    )
                    event["review"] = {
                        "reviewer": review["reviewer"],
                        "visual_reason": claim["visual_reason"],
                    }
                output.append(event)
            return output

        scene = copy.deepcopy(branches["scene_facts"])
        scene["events"] = events(scene["events"], "scene")
        layers = {
            "action_events": {
                "status": "available",
                "events": events(branches["action_events"], "action"),
            },
            "occlusion_events": {
                "status": occlusion["status"],
                "events": events(occlusion["events"], "occlusion"),
            },
            "scene_facts": scene,
        }
        consumers = (
            branches["action_events"]
            + occlusion["events"]
            + scene["events"]
            + scene["locations"]
            + scene["relations"]
        )
        overlays = _overlays(overlay_references, consumers, duration)
        if include_fine_segments:
            fine = []
            for segment in segments:
                # Fine evidence is not promoted into a second primary action set.
                action = next(
                    event
                    for event in branches["action_events"]
                    if segment["segment_index"] in event["source_segment_indices"]
                )
                fine.append(
                    {
                        **copy.deepcopy(segment),
                        "id": f"fine_{segment['segment_index']}",
                        "branch": "fine",
                        "model_stage": "embodied_enrichment",
                        "evidence_mode": "vlm_only",
                        "source_segment_indices": [segment["segment_index"]],
                        "source_track_ids": [],
                        "source_keyframe_ids": [],
                        "repair_history": list(action["repair_history"]),
                        "review_status": "not_required",
                        "review": None,
                    }
                )
            layers["fine_segments"] = {"status": "available", "events": fine}
        output = {
            "schema_version": "comparison_viewer_hybrid_v1",
            "sample": {"sample_id": sample_id, "duration_seconds": duration},
            "layers": layers,
            "provenance": {
                "source_video_sha256": source_sha256,
                "source_result_sha256": original_sha,
                "canonical_result_sha256": canonical_result_sha,
                "model_identity": model_identity,
                "cv_evidence": copy.deepcopy(result["cv_evidence"]),
                "review_sha256": digest(review) if review is not None else None,
                "performance": {
                    key: result["performance"][key]
                    for key in ("total_seconds", "repair_count", "degradation_count")
                },
                "overlays": overlays,
            },
            "warnings": copy.deepcopy(result.get("warnings", [])),
        }
        # Canonical serialization is the final finite/JSON-type boundary; every
        # included branch above passed the strict canonical source validator.
        canonical_json(output)
        return output
    except (KeyError, TypeError, AttributeError, ValueError, RecursionError):
        raise ValueError("invalid hybrid viewer projection") from None
