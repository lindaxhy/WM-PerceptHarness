"""Bounded, claim-free spatial provenance choices for scene generation.

The summary's public projection is revalidated once. The existing provenance
boundary checks each offered row; semantic placeholders exist only for structural
validation and never turn detector geometry into a scene assertion.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

from ..cv.summary import CvEvidenceSummary
from .hybrid_result import _name, _reject_artifact_text, validate_event_provenance
from .scene_semantics import (
    SceneObject,
    SceneSemantics,
    trusted_target_skeleton,
    validate_scene_semantics,
)

MAX_OPTIONS = 64
MAX_OPTION_BYTES = 24_000
# Selection work stays bounded even when every exact segment list is too large.
MAX_SELECTION_ATTEMPTS = 512
_FLAT = {
    "evidence_mode": "hybrid",
    "branch": "scene",
    "model_stage": "scene_semantics",
    "repair_history": ["initial"],
    "review_status": "not_required",
}


def _json(value: object, *, sort_keys: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=(",", ":"),
        allow_nan=False,
    )


def _spread(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Visit early, middle, late, then recursively fill temporal gaps."""
    if not values:
        return []
    indices = [0, len(values) // 2, len(values) - 1]
    ranges = [(0, len(values) // 2), (len(values) // 2, len(values) - 1)]
    for left, right in ranges:
        if right - left > 1:
            middle = (left + right) // 2
            indices.append(middle)
            ranges.extend(((left, middle), (middle, right)))
    return [values[i] for i in dict.fromkeys(indices)]


def _claim_shape(option: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in option.items()
        if key not in {"option_id", "kind", "object_ids", "object_names"}
    }
    row.update(deepcopy(_FLAT))
    row.update(
        visual_evidence="replace with independently visible evidence", confidence=0.0
    )
    if option["kind"] == "location":
        row.update(
            object_id=option["object_ids"][0],
            location="replace with visibly supported location",
        )
    else:
        row.update(
            subject_object_id=option["object_ids"][0],
            object_object_id=option["object_ids"][1],
            relation="unknown",
        )
    return row


def scene_spatial_prompt_data(
    summary: CvEvidenceSummary | None,
    segments: list[dict[str, Any]],
    *,
    duration: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (optional choices envelope, exact original summary JSON suffix).

    JSON null is fixed prompt syntax when not even an empty envelope fits. The
    evidence payload is never compacted or rewritten to create extra headroom.
    """
    if type(segments) is not list or len(segments) > 10_000:
        raise ValueError("scene segment table is not materializable")
    for row in segments:
        if (
            not isinstance(row, dict)
            or any(
                type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                for k in ("start", "end")
            )
            or not 0 <= row["start"] < row["end"] <= duration
        ):
            raise ValueError("scene segment bounds are invalid")
        if type(row.get("segment_index")) is not int or row["segment_index"] < 0:
            raise ValueError("scene segment index is invalid")
    envelope = {"options": [], "options_complete": True}
    if summary is None:
        return envelope, None
    if type(summary) is not CvEvidenceSummary or summary.status != "available":
        raise ValueError("CV evidence must be an available bounded summary")
    # prompt_record preflights model_construct/subclass attacks and revalidates
    # the entire immutable source once, including its content identity.
    suffix = _json(summary.prompt_record())
    budget = min(MAX_OPTION_BYTES, summary.prompt_char_limit - len(suffix))
    if len(_json({"options": [], "options_complete": False}).encode()) > budget:
        return None, suffix

    tokens = defaultdict(set)
    for entity in summary.entities:
        for token in (entity.entity_id, entity.canonical_label, *entity.aliases):
            tokens[_name(token)].add(entity.entity_id)
    bindings = {}
    entity_bindings = defaultdict(list)
    targets = trusted_target_skeleton(segments)
    reserved_ids = {target["object_id"] for target in targets}
    for target in targets:
        matches = (
            tokens[_name(target["name"])]
            if _name(target["name"]) not in {"none", "unknown"}
            else set()
        )
        if len(matches) == 1:
            entity_bindings[next(iter(matches))].append(target)
    # Multiple scene targets matching one entity do not establish distinct
    # physical identities. Never form a relation through this ambiguity.
    for entity_id, values in entity_bindings.items():
        if len(values) == 1:
            bindings[entity_id] = values[0]
    for entity in sorted(summary.entities, key=lambda e: e.entity_id):
        if (
            entity.entity_id not in entity_bindings
            and entity.entity_id not in reserved_ids
            and tokens[_name(entity.canonical_label)] == {entity.entity_id}
            and _name(entity.canonical_label) not in {"none", "unknown"}
        ):
            bindings[entity.entity_id] = {
                "object_id": entity.entity_id,
                "name": entity.canonical_label,
            }
    for entity_id, binding in tuple(bindings.items()):
        try:
            SceneObject.model_validate(dict(binding, description="visible entity"))
            _reject_artifact_text(binding)
        except ValueError:
            # Such a binding cannot be copied into the unchanged scene schema.
            del bindings[entity_id]
    tracks = {
        t.track_id: t
        for t in summary.tracks
        if t.status == "available" and t.entity_id in bindings
    }
    times = sorted(
        frame.timestamp_seconds
        for frame in summary.observed_clock
        if frame.timestamp_seconds <= duration
    )
    boundaries = sorted(
        {0.0, duration, *(s["start"] for s in segments), *(s["end"] for s in segments)}
    )
    windows = {}

    def window(witness):
        if witness >= duration:
            return None
        if witness in windows:
            return windows[witness]
        later = bisect_right(times, witness)
        if later == len(times):
            windows[witness] = None
            return None
        boundary = bisect_right(boundaries, witness)
        left = boundaries[boundary - 1]
        right = boundaries[boundary] if boundary < len(boundaries) else duration
        start_index = bisect_left(times, left)
        end_index = bisect_right(times, right) - 1
        start, end = times[start_index], times[end_index]
        result = (start, end) if start <= witness < end else (witness, times[later])
        windows[witness] = result
        return result

    # Only traverse retained observations/relations, never an object Cartesian
    # product. Per group/window, retain one stable eligible track per object.
    groups = defaultdict(dict)

    def add(kind, track_ids, witness):
        interval = window(witness)
        if interval is None:
            return
        ordered = sorted(
            (bindings[tracks[t].entity_id]["object_id"], t) for t in track_ids
        )
        ids = tuple(item[0] for item in ordered)
        if len(set(ids)) != len(ids):
            return
        key = (kind, ids)
        selected_tracks = tuple(item[1] for item in ordered)
        old = groups[key].get(interval)
        if old is None or selected_tracks < old:
            groups[key][interval] = selected_tracks

    for track_id, track in sorted(tracks.items()):
        for observation in track.observations:
            add("location", (track_id,), observation.timestamp_seconds)
    for relation in summary.relations:
        pair = (relation.subject_track_id, relation.object_track_id)
        if (
            all(t in tracks for t in pair)
            and max(
                relation.bbox_iou,
                relation.subject_bbox_covered_fraction,
                relation.object_bbox_covered_fraction,
            )
            > 0
        ):
            add("relation", pair, relation.timestamp_seconds)

    queues = []
    # Alternate kinds so location groups cannot monopolize the first round.
    kinds = [
        [key for key in sorted(groups) if key[0] == kind]
        for kind in ("location", "relation")
    ]
    group_order = []
    for i in range(max(map(len, kinds), default=0)):
        group_order.extend(keys[i] for keys in kinds if i < len(keys))
    for ordinal, key in enumerate(group_order):
        intervals = _spread(sorted(groups[key]))
        # Under a tight cap, first choices themselves cover the whole video.
        if intervals:
            offset = ordinal % min(3, len(intervals))
            intervals = (
                intervals[offset : offset + 1]
                + intervals[:offset]
                + intervals[offset + 1 :]
            )
        queues.append((key, intervals))
    total = sum(len(intervals) for _, intervals in queues)
    if not total:
        return envelope, suffix
    envelope.update(options_complete=False, flat_provenance=deepcopy(_FLAT))
    if len(_json(envelope).encode()) > budget:
        return {"options": [], "options_complete": False}, suffix
    selected = []
    identities = {}
    attempts = 0
    clock = {
        frame.frame_index: frame.timestamp_seconds for frame in summary.observed_clock
    }
    names = {binding["object_id"]: binding["name"] for binding in bindings.values()}
    overlays = []
    for overlay in summary.overlays:
        stem = PurePosixPath(overlay.path).stem
        try:
            _reject_artifact_text(stem)
        except ValueError:
            # The scene boundary is stricter than the retained path grammar;
            # an empty keyframe subset is legal, an uncopyable stem is not.
            continue
        overlays.append((overlay, stem))
    participant_counts = Counter()
    last_kind = None

    def priority(item):
        ordinal, (key, _, depth) = item
        counts = [participant_counts[obj] for obj in key[1]]
        uncovered = sum(count == 0 for count in counts)
        return (
            not uncovered,
            depth,
            -uncovered,
            max(counts),
            key[0] == last_kind,
            sum(counts),
            ordinal,
        )

    # Keep one head per retained group. Unrepresented objects can advance to
    # a later fitting window before any already-covered group consumes repeats.
    # Once coverage is equal, temporal round depth balances pairs and windows.
    pending = {
        ordinal: (key, intervals, 0)
        for ordinal, (key, intervals) in enumerate(queues)
        if intervals
    }
    while pending:
        if len(selected) >= MAX_OPTIONS or attempts >= MAX_SELECTION_ATTEMPTS:
            break

        # Rank only retained groups, never synthesized object pairs. The
        # 512-attempt cap limits scans over the already bounded source groups.
        # Coverage counts change only after an option fits and validates,
        # so rejected large rows cannot falsely mark an object represented.
        ordinal, _ = min(pending.items(), key=priority)
        key, intervals, depth = pending.pop(ordinal)
        if depth + 1 < len(intervals):
            pending[ordinal] = (key, intervals, depth + 1)
        attempts += 1
        start, end = intervals[depth]
        track_ids = groups[key][(start, end)]
        option = {
            "kind": key[0],
            "object_ids": list(key[1]),
            "object_names": [names[i] for i in key[1]],
            "start": start,
            "end": end,
            "source_track_ids": sorted(track_ids),
            "source_keyframe_ids": sorted(
                {
                    stem
                    for o, stem in overlays
                    if o.track_id in track_ids and start <= clock[o.frame_index] < end
                }
            )[:2],
            "source_segment_indices": [
                s["segment_index"]
                for s in segments
                if s["start"] < end and s["end"] > start
            ],
        }
        canonical = _json(option, sort_keys=True)
        identity = "spv_" + hashlib.sha256(canonical.encode()).hexdigest()
        if identity in identities and identities[identity] != canonical:
            raise ValueError("spatial option identity collision")
        option["option_id"] = identity
        trial = dict(
            envelope,
            options=selected + [option],
            field_shape_example=_claim_shape(selected[0] if selected else option),
        )
        if len(_json(trial).encode()) > budget:
            continue
        claim = _claim_shape(option)
        validate_event_provenance(
            claim,
            segments=segments,
            summary=summary,
            frame_pts=None,
            spatial=True,
            expected_names=option["object_names"],
        )
        identities[identity] = canonical
        selected.append(option)
        participant_counts.update(key[1])
        last_kind = key[0]
    example = _claim_shape(selected[0]) if selected else None
    selected.sort(
        key=lambda o: (
            o["kind"],
            o["start"],
            o["end"],
            o["object_ids"],
            o["source_track_ids"],
        )
    )
    envelope["options"] = selected
    envelope["options_complete"] = len(selected) == total
    if selected:
        # Validate all offered shapes together, without revalidating the source
        # summary for each row or changing the downstream acceptance boundary.
        shape = {
            "objects": [
                {"object_id": k, "name": v, "description": "visible entity"}
                for k, v in names.items()
            ],
            "initial_state": [],
            "final_state": [],
            "locations": [],
            "relations": [],
            "outcome": {
                "status": "unknown",
                "description": "unasserted",
                "confidence": 0.0,
            },
            "semantic_events": [],
        }
        for option in selected:
            shape["locations" if option["kind"] == "location" else "relations"].append(
                _claim_shape(option)
            )
        parsed = SceneSemantics.model_validate(shape)
        validate_scene_semantics(parsed, duration, spatial_evidence_available=True)
        _reject_artifact_text(shape)
        envelope["field_shape_example"] = example
    return envelope, suffix
