"""Bounded retained-geometry hypotheses; never physical identity verdicts.

The caller preflights and authenticates the summary before building indexes.
No masks, appearance features, inferred native-frame adjacency, or class gates.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator
import json
import math
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, StrictBool, model_validator

from .contracts import Fraction, NonnegativeInt, StrictModel, Timestamp, TrackId

if TYPE_CHECKING:
    from .summary import CvEvidenceSummary, OcclusionCandidate, SummaryObservation

MAX_IDENTITY_ROWS = 64
MAX_IDENTITY_BYTES = 24_000
MAX_CONTINUATION_PROBES = 262_144
MAX_RELATION_PROBES = 262_144


class ContinuationCue(StrictModel):
    other_track_id: TrackId
    target_frame_index: NonnegativeInt
    other_frame_index: NonnegativeInt
    target_timestamp_seconds: Timestamp
    other_timestamp_seconds: Timestamp
    bbox_iou: Fraction
    area_similarity: Fraction
    center_distance_fraction: Fraction


class CrossLabelCue(StrictModel):
    left_track_id: TrackId
    right_track_id: TrackId
    frame_index: NonnegativeInt
    timestamp_seconds: Timestamp
    bbox_iou: Fraction
    area_similarity: Fraction
    center_distance_fraction: Fraction

    @model_validator(mode='after')
    def ordered_tracks(self):
        if self.left_track_id >= self.right_track_id:
            raise ValueError('cross-label tracks must be distinct and canonical')
        return self


def continuation_key(c: ContinuationCue) -> tuple[int, int, str]:
    return (c.target_frame_index, c.other_frame_index, c.other_track_id)


def cross_label_key(c: CrossLabelCue) -> tuple[int, str, str]:
    return (c.frame_index, c.left_track_id, c.right_track_id)


class IdentityEvidence(StrictModel):
    basis: Literal['retained_geometry'] = 'retained_geometry'
    target_scope: Literal['predictor_track'] = 'predictor_track'
    continuation_cues: Annotated[tuple[ContinuationCue, ...], Field(max_length=2)]
    cross_label_cues: Annotated[tuple[CrossLabelCue, ...], Field(max_length=2)]
    complete: StrictBool

    @model_validator(mode='before')
    @classmethod
    def bounded_plain_evidence(cls, value):
        preflight_identity(value, raw=True)
        return value

    @model_validator(mode='after')
    def canonical_rows(self):
        for rows, key in ((self.continuation_cues, continuation_key),
                          (self.cross_label_cues, cross_label_key)):
            keys = tuple(key(c) for c in rows)
            if keys != tuple(sorted(set(keys))):
                raise ValueError('identity cues must be unique and canonical')
        return self


def preflight_identity(value: object, *, raw: bool = False) -> None:
    """Reject hostile containers/scalars before model traversal or projection."""
    if value is None:
        return
    if type(value) is IdentityEvidence:
        get = lambda name: getattr(value, name, None)
    elif raw and type(value) is dict:
        if len(value) > 5 or any(type(k) is not str for k in value):
            raise ValueError('identity evidence fields exceed their bound')
        get = value.get
    else:
        raise ValueError('identity evidence exceeds the CV structural input bound')
    for name in ('basis', 'target_scope'):
        item = get(name)
        if item is not None and type(item) is not str:
            raise ValueError('identity scope must use plain strings')
    if type(get('complete')) is not bool:
        raise ValueError('identity completeness must be a plain bool')
    for name, cls in (('continuation_cues', ContinuationCue), ('cross_label_cues', CrossLabelCue)):
        rows = get(name)
        if type(rows) not in ((list, tuple) if raw else (tuple,)) or len(rows) > 2:
            raise ValueError('identity rows exceed the CV structural input bound')
        for row in rows:
            if type(row) is cls:
                fields = {name: getattr(row, name, None) for name in cls.model_fields}
            elif raw and type(row) is dict:
                if len(row) > len(cls.model_fields) or any(type(k) is not str for k in row):
                    raise ValueError('identity row fields are invalid')
                fields = row
            else:
                raise ValueError('identity row has a noncanonical model type')
            for field, val in fields.items():
                if field.endswith('track_id'):
                    valid = type(val) is str and len(val) <= 128
                elif field.endswith('frame_index'):
                    valid = type(val) is int and 0 <= val <= 2**63 - 1
                else:
                    valid = type(val) in (float, int) and math.isfinite(val) and val >= 0
                if not valid:
                    raise ValueError('identity row requires bounded plain scalar values')


def identity_record(evidence: IdentityEvidence | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return { 'basis': evidence.basis, 'target_scope': evidence.target_scope,
        'continuation_cues': [{name: getattr(c, name) for name in ContinuationCue.model_fields}
                              for c in evidence.continuation_cues],
        'cross_label_cues': [{name: getattr(c, name) for name in CrossLabelCue.model_fields}
                             for c in evidence.cross_label_cues],
        'complete': evidence.complete }


def identity_bytes(evidence: IdentityEvidence | None) -> int:
    return len(json.dumps(identity_record(evidence), ensure_ascii=False,
                          separators=(',', ':'), sort_keys=True).encode('utf-8')) if evidence else 0


def _geometry(a: SummaryObservation, b: SummaryObservation) -> dict[str, float]:
    al, at, ar, ab = a.bbox_xyxy
    bl, bt, br, bb = b.bbox_xyxy
    intersection = max(0., min(ar, br)-max(al, bl)) * max(0., min(ab, bb)-max(at, bt))
    union = (ar-al)*(ab-at) + (br-bl)*(bb-bt) - intersection
    area = max(a.area_fraction, b.area_fraction)
    return dict(bbox_iou=min(1., max(0., intersection/union if union > 0 else 0.)),
                area_similarity=min(a.area_fraction,b.area_fraction)/area if area else 1.,
                center_distance_fraction=min(1., math.hypot(a.center_xy[0]-b.center_xy[0],
                                                          a.center_xy[1]-b.center_xy[1])/math.sqrt(2.)))


def _spread_indices(count: int) -> Iterator[int]:
    """Even time coverage before filling gaps; independent of source order."""
    pending = [(0, count-1)]
    if count:
        yield 0
    if count > 1:
        yield count-1
    while pending:
        following = []
        for left, right in pending:
            if right-left > 1:
                middle = (left+right)//2
                yield middle
                following.extend(((left,middle),(middle,right)))
        pending = following


def derive_identity_evidence(
    summary: CvEvidenceSummary, candidates: tuple[OcclusionCandidate, ...]
) -> tuple[IdentityEvidence | None, ...]:
    """Indexed bounded scan, local selection, then fair global allocation."""
    tracks = {t.track_id: t for t in summary.tracks}
    visible = {tid: tuple(o for o in t.observations if o.visible) for tid,t in tracks.items()}
    times = {tid: tuple(o.timestamp_seconds for o in obs) for tid,obs in visible.items()}
    observations = {(tid,o.frame_index): o for tid,obs in visible.items() for o in obs}
    same_entity = {}
    for tid,t in sorted(tracks.items()):
        same_entity.setdefault(t.entity_id, []).append(tid)
    relations = {}
    for relation in summary.relations:
        for tid in (relation.subject_track_id, relation.object_track_id):
            relations.setdefault((relation.frame_index,tid), []).append(relation)
    frames_by_time = {f.timestamp_seconds:f.frame_index for f in summary.observed_clock}
    pools = []
    continuation_probes = relation_probes = 0
    for candidate in candidates:
        complete = summary.candidate_search_complete and summary.relations_complete
        start = min(o.start for o in candidate.allowed_event_intervals)
        end = max(o.end for o in candidate.allowed_event_intervals)
        anchors = [observations.get((candidate.target_track_id, frame)) for frame in
                   (candidate.last_visible_frame, candidate.first_revisible_frame) if frame is not None]
        complete = complete and all(a is not None for a in anchors)
        continuations = []
        anchor_rankings = []
        for anchor in (a for a in anchors if a is not None):
            ranked = []
            for tid in same_entity.get(candidate.target_entity_id, ()):
                if tid == candidate.target_track_id:
                    continue
                complete = complete and tracks[tid].candidate_search_complete
                index = bisect_left(times[tid], anchor.timestamp_seconds)
                for offset in (index-1,index):
                    if not 0 <= offset < len(visible[tid]):
                        continue
                    if continuation_probes >= MAX_CONTINUATION_PROBES:
                        complete = False
                        continue
                    continuation_probes += 1
                    other = visible[tid][offset]
                    if not start <= other.timestamp_seconds <= end:
                        continue
                    cue = ContinuationCue(other_track_id=tid, target_frame_index=anchor.frame_index,
                        other_frame_index=other.frame_index, target_timestamp_seconds=anchor.timestamp_seconds,
                        other_timestamp_seconds=other.timestamp_seconds, **_geometry(anchor, other))
                    rank = (abs(other.timestamp_seconds-anchor.timestamp_seconds), -cue.bbox_iou,
                            -cue.area_similarity, cue.center_distance_fraction, continuation_key(cue))
                    ranked.append((rank,cue))
            ranked.sort(key=lambda r:r[0])
            anchor_rankings.append(ranked)
        # Cover both anchors before filling a second row from one anchor.
        for offset in range(2):
            for ranked in anchor_rankings:
                if len(ranked) > offset and len(continuations) < 2:
                    continuations.append(ranked[offset][1])
        if sum(map(len, anchor_rankings)) > len(continuations):
            complete = False
        frames = {candidate.last_visible_frame}
        frames.update(frames_by_time[t] for option in candidate.allowed_event_intervals
                      for t in (option.start,option.end) if t in frames_by_time)
        if candidate.first_revisible_frame is not None:
            frames.add(candidate.first_revisible_frame)
        for provenance in candidate.possible_occluders:
            frames.update(provenance.supporting_frames)
        frames = sorted(frames)
        if len(frames)>4:
            complete = False
            frames = frames[:2]+frames[-2:]
        involved = {candidate.target_track_id, *(p.track_id for p in candidate.possible_occluders)}
        conflicts = {}
        for frame in frames:
            for tid in sorted(involved):
                for relation in relations.get((frame,tid), ()):
                    if relation_probes >= MAX_RELATION_PROBES:
                        complete = False
                        break
                    relation_probes += 1
                    left,right = relation.subject_track_id,relation.object_track_id
                    if tracks[left].entity_id == tracks[right].entity_id:
                        continue
                    if (left,frame) not in observations or (right,frame) not in observations:
                        complete = False
                        continue
                    cue = CrossLabelCue(left_track_id=left,right_track_id=right,frame_index=frame,
                        timestamp_seconds=relation.timestamp_seconds,bbox_iou=relation.bbox_iou,
                        area_similarity=relation.area_similarity,center_distance_fraction=relation.center_distance_fraction)
                    conflicts[cross_label_key(cue)] = cue
        ranked_conflicts = sorted(conflicts.values(), key=lambda c:(-c.bbox_iou,-c.area_similarity,
                                    c.center_distance_fraction,cross_label_key(c)))
        if len(ranked_conflicts)>2:
            complete=False
        selected_cross = ranked_conflicts[:1]
        if len(ranked_conflicts)>1:
            # Preserve a different time witness where available.
            selected_cross.append(next((c for c in ranked_conflicts[1:] if c.frame_index != selected_cross[0].frame_index), ranked_conflicts[1]))
        pools.append((continuations,selected_cross,complete))
    order = sorted(range(len(candidates)), key=lambda i:(candidates[i].last_visible_frame,
                   candidates[i].target_entity_id,candidates[i].target_track_id,i))
    fair_order = [order[i] for i in _spread_indices(len(order))]
    kept = [[[],[]] for _ in candidates]
    result = [None]*len(candidates)
    rows_used = bytes_used = 0
    # Interleave categories locally, then distribute one available row per
    # candidate per round, including candidates with only cross-label context.
    local_rows = [tuple((category, rows[category][offset])
                        for offset in range(2) for category in range(2)
                        if len(rows[category]) > offset) for rows in pools]
    for step in range(4):
        for i in fair_order:
            if len(local_rows[i]) <= step:
                continue
            category, row = local_rows[i][step]
            selected = [list(v) for v in kept[i]]
            selected[category].append(row)
            evidence = IdentityEvidence(continuation_cues=tuple(sorted(selected[0],key=continuation_key)),
                cross_label_cues=tuple(sorted(selected[1],key=cross_label_key)),
                complete=pools[i][2] and sum(map(len,selected)) == len(pools[i][0])+len(pools[i][1]))
            added = identity_bytes(evidence)-identity_bytes(result[i])
            if rows_used < MAX_IDENTITY_ROWS and bytes_used+added <= MAX_IDENTITY_BYTES:
                kept[i]=selected
                result[i]=evidence
                rows_used+=1
                bytes_used+=added
    for i in fair_order:
        if result[i] is None and not pools[i][0] and not pools[i][1]:
            evidence = IdentityEvidence(continuation_cues=(), cross_label_cues=(),complete=pools[i][2])
            if bytes_used+identity_bytes(evidence) <= MAX_IDENTITY_BYTES:
                result[i]=evidence
                bytes_used+=identity_bytes(evidence)
    return tuple(result)
