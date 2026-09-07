"""Deterministic, input-only hints derived from the full authenticated scene.

No source context, offer capacity, output schema or projection is changed here.
Existing source preflights bound traversal before any compact materialization.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..cv.summary import CvEvidenceSummary
from .scene_choices import authenticate_scene_context, canonical, compact_scene_options
from .scene_provenance import scene_object_bindings

VIEW_VERSION = 'scene-model-view-v1'
FULL_VERSION = 'scene-full-view-v1'
MAX_DATA_BYTES = 49_152
MAX_PROMPT_BYTES = 65_536
# Reserved independent of actual repair codes: initial/repair cannot change mode.
MAX_REPAIR_BYTES = 4_096
LIMITS = dict(options=64, observations=128, pairs=64, tracks=32,
              endpoints=64, runs=128, gaps=64)
SEGMENT_KEYS = ('segment_index', 'start', 'end', 'actor', 'target', 'description')


def _observation(row):
    return dict(time=row.timestamp_seconds, bbox=list(row.bbox_xyxy),
                visible=row.visible, confidence=row.confidence)


def full_model_view(context, *, reason, counts):
    encoded = canonical(context).encode('utf-8')
    return dict(data=None, metadata=dict(mode='full_fallback', version=FULL_VERSION,
        reason=reason, data_sha256=hashlib.sha256(encoded).hexdigest(),
        data_bytes=len(encoded), counts=counts))


def build_scene_model_view(context: Any) -> dict[str, Any]:
    context = authenticate_scene_context(context)
    summary = (CvEvidenceSummary.model_validate(context['evidence_summary'])
               if context['evidence_summary'] is not None else None)
    tracks = sorted((t for t in summary.tracks if t.status == 'available'),
                    key=lambda t: t.track_id) if summary else []
    offers = context['spatial_options']
    options = offers['options'] if offers else []
    counts = dict(options=len(options), observations=0, pairs=0, tracks=len(tracks),
        endpoints=sum(min(2, len(t.observations)) for t in tracks),
        runs=sum(len(t.visibility_runs) for t in tracks),
        gaps=sum(len(t.missing_intervals) for t in tracks))
    if any(counts[k] > cap for k, cap in LIMITS.items()):
        return full_model_view(context, reason='row_limits', counts=counts)
    bindings = scene_object_bindings(summary, context['segments']) if summary else {}
    by_track = {t.track_id: t for t in tracks}
    # At most 64 offers times the authenticated bounded source rows; no Cartesian
    # geometry calculation and no fabricated same-frame evidence.
    observations, pairs, links = {}, {}, []
    for index, option in enumerate(options):
        track_ids = option['source_track_ids']
        if len(track_ids) > 2 or any(t not in by_track or by_track[t].entity_id not in bindings
                                    for t in track_ids):
            return full_model_view(context, reason='binding_ineligible', counts=counts)
        resolved = [bindings[by_track[t].entity_id]['object_id'] for t in track_ids]
        if sorted(resolved) != option['object_ids']:
            return full_model_view(context, reason='binding_ineligible', counts=counts)
        start, end = option['start'], option['end']
        center = start + (end - start) / 2
        obs_keys, pair_keys = [], []
        for track_id, object_id in zip(track_ids, resolved):
            candidates = (r for r in by_track[track_id].observations
                          if start <= r.timestamp_seconds < end)
            row = min(candidates, key=lambda r: (abs(r.timestamp_seconds-center),
                      r.timestamp_seconds, r.frame_index), default=None)
            if row is not None:
                key = (track_id, row.frame_index)
                observations[key] = dict(object_id=object_id, **_observation(row))
                obs_keys.append(key)
        if option['kind'] == 'relation':
            candidates = (r for r in summary.relations
                if {r.subject_track_id, r.object_track_id} == set(track_ids)
                and start <= r.timestamp_seconds < end)
            row = min(candidates, key=lambda r: (abs(r.timestamp_seconds-center),
                r.timestamp_seconds, r.frame_index, r.subject_track_id, r.object_track_id), default=None)
            if row is not None:
                key = (row.subject_track_id, row.object_track_id, row.frame_index)
                pairs[key] = dict(
                    subject_object_id=bindings[by_track[row.subject_track_id].entity_id]['object_id'],
                    object_object_id=bindings[by_track[row.object_track_id].entity_id]['object_id'],
                    time=row.timestamp_seconds, bbox_iou=row.bbox_iou,
                    subject_covered=row.subject_bbox_covered_fraction,
                    object_covered=row.object_bbox_covered_fraction)
                pair_keys.append(key)
        links.append((index, obs_keys, pair_keys))
    counts.update(observations=len(observations), pairs=len(pairs))
    if any(counts[k] > cap for k, cap in LIMITS.items()):
        return full_model_view(context, reason='row_limits', counts=counts)
    obs_index = {key: i for i, key in enumerate(sorted(observations))}
    pair_index = {key: i for i, key in enumerate(sorted(pairs))}
    lifecycle = []
    for index, track in enumerate(tracks):
        endpoints = [track.observations[0]] if track.observations else []
        if len(track.observations) > 1:
            endpoints.append(track.observations[-1])
        lifecycle.append(dict(track_ref=index, entity_id=track.entity_id,
            visibility_lifecycle_complete=track.visibility_lifecycle_complete,
            candidate_search_complete=track.candidate_search_complete,
            endpoints=[_observation(r) for r in endpoints],
            runs=[{k: getattr(r, k) for k in ('start_time', 'end_time', 'state',
                  'minimum_confidence')} for r in track.visibility_runs],
            gaps=[{k: getattr(r, k) for k in ('last_visible_time', 'first_missing_time',
                  'last_missing_time', 'first_revisible_time', 'minimum_confidence',
                  'edge_departure')} for r in track.missing_intervals]))
    data = dict(version=VIEW_VERSION, duration=context['duration'], cv_available=summary is not None,
        known_targets=context['known_targets'],
        segments=[{k: row[k] for k in SEGMENT_KEYS if k in row} for row in context['segments']],
        options=compact_scene_options(context),
        evidence=dict(observations=[observations[k] for k in obs_index],
            pairs=[pairs[k] for k in pair_index],
            option_witness_columns=['option_index', 'observation_indices', 'proxy_indices'],
            option_witnesses=[[i, [obs_index[k] for k in obs], [pair_index[k] for k in pair]]
                              for i, obs, pair in links]),
        lifecycle=lifecycle,
        entities=[dict(entity_id=e.entity_id, canonical_label=e.canonical_label, role=e.role.value)
                  for e in summary.entities] if summary else [],
        completeness=dict(witness_selection='representative_subset',
            endpoint_selection='first_last_retained_observations',
            retained_lifecycle_rows_complete=True,
            candidate_search_complete=summary.candidate_search_complete if summary else None,
            relations_complete=summary.relations_complete if summary else None,
            options_complete=offers['options_complete'] if offers else None))
    encoded = canonical(data).encode('utf-8')
    if len(encoded) > MAX_DATA_BYTES:
        return full_model_view(context, reason='data_bytes', counts=counts)
    return dict(data=data, metadata=dict(mode='compact', version=VIEW_VERSION,
        reason='eligible', data_sha256=hashlib.sha256(encoded).hexdigest(),
        data_bytes=len(encoded), counts=counts))
