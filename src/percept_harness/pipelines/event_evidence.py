"""Mandatory SAM evidence and bounded original-clock local rechecks."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from ..cv.artifacts import CvArtifactStore, CvArtifactHandle, cv_cache_key
from ..cv.contracts import CvEvidenceRequest, SamplingPolicy, EvidenceThresholds, FrameTimeline
from ..cv.entities import EntityCandidate, normalize_entities
from ..cv.timeline import probe_frame_timeline
from ..domain import InferenceJobSpec
from ..models.base import VisualEvidence
from .base import SafePipelineError

MAX_RECHECKS = 3
MAX_RECHECK_SECONDS = 30.0
CONTEXT_SECONDS = 0.5


@dataclass(frozen=True)
class Evidence:
    audit: dict
    prompt_record: dict
    images: tuple[VisualEvidence, ...]


def merged_windows(review, duration):
    """Contextualize and merge requests before applying a bounded work budget."""
    windows = []
    for concern in sorted(review.concerns, key=lambda c: c.start):
        start, end = max(0.0, concern.start - CONTEXT_SECONDS), min(duration, concern.end + CONTEXT_SECONDS)
        if windows and start <= windows[-1]['end']:
            windows[-1]['end'] = max(windows[-1]['end'], end)
            windows[-1]['new_entities'].extend(concern.new_entities)
        else:
            windows.append({'start': start, 'end': end, 'new_entities': list(concern.new_entities)})
    selected, unresolved, spent = [], [], 0.0
    for window in windows:
        if len(selected) >= MAX_RECHECKS or spent + window['end'] - window['start'] > MAX_RECHECK_SECONDS:
            unresolved.append(f"Recheck budget exhausted for [{window['start']}, {window['end']}] seconds")
            continue
        selected.append(window)
        spent += window['end'] - window['start']
    return selected, unresolved


class EvidenceOrchestrator:
    def __init__(self, task, context, media_path, duration, wait_jobs):
        self.task, self.context, self.media_path, self.duration, self.wait_jobs = task, context, media_path, duration, wait_jobs
        self.timeline = probe_frame_timeline(media_path)
        with media_path.open('rb') as stream:
            self.video_sha = hashlib.file_digest(stream, 'sha256').hexdigest()

    def run(self, candidates, *, ordinal=0, window=None):
        settings = self.context.settings
        normalized = normalize_entities(candidates, limit=settings.cv_entity_limit)
        empty = not normalized.entities
        if empty:
            # A broad visual probe actually runs the provider even when nomination
            # found nothing. This label is provenance, never a semantic assertion.
            normalized = normalize_entities([EntityCandidate(name='visible objects', aliases=(), role='other')], limit=settings.cv_entity_limit)
        selected = self.timeline
        if window is not None:
            points = tuple(p for p in self.timeline.frames if window['start'] <= p.timestamp_seconds <= window['end'])
            if not points:
                raise SafePipelineError('Local SAM window contains no original frames')
            selected = FrameTimeline(frames=points)
        request = CvEvidenceRequest(schema_version='cv_request_v1', provider=settings.cv_provider,
            model_identity=settings.cv_model_alias, video_path=self.media_path, video_sha256=self.video_sha,
            duration_seconds=self.duration, frame_count=len(self.timeline.frames), checkpoint_sha256=settings.cv_checkpoint_sha256,
            timeline=selected, entities=normalized.entities,
            sampling=SamplingPolicy(short_video_seconds=settings.cv_short_video_seconds,
                scan_fps=settings.cv_max_fps if window else settings.cv_scan_fps, max_fps=settings.cv_max_fps,
                refinement_radius_seconds=settings.cv_refinement_radius_seconds),
            thresholds=EvidenceThresholds(min_confidence=settings.cv_min_confidence,
                min_area_fraction=settings.cv_min_area_fraction, occlusion_visibility_drop=settings.cv_occlusion_visibility_drop))
        [job] = self.context.store.create_inference_jobs(self.task.task_id, [InferenceJobSpec(
            stage='cv_evidence', ordinal=ordinal, payload=request.model_dump(mode='json'), model_name=settings.cv_model_alias)])
        [result] = self.wait_jobs(self.context.store, self.task.task_id, [job.job_id], settings.cv_timeout_seconds)
        if result.get('status') != 'available' or result.get('artifact_key') != cv_cache_key(request):
            raise SafePipelineError('Mandatory SAM evidence unavailable or mismatched')
        handle = CvArtifactHandle(result['artifact_key'], result['manifest_sha256'])
        scope = 'baseline' if ordinal == 0 else f'local_{ordinal}'
        with CvArtifactStore(settings.cv_cache_root, max_files=settings.cv_cache_max_files, max_bytes=settings.cv_cache_max_bytes) as store:
            artifact = store.load(handle)
            if artifact.video_sha256 != self.video_sha or artifact.entities != request.entities or artifact.provider != request.provider:
                raise SafePipelineError('SAM artifact does not match the evidence request')
            if artifact.processed_timeline is None:
                raise SafePipelineError('SAM evidence has no observed source frames')
            expected = {p.frame_index: p.timestamp_seconds for p in selected.frames}
            if any(expected.get(p.frame_index) != p.timestamp_seconds for p in artifact.processed_timeline.frames):
                raise SafePipelineError('SAM frame mapping does not match original video')
            overlay_bytes = store.read_overlays(handle, tuple(record.path for record in artifact.overlay_records))
        tracks = {track.track_id: track for track in artifact.tracks}
        images = []
        for record in artifact.overlay_records:
            track = tracks[record.track_id]
            observation = next(p for p in track.observations if p.frame_index == record.frame_index)
            images.append(VisualEvidence(png_bytes=overlay_bytes[record.path], timestamp=observation.timestamp_seconds,
                          entity_id=track.entity_id, track_id=f'{scope}:{track.track_id}', source=scope))
        if artifact.tracks and not images:
            raise SafePipelineError('SAM tracks have no authenticated visual overlays')
        # Zero detections can be valid. Record this explicitly; original imagery
        # is still supplied and absence of tracks never establishes object absence.
        audit = {**result, 'scope': scope, 'provider': artifact.provider, 'video_sha256': self.video_sha,
                 'window': [window['start'], window['end']] if window else [0.0, self.duration],
                 'empty_nomination_scene_probe': empty, 'no_tracks': not bool(artifact.tracks),
                 'entities': [e.model_dump(mode='json') for e in artifact.entities],
                 'entity_mapping': {e.entity_id: e.canonical_label for e in artifact.entities},
                 'frame_mapping': [{'provider_frame_index': p.frame_index, 'original_frame_index': p.frame_index,
                                    'timestamp_seconds': p.timestamp_seconds} for p in artifact.processed_timeline.frames],
                 'normalization_warnings': list(normalized.warnings),
                 'overlay_count': len(images)}
        # Bound prompt observations to the actual overlay frames. Full raw tracks
        # remain in the immutable artifact; no hidden local-to-global rewriting.
        visible_frames = {r.frame_index for r in artifact.overlay_records}
        prompt_tracks = [{'track_id': f'{scope}:{t.track_id}', 'entity_id': t.entity_id,
                          'observations': [o.model_dump(mode='json', exclude={'mask_ref'}) for o in t.observations if o.frame_index in visible_frames]}
                         for t in artifact.tracks]
        return Evidence(audit, {'scope': scope, 'window': audit['window'], 'entities': audit['entities'],
                        'tracks': prompt_tracks, 'no_tracks': audit['no_tracks'],
                        'empty_nomination_scene_probe': empty,
                        'warning': 'Track geometry is supporting evidence only; absence, cause, and semantic boundaries require visual review.'}, tuple(images))
