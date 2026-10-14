"""Targeted SAM probes and tracks on the original playback clock."""
from dataclasses import dataclass
from pathlib import Path
from ..cv.artifacts import CvArtifactStore, CvArtifactHandle, cv_cache_key
from ..cv.contracts import CvEvidenceRequest, SamplingPolicy, EvidenceThresholds, FrameTimeline, SpatialPrompt
from ..cv.entities import EntityCandidate, normalize_entities
from ..cv.timeline import probe_frame_timeline
from ..domain import InferenceJobSpec
from ..models.base import VisualEvidence
from .base import SafePipelineError

MAX_RECHECKS = 3
MAX_RECHECK_SECONDS = 30.0
CONTEXT_SECONDS = 0.5


def local_asset_identity(path):
    """Cache identity from local metadata; this does not verify file contents."""
    path = Path(path).resolve()
    try:
        status = path.stat()
    except FileNotFoundError:
        return str(path)
    return f"{path}:{status.st_size}:{status.st_mtime_ns}"


@dataclass(frozen=True)
class Evidence:
    audit: dict
    prompt_record: dict
    images: tuple[VisualEvidence, ...]


def targeted_windows(review, duration):
    """Budget only actionable object questions, never editorial/camera issues."""
    selected, unresolved, spent = [], [], 0.0
    for index, concern in enumerate(review.concerns):
        if concern.target is None:
            continue
        start = max(0.0, concern.start - CONTEXT_SECONDS)
        end = min(duration, concern.end + CONTEXT_SECONDS)
        if len(selected) >= MAX_RECHECKS or spent + end - start > MAX_RECHECK_SECONDS:
            unresolved.append(f'SAM budget exhausted for concern {index}: {concern.reason}')
            continue
        selected.append({'start': start, 'end': end, 'concern_index': index,
                         'question': concern.reason, 'target': concern.target.model_dump(mode='json')})
        spent += end - start
    return selected, unresolved


class EvidenceOrchestrator:
    def __init__(self, task, context, media_path, duration, wait_jobs):
        self.task, self.context, self.media_path, self.duration, self.wait_jobs = task, context, media_path, duration, wait_jobs
        self.timeline = probe_frame_timeline(media_path)
        self.video_identity = local_asset_identity(media_path)

    def run(self, candidates, *, ordinal, window, target, mode):
        settings = self.context.settings
        normalized = normalize_entities(candidates, limit=settings.cv_entity_limit)
        if len(normalized.entities) != 1:
            raise SafePipelineError('Targeted SAM requires exactly one object')
        points = tuple(p for p in self.timeline.frames if window['start'] <= p.timestamp_seconds <= window['end'])
        if not points:
            raise SafePipelineError('Local SAM window contains no original frames')
        reference = min(points, key=lambda p: abs(p.timestamp_seconds - target.reference_time))
        selected = FrameTimeline(frames=(reference,) if mode == 'probe' else points)
        spatial = SpatialPrompt(entity_id=normalized.entities[0].entity_id,
                                frame_index=reference.frame_index, bbox_xyxy=target.bbox_xyxy)
        request = CvEvidenceRequest(schema_version='cv_request_v1', provider=settings.cv_provider,
            model_identity=settings.cv_model_alias, video_path=self.media_path, video_identity=self.video_identity,
            duration_seconds=self.duration, frame_count=len(self.timeline.frames), checkpoint_identity=(local_asset_identity(settings.cv_checkpoint_path)
                if settings.cv_checkpoint_path else settings.cv_model_alias),
            timeline=selected, entities=normalized.entities, mode=mode, spatial_prompts=(spatial,),
            sampling=SamplingPolicy(short_video_seconds=settings.cv_short_video_seconds,
                scan_fps=settings.cv_max_fps if window else settings.cv_scan_fps, max_fps=settings.cv_max_fps,
                refinement_radius_seconds=settings.cv_refinement_radius_seconds),
            thresholds=EvidenceThresholds(min_confidence=settings.cv_min_confidence,
                min_area_fraction=settings.cv_min_area_fraction, occlusion_visibility_drop=settings.cv_occlusion_visibility_drop))
        [job] = self.context.store.create_inference_jobs(self.task.task_id, [InferenceJobSpec(
            stage='cv_evidence', ordinal=ordinal, payload=request.model_dump(mode='json'), model_name=settings.cv_model_alias)])
        [result] = self.wait_jobs(self.context.store, self.task.task_id, [job.job_id], settings.cv_timeout_seconds)
        if result.get('status') != 'available' or result.get('artifact_key') != cv_cache_key(request):
            raise SafePipelineError('Targeted SAM evidence unavailable or mismatched')
        handle = CvArtifactHandle(result['artifact_key'])
        scope = f'{mode}_{ordinal}'
        with CvArtifactStore(settings.cv_cache_root, max_files=settings.cv_cache_max_files, max_bytes=settings.cv_cache_max_bytes) as store:
            artifact = store.load(handle)
            if artifact.video_identity != self.video_identity or artifact.entities != request.entities or artifact.provider != request.provider:
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
            raise SafePipelineError('SAM tracks have no visual overlays')
        # Zero detections can be valid. Record this explicitly; original imagery
        # is still supplied and absence of tracks never establishes object absence.
        audit = {**result, 'scope': scope, 'provider': artifact.provider, 'video_identity': self.video_identity,
                 'time_basis': 'container-relative playback seconds; original decoder frame indices',
                 'window': [window['start'], window['end']] if window else [0.0, self.duration],
                 'mode': mode, 'question': window['question'],
                 'spatial_prompt': spatial.model_dump(mode='json'),
                 'reference_time': reference.timestamp_seconds, 'no_tracks': not bool(artifact.tracks),
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
                        'mode': mode, 'reference_time': reference.timestamp_seconds,
                        'warning': 'Track geometry is supporting evidence only; absence, cause, and semantic boundaries require visual review.'}, tuple(images))
