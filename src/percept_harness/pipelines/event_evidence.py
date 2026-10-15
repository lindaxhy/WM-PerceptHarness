"""Targeted SAM probes and tracks on the original playback clock."""
from dataclasses import dataclass
from pathlib import Path
from ..cv.artifacts import CvArtifactStore, CvArtifactHandle, cv_cache_key
from ..cv.contracts import CvEvidenceRequest, SamplingPolicy, FrameTimeline, SpatialPrompt, EntityPrompt
from ..cv.timeline import probe_frame_timeline
from ..models.base import VisualEvidence
from .base import SafePipelineError

MAX_RECHECKS = 32
MAX_RECHECK_SECONDS = 300.0
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


def targeted_windows(concerns, duration, *, max_windows=MAX_RECHECKS, max_seconds=MAX_RECHECK_SECONDS):
    """Queue every concern for local target planning, subject to explicit budgets."""
    selected, unresolved, spent = [], [], 0.0
    for index, concern in enumerate(concerns):
        start = max(0.0, concern.start - CONTEXT_SECONDS)
        end = min(duration, concern.end + CONTEXT_SECONDS)
        if len(selected) >= max_windows or spent + end - start > max_seconds:
            unresolved.append(f'SAM budget exhausted for concern {index}: {concern.reason}')
            continue
        selected.append({'start': start, 'end': end, 'concern_index': index,
                         'question': concern.reason, 'kind': concern.kind,
                         'target': concern.target.model_dump(mode='json') if concern.target else None})
        spent += end - start
    return selected, unresolved


class EvidenceOrchestrator:
    def __init__(self, context, media_path, duration):
        self.context, self.media_path, self.duration = context, media_path, duration
        self.timeline = probe_frame_timeline(media_path)
        self.video_identity = local_asset_identity(media_path)

    def run(self, *, ordinal, window, target, mode):
        settings = self.context.settings
        entity = EntityPrompt(entity_id="target", canonical_label=target.description)
        points = tuple(p for p in self.timeline.frames if window['start'] <= p.timestamp_seconds <= window['end'])
        if not points:
            raise SafePipelineError('Local SAM window contains no original frames')
        reference = min(points, key=lambda p: abs(p.timestamp_seconds - target.reference_time))
        selected = FrameTimeline(frames=(reference,) if mode == 'probe' else points)
        spatial = SpatialPrompt(entity_id=entity.entity_id,
                                frame_index=reference.frame_index, bbox_xyxy=target.bbox_xyxy)
        request = CvEvidenceRequest(schema_version='cv_request_v1', provider=settings.cv_provider,
            model_identity="sam3.1", video_path=self.media_path, video_identity=self.video_identity,
            duration_seconds=self.duration, frame_count=len(self.timeline.frames), checkpoint_identity=local_asset_identity(settings.cv_checkpoint_path),
            timeline=selected, entities=(entity,), mode=mode, spatial_prompts=(spatial,),
            sampling=SamplingPolicy(max_fps=settings.cv_max_fps))
        result = self.context.executor.execute('cv_evidence', request.model_dump(mode='json'))
        if result.get('status') != 'available' or result.get('artifact_key') != cv_cache_key(request):
            raise SafePipelineError('Targeted SAM evidence unavailable or mismatched')
        handle = CvArtifactHandle(result['artifact_key'])
        scope = f'{mode}_{ordinal}'
        with CvArtifactStore(settings.cv_cache_root) as store:
            artifact = store.load(handle)
            if artifact.processed_timeline is None:
                raise SafePipelineError('SAM evidence has no observed source frames')
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
                 'window': [window['start'], window['end']],
                 'mode': mode, 'question': window['question'],
                 'spatial_prompt': spatial.model_dump(mode='json'),
                 'reference_time': reference.timestamp_seconds, 'no_tracks': not bool(artifact.tracks),
                 'entities': [e.model_dump(mode='json') for e in artifact.entities],
                 'entity_mapping': {e.entity_id: e.canonical_label for e in artifact.entities},
                 'frame_mapping': [{'provider_frame_index': p.frame_index, 'original_frame_index': p.frame_index,
                                    'timestamp_seconds': p.timestamp_seconds} for p in artifact.processed_timeline.frames],
                 'normalization_warnings': [],
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
