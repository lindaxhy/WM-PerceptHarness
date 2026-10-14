"""Targeted SAM regressions: original clocks, probe masks, and cache isolation."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError
try:
    import numpy
except ImportError:
    numpy = None

from percept_harness.cv.contracts import CvEvidenceRequest, CvEvidenceArtifact
from percept_harness.cv.artifacts import cv_cache_key, CvArtifactStore, CvArtifactError, _cache_identity, _canonical_json
from percept_harness.cv.sam31 import Sam31EvidenceProvider
from percept_harness.cv.timeline import SampledFrame, SampledFrameSet


def request(**updates):
    payload = dict(schema_version='cv_request_v1', provider='sam31', model_identity='sam3.1',
        video_path=Path('/tmp/source.mp4'), video_identity='source-video', checkpoint_identity='fixture-checkpoint',
        duration_seconds=20.0, frame_count=100,
        timeline={'frames': [{'frame_index': i, 'timestamp_seconds': t} for i,t in [(20,2.0),(30,3.0),(40,4.0)]]},
        entities=[{'entity_id':'cup','canonical_label':'red cup'}],
        sampling={'max_fps':4.0},
        spatial_prompts=[spatial()])
    payload.update(updates)
    return CvEvidenceRequest.model_validate(payload)


def spatial(frame=30, box=(0.25,0.25,0.75,0.75)):
    return {'entity_id':'cup', 'frame_index':frame, 'bbox_xyxy':box}


class Predictor:
    """Small external SAM boundary; masks expose which frames really ran."""
    def __init__(self):
        self.calls=[]
    def response(self, index):
        return {'frame_index':index, 'outputs':{'out_obj_ids':numpy.array([1],dtype=numpy.int64),
            'out_probs':numpy.array([0.9]),'out_boxes_xywh':numpy.array([[0.25,0.25,0.5,0.5]]),
            'out_binary_masks':numpy.ones((1,4,4),dtype=numpy.bool_)}}
    def handle_request(self, payload):
        self.calls.append(payload)
        if payload['type']=='start_session':
            self.count=len(list(Path(payload['resource_path']).glob('*.jpg')))
            return {'session_id':'test'}
        if payload['type']=='add_prompt':
            return self.response(payload['frame_index'])
        return {}
    def handle_stream_request(self, payload):
        self.calls.append(payload)
        start=payload['start_frame_index']
        direction=payload['propagation_direction']
        if direction=='both':
            indices=list(range(start,self.count))+list(range(start-1,-1,-1))
        elif direction=='backward':
            indices=range(start,-1,-1)
        else:
            indices=range(start,self.count)
        return iter(self.response(i) for i in indices)


def materialize(video, timeline, indices, destination):
    destination.mkdir(parents=True)
    times={f.frame_index:f.timestamp_seconds for f in timeline.frames}
    frames=[]
    for local, original in enumerate(indices):
        path=destination/f'{local:06d}.jpg'
        from PIL import Image
        Image.new('RGB', (4, 4), 'red').save(path)
        frames.append(SampledFrame(local,original,times[original],path))
    return SampledFrameSet(tuple(frames))


class TargetedCvTests(unittest.TestCase):
    def test_sampling_keeps_source_indices_at_pts_targets_and_endpoints(self):
        from percept_harness.cv.contracts import FrameTimeline, SamplingPolicy
        from percept_harness.cv.timeline import initial_sample_indices

        for timestamps, rate, expected in (
            ((2.0,), 4.0, (20,)),
            ((2.0, 2.5, 3.0), 4.0, (20, 21, 22)),
            ((2.0, 2.1, 2.7, 2.8, 3.1), 2.0, (20, 22, 24)),
        ):
            timeline = FrameTimeline(frames=tuple(
                {'frame_index': 20 + index, 'timestamp_seconds': timestamp}
                for index, timestamp in enumerate(timestamps)
            ))
            with self.subTest(timestamps=timestamps, rate=rate):
                self.assertEqual(
                    initial_sample_indices(timeline, SamplingPolicy(max_fps=rate)),
                    expected,
                )

    def test_cache_separates_selected_timeline_and_source_clock(self):
        original=request()
        for changed in [request(timeline={'frames':[{'frame_index':30,'timestamp_seconds':3.0}]}),
                        request(duration_seconds=21.0),request(frame_count=101)]:
            self.assertNotEqual(cv_cache_key(original),cv_cache_key(changed))

    def test_spatial_contract_and_cache_separate_modes_boxes(self):
        probe=request(mode='probe',spatial_prompts=[spatial()])
        track=request(mode='track',spatial_prompts=[spatial()])
        shifted=request(mode='probe',spatial_prompts=[spatial(box=(0.1,0.1,0.5,0.5))])
        self.assertEqual(len({cv_cache_key(x) for x in [probe,track,shifted]}),3)
        self.assertEqual(_cache_identity(probe)['mode'],'probe')
        for invalid in [spatial(frame=31),spatial(box=(0.7,0.2,0.1,0.8)),spatial(box=(0.1,0.2,1.1,0.8)),dict(spatial(),entity_id='other')]:
            with self.assertRaises(ValidationError):
                request(mode='probe',spatial_prompts=[invalid])
        with self.assertRaises(ValidationError):
            request(mode='probe', spatial_prompts=[])
        with self.assertRaises(ValidationError):
            request(spatial_prompts=[spatial(),spatial()])

    def test_incomplete_legacy_cache_is_a_miss(self):
        current=request()
        legacy={key:value for key,value in _cache_identity(current).items()
                if key not in {'timeline','duration_seconds','frame_count','mode','spatial_prompts'}}
        key = hashlib.sha256(_canonical_json(legacy)).hexdigest()
        self.assertNotEqual(key, cv_cache_key(current))
        with tempfile.TemporaryDirectory() as root, CvArtifactStore(Path(root)/'cache') as store:
            entry = store._entry_path(key)
            entry.parent.mkdir(mode=0o700)
            entry.mkdir(mode=0o700)
            (entry/'manifest.json').write_bytes(_canonical_json({'cache_identity': legacy, 'artifact': {}}))
            self.assertIsNone(store.lookup(key))

    def test_manifest_rejects_evidence_outside_selected_window(self):
        probe=request(mode='probe',spatial_prompts=[spatial()])
        artifact=CvEvidenceArtifact(schema_version='cv_evidence_v1',status='available',
            provider=probe.provider,model_identity=probe.model_identity,
            video_identity=probe.video_identity,checkpoint_identity=probe.checkpoint_identity,
            processed_timeline={'frames':[{'frame_index':20,'timestamp_seconds':2.0}]},
            entities=probe.entities,tracks=(),files=(),overlay_records=())
        with self.assertRaises(ValueError):
            CvArtifactStore._validate_artifact_matches_request(probe,artifact)

    def provider(self, predictor):
        return Sam31EvidenceProvider(predictor=predictor,torch_module=SimpleNamespace(),
            materialize_frames=materialize,initial_sampler=lambda *args:(20,40),
            overlay_renderer=lambda **kwargs:kwargs['destination'].write_bytes(b'PNG fixture'))

    @unittest.skipIf(numpy is None, 'SAM predictor tests require NumPy')
    def test_probe_returns_reference_mask_without_propagation(self):
        predictor=Predictor()
        with tempfile.TemporaryDirectory() as root:
            artifact=self.provider(predictor).analyze(request(mode='probe',spatial_prompts=[spatial()]),Path(root).resolve())
            self.assertEqual([f.frame_index for f in artifact.processed_timeline.frames],[30])
            self.assertEqual([o.frame_index for o in artifact.tracks[0].observations],[30])
            self.assertTrue(artifact.overlay_records,'probe must supply an image for prompt verification')
            self.assertFalse(any(c['type']=='propagate_in_video' for c in predictor.calls))
            prompt=next(c for c in predictor.calls if c['type']=='add_prompt')
            self.assertEqual(prompt['bounding_boxes'],[[0.25,0.25,0.5,0.5]])
            self.assertEqual(prompt['bounding_box_labels'],[1])
            self.assertEqual(prompt['text'],'red cup')

    @unittest.skipIf(numpy is None, 'SAM predictor tests require NumPy')
    def test_middle_reference_does_not_extend_pinned_detector_past_video_end(self):
        class PinnedBoundsPredictor(Predictor):
            def handle_stream_request(self, payload):
                # The pinned detector uses start + limit without clamping to
                # num_frames; None selects exactly the full source range.
                limit = payload.get('max_frame_num_to_track')
                end = (self.count if limit is None
                       else payload['start_frame_index'] + limit)
                if end > self.count:
                    raise IndexError('pinned detector requested a nonexistent image')
                return super().handle_stream_request(payload)

        predictor = PinnedBoundsPredictor()
        with tempfile.TemporaryDirectory() as root:
            artifact = self.provider(predictor).analyze(
                request(spatial_prompts=[spatial()]), Path(root).resolve()
            )
            self.assertEqual(
                [o.frame_index for o in artifact.tracks[0].observations],
                [20, 30, 40],
            )

    @unittest.skipIf(numpy is None, 'SAM predictor tests require NumPy')
    def test_track_keeps_unsampled_reference_and_covers_both_sides(self):
        predictor=Predictor()
        with tempfile.TemporaryDirectory() as root:
            artifact=self.provider(predictor).analyze(request(spatial_prompts=[spatial()]),Path(root).resolve())
            self.assertEqual([o.frame_index for o in artifact.tracks[0].observations],[20,30,40])
            prompt=next(c for c in predictor.calls if c['type']=='add_prompt')
            self.assertEqual(prompt['frame_index'],1)
            self.assertEqual(prompt['bounding_boxes'],[[0.25,0.25,0.5,0.5]])



class TargetedSchemaTests(unittest.TestCase):
    def test_targeted_request_rejects_retired_options_and_multiple_targets(self):
        for field, update in (
            ('entities', [{'entity_id': 'cup', 'canonical_label': 'cup', 'aliases': ['mug']}]),
            ('entities', [{'entity_id': 'cup', 'canonical_label': 'cup', 'role': 'other'}]),
            ('sampling', {'max_fps': 4.0, 'scan_fps': 1.0}),
            ('thresholds', {'min_confidence': 0.1, 'min_area_fraction': 0.01}),
            ('thresholds', {'min_confidence': 0.1, 'occlusion_visibility_drop': 0.5}),
            ('entities', []),
            ('entities', [{'entity_id': 'cup', 'canonical_label': 'cup'},
                          {'entity_id': 'plate', 'canonical_label': 'plate'}]),
            ('spatial_prompts', []),
        ):
            with self.assertRaises(ValidationError):
                request(**{field: update})
        original = request()
        self.assertEqual(CvEvidenceRequest.model_validate_json(original.model_dump_json()), original)


    def test_artifact_references_and_resource_bounds_survive_schema_simplification(self):
        current = request()
        base = dict(schema_version='cv_evidence_v1', status='available',
                    provider=current.provider, model_identity=current.model_identity,
                    video_identity=current.video_identity, checkpoint_identity=current.checkpoint_identity,
                    processed_timeline=current.timeline, entities=current.entities,
                    tracks=[], files=[], overlay_records=[])
        self.assertEqual(CvEvidenceArtifact.model_validate(base).processed_timeline, current.timeline)
        observation = dict(frame_index=30, timestamp_seconds=3.0, bbox_xyxy=[0.1, 0.1, 0.8, 0.8],
                           mask_ref='masks/cup.npz', visible=True, confidence=0.9,
                           area_fraction=0.1, center_xy=[0.5, 0.5])
        track = dict(track_id='cup_1', entity_id='cup', observations=[observation])
        with self.assertRaisesRegex(ValidationError, 'mask_ref must reference'):
            CvEvidenceArtifact.model_validate({**base, 'tracks': [track]})
        with self.assertRaisesRegex(ValidationError, 'processed timeline'):
            CvEvidenceArtifact.model_validate({**base, 'tracks': [{**track, 'observations': [
                {**observation, 'timestamp_seconds': 3.1, 'mask_ref': None}]}]})
        with self.assertRaisesRegex(ValidationError, 'at most 24'):
            CvEvidenceArtifact.model_validate({**base, 'overlay_records': [{}] * 25})
        with self.assertRaisesRegex(ValidationError, 'too many total observations'):
            CvEvidenceArtifact.model_validate({**base, 'tracks': [{'observations': [None] * 64_001}]})


if __name__=='__main__':
    unittest.main()
