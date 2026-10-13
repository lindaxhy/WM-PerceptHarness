"""Targeted SAM regressions: original clocks, probe masks, and cache isolation."""
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from percept_harness.cv.contracts import CvEvidenceRequest, CvEvidenceArtifact
from percept_harness.cv.base import FakeCvEvidenceProvider
from percept_harness.cv.artifacts import cv_cache_key, CvArtifactStore, _cache_identity, _canonical_json
from percept_harness.cv.sam31 import Sam31EvidenceProvider
from percept_harness.cv.timeline import SampledFrame, SampledFrameSet


def request(**updates):
    payload = dict(schema_version='cv_request_v1', provider='sam31', model_identity='sam3.1',
        video_path=Path('/tmp/source.mp4'), video_sha256='a'*64, checkpoint_sha256='b'*64,
        duration_seconds=20.0, frame_count=100,
        timeline={'frames': [{'frame_index': i, 'timestamp_seconds': t} for i,t in [(20,2.0),(30,3.0),(40,4.0)]]},
        entities=[{'entity_id':'cup','canonical_label':'red cup','aliases':['mug'],'role':'other'}],
        sampling={'short_video_seconds':30.0,'scan_fps':1.0,'max_fps':4.0,'refinement_radius_seconds':1.0},
        thresholds={'min_confidence':0.1,'min_area_fraction':0.001,'occlusion_visibility_drop':0.5})
    payload.update(updates)
    return CvEvidenceRequest.model_validate(payload)


def spatial(frame=30, box=(0.25,0.25,0.75,0.75)):
    return {'entity_id':'cup', 'frame_index':frame, 'bbox_xyxy':box}


class Array:
    def __init__(self, values, shape, kind):
        self.values=values
        self.shape=shape
        self.dtype=SimpleNamespace(kind=kind)
    def __getitem__(self, index):
        value=self.values
        for offset in index if isinstance(index,tuple) else (index,):
            value=value[offset]
        return value


class Predictor:
    """Small external SAM boundary; masks expose which frames really ran."""
    def __init__(self):
        self.calls=[]
    def response(self, index):
        return {'frame_index':index, 'outputs':{'out_obj_ids':Array([1],(1,),'i'),
            'out_probs':Array([0.9],(1,),'f'),'out_boxes_xywh':Array([[0.25,0.25,0.5,0.5]],(1,4),'f'),
            'out_binary_masks':Array([[bytes([1]*4)]*4],(1,4,4),'b')}}
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
        path.write_bytes(b'\xff\xd8\xff\xc0\x00\x07\x08\x00\x04\x00\x04\xff\xd9')
        frames.append(SampledFrame(local,original,times[original],path))
    return SampledFrameSet(tuple(frames))


class TargetedCvTests(unittest.TestCase):
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
        normalized,_=CvArtifactStore._validated_cache_identity(_cache_identity(probe))
        self.assertEqual(normalized['mode'],'probe')
        for invalid in [spatial(frame=31),spatial(box=(0.7,0.2,0.1,0.8)),spatial(box=(0.1,0.2,1.1,0.8)),dict(spatial(),entity_id='other')]:
            with self.assertRaises(ValidationError):
                request(mode='probe',spatial_prompts=[invalid])
        with self.assertRaises(ValidationError):
            request(mode='probe')
        with self.assertRaises(ValidationError):
            request(spatial_prompts=[spatial(),spatial()])

    def test_fake_probe_uses_reference_and_spatial_box(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root).resolve()
            video=root/'clip.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'color=c=red:s=32x32:r=10:d=5','-c:v','mpeg4',str(video)],check=True)
            probe=request(provider='fake',video_path=video,mode='probe',spatial_prompts=[spatial()])
            artifact=FakeCvEvidenceProvider().analyze(probe,root/'stage')
            self.assertEqual([f.frame_index for f in artifact.processed_timeline.frames],[30])
            self.assertEqual(artifact.tracks[0].observations[0].bbox_xyxy,(0.25,0.25,0.75,0.75))
            self.assertEqual([o.frame_index for o in artifact.tracks[0].observations],[30])
            with CvArtifactStore(root/'cache') as store:
                key=cv_cache_key(probe)
                with store.staging(key) as staging:
                    artifact=FakeCvEvidenceProvider().analyze(probe,staging)
                    handle=store.publish(probe,staging,artifact)
                self.assertEqual(store.lookup(key),handle)
                self.assertIsNone(store.lookup(cv_cache_key(request(provider='fake',video_path=video,
                    mode='track',spatial_prompts=[spatial()]))))

    def test_legacy_manifest_identity_remains_readable_but_cannot_hit_new_key(self):
        current=request()
        legacy={key:value for key,value in _cache_identity(current).items()
                if key not in {'timeline','duration_seconds','frame_count','mode','spatial_prompts'}}
        normalized,entities=CvArtifactStore._validated_cache_identity(legacy)
        self.assertEqual(normalized,legacy)
        self.assertEqual(entities,current.entities)
        self.assertNotEqual(hashlib.sha256(_canonical_json(legacy)).hexdigest(),cv_cache_key(current))

    def test_manifest_rejects_evidence_outside_selected_window(self):
        probe=request(mode='probe',spatial_prompts=[spatial()])
        artifact=CvEvidenceArtifact(schema_version='cv_evidence_v1',status='available',
            provider=probe.provider,model_identity=probe.model_identity,
            video_sha256=probe.video_sha256,checkpoint_sha256=probe.checkpoint_sha256,
            processed_timeline={'frames':[{'frame_index':20,'timestamp_seconds':2.0}]},
            entities=probe.entities,tracks=(),files=(),overlay_records=())
        with self.assertRaises(ValueError):
            CvArtifactStore._validate_artifact_matches_request(probe,artifact)

    def provider(self, predictor):
        return Sam31EvidenceProvider(predictor=predictor,torch_module=SimpleNamespace(),
            materialize_frames=materialize,initial_sampler=lambda *args:(20,40),
            overlay_renderer=lambda **kwargs:kwargs['destination'].write_bytes(b'PNG fixture'))

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

    def test_track_keeps_unsampled_reference_and_covers_both_sides(self):
        predictor=Predictor()
        with tempfile.TemporaryDirectory() as root:
            artifact=self.provider(predictor).analyze(request(spatial_prompts=[spatial()]),Path(root).resolve())
            self.assertEqual([o.frame_index for o in artifact.tracks[0].observations],[20,30,40])
            prompt=next(c for c in predictor.calls if c['type']=='add_prompt')
            self.assertEqual(prompt['frame_index'],1)
            self.assertEqual(prompt['bounding_boxes'],[[0.25,0.25,0.5,0.5]])

if __name__=='__main__':
    unittest.main()
