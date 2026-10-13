import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from percept_harness.config import Settings
from percept_harness.cv.artifacts import CvArtifactStore
from percept_harness.cv.base import FakeCvEvidenceProvider
from percept_harness.cv.executor import SyncCvExecutor
from percept_harness.models.fake import FakeVideoModel
from percept_harness.runner import SyncRunner


class RecordingModel(FakeVideoModel):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


class GroundedAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.video = self.root / 'clip.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'color=c=red:s=64x64:r=4:d=4', '-c:v', 'mpeg4', str(self.video)], check=True)
        self.settings = Settings(work_root=self.root / 'work', cv_cache_root=self.root / 'cv', cv_provider='fake')
        self.artifacts = CvArtifactStore(self.settings.cv_cache_root)
        self.addCleanup(self.artifacts.close)
        self.executor = SyncCvExecutor(FakeCvEvidenceProvider(), self.artifacts)

    def run_model(self, model, executor=True):
        return SyncRunner(model, self.settings, model_alias='fake',
                          cv_executor=self.executor if executor else None).evaluate(
                              self.video, 'embodied_action_captioning')

    def test_grounded_stages_carry_authenticated_images_and_minimal_public_data(self):
        model = RecordingModel()
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(set(result.data), {'duration', 'events'})
        self.assertEqual([r.stage for r in model.requests],
                         ['entity_nomination', 'grounded_draft', 'evidence_review', 'grounded_revision'])
        for request in model.requests[1:]:
            self.assertIsNotNone(request.response_contract, 'grounded stages must declare structured schemas')
            self.assertTrue(request.visual_evidence)
            self.assertEqual(request.video_path, self.video)
            self.assertTrue(all(e.png_bytes.startswith(b'\x89PNG') for e in request.visual_evidence))
        self.assertTrue(result.metadata['synthetic'])
        self.assertEqual(result.metadata['phases'], ['entity_nomination', 'baseline_sam', 'grounded_draft',
                                                     'evidence_review', 'local_sam_rechecks', 'grounded_revision'])

    def test_successful_zero_detection_preserves_no_tracks_provenance(self):
        class ZeroDetectionProvider(FakeCvEvidenceProvider):
            def analyze(self, request, staging):
                from percept_harness.cv.contracts import CvEvidenceArtifact, EvidenceStatus
                return CvEvidenceArtifact(schema_version='cv_evidence_v1', status=EvidenceStatus.AVAILABLE,
                    provider=request.provider, model_identity=request.model_identity,
                    video_sha256=request.video_sha256, checkpoint_sha256=request.checkpoint_sha256,
                    processed_timeline=request.timeline, entities=request.entities,
                    tracks=(), files=(), overlay_records=())
        self.executor = SyncCvExecutor(ZeroDetectionProvider(), self.artifacts)
        model = RecordingModel()
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertTrue(result.metadata['evidence'][0]['no_tracks'])
        self.assertEqual(result.metadata['evidence'][0]['overlay_count'], 0)
        self.assertEqual(model.requests[1].visual_evidence, ())
        self.assertEqual(model.requests[1].video_path, self.video)
        self.assertIn('no_tracks', model.requests[1].prompt)

    def test_recheck_budget_reports_selected_windows(self):
        model = RecordingModel(failure_script={'evidence_review': [{'concerns': [
            {'reason': 'Check boundary', 'event_indices': [0], 'start': 1, 'end': 2, 'new_entities': []}], 'unresolved': []}]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(result.metadata['recheck_budget']['selected_windows'], 1)
        self.assertNotIn('requested_windows', result.metadata['recheck_budget'])

    def test_missing_mandatory_cv_fails_instead_of_degrading(self):
        result = self.run_model(FakeVideoModel(), executor=False)
        self.assertEqual(result.status, 'failed')
        self.assertIsNone(result.data)

    def test_empty_annotation_still_has_baseline_and_review(self):
        empty = {'duration': 4.0, 'events': []}
        model = RecordingModel(failure_script={'entity_nomination': [{'entities': []}],
                                               'grounded_draft': [empty], 'grounded_revision': [empty]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(result.data, empty)
        self.assertTrue(result.metadata['evidence'][0]['empty_nomination_scene_probe'])
        self.assertTrue(model.requests[2].visual_evidence)

    def test_review_merges_context_windows_and_adds_entities_on_original_clock(self):
        review = {'concerns': [
            {'reason': 'Object obscured', 'event_indices': [0], 'start': 1.5, 'end': 2.0, 'new_entities': ['blue cup']},
            {'reason': 'Completion uncertain', 'event_indices': [0], 'start': 2.0, 'end': 2.5, 'new_entities': []}],
            'unresolved': []}
        model = RecordingModel(failure_script={'evidence_review': [review]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        evidence = result.metadata['evidence']
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[1]['window'], [1.0, 3.0])
        self.assertIn('blue cup', [e['canonical_label'] for e in evidence[1]['entities']])
        self.assertTrue(all(1.0 <= p['timestamp_seconds'] <= 3.0 for p in evidence[1]['frame_mapping']))
        self.assertTrue(all(p['original_frame_index'] == p['provider_frame_index'] for p in evidence[1]['frame_mapping']))
        self.assertIn('local_1:', model.requests[-1].prompt)

    def test_local_new_entity_is_not_dropped_when_baseline_fills_limit(self):
        self.settings = self.settings.model_copy(update={'cv_entity_limit': 1})
        model = RecordingModel(failure_script={'evidence_review': [{'concerns': [
            {'reason': 'Check newly visible blue cup', 'event_indices': [0], 'start': 1.0, 'end': 2.0, 'new_entities': ['blue cup']}], 'unresolved': []}]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(result.metadata['evidence'][1]['entities'][0]['canonical_label'], 'blue cup')
        self.assertTrue(result.metadata['unresolved_concerns'])

    def test_local_failure_is_audited_and_final_revision_is_conservative(self):
        from percept_harness.cv.base import CvProviderError
        class FailedRecheck(FakeCvEvidenceProvider):
            def analyze(self, request, staging):
                if request.timeline.frames[0].timestamp_seconds > 0:
                    raise CvProviderError('fixture recheck unavailable')
                return super().analyze(request, staging)
        self.executor = SyncCvExecutor(FailedRecheck(), self.artifacts)
        model = RecordingModel(failure_script={'evidence_review': [{'concerns': [
            {'reason': 'Completion uncertain', 'event_indices': [0], 'start': 2.0, 'end': 2.5, 'new_entities': []}], 'unresolved': []}]})
        result=self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(len(result.metadata['evidence']), 1)
        self.assertTrue(any('unavailable' in x for x in result.metadata['unresolved_concerns']))
        self.assertIn('Completion uncertain', model.requests[-1].prompt)
        self.assertIn('Local SAM recheck 1 unavailable', model.requests[-1].prompt)

    def test_strict_event_schema_rejects_categories_and_bad_times_but_allows_overlap(self):
        from percept_harness.pipelines import output_validation
        registry = output_validation.DEFAULT_OUTPUT_SCHEMAS
        valid = {'duration': 4.0, 'events': [{'start': 0.0, 'end': 3.0, 'description': 'Hand moves.'},
                                           {'start': 1.0, 'end': 4.0, 'description': 'Cup falls.'}]}
        self.assertEqual(registry.sanitize('GroundedEvents', valid, {'duration': 4.0}), valid)
        for patch in ({'start': True}, {'end': float('inf')}, {'description': ' '}, {'skill': 'move'}, {'end': 5.0}):
            value = json.loads(json.dumps(valid))
            value['events'][0].update(patch)
            self.assertIsNotNone(registry.failure_codes('GroundedEvents', registry.sanitize('GroundedEvents', value, {'duration': 4.0})))

    def test_settings_default_does_not_preflight_gpu_assets(self):
        self.assertEqual(Settings().cv_provider, 'sam31')
        self.assertEqual(Settings().cv_bpe_path, Settings().cv_repository_path / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz')


if __name__ == '__main__':
    unittest.main()

class AnnotationGuardTests(unittest.TestCase):
    def test_production_model_cannot_use_fake_cv(self):
        class ProductionModel:
            def generate(self, request):
                raise AssertionError('must reject before model execution')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'video.mp4'; path.write_bytes(b'not needed')
            result=SyncRunner(ProductionModel(),Settings(cv_provider='fake'),model_alias='prod').evaluate(path,'embodied_action_captioning')
            self.assertEqual(result.status,'failed')
            self.assertIn('Fake CV',result.error)

    def test_annotation_specific_preflight_rejects_disabled_and_missing_sam(self):
        from argparse import Namespace
        from percept_harness.cli import _cv_executor
        with self.assertRaisesRegex(ValueError, 'requires SAM'):
            with _cv_executor('disabled',Namespace(backend='fake',cv_device=None),Settings(cv_provider='disabled')):
                pass
        with self.assertRaisesRegex(ValueError, 'cv_repository_path'):
            with _cv_executor('sam31',Namespace(backend='openai',cv_device=None),Settings(cv_repository_path=Path('/nonexistent/sam'))):
                pass

    def test_fake_cv_preflight_rejects_production_backend(self):
        from argparse import Namespace
        from percept_harness.cli import _cv_executor
        with self.assertRaisesRegex(ValueError, 'fake model'):
            with _cv_executor('fake',Namespace(backend='openai',cv_device=None),Settings(cv_provider='fake')):
                pass
