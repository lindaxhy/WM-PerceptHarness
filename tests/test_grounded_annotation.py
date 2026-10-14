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


def concern(kind='object_identity', target=True):
    return {'kind': kind, 'reason': 'Which pen is held by the hand?', 'event_indices': [0],
            'start': 1.0, 'end': 3.0,
            'target': {'description': 'red container', 'reference_time': 2.0,
                       'bbox_xyxy': [0.1, 0.1, 0.8, 0.8]} if target else None}


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
        self.cv_requests = []
        captured = self.cv_requests
        class RecordingProvider(FakeCvEvidenceProvider):
            def analyze(self, request, staging):
                captured.append(request)
                return super().analyze(request, staging)
        self.executor = SyncCvExecutor(RecordingProvider(), self.artifacts)

    def run_model(self, model, executor=True):
        return SyncRunner(model, self.settings, model_alias='fake',
                          cv_executor=self.executor if executor else None).evaluate(
                              self.video)

    def test_clear_video_needs_no_sam_executor_or_visual_evidence(self):
        model = RecordingModel()
        result = self.run_model(model, executor=False)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(set(result.data), {'duration', 'events'})
        self.assertEqual([r.stage for r in model.requests], ['grounded_draft', 'evidence_review', 'grounded_revision'])
        self.assertFalse(result.metadata['evidence'])
        self.assertFalse(self.cv_requests)
        self.assertTrue(all(not r.visual_evidence for r in model.requests))

    def test_target_is_probed_and_visually_checked_before_tracking(self):
        model = RecordingModel(failure_script={'evidence_review': [{'concerns': [concern()], 'unresolved': [], 'motion_evidence': 'Synthetic fixture: no camera uncertainty.'}]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual([r.mode for r in self.cv_requests], ['probe', 'track'])
        self.assertEqual(len(self.cv_requests[0].timeline.frames), 1)
        self.assertEqual(self.cv_requests[0].spatial_prompts, self.cv_requests[1].spatial_prompts)
        self.assertEqual([r.stage for r in model.requests], ['grounded_draft', 'evidence_review', 'sam_prompt_review', 'grounded_revision'])
        check = model.requests[2]
        self.assertTrue(check.visual_evidence)
        self.assertTrue(all(e.png_bytes.startswith(b'\x89PNG') for e in check.visual_evidence))
        self.assertEqual(len(self.cv_requests[1].entities), 1)
        self.assertEqual(result.metadata['recheck_budget']['completed_windows'], 1)

    def test_camera_and_editorial_concerns_never_trigger_sam(self):
        review = {'concerns': [concern('camera_motion', False), concern('event_boundary', False)], 'unresolved': [], 'motion_evidence': 'Synthetic fixture: no camera uncertainty.'}
        model = RecordingModel(failure_script={'evidence_review': [review]})
        result = self.run_model(model, executor=False)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertFalse(self.cv_requests)
        self.assertIn('Which pen', model.requests[-1].prompt)

    def test_rejected_segmentation_has_bounded_retry_and_never_propagates(self):
        target = concern()['target']
        target['bbox_xyxy'] = [0.2, 0.2, 0.9, 0.9]
        rejection = {'accepted': False, 'reason': 'Mask follows the wrong object.', 'correction': target}
        model = RecordingModel(failure_script={
            'evidence_review': [{'concerns': [concern()], 'unresolved': [], 'motion_evidence': 'Synthetic fixture: no camera uncertainty.'}],
            'sam_prompt_review': [rejection, rejection]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual([r.mode for r in self.cv_requests], ['probe', 'probe'])
        self.assertEqual(result.metadata['recheck_budget']['completed_windows'], 0)
        self.assertTrue(result.metadata['unresolved_concerns'])
        self.assertFalse(model.requests[-1].visual_evidence, 'rejected masks must not support final event claims')

    def test_missing_requested_sam_is_audited_and_revision_remains_possible(self):
        model = RecordingModel(failure_script={'evidence_review': [{'concerns': [concern()], 'unresolved': [], 'motion_evidence': 'Synthetic fixture: no camera uncertainty.'}]})
        result = self.run_model(model, executor=False)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertTrue(any('unavailable' in x for x in result.metadata['unresolved_concerns']))
        self.assertEqual(result.metadata['recheck_budget']['completed_windows'], 0)

    def test_empty_annotation_still_gets_visual_review_without_scene_probe(self):
        empty = {'duration': 4.0, 'events': []}
        model = RecordingModel(failure_script={'grounded_draft': [empty], 'grounded_revision': [empty]})
        result = self.run_model(model, executor=False)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(result.data, empty)
        self.assertFalse(self.cv_requests)
        self.assertIn('evidence_review', [r.stage for r in model.requests])

    def test_audit_preserves_draft_claims_when_revision_changes_events(self):
        final = {'duration': 4.0, 'events': []}
        model = RecordingModel(failure_script={'grounded_revision': [final]})
        result = self.run_model(model, executor=False)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(result.data, final)
        self.assertEqual(result.metadata['draft']['events'][0]['description'], 'A hand moves the red container.')

    def test_failed_final_revision_retains_review_and_sam_audit(self):
        model = RecordingModel(failure_script={
            'evidence_review': [{'concerns': [concern()], 'unresolved': [], 'motion_evidence': 'Synthetic fixture: no camera uncertainty.'}],
            'grounded_revision': [RuntimeError('fixture final request failed')]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'failed')
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.metadata)
        self.assertEqual(len(result.metadata['evidence']), 2)
        self.assertTrue(result.metadata['sam_checks'][0]['tracked'])
        self.assertEqual(result.metadata['review']['concerns'][0]['kind'], 'object_identity')
        self.assertNotIn('grounded_revision', result.metadata['phases'])

    def test_zero_detection_does_not_trigger_tracking_or_claim_absence(self):
        class ZeroDetection(FakeCvEvidenceProvider):
            def analyze(self, request, staging):
                from percept_harness.cv.contracts import CvEvidenceArtifact
                return CvEvidenceArtifact(schema_version='cv_evidence_v1', status='available',
                    provider=request.provider, model_identity=request.model_identity,
                    video_identity=request.video_identity, checkpoint_identity=request.checkpoint_identity,
                    processed_timeline=request.timeline, entities=request.entities,
                    tracks=(), files=(), overlay_records=())
        self.executor = SyncCvExecutor(ZeroDetection(), self.artifacts)
        model = RecordingModel(failure_script={'evidence_review': [{'concerns': [concern()], 'unresolved': [], 'motion_evidence': 'Synthetic fixture: no camera uncertainty.'}]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        self.assertEqual(result.metadata['recheck_budget']['sam_calls'], 1)
        self.assertEqual(result.metadata['recheck_budget']['completed_windows'], 0)
        self.assertTrue(result.metadata['evidence'][0]['no_tracks'])
        self.assertFalse(model.requests[-1].visual_evidence)

    def test_target_budget_limits_work_and_reports_unchecked_questions(self):
        from percept_harness.pipelines.event_schemas import EvidenceReview
        from percept_harness.pipelines.event_evidence import targeted_windows
        questions = []
        for start in (0.0, 12.0, 24.0, 36.0):
            c = concern()
            c.update(start=start, end=start+10.0)
            c['target']['reference_time'] = start+1.0
            questions.append(c)
        windows, omitted = targeted_windows(EvidenceReview.model_validate({'concerns': questions, 'unresolved': [], 'motion_evidence': 'Synthetic fixture: no camera uncertainty.'}), 50.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(len(omitted), 2)

    def test_response_contract_preserves_nullable_target_and_numeric_box_items(self):
        from percept_harness.pipelines.event_schemas import EvidenceReview
        from percept_harness.models.response_contract import compile_local_schema
        schema = compile_local_schema(EvidenceReview.model_json_schema())
        target = schema['properties']['concerns']['items']['properties']['target']['anyOf']
        self.assertEqual(target[1], {'type': 'null'})
        box = target[0]['properties']['bbox_xyxy']
        self.assertEqual(box['items'], {'type': 'number', 'minimum': 0.0, 'maximum': 1.0})
        self.assertEqual((box['minItems'], box['maxItems']), (4, 4))

    def test_submicrosecond_model_rounding_preserves_authoritative_clock(self):
        from percept_harness.pipelines.event_schemas import validate_stage, GroundedEvents
        actual = 9.840233333333334
        result = validate_stage(GroundedEvents, {'duration': 9.840233, 'events': [
            {'start': 0.0, 'end': 9.840234, 'description': 'Hand moves the magnet.'}]}, {'duration': actual})
        self.assertEqual(result['duration'], actual)
        self.assertEqual(result['events'][0]['end'], actual)
        with self.assertRaises(ValueError):
            validate_stage(GroundedEvents, {'duration': 9.84, 'events': []}, {'duration': actual})

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


if __name__ == '__main__':
    unittest.main()
