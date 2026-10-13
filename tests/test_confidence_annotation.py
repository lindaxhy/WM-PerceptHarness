import unittest
import test_grounded_annotation as fixtures
from test_grounded_annotation import RecordingModel, concern


def draft(confidence=0.9, kind='object_identity', target=True):
    return {'duration': 4.0, 'events': [{'start': 0.0, 'end': 4.0,
        'description': 'A hand moves the red container.', 'confidence': confidence,
        'uncertainty': {k: v for k, v in concern(kind, target).items() if k != 'event_indices'} if confidence <= 0.6 else None}]}


class ConfidenceAnnotationTests(unittest.TestCase):
    def setUp(self):
        fixtures.GroundedAnnotationTests.setUp(self)
        self.settings = self.settings.model_copy(update={'event_workflow': 'confidence-gated'})

    run_model = fixtures.GroundedAnnotationTests.run_model

    def run_draft(self, data, **extra):
        model = RecordingModel(failure_script={'confidence_draft': [data], **extra})
        result = self.run_model(model)
        self.assertEqual(result.status, 'completed', result.error)
        return result, model

    def test_high_confidence_returns_public_draft_without_review_or_sam(self):
        result, model = self.run_draft(draft(0.9))
        self.assertEqual([r.stage for r in model.requests], ['confidence_draft'])
        self.assertFalse(self.cv_requests)
        self.assertEqual(set(result.data['events'][0]), {'start', 'end', 'description'})
        self.assertEqual(result.metadata['draft']['events'][0]['confidence'], 0.9)

    def test_threshold_is_inclusive_and_masks_checked_before_tracking(self):
        result, model = self.run_draft(draft(0.6))
        self.assertEqual([r.stage for r in model.requests], ['confidence_draft', 'sam_prompt_review', 'grounded_revision'])
        self.assertEqual([r.mode for r in self.cv_requests], ['probe', 'track'])
        self.assertTrue(model.requests[1].visual_evidence)
        self.assertEqual(result.metadata['recheck_budget']['sam_calls'], 2)

    def test_high_confidence_question_still_checks_candidate(self):
        data = draft(0.6); data['events'][0]['confidence'] = 0.92
        result, model = self.run_draft(data)
        self.assertEqual([r.mode for r in self.cv_requests], ['probe', 'track'])
        self.assertEqual(len(model.requests), 3)

    def test_low_confidence_editorial_question_is_preserved_without_sam(self):
        result, model = self.run_draft(draft(0.4, 'event_boundary', False))
        self.assertFalse(self.cv_requests)
        self.assertEqual([r.stage for r in model.requests], ['confidence_draft', 'grounded_revision'])
        self.assertTrue(result.metadata['unresolved_concerns'])

    def test_rejected_mask_never_tracks_but_revises(self):
        result, model = self.run_draft(draft(0.4), sam_prompt_review=[
            {'accepted': False, 'reason': 'Wrong object.', 'correction': None}])
        self.assertEqual([r.mode for r in self.cv_requests], ['probe'])
        self.assertEqual([r.stage for r in model.requests], ['confidence_draft', 'sam_prompt_review', 'grounded_revision'])
        self.assertFalse(model.requests[-1].visual_evidence)
        self.assertTrue(result.metadata['unresolved_concerns'])

    def test_empty_draft_stays_successful_and_needs_one_call(self):
        result, model = self.run_draft({'duration': 4.0, 'events': []})
        self.assertEqual(result.data, {'duration': 4.0, 'events': []})
        self.assertEqual(len(model.requests), 1)

    def test_failed_final_keeps_confidence_draft(self):
        model = RecordingModel(failure_script={'confidence_draft': [draft(0.5)],
            'grounded_revision': [RuntimeError('failed final')]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.metadata['draft']['events'][0]['confidence'], 0.5)

    def test_single_corrected_mask_then_track(self):
        target = concern()['target']; target['reference_time'] = 1.5
        result, model = self.run_draft(draft(0.3), sam_prompt_review=[
            {'accepted': False, 'reason': 'Select the other object.', 'correction': target},
            {'accepted': True, 'reason': 'Correct target now.', 'correction': None}])
        self.assertEqual([r.mode for r in self.cv_requests], ['probe', 'probe', 'track'])
        self.assertEqual(len(model.requests), 4)
        self.assertEqual(result.metadata['recheck_budget']['sam_calls'], 3)

    def test_three_target_budget_keeps_omitted_uncertainty(self):
        data = draft(0.4)
        import copy
        data['events'] = [copy.deepcopy(data['events'][0]) for _ in range(4)]
        for i, e in enumerate(data['events']):
            e['uncertainty']['target']['bbox_xyxy'][0] = 0.1 + i * 0.1
        result, model = self.run_draft(data)
        self.assertEqual(len(self.cv_requests), 6)
        self.assertEqual(result.metadata['recheck_budget']['selected_windows'], 3)
        self.assertTrue(any('budget exhausted' in s for s in result.metadata['unresolved_concerns']))

    def test_invalid_confidence_or_target_cannot_enter_pipeline(self):
        from percept_harness.pipelines.event_schemas import ConfidenceDraft, validate_stage
        for value in (-0.1, 1.1, float('nan'), True, '0.5'):
            data = draft(); data['events'][0]['confidence'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_stage(ConfidenceDraft, data, {'duration': 4.0})
        data = draft(0.4); data['events'][0]['uncertainty'] = None
        with self.assertRaises(ValueError): validate_stage(ConfidenceDraft, data, {'duration': 4.0})
        data = draft(0.4); data['events'][0]['end'] = 4.1
        with self.assertRaises(ValueError): validate_stage(ConfidenceDraft, data, {'duration': 4.0})

    def test_draft_response_schema_compiles_for_real_backend(self):
        from percept_harness.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
        contract = DEFAULT_OUTPUT_SCHEMAS.model_response_contract('ConfidenceDraft', {'duration': 4.0})
        self.assertEqual(contract.format()['name'], 'ConfidenceDraft')

    def test_cli_manifest_confidence_draft_only_is_completed(self):
        import json, os, subprocess, sys
        from pathlib import Path
        manifest = self.root / 'input.jsonl'
        manifest.write_text(json.dumps({'job_id': 'one', 'video_path': str(self.video)}) + '\n')
        process = subprocess.run([sys.executable, '-m', 'percept_harness.cli', 'annotate',
            '--manifest', str(manifest), '--event-workflow', 'confidence-gated',
            '--output', str(self.root/'out'), '--backend', 'fake', '--model', 'fake', '--cv', 'sam31',
            '--cv-devices', '0', '--workers-per-gpu', '1', '--request-concurrency', '1'],
            env={**os.environ, 'PYTHONPATH': str(Path('src').resolve()),
                 'PERCEPT_WORK_ROOT': str(self.root/'batch-work'), 'PERCEPT_CV_CACHE_ROOT': str(self.root/'batch-cv')},
            text=True, capture_output=True, timeout=45)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        audit = json.loads((self.root/'out/one/clip.audit.json').read_text())
        self.assertEqual(audit['metadata']['phases'], ['confidence_draft'])
        self.assertEqual(audit['provenance']['run']['settings']['event_workflow'], 'confidence-gated')

    def test_submicrosecond_rounding_canonicalizes_nested_target(self):
        from percept_harness.pipelines.event_schemas import ConfidenceDraft, validate_stage
        data = draft(0.4); endpoint = 4.0000005
        data['duration'] = endpoint
        event = data['events'][0]; event['end'] = endpoint
        event['uncertainty']['end'] = endpoint
        event['uncertainty']['target']['reference_time'] = endpoint
        output = validate_stage(ConfidenceDraft, data, {'duration': 4.0})
        self.assertEqual(output['events'][0]['uncertainty']['end'], 4.0)
        self.assertEqual(output['events'][0]['uncertainty']['target']['reference_time'], 4.0)
        data['events'][0]['uncertainty']['end'] = 4.001
        with self.assertRaises(ValueError): validate_stage(ConfidenceDraft, data, {'duration': 4.0})

    def test_high_confidence_without_target_still_revises(self):
        data = draft(0.4, 'event_boundary', False)
        data['events'][0]['confidence'] = 0.9
        result, model = self.run_draft(data)
        self.assertEqual([r.stage for r in model.requests], ['confidence_draft', 'grounded_revision'])
        self.assertFalse(self.cv_requests)
        self.assertEqual(result.metadata['confidence_gate']['route'], 'revision_only')
        self.assertEqual(result.metadata['confidence_gate']['uncertain_event_indices'], [0])
        from percept_harness.batch import required_event_phases
        self.assertIn('grounded_revision', required_event_phases(result.metadata))

    def test_revision_failure_without_sam_is_not_successful_draft(self):
        data = draft(0.4, 'event_boundary', False)
        data['events'][0]['confidence'] = 0.9
        model = RecordingModel(failure_script={'confidence_draft': [data],
            'grounded_revision': [RuntimeError('failed revision')]})
        result = self.run_model(model)
        self.assertEqual(result.status, 'failed')
        self.assertEqual(result.metadata['draft'], data)
        self.assertFalse(self.cv_requests)

    def test_nested_question_gets_event_index_from_its_parent(self):
        data = draft(0.4)
        data['events'][0]['uncertainty'].pop('event_indices', None)
        result, model = self.run_draft(data)
        self.assertEqual(result.metadata['confidence_gate']['uncertain_event_indices'], [0])
        self.assertEqual([r.mode for r in self.cv_requests], ['probe', 'track'])

    def test_outside_reference_drops_only_target_and_revises_valid_events(self):
        data = draft(0.4)
        data['events'][0]['uncertainty']['target']['reference_time'] = 3.5
        result, model = self.run_draft(data)
        self.assertFalse(self.cv_requests)
        self.assertEqual([r.stage for r in model.requests], ['confidence_draft', 'grounded_revision'])
        self.assertIsNone(result.metadata['draft']['events'][0]['uncertainty']['target'])
        skipped = result.metadata['discarded_sam_targets'][0]
        self.assertEqual(skipped['event_index'], 0)
        self.assertEqual(skipped['target']['reference_time'], 3.5)
        self.assertEqual(skipped['reason'], 'invalid_optional_sam_target')

    def test_bad_target_does_not_suppress_other_valid_request(self):
        import copy
        data = draft(0.4)
        data['events'].append(copy.deepcopy(data['events'][0]))
        data['events'][0]['uncertainty']['target']['bbox_xyxy'] = [0.8, 0.1, 0.2, 0.9]
        result, model = self.run_draft(data)
        self.assertEqual([r.mode for r in self.cv_requests], ['probe', 'track'])
        self.assertEqual(len(result.metadata['discarded_sam_targets']), 1)
        self.assertEqual(result.metadata['sam_checks'][0]['concern_index'], 1)

    def test_invalid_question_is_not_hidden_by_target_fallback(self):
        from percept_harness.pipelines.event_schemas import ConfidenceDraft, validate_stage
        data = draft(0.4)
        data['events'][0]['uncertainty']['end'] = 0.5
        with self.assertRaises(ValueError):
            validate_stage(ConfidenceDraft, data, {'duration': 4.0})
