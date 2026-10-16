import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from percept_harness import batch, results
from percept_harness.config import Settings
from percept_harness.runner import SyncRunner, run_batch
from percept_harness.pipelines.event_schemas import ReviewedRevision, RevisionVerification, validate_stage


DRAFT = {'duration': 10.0, 'events': [
    {'start': 0.0, 'end': 5.0, 'description': 'Camera moves over stationary spoons.'},
    {'start': 5.0, 'end': 10.0, 'description': 'Camera continues moving over stationary spoons.'}]}
CONCERN = {'kind': 'event_boundary', 'reason': 'No visible transition at 5 seconds.',
           'start': 0.0, 'end': 10.0, 'target': None, 'event_indices': [0, 1]}


class ScriptedModel:
    def __init__(self, *, passes_on=1, malformed=False, confidence=False, empty=False, new_issue=False, target=False, kind="event_boundary"):
        self.calls = []
        self.requests = []
        self.passes_on, self.malformed = passes_on, malformed
        self.confidence, self.empty, self.new_issue = confidence, empty, new_issue
        self.attempt = 0
        self.target, self.kind = target, kind

    def generate(self, request):
        self.calls.append(request.stage)
        self.requests.append(request)
        # Also exercise schema compilation and the transport allowlist.
        self_contract = request.response_contract.format()
        assert self_contract['name'] == request.schema_name
        payload = json.loads(request.prompt.split('INPUT_DATA_JSON\n')[1])
        if request.stage == 'grounded_draft':
            return DRAFT
        if request.stage == 'confidence_draft':
            uncertainty = {k: v for k, v in CONCERN.items() if k != 'event_indices'}
            return {'duration': 10.0, 'events': [{**DRAFT['events'][0], 'confidence': 0.9,
                     'uncertainty': None if self.empty else uncertainty}]}
        if request.stage == 'sam_target_plan':
            return {'target': {'description': 'spoon', 'reference_time': 5.0, 'bbox_xyxy': [0.2,0.2,0.6,0.6]} if self.target else None, 'reason': 'Visible relevant object.' if self.target else 'No reliable relevant visible reference in this mocked clip.'}
        if request.stage == 'evidence_review':
            return {'motion_evidence': 'Spoons stationary relative to table.',
                    'concerns': [] if self.empty else [{**CONCERN, 'kind': self.kind}], 'unresolved': []}
        if request.stage == 'sam_prompt_review':
            return {'accepted': True, 'reason': 'Correct target.', 'correction': None}
        ledger = payload['review']['concerns']
        if request.stage == 'grounded_revision':
            self.attempt += 1
            events = DRAFT['events'] if self.attempt < self.passes_on else [
                {'start': 0.0, 'end': 10.0, 'description': 'Camera moves over stationary spoons.'}]
            return {'duration': 10.0, 'events': events, 'responses': [] if self.malformed else [
                {'concern_id': c['concern_id'], 'action': 'changed',
                 'rationale': 'Merged arbitrary segments.', 'visual_times': [5.0]} for c in ledger]}
        if request.stage == 'revision_verification':
            assert 'responses' not in payload['review']
            assert 'previous_verification' not in payload['review']
            return {'verdicts': [{'concern_id': c['concern_id'], 'resolved': self.attempt >= self.passes_on,
                                 'reason': 'Boundary checked against frames.'} for c in ledger],
                    'new_issues': [{**CONCERN, 'kind': 'description', 'reason': 'Unsupported description.', 'event_indices': [0]}] if self.new_issue and self.attempt == 1 else []}
        raise AssertionError(request.stage)


class FakeSlot:
    def __init__(self, context, settings, gpu):
        self.gpu, self.ready = gpu, False
        self.process = SimpleNamespace(pid=0, is_alive=lambda: True)
    def wait_ready(self): self.ready = True
    def close(self): pass


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / 'video.mp4'
        self.video.write_bytes(b'fake video for stubbed model')
        self.probe = patch('percept_harness.pipelines.embodied.probe_video', return_value=SimpleNamespace(duration=10.0))
        self.probe.start()
        self.addCleanup(self.probe.stop)
        self.settings = Settings(semantic_result_cache=False, work_root=self.root/'work')

    def run_model(self, model):
        settings = self.settings.model_copy(update={'event_workflow': 'confidence-gated' if model.confidence else 'reviewed'})
        runner = SyncRunner(model, settings, model_alias='test')
        return run_batch(runner, [self.video], self.root/'out', log=lambda _: None)[0]

    def test_verified_revision_passes_and_can_resume(self):
        model = ScriptedModel()
        outcome = self.run_model(model)
        self.assertEqual(outcome.status, 'completed')
        self.assertEqual(len(outcome.data['events']), 1)
        self.assertEqual(outcome.metadata['unresolved_concerns'], [])
        self.assertEqual(self.run_model(ScriptedModel()).status, 'skipped')

    def assessed_model(self, category='visual_uncertainty', verifier_category=None, stale_quote=False):
        class AssessedModel(ScriptedModel):
            def generate(self, request):
                data = super().generate(request)
                description = 'A blue region becomes distorted; whether the same object remains is unclear.'
                assessment = {'category': category, 'start': 4.0, 'end': 8.0,
                    'reason': 'Original frames show merging blue regions with no stable object contour.',
                    'visual_times': [4.5, 6.0, 7.5], 'event_index': 0, 'candidate_quote': description}
                if request.stage == 'grounded_revision':
                    data['events'][0]['description'] = description
                    for response in data['responses']:
                        response['assessment'] = assessment
                if request.stage == 'revision_verification':
                    payload = json.loads(request.prompt.split('INPUT_DATA_JSON\n')[1])
                    assert 'original_draft' not in payload['review']
                    for verdict in data['verdicts']:
                        verdict['assessment'] = {**assessment,
                            'category': verifier_category or category,
                            'candidate_quote': 'Old removed claim.' if stale_quote else description}
                return data
        return AssessedModel()

    def test_independently_verified_uncertainty_is_completed_and_persisted(self):
        outcome = self.run_model(self.assessed_model())
        self.assertEqual(outcome.status, 'completed', outcome.error)
        self.assertEqual(outcome.metadata['annotation_outcome'], 'completed_with_uncertainty')
        self.assertEqual(outcome.metadata['unresolved_concerns'], [])
        self.assertEqual(outcome.metadata['annotation_assessments'][0]['category'], 'visual_uncertainty')
        self.assertIn('unclear', outcome.data['events'][0]['description'])
        record = results.completed_result(self.root/'out'/outcome.result_file, self.video)
        self.assertEqual(record['metadata']['annotation_outcome'], 'completed_with_uncertainty')

    def test_visible_defect_is_a_valid_observation(self):
        outcome = self.run_model(self.assessed_model('observable_defect'))
        self.assertEqual(outcome.status, 'completed', outcome.error)
        self.assertEqual(outcome.metadata['annotation_outcome'], 'completed_with_observable_defects')

    def test_pipeline_error_cannot_be_waived_even_if_verifier_says_resolved(self):
        outcome = self.run_model(self.assessed_model('pipeline_error'))
        self.assertEqual(outcome.status, 'needs_review')
        self.assertEqual(outcome.metadata['annotation_assessments'], [])

    def test_taxonomy_difference_does_not_reject_verified_description(self):
        outcome = self.run_model(self.assessed_model(verifier_category='observable_defect'))
        self.assertEqual(outcome.status, 'completed')
        self.assertEqual(outcome.metadata['annotation_outcome'], 'completed_with_uncertainty')
        self.assertTrue(outcome.metadata['assessment_disagreements'])
        self.assertEqual(outcome.metadata['annotation_assessments'][0]['categories'], ['observable_defect', 'visual_uncertainty'])

    def test_stale_candidate_quote_fails_closed(self):
        outcome = self.run_model(self.assessed_model(stale_quote=True))
        self.assertEqual(outcome.status, 'failed')

    def test_assessment_time_and_candidate_references_are_validated(self):
        assessment = {'category': 'visual_uncertainty', 'start': 1.0, 'end': 4.0,
            'reason': 'Object hidden.', 'visual_times': [2.0], 'event_index': 0,
            'candidate_quote': DRAFT['events'][0]['description']}
        context = {'duration': 10.0, 'concern_ids': ['c0000'], 'candidate_events': DRAFT['events']}
        for update in ({'end': 11.0}, {'visual_times': [11.0]}, {'event_index': 5},
                       {'candidate_quote': 'deleted claim'}, {'start': 6.0, 'end': 8.0, 'visual_times': [7.0]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_stage(RevisionVerification, {'verdicts': [{'concern_id': 'c0000',
                    'resolved': True, 'reason': 'Checked.', 'assessment': {**assessment, **update}}],
                    'new_issues': []}, context)
        # Before/after frames can establish identity or contrast while the
        # uncertainty itself remains confined to a different interval.
        validate_stage(RevisionVerification, {'verdicts': [{'concern_id': 'c0000',
            'resolved': True, 'reason': 'Compared to clear before/after frames.',
            'assessment': {**assessment, 'visual_times': [0.0, 5.0]}}], 'new_issues': []}, context)

    def test_missing_required_verifier_assessment_is_not_silent_disagreement(self):
        context = {'duration': 10.0, 'concern_ids': ['c0000'], 'candidate_events': DRAFT['events'],
                   'assessment_required_ids': ['c0000']}
        with self.assertRaisesRegex(ValueError, 'assessment required'):
            validate_stage(RevisionVerification, {'verdicts': [{'concern_id': 'c0000',
                'resolved': True, 'reason': 'The uncertainty is responsibly preserved.', 'assessment': None}],
                'new_issues': []}, context)

    def test_prior_questions_survive_an_empty_fresh_review(self):
        prior = self.root/'prior.audit.json'
        prior.write_text(json.dumps({'video_path': str(self.video.resolve()), 'status': 'needs_review',
            'metadata': {'concern_ledger': [{'concern_id': 'c0000', **CONCERN, 'event_indices': [999]}],
                         'unresolved_concerns': ['c0000: No visible change at this boundary.']}}))
        model = ScriptedModel(empty=True)
        runner = SyncRunner(model, self.settings, model_alias='test')
        outcome = run_batch(runner, [self.video], self.root/'seeded', review_from=prior, log=lambda _:None)[0]
        self.assertEqual(outcome.status, 'completed', outcome.error)
        retained = outcome.metadata['retained_prior_concerns']
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]['event_indices'], [])
        self.assertIn('No visible change', retained[0]['reason'])
        self.assertIn('sam_target_plan', model.calls)
        self.assertTrue(outcome.metadata['revision_attempts'][0]['revision']['responses'])

    def test_prior_review_from_another_video_fails_closed(self):
        prior = self.root/'wrong.audit.json'
        prior.write_text(json.dumps({'video_path': '/different.mp4', 'status': 'needs_review'}))
        outcome = SyncRunner(ScriptedModel(), self.settings, model_alias='test').evaluate(self.video, review_from=prior)
        self.assertEqual(outcome.status, 'failed')

    def test_verifier_can_explicitly_disagree_that_source_limitation_exists(self):
        model = self.assessed_model(verifier_category='no_source_limitation')
        generate = model.generate
        def reject(request):
            data = generate(request)
            if request.stage == 'revision_verification':
                for verdict in data['verdicts']:
                    verdict.update(resolved=False, reason='Original frames clearly establish identity; the candidate uncertainty is unsupported.')
            return data
        model.generate = reject
        outcome = self.run_model(model)
        self.assertEqual(outcome.status, 'needs_review')

    def test_self_reported_fix_does_not_override_failed_verifier(self):
        model = ScriptedModel(passes_on=3)
        outcome = self.run_model(model)
        self.assertEqual(outcome.status, 'needs_review')
        self.assertEqual(model.calls.count('grounded_revision'), 2)
        self.assertEqual(model.calls.count('revision_verification'), 2)
        self.assertTrue(outcome.metadata['unresolved_concerns'])
        path = self.root/'out'/outcome.result_file
        self.assertTrue(path.exists())
        self.assertEqual(results.annotation_record(path)['status'], 'needs_review')
        self.assertIsNone(results.completed_result(path, self.video))

    def test_second_revision_can_resolve_concern(self):
        outcome = self.run_model(ScriptedModel(passes_on=2))
        self.assertEqual(outcome.status, 'completed')
        self.assertEqual(outcome.metadata['revision_validation']['attempts'], 2)

    def test_new_verifier_issue_is_added_to_next_ledger(self):
        outcome = self.run_model(ScriptedModel(new_issue=True))
        self.assertEqual(outcome.status, 'completed')
        self.assertIn('Unsupported description.', [c['reason'] for c in outcome.metadata['concern_ledger']])
        self.assertEqual(len(outcome.metadata['revision_attempts'][1]['revision']['responses']), len(outcome.metadata['concern_ledger']))

    def test_missing_response_fails_closed(self):
        self.assertEqual(self.run_model(ScriptedModel(malformed=True)).status, 'failed')

    def test_confidence_paths(self):
        outcome = self.run_model(ScriptedModel(confidence=True, empty=True))
        self.assertEqual(outcome.status, 'completed')
        self.assertEqual(outcome.metadata['revision_validation']['status'], 'not_required')
        # Separate output so the completed draft-only result cannot be reused.
        self.root = self.root/'other'; self.root.mkdir()
        outcome = self.run_model(ScriptedModel(confidence=True, passes_on=3))
        self.assertEqual(outcome.status, 'needs_review')

    def test_old_result_does_not_bypass_new_validation(self):
        path = self.root/'old.json'
        results.write_annotation(path, {'status': 'completed', 'video_path': str(self.video),
            'data': DRAFT, 'metadata': {'annotation_version': 'grounded-events-v3'}})
        self.assertIsNone(results.completed_result(path, self.video))

    def test_duplicate_unknown_and_out_of_range_responses_rejected(self):
        response = {'concern_id': 'c0000', 'action': 'retained', 'rationale': 'Visible change.', 'visual_times': [5.0]}
        context = {'duration': 10.0, 'concern_ids': ['c0000']}
        for responses in ([response, response], [{**response, 'concern_id': 'unknown'}],
                          [{**response, 'visual_times': [11.0]}]):
            with self.subTest(responses=responses), self.assertRaises(ValueError):
                validate_stage(ReviewedRevision, {**DRAFT, 'responses': responses}, context)
        with self.assertRaises(ValueError):
            validate_stage(RevisionVerification, {'verdicts': [], 'new_issues': []}, context)

    def test_openai_transport_and_cache_support_verification_stage(self):
        import httpx
        from percept_harness.media import FrameRef
        from percept_harness.models.openai_compat import OpenAICompatVideoModel
        scripted = ScriptedModel()
        stages = {'GroundedEvents': 'grounded_draft', 'EvidenceReview': 'evidence_review',
                  'ReviewedRevision': 'grounded_revision', 'RevisionVerification': 'revision_verification', 'SamTargetPlan': 'sam_target_plan'}
        def handle(request):
            body = json.loads(request.content)
            name = body['response_format']['json_schema']['name']
            self.assertGreater(body['max_tokens'], 0)
            fake = SimpleNamespace(stage=stages[name], schema_name=name,
                prompt=body['messages'][0]['content'][0]['text'],
                response_contract=SimpleNamespace(format=lambda: {'name': name}))
            data = scripted.generate(fake)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(data)}}]})
        def extract(video, span, fps, output):
            path = output/'frame.jpg'; path.write_bytes(b'fake image for mocked HTTP transport')
            return [FrameRef(path, 0.0)]
        model = OpenAICompatVideoModel(base_url='https://example.invalid/v1', api_key='test',
            model_name='test', response_format='json_schema', transport=httpx.MockTransport(handle),
            frame_extractor=extract)
        self.addCleanup(model.close)
        self.settings = self.settings.model_copy(update={'semantic_result_cache': True})
        runner = SyncRunner(model, self.settings, model_alias='test')
        outcome = run_batch(runner, [self.video], self.root/'transport', log=lambda _: None)[0]
        self.assertEqual(outcome.status, 'completed', outcome.error)
        self.assertIn('revision_verification', scripted.calls)
        self.assertIn('sam_target_plan', scripted.calls)
        self.assertFalse(any('error' in c for c in outcome.metadata['sam_checks']))

    def fake_orchestrator(self, empty=False):
        from percept_harness.models.base import VisualEvidence
        calls = []
        class Orchestrator:
            def __init__(self, *args): pass
            def run(self, *, ordinal, window, target, mode):
                calls.append((ordinal, mode, window['start'], window['end']))
                scope = f'{mode}_{ordinal}'
                return SimpleNamespace(audit={'scope': scope, 'reference_time': 5.0},
                    prompt_record={'scope': scope}, images=() if empty else (VisualEvidence(
                        png_bytes=b'\x89PNG\r\n\x1a\nmock', timestamp=5.0, entity_id='target',
                        track_id=scope, source=scope),))
        return Orchestrator, calls

    def test_all_concern_kinds_enter_local_planning_and_sam(self):
        original_root = self.root
        for kind in ('event_boundary', 'description', 'missing_event', 'camera_motion',
                     'object_identity', 'object_location', 'occlusion', 'trajectory'):
            with self.subTest(kind=kind):
                self.root = original_root/kind; self.root.mkdir()
                provider,calls = self.fake_orchestrator()
                with patch('percept_harness.pipelines.embodied.EvidenceOrchestrator', provider):
                    outcome = self.run_model(ScriptedModel(target=True, kind=kind))
                self.assertEqual(outcome.status,'completed',outcome.error)
                self.assertEqual([c[1] for c in calls],['probe','track'])
                self.assertEqual(outcome.metadata['recheck_budget']['completed_windows'],1)

    def test_planner_and_sam_use_only_the_question_window(self):
        provider,calls = self.fake_orchestrator()
        model=ScriptedModel(target=True)
        with patch.dict(CONCERN, start=4.0, end=6.0), \
             patch('percept_harness.pipelines.embodied.EvidenceOrchestrator',provider):
            outcome=self.run_model(model)
        self.assertEqual(outcome.status,'completed',outcome.error)
        request=next(r for r in model.requests if r.stage=='sam_target_plan')
        self.assertEqual((request.span.start,request.span.end),(3.5,6.5))
        self.assertEqual(request.visual_evidence,())
        self.assertTrue(all((c[2],c[3])==(3.5,6.5) for c in calls))

    def test_external_reference_reaches_planner_probe_and_track(self):
        original_root = self.root
        for reference, expected in [(1.0, (0.5, 8.5)), (9.0, (5.5, 9.5))]:
            self.root = original_root / str(reference); self.root.mkdir()
            target = dict(description='spoon', reference_time=reference, bbox_xyxy=[0.2,0.2,0.6,0.6])
            class Model(ScriptedModel):
                def generate(self, request):
                    data = super().generate(request)
                    if request.stage == 'sam_target_plan': data['target'] = target
                    return data
            model = Model(target=True)
            provider, calls = self.fake_orchestrator()
            with patch.dict(CONCERN, start=6.0, end=8.0, target=target), \
                 patch('percept_harness.pipelines.embodied.EvidenceOrchestrator', provider):
                outcome = self.run_model(model)
            self.assertEqual(outcome.status, 'completed', outcome.error)
            request = next(r for r in model.requests if r.stage == 'sam_target_plan')
            self.assertEqual((request.span.start, request.span.end), expected)
            self.assertEqual([c[1] for c in calls], ['probe', 'track'])
            self.assertTrue(all((c[2],c[3]) == expected for c in calls))
            self.assertEqual(outcome.metadata['sam_checks'][0]['concern_window'], [6.,8.])

    def test_invalid_review_retries_only_failed_stage_with_field_feedback(self):
        class Model(ScriptedModel):
            def generate(self, request):
                data = super().generate(request)
                if request.stage == 'evidence_review' and self.calls.count('evidence_review') == 1:
                    data['motion_evidence'] = 'x' * 801
                return data
        model = Model()
        outcome = self.run_model(model)
        self.assertEqual(outcome.status, 'completed', outcome.error)
        self.assertEqual(model.calls.count('grounded_draft'), 1)
        self.assertEqual(model.calls.count('evidence_review'), 2)
        records = outcome.metadata['model_output_failures']
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['recovered'])
        detail = records[0]['failure']['details'][0]
        self.assertEqual(detail['location'], ['motion_evidence'])
        self.assertNotIn('input', detail)
        retry = [r for r in model.requests if r.stage == 'evidence_review'][1]
        self.assertIn('motion_evidence', retry.prompt.split('INPUT_DATA_JSON')[0])
        self.assertEqual(retry.response_contract, model.requests[1].response_contract)

    def test_persistent_invalid_output_stops_after_three_attempts(self):
        model = ScriptedModel(malformed=True)
        outcome = self.run_model(model)
        self.assertEqual(outcome.status, 'failed')
        self.assertEqual(model.calls.count('grounded_revision'), 3)
        self.assertEqual(model.calls.count('grounded_draft'), 1)
        self.assertEqual(len(outcome.metadata['model_output_failures']), 3)
        self.assertFalse(any(r['recovered'] for r in outcome.metadata['model_output_failures']))

    def test_new_verifier_question_gets_sam_before_second_revision(self):
        provider,calls = self.fake_orchestrator()
        model=ScriptedModel(target=True,new_issue=True)
        with patch('percept_harness.pipelines.embodied.EvidenceOrchestrator', provider):
            outcome=self.run_model(model)
        self.assertEqual(outcome.status,'completed',outcome.error)
        self.assertEqual([c[0] for c in calls],[1,3,4,6])
        self.assertEqual(outcome.metadata['sam_checks'][1]['source'],'revision_verification')
        self.assertEqual(model.calls.count('sam_target_plan'),2)

    def test_empty_sam_mask_is_rejected_not_treated_as_disappearance(self):
        provider,calls=self.fake_orchestrator(empty=True)
        with patch('percept_harness.pipelines.embodied.EvidenceOrchestrator',provider):
            outcome=self.run_model(ScriptedModel(target=True))
        self.assertEqual([c[1] for c in calls],['probe'])
        self.assertFalse(outcome.metadata['sam_checks'][0]['tracked'])
        self.assertFalse(outcome.metadata['sam_checks'][0]['attempts'][0]['accepted'])

    def test_sam_budget_is_shared_across_review_and_verification(self):
        provider,calls=self.fake_orchestrator()
        with patch('percept_harness.pipelines.embodied.EvidenceOrchestrator',provider), \
             patch('percept_harness.pipelines.embodied.MAX_RECHECKS',1):
            outcome=self.run_model(ScriptedModel(target=True,new_issue=True))
        self.assertEqual(outcome.metadata['recheck_budget']['selected_windows'],1)
        self.assertEqual(len(calls),2)

    def test_manifest_reports_needs_review_and_nonzero_exit(self):
        manifest=self.root/'manifest.jsonl'
        manifest.write_text(json.dumps({'job_id':'one','video_path':str(self.video)})+'\n')
        args=SimpleNamespace(workers_per_gpu=1, request_concurrency=1, shard_index=0, shard_count=1,
            cv='sam31', cv_devices='0', event_workflow='reviewed',manifest=manifest,output=self.root/'batch',force=False)
        with patch.object(batch, 'Settings',return_value=self.settings) as configured, patch.object(batch,'GpuSlot',FakeSlot), \
             patch('percept_harness.cli._load_backend',return_value=(ScriptedModel(passes_on=3),'test',lambda:None)):
            configured.model_validate.side_effect = Settings.model_validate
            code=batch.run_manifest(args)
        self.assertEqual(code,1)
        state=json.loads((args.output/'shard-00.status.json').read_text())
        self.assertEqual(state['counts']['needs_review'],1)
        self.assertEqual(state['jobs']['one']['status'],'needs_review')


if __name__ == '__main__': unittest.main()
