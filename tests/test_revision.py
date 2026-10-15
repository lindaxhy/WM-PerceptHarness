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
    def __init__(self, *, passes_on=1, malformed=False, confidence=False, empty=False, new_issue=False):
        self.calls = []
        self.passes_on, self.malformed = passes_on, malformed
        self.confidence, self.empty, self.new_issue = confidence, empty, new_issue
        self.attempt = 0

    def generate(self, request):
        self.calls.append(request.stage)
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
        if request.stage == 'evidence_review':
            return {'motion_evidence': 'Spoons stationary relative to table.',
                    'concerns': [] if self.empty else [CONCERN], 'unresolved': []}
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
                    'new_issues': ['Unsupported description.'] if self.new_issue and self.attempt == 1 else []}
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
        self.assertEqual(len(outcome.metadata['concern_ledger']), 2)
        self.assertEqual(len(outcome.metadata['revision_attempts'][1]['revision']['responses']), 2)

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
                  'ReviewedRevision': 'grounded_revision', 'RevisionVerification': 'revision_verification'}
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
