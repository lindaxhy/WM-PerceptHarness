"""Characterization tests: request/cache identities, CLI exits, inputs and result files."""
import hashlib
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from percept_harness import results
from percept_harness.config import Settings
from percept_harness.media import TimeSpan
from percept_harness.models.base import ModelRequest
from percept_harness.pipelines.event_prompts import prompt
from percept_harness.pipelines import output_validation
from percept_harness.execution import SemanticResultCache, StageExecutor
from percept_harness.runner import collect_videos

DRAFT = {'duration': 10.0, 'events': [{'start': 0.0, 'end': 5.0, 'description': 'A hand lifts a cup.'}]}
REVIEW = {'concerns': [{'concern_id': 'c0000', 'kind': 'event_boundary', 'reason': 'Boundary?',
                        'start': 0.0, 'end': 10.0, 'target': None, 'event_indices': [0]}]}
TARGET = {'start': 1.0, 'end': 4.0, 'question': 'Which cup?'}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class RequestIdentityTests(unittest.TestCase):
    """Prompts and response schemas enter the semantic cache key byte for byte."""

    def test_stage_prompts_are_unchanged(self):
        expected = {
            'confidence_draft': '6fd3a5b9c76e86e1c02beab8fe880045d1ed4d4d09870c07479ebf1434895c32',
            'grounded_draft': 'da7976a5e41bf38ef9fe4aa9e468ca1419ab2261e8c120e2fe8a51809903d51a',
            'evidence_review': 'a70c136a0916b2b010cabbbd15a881fb0d676bb26d8aba96d48983b0159f7457',
            'sam_target_plan': '17d2aedb1934a2ddf2f8f70f8f4c86fd43185e3f3bb237882f5019e6edc9dc9f',
            'sam_prompt_review': 'b24fe7b4ab370471d51c14ab9d7f39ad482c2a4c82479284ea7e93eb08cb4a82',
            'grounded_revision': '5d02bb804764331638dca11afa0294d8ab73d894597b57f77ccc9b3d28e0b7d8',
            'revision_verification': '0cc3f25242804b594c71b68e8339def2a1b309db138a4f25b3e309441cc41bac',
        }
        actual = {stage: sha(prompt(stage, duration=10.0, context='mugs on a table', evidence=[{'scope': 'track_3'}],
                                    draft=DRAFT, review=REVIEW, target=TARGET, unresolved=['u']))
                  for stage in expected}
        self.assertEqual(actual, expected)

    def test_response_contracts_are_unchanged(self):
        expected = {
            'GroundedEvents': '8f5db187ec4dde118e2c11dc6131943c425bceaa7c5bbc8943b94741747f3463',
            'ConfidenceDraft': '932cef2be30e4c1a148969e60eae10925ddf1c0f319dc5f778120bcb6f2f3e16',
            'EvidenceReview': 'efe63e71b4e9c38a7787d8b17bd938c2a7671ac912d9f032cafb140b84cc3f68',
            'SamPromptReview': 'dc590208b937707ee774f1b46a3df7d41b80a9cf2d96e0a86c9b86635b8dbc06',
            'ReviewedRevision': '86d3bbd764a0294dd9e7dab11336b570eb82c009eea0b3be70e334a2640c814a',
            'RevisionVerification': '169a4b8aae251b5f4c1ad1907cad792a7f53d11654e12c69e0cbdcdf98054b53',
            'SamTargetPlan': 'c72e51a6e547a7e650ed92052ffa1bfa38a3230d1eace0bde77b23e83c92cb92',
        }
        actual = {name: sha(output_validation.response_contract(name).schema_json) for name in expected}
        self.assertEqual(actual, expected)
        self.assertIsNone(output_validation.response_contract('Unknown'))


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = self.root / 'video.mp4'
        self.video.write_bytes(b'fake video')
        os.utime(self.video, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))

    def payload(self):
        return {'video_path': str(self.video), 'span': {'start': 0.0, 'end': 10.0}, 'fps': 2.0,
                'schema_name': 'GroundedEvents', 'schema_context': {'duration': 10.0, 'event_count': 0},
                'prompt': 'describe', 'visual_evidence': []}

    def model(self, outputs):
        calls = []
        class Model:
            supports_semantic_result_cache = True
            def semantic_cache_identity(self, request):
                return {'backend': 'fake', 'contract': request.response_contract.cache_identity()['name']}
            def generate(self, request):
                calls.append(request)
                return outputs.pop(0)
        return Model(), calls

    def test_key_identity_layout(self):
        cache = SemanticResultCache(self.root / 'cache')
        request = ModelRequest(stage='grounded_draft', video_path=self.video, span=TimeSpan(0.0, 10.0), fps=2.0,
                               prompt='describe', schema_name='GroundedEvents', model_name='alias')
        identity = {'cache_version': 3, 'video': [str(self.video.resolve()), 10, 1_700_000_000_000_000_000],
                    'span': [0.0, 10.0], 'fps': 2.0, 'stage': 'grounded_draft', 'prompt': 'describe',
                    'schema_name': 'GroundedEvents', 'schema_context': {'a': 1}, 'backend': {'b': 2},
                    'visual_evidence': []}
        expected = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.assertEqual(cache.key(request, {'b': 2}, {'a': 1}), expected)

    def test_valid_output_is_replayed_and_invalid_output_is_not_cached(self):
        raw = json.loads(json.dumps(DRAFT))
        model, calls = self.model([{'duration': 10.0}, raw])
        cache = SemanticResultCache(self.root / 'cache')
        executor = StageExecutor(model, default_model_alias='alias', semantic_cache=cache)
        invalid = executor.execute('grounded_draft', self.payload())
        self.assertIsNotNone(output_validation.failure_codes('GroundedEvents', invalid))
        first = executor.execute('grounded_draft', self.payload())
        second = executor.execute('grounded_draft', self.payload())
        self.assertEqual(len(calls), 2)
        self.assertEqual(first, second)
        self.assertEqual(first, DRAFT)
        entry = next((self.root / 'cache').rglob('*.json'))
        self.assertEqual(json.loads(entry.read_text()), {'format_version': 2, 'result': raw})

    def test_cache_is_disabled_for_models_without_support(self):
        model, calls = self.model([DRAFT, DRAFT])
        type(model).supports_semantic_result_cache = False
        executor = StageExecutor(model, default_model_alias='alias', semantic_cache=SemanticResultCache(self.root / 'c'))
        executor.execute('grounded_draft', self.payload())
        executor.execute('grounded_draft', self.payload())
        self.assertEqual(len(calls), 2)


class InputTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_collect_videos_recurses_filters_and_deduplicates(self):
        (self.root / 'nested').mkdir()
        for name in ('b.mp4', 'nested/a.MOV', 'nested/notes.txt'):
            (self.root / name).write_bytes(b'x')
        videos = collect_videos([self.root, self.root / 'b.mp4'])
        self.assertEqual(videos, [(self.root / 'b.mp4').resolve(), (self.root / 'nested/a.MOV').resolve()])
        with self.assertRaises(FileNotFoundError):
            collect_videos([self.root / 'missing.mp4'])

    def test_manifest_rows_are_validated(self):
        from percept_harness.batch import read_manifest
        video = self.root / 'v.mp4'
        video.write_bytes(b'x')
        manifest = self.root / 'm.jsonl'
        def rows(*items):
            manifest.write_text('\n'.join(json.dumps(item) for item in items) + '\n\n')
            return read_manifest(manifest)
        self.assertEqual(rows({'job_id': 'a/b', 'video_path': str(video)}),
                         [{'job_id': 'a/b', 'video_path': str(video)}])
        for bad, message in (({'video_path': str(video)}, 'requires a job_id'),
                             ({'job_id': '../x', 'video_path': str(video)}, 'unsafe job_id'),
                             ({'job_id': '/x', 'video_path': str(video)}, 'unsafe job_id'),
                             ({'job_id': 'a', 'video_path': 'v.mp4'}, 'missing absolute video path')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                rows(bad)
        with self.assertRaisesRegex(ValueError, 'duplicate job_id at row 2'):
            rows({'job_id': 'a', 'video_path': str(video)}, {'job_id': 'a', 'video_path': str(video)})

    def test_settings_aliases_and_exclusive_model_settings(self):
        self.assertEqual(Settings(openai_model='org/Model v1').openai_model_aliases, {'org-Model-v1': 'org/Model v1'})
        self.assertEqual(Settings(openai_model_registry={'a': 'b'}).openai_model_aliases, {'a': 'b'})
        with self.assertRaises(ValueError):
            Settings(openai_model='m', openai_model_registry={'a': 'b'})


class ResultFileTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def record(self, video, status='completed'):
        return {'video_path': str(video), 'status': status, 'data': DRAFT if status == 'completed' else None,
                'metadata': {}, 'error': None, 'elapsed_seconds': 1.0, 'result_file': None}

    def test_public_file_holds_only_the_annotation(self):
        video = self.root / 'clip.mp4'
        video.write_bytes(b'x')
        path = results.result_path(self.root / 'out', video)
        self.assertEqual(path.name, 'clip.json')
        path.parent.mkdir()
        results.write_annotation(path, self.record(video.resolve()))
        self.assertEqual(json.loads(path.read_text()), DRAFT)
        audit = json.loads(results.audit_path(path).read_text())
        self.assertNotIn('data', audit)
        self.assertEqual(results.annotation_record(path), {**audit, 'data': DRAFT})
        self.assertEqual(results.result_path(self.root / 'out', video), path)

    def test_failure_removes_previous_annotation(self):
        video = (self.root / 'clip.mp4')
        path = self.root / 'clip.json'
        results.write_annotation(path, self.record(video))
        results.write_annotation(path, self.record(video, status='failed'))
        self.assertFalse(path.exists())
        self.assertEqual(results.annotation_record(path)['status'], 'failed')

    def test_name_collisions_get_suffixes(self):
        output = self.root / 'out'
        output.mkdir()
        first, second = self.root / 'a' / 'clip.mp4', self.root / 'b' / 'clip.mp4'
        for video in (first, second):
            video.parent.mkdir()
            video.write_bytes(b'x')
        path = results.result_path(output, first)
        results.write_annotation(path, self.record(first.resolve()))
        self.assertEqual(results.result_path(output, second).name, 'clip--1.json')

    def test_plain_annotation_is_read(self):
        path = self.root / 'human.json'
        results.write_json(path, DRAFT)
        self.assertEqual(results.annotation_record(path), DRAFT)

    def test_single_file_envelope_is_not_an_annotation(self):
        path = self.root / 'old.json'
        results.write_json(path, {'status': 'completed', 'data': DRAFT, 'video_path': '/v.mp4'})
        self.assertIsNone(results.annotation_record(path))
        self.assertFalse(results.is_annotation_record({'data': None}))


class RequestGateTests(unittest.TestCase):
    def test_gate_ramps_after_healthy_calls_and_halves_on_429(self):
        from percept_harness.batch import RequestGate, _ThrottleHandler
        gate = RequestGate(5)
        self.assertEqual(gate.limit, 2)
        for _ in range(4):
            with gate:
                pass
        self.assertEqual(gate.limit, 3)
        logger = logging.getLogger('percept_harness.test_gate')
        logger.addHandler(_ThrottleHandler(gate))
        self.addCleanup(logger.handlers.clear)
        logger.warning('request failed: service status 429')
        logger.warning('request failed: service status 500')
        self.assertEqual(gate.snapshot()['limit'], 1)
        self.assertEqual(gate.snapshot()['rate_limits'], 1)


class CliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.video = self.root / 'clip.mp4'
        self.video.write_bytes(b'x')
        clean = {k: v for k, v in os.environ.items() if not k.startswith('PERCEPT_')}
        environment = patch.dict(os.environ, {**clean, 'PERCEPT_WORK_ROOT': str(self.root / 'work')}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def run_cli(self, *extra):
        from percept_harness.cli import main
        return main(['annotate', '--videos', str(self.video), '--output', str(self.root / 'out'), *extra])

    def test_missing_configuration_exits_2(self):
        with patch('sys.stderr'):
            self.assertEqual(self.run_cli(), 2)
            from percept_harness.cli import main
            self.assertEqual(main(['annotate', '--videos', str(self.root / 'none.mp4'), '--output', str(self.root)]), 2)

    def test_exit_code_reflects_outcomes(self):
        from percept_harness.runner import EvalOutcome
        cases = {'completed': 0, 'needs_review': 1, 'failed': 1}
        for status, code in cases.items():
            outcome = EvalOutcome(video_path=self.video, status=status, data=DRAFT if status != 'failed' else None, metadata={})
            with self.subTest(status=status), \
                 patch('percept_harness.backends.load_model', return_value=(object(), 'alias', lambda: None)), \
                 patch('percept_harness.runner.SyncRunner.evaluate', return_value=outcome), \
                 patch('builtins.print'):
                self.assertEqual(self.run_cli('--force'), code)
