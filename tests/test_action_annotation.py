import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from percept_harness.cli import _parser
from percept_harness.config import Settings
from percept_harness.models.fake import FakeVideoModel
from percept_harness.runner import SyncRunner, run_batch
from percept_harness.cv.base import FakeCvEvidenceProvider
from percept_harness.cv.executor import SyncCvExecutor
from percept_harness.cv.artifacts import CvArtifactStore


class ActionAnnotationTests(unittest.TestCase):
    def test_annotation_defaults_to_action_pipeline(self):
        args = _parser().parse_args(['annotate', '--videos', 'clip.mp4', '--output', 'out'])
        self.assertFalse(hasattr(args, 'template'))

    def test_resume_flag_is_removed(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            _parser().parse_args(['annotate', '--videos', 'clip.mp4', '--output', 'out', '--resume'])
        self.assertEqual(error.exception.code, 2)

    def test_retired_templates_rejected_by_cli(self):
        for template in ('embodied_action_captioning', 'general_video_captioning', 'embodied_active_object_detection'):
            with self.subTest(template=template), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    _parser().parse_args(['annotate', '--videos', 'clip.mp4', '--output', 'out', '--template', template])
                self.assertEqual(error.exception.code, 2)

    def test_action_annotation_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video = root / 'clip.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=c=red:s=64x64:r=4:d=2', '-c:v', 'mpeg4', str(video)], check=True)
            settings = Settings(work_root=root / 'work', cv_provider='fake', cv_cache_root=root / 'cv')
            artifacts = CvArtifactStore(settings.cv_cache_root)
            self.addCleanup(artifacts.close)
            runner = SyncRunner(FakeVideoModel(), settings, model_alias='fake',
                                cv_executor=SyncCvExecutor(FakeCvEvidenceProvider(), artifacts))
            first = list(run_batch(runner, [video], root / 'out', log=lambda message: None))
            self.assertEqual(first[0].status, 'completed', first[0].error)
            data = first[0].data
            self.assertEqual(set(data), {'duration', 'events'})
            self.assertTrue(data['events'])
            self.assertEqual(data['events'][0]['description'], 'A hand moves the red container.')
            second = list(run_batch(runner, [video], root / 'out', log=lambda message: None))
            self.assertEqual(second[0].status, 'skipped')
            self.assertFalse(second[0].metadata['evidence'])
            stored = json.loads((root / 'out' / 'clip.json').read_text())
            self.assertEqual(stored['events'], data['events'])
            audit = json.loads((root / 'out' / 'clip.audit.json').read_text())
            self.assertTrue(audit['metadata']['synthetic'])
            self.assertNotIn('template', audit)
            self.assertNotIn('provenance', audit)
            aggregate = json.loads((root / 'out' / 'results.jsonl').read_text().splitlines()[0])
            self.assertNotIn('template', aggregate)


if __name__ == '__main__':
    unittest.main()


def test_direct_stage_execution_reuses_cache_and_retries_invalid_output(tmp_path):
    from percept_harness.models.base import ModelOutputError
    from percept_harness.models.fake import FakeVideoModel
    from percept_harness.runner import StageExecutor, SemanticResultCache
    from percept_harness.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    class CachedModel(FakeVideoModel):
        supports_semantic_result_cache = True

        def __init__(self):
            super().__init__(failure_script={'grounded_draft': [ModelOutputError('invalid fixture')]})
            self.calls = 0

        def semantic_cache_identity(self, request):
            return {'fixture': 'direct-stage-v1'}

        def generate(self, request):
            self.calls += 1
            return super().generate(request)

    video = tmp_path / 'video.mp4'
    video.write_bytes(b'cache identity fixture')
    payload = {'video_path': str(video), 'span': {'start': 0.0, 'end': 2.0},
               'schema_name': 'GroundedEvents', 'schema_context': {'duration': 2.0},
               'fps': 2.0, 'prompt': 'describe'}
    model = CachedModel()
    executor = StageExecutor(model, default_model_alias='fake',
                             semantic_cache=SemanticResultCache(tmp_path / 'cache'))
    invalid = executor.execute('grounded_draft', payload)
    assert DEFAULT_OUTPUT_SCHEMAS.failure_codes('GroundedEvents', invalid)
    valid = executor.execute('grounded_draft', payload)
    assert valid['events']
    assert executor.execute('grounded_draft', payload) == valid
    assert model.calls == 2
    for entry in (tmp_path / 'cache').rglob('*.json'):
        entry.write_text('{broken')
    assert executor.execute('grounded_draft', payload) == valid
    assert model.calls == 3
    for entry in (tmp_path / 'cache').rglob('*.json'):
        record = json.loads(entry.read_text())
        record['result'] = {'duration': 2.0, 'events': [{'start': 0.0, 'end': 3.0, 'description': 'invalid'}]}
        entry.write_text(json.dumps(record))
    assert executor.execute('grounded_draft', payload) == valid
    assert model.calls == 4
    executor.execute('grounded_draft', {**payload, 'prompt': 'different input'})
    assert model.calls == 5


def test_single_target_identity_is_stable_and_bounded():
    from percept_harness.pipelines.event_evidence import target_entity
    from percept_harness.pipelines.base import SafePipelineError
    import pytest

    assert target_entity('  Red   Container ').entity_id == 'red_container'
    assert target_entity('Café').entity_id == 'cafe'
    assert target_entity('𝐀 cup').entity_id == 'a_cup'
    assert target_entity('123 cup').entity_id == 'entity_123_cup'
    assert target_entity('杯子').canonical_label == '杯子'
    first, second = target_entity('a' * 255 + 'x'), target_entity('a' * 255 + 'y')
    assert first.entity_id != second.entity_id
    assert len(first.entity_id) <= 125
    with pytest.raises(SafePipelineError):
        target_entity('unknown')
