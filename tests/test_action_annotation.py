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
        self.assertEqual(args.template, 'embodied_action_captioning')

    def test_retired_templates_rejected_by_cli(self):
        for template in ('general_video_captioning', 'embodied_active_object_detection'):
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
            first = list(run_batch(runner, [video], 'embodied_action_captioning', root / 'out'))
            self.assertEqual(first[0].status, 'completed', first[0].error)
            data = first[0].data
            self.assertEqual(set(data), {'duration', 'events'})
            self.assertTrue(data['events'])
            self.assertEqual(data['events'][0]['description'], 'A hand moves the red container.')
            second = list(run_batch(runner, [video], 'embodied_action_captioning', root / 'out'))
            # CV execution identity is not yet reusable; rerun honestly, with CV cache.
            self.assertEqual(second[0].status, 'completed')
            self.assertTrue(second[0].metadata['evidence'][0]['cache_hit'])
            stored = json.loads((root / 'out' / 'clip.json').read_text())
            self.assertEqual(stored['data']['events'], data['events'])
            self.assertTrue(stored['metadata']['synthetic'])


if __name__ == '__main__':
    unittest.main()
