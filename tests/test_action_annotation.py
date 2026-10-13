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
            root = Path(temporary)
            video = root / 'clip.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=c=red:s=64x64:r=4:d=2', '-c:v', 'mpeg4', str(video)], check=True)
            runner = SyncRunner(FakeVideoModel(), Settings(work_root=root / 'work'), model_alias='fake')
            first = list(run_batch(runner, [video], 'embodied_action_captioning', root / 'out'))
            self.assertEqual(first[0].status, 'completed', first[0].error)
            data = first[0].data
            self.assertEqual(data['task_description'], 'move the red container')
            self.assertTrue(data['segments'])
            self.assertEqual(data['segments'][0]['target'], 'red container')
            second = list(run_batch(runner, [video], 'embodied_action_captioning', root / 'out'))
            self.assertEqual(second[0].status, 'skipped')
            stored = json.loads((root / 'out' / 'clip.json').read_text())
            self.assertEqual(stored['data']['segments'], data['segments'])


if __name__ == '__main__':
    unittest.main()
