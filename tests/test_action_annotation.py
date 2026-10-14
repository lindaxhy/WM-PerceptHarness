import unittest
from contextlib import redirect_stderr
from io import StringIO
from percept_harness.cli import _parser


class ContractTests(unittest.TestCase):
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
