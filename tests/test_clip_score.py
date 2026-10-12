"""Offline contract tests for the official VBench competition CLIP wrapper."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import uuid
from unittest import mock

from percept_harness.video_metrics import vbench_clip as module

SCRIPT = Path(module.__file__)


class ClipScoreWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.gettempdir()).resolve() / ("clip-score-wrapper-" + uuid.uuid4().hex)
        self.tmp.mkdir()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.source = self.tmp / "source"
        (self.source / "competitions").mkdir(parents=True)
        (self.source / "vbench").mkdir()
        (self.source / "vbench2_beta_long").mkdir()
        for p in ("competitions/clip_score.py", "vbench/__init__.py", "vbench2_beta_long/utils.py"):
            path = self.source / p
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        self.pin = mock.patch.object(module, "SOURCE_SHA256", module.source_fingerprint(self.source))
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.video = self.tmp / "video.mp4"
        self.video.write_bytes(b"video")
        self.home = self.tmp / "home"
        self.model = self.home / ".cache" / "clip" / module.CLIP_FILENAME
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b"checkpoint")
        self.output = self.tmp / "out"

    def test_preflight_never_runs_runtime(self):
        with mock.patch.object(module.subprocess, "run") as runner:
            result = module.run_official(vbench_root=self.source, videos_path=self.video,
                                         prompt="move the cup", clip_home=self.home,
                                         output_dir=self.output, check_only=True,
                                         clip_sha256=module.sha256(self.model))
        runner.assert_not_called()
        self.assertEqual(result["status"], "preflight_passed")
        self.assertEqual(result["official_full_info"][0]["dimension"], ["clip_score"])

    def test_input_and_checkpoint_guards(self):
        for kwargs in ({"prompt": ""}, {"prompt": "None"}, {"videos_path": self.tmp}):
            values = dict(vbench_root=self.source, videos_path=self.video, prompt="ok",
                          clip_home=self.home, output_dir=self.output, check_only=True,
                          clip_sha256=module.sha256(self.model))
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, FileNotFoundError)):
                module.run_official(**values)
        self.model.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            module.verify_clip_cache(self.home)

    def test_official_result_validation(self):
        result = self.tmp / "result.json"
        result.write_text(json.dumps([0.25, [{"video_path": str(self.video), "video_results": 0.25}]]))
        self.assertEqual(module.validate_result(result, self.video)["score"], 0.25)
        result.write_text(json.dumps([0.25, [{"video_path": str(self.video), "video_results": 0.2}]]))
        with self.assertRaises(ValueError):
            module.validate_result(result, self.video)


if __name__ == "__main__":
    unittest.main()
