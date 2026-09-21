"""Wrapper contract tests; no GPU, model loading or network access required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_vbench_motion_smoothness.py"
from percept_harness.video_metrics import motion as module


class MotionWrapperTests(unittest.TestCase):
    def test_help_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_result_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            video = (Path(directory) / "video.mp4").resolve()
            result = Path(directory) / "result.json"
            def write(score=0.987654321, mean=0.987654321, path=video, count=1):
                result.write_text(json.dumps({"motion_smoothness": [mean, [{"video_path": str(path), "video_results": score}] * count]}))
            write()
            self.assertEqual(module.read_score(result, video), 0.987654321)
            for kwargs in ({"count": 0}, {"count": 2}, {"path": video.with_name("other.mp4")},
                           {"score": float("nan")}, {"score": True}, {"score": 2}, {"mean": 0.1}):
                with self.subTest(kwargs=kwargs):
                    write(**kwargs)
                    with self.assertRaises(ValueError):
                        module.read_score(result, video)

    def test_existing_output_prevents_execution(self):
        with mock.patch.object(module.subprocess, "run") as runner:
            with self.assertRaises(FileExistsError):
                module.score_video(SCRIPT, SCRIPT.parent, SCRIPT)
            runner.assert_not_called()

    def test_wrapper_contract_and_failures(self):
        for case in ("success", "cli_error", "no_result", "changed_video", "bad_checkpoint"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                video = root / "input with spaces.mp4"
                video.write_bytes(b"test video")
                cache = root / "cache"
                (cache / "amt_model").mkdir(parents=True)
                checkpoint = cache / "amt_model/amt-s.pth"
                checkpoint.write_bytes(b"test checkpoint")
                binary = root / ("vbench.exe" if module.os.name == "nt" else "vbench")
                binary.touch()
                output = root / "score.json"
                expected_hash = module.sha256(checkpoint) if case != "bad_checkpoint" else "wrong"
                def fake_run(command, *, env, stdout, stderr):
                    self.assertEqual(command[:2], [str(binary), "evaluate"])
                    self.assertEqual(command[command.index("--videos_path") + 1], str(video))
                    self.assertEqual(command[command.index("--dimension") + 1], "motion_smoothness")
                    self.assertEqual(command[command.index("--mode") + 1], "custom_input")
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
                    self.assertEqual(env["VBENCH_CACHE_DIR"], str(cache))
                    self.assertEqual(env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
                    self.assertNotIn("TORCH_FORCE_WEIGHTS_ONLY_LOAD", env)
                    if case not in ("cli_error", "no_result"):
                        run_dir = Path(command[command.index("--output_path") + 1])
                        (run_dir / "results_test_eval_results.json").write_text(json.dumps({
                            "motion_smoothness": [0.987654321, [{"video_path": str(video), "video_results": 0.987654321}]]}))
                    if case == "changed_video":
                        video.write_bytes(b"changed")
                    return subprocess.CompletedProcess(command, 1 if case == "cli_error" else 0)
                with mock.patch.object(module, "AMT_SHA256", expected_hash), \
                     mock.patch.object(module.sysconfig, "get_path", return_value=str(root)), \
                     mock.patch.object(module.importlib.metadata, "version", return_value="0.1.5"), \
                     mock.patch.object(module.subprocess, "run", side_effect=fake_run) as runner:
                    if case == "success":
                        payload = module.score_video(video, cache, output, "2")
                        self.assertEqual(payload["score"], 0.987654321)
                        self.assertTrue(Path(payload["official_result"]).is_file())
                        self.assertEqual(json.loads(output.read_text()), payload)
                    else:
                        with self.assertRaises((RuntimeError, ValueError)):
                            module.score_video(video, cache, output, "2")
                        self.assertFalse(output.exists())
                        if case == "bad_checkpoint":
                            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
