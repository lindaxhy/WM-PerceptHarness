"""Tests for the standalone CLIP-IQA+ video scoring script.

The default tests deliberately use only the Python standard library.  Tests
which exercise the actual OpenCV/PyTorch protocol are skipped when the metric
runtime is not installed; they use a fake ``pyiqa`` module so no checkpoint is
downloaded.
"""

from __future__ import annotations

from contextlib import contextmanager
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_clipiqa_plus.py"


def load_script():
    """Load the script without importing optional metric dependencies."""
    from percept_harness.video_metrics import clipiqa
    return clipiqa


def optional_runtime_or_skip(test_case: unittest.TestCase):
    """Return optional runtime modules, or skip this protocol test."""
    try:
        import cv2
        import numpy as np
        import torch
        import torchvision
        from torchvision.transforms import ToTensor  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on test machine
        test_case.skipTest(f"optional metric runtime unavailable: {exc}")
    return cv2, np, torch, torchvision


@contextmanager
def fake_pyiqa(create_metric):
    """Install a fake pyiqa module for tests without downloading weights."""
    fake = types.ModuleType("pyiqa")
    fake.create_metric = create_metric
    with mock.patch.dict(sys.modules, {"pyiqa": fake}):
        yield fake


class ScriptSmokeTests(unittest.TestCase):
    def test_import_and_argument_validation_need_no_gpu_runtime(self):
        module = load_script()
        self.assertEqual(module._positive_int("4"), 4)
        with self.assertRaises(argparse.ArgumentTypeError):
            module._positive_int("0")
        with self.assertRaises(ValueError):
            module._positive_int("not-an-int")

    def test_help_does_not_import_optional_runtime(self):
        completed = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--video", completed.stdout)
        self.assertIn("CLIP-IQA+", completed.stdout)

    def test_existing_output_is_never_overwritten(self):
        module = load_script()
        # Use this already-existing source file so the smoke test does not
        # need to create a temporary file (and cannot accidentally write it).
        output = SCRIPT
        before = output.read_bytes()
        with mock.patch.object(module, "score_video") as scorer:
            result = module.main(
                ["--video", str(SCRIPT.with_name("input.mp4")), "--output", str(output)]
            )

        self.assertEqual(result, 1)
        scorer.assert_not_called()
        self.assertEqual(output.read_bytes(), before)


class ProtocolTests(unittest.TestCase):
    def _write_video(self, directory: Path):
        cv2, np, _torch, _torchvision = optional_runtime_or_skip(self)
        path = directory / "colours.avi"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (64, 32),
        )
        if not writer.isOpened():  # codec availability is environmental
            writer.release()
            self.skipTest("OpenCV MJPG writer is unavailable")
        # Make frames 0, 4 and 8 visually distinct.  Intermediate frames are
        # valid filler; sample_step=4 should select exactly these three frames.
        colours_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for index in range(9):
            colour = colours_bgr[index // 4] if index in (0, 4, 8) else (20, 20, 20)
            writer.write(np.full((32, 64, 3), colour, dtype=np.uint8))
        writer.release()
        if not path.is_file() or path.stat().st_size == 0:
            self.skipTest("OpenCV did not produce a video")
        return path

    def test_video_protocol_samples_0_4_8_converts_rgb_and_resizes(self):
        cv2, np, torch, torchvision = optional_runtime_or_skip(self)
        module = load_script()
        captured = []
        values = iter((0.12344, 0.56789, 0.99999))

        class FakeMetric:
            def eval(self):
                return self

            def __call__(self, tensor):
                captured.append(tensor.detach().cpu())
                return torch.tensor([next(values)], dtype=torch.float32, device=tensor.device)

        with tempfile.TemporaryDirectory() as directory:
            video = self._write_video(Path(directory))
            with fake_pyiqa(lambda name, device: self._metric_factory(name, device, FakeMetric)):
                with mock.patch.object(module.importlib.metadata, "version", return_value="0.1.16"):
                    result = module.score_video(video, device="cpu", sample_step=4, max_side=32)

        self.assertEqual(result["frame_count"], 9)
        self.assertEqual(result["sampled_frames"], 3)
        self.assertEqual(result["score"], 0.5638)
        self.assertEqual([tuple(t.shape) for t in captured], [(1, 3, 16, 32)] * 3)
        # OpenCV stores BGR; the first/second/third sampled frames are red,
        # green, blue respectively after the script's BGR->RGB conversion.
        means = [tensor[0].mean(dim=(1, 2)).tolist() for tensor in captured]
        self.assertGreater(means[0][0], 0.75)
        self.assertLess(means[0][2], 0.25)
        self.assertGreater(means[1][1], 0.75)
        self.assertGreater(means[2][2], 0.75)
        self.assertEqual(result["sampling"]["max_side"], 32)
        self.assertEqual(result["versions"]["pyiqa"], "0.1.16")

    @staticmethod
    def _metric_factory(name, device, metric_type):
        if name != "clipiqa+":
            raise AssertionError(f"unexpected metric fallback: {name}")
        return metric_type()

    def test_model_loading_error_is_not_replaced_by_fallback(self):
        _cv2, np, _torch, _torchvision = optional_runtime_or_skip(self)
        module = load_script()
        calls = []

        def create_metric(name, device):
            calls.append((name, device))
            raise RuntimeError("checkpoint unavailable")

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            video = directory / "input.avi"
            # VideoReader is replaced below, so this only needs to be a file
            # for score_video's path validation and hash code.
            video.write_bytes(b"placeholder")

            class Reader:
                num_frames = 1

                def __init__(self, *_args, **_kwargs):
                    pass

                def get(self, _index):
                    return np.zeros((2, 2, 3), dtype=np.uint8)

                def close(self):
                    pass

            with fake_pyiqa(create_metric):
                with mock.patch.object(module, "VideoReader", Reader):
                    with mock.patch.object(module.importlib.metadata, "version", return_value="0.1.16"):
                        with self.assertRaisesRegex(RuntimeError, "checkpoint unavailable"):
                            module.score_video(video, device="cpu")
        self.assertEqual(calls, [("clipiqa+", "cpu")])

    def test_nan_score_is_rejected_and_reader_is_released(self):
        _cv2, np, torch, _torchvision = optional_runtime_or_skip(self)
        module = load_script()
        state = {"reader": None, "closed": False}

        class Reader:
            num_frames = 1

            def __init__(self, *_args, **_kwargs):
                state["reader"] = self

            def get(self, _index):
                return np.zeros((2, 2, 3), dtype=np.uint8)

            def close(self):
                state["closed"] = True

        class NaNMetric:
            def eval(self):
                return self

            def __call__(self, _tensor):
                return torch.tensor([float("nan")])

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.avi"
            video.write_bytes(b"placeholder")
            with fake_pyiqa(lambda name, device: NaNMetric()):
                with mock.patch.object(module, "VideoReader", Reader):
                    with mock.patch.object(module.importlib.metadata, "version", return_value="0.1.16"):
                        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                            module.score_video(video, device="cpu")
        self.assertIsNotNone(state["reader"])
        self.assertTrue(state["closed"])


if __name__ == "__main__":
    unittest.main()
