"""Wrapper contract tests; no DOVER dependencies, downloads, or inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_dover_official as dover


class DoverWrapperTests(unittest.TestCase):
    def setUp(self):
        # Ordinary inherited ACLs also work under Windows restricted tokens;
        # tempfile's explicit 0700 ACL can make its child inaccessible there.
        self.base = Path(tempfile.gettempdir()).resolve() / ("dover-wrapper-test-" + uuid4().hex)
        self.base.mkdir()
        self.addCleanup(shutil.rmtree, self.base)
        self.root = self.base / "DOVER"
        self.root.mkdir()
        (self.root / "pretrained_weights").mkdir()
        (self.root / "pretrained_weights" / "DOVER.pth").write_bytes(b"dover fixture")
        self.torch_home = self.base / "torch"
        (self.torch_home / "hub" / "checkpoints").mkdir(parents=True)
        (self.torch_home / "hub" / "checkpoints" / dover.CONVNEXT_FILENAME).write_bytes(b"convnext fixture")
        self.video = self.base / "video.mp4"
        self.video.write_bytes(b"video fixture")
        self.output = self.base / "output"
        self.kwargs = dict(dover_root=self.root, video=self.video,
                           torch_home=self.torch_home, output_dir=self.output)

    def run_fixture(self, text=dover.SCORE_LABEL + " 0.75\n", returncode=0, **kwargs):
        def fake_run(command, **options):
            self.assertIn("-f", command)
            self.assertEqual(options["cwd"], self.root)
            self.assertEqual(options["env"]["TORCH_HOME"], str(self.torch_home))
            options["stdout"].write(text.encode("utf-8"))
            return subprocess.CompletedProcess(command, returncode)

        runtime = {"torch_1_13": True, "problems": [], "model_loading_tested": False}
        with patch.object(dover, "verify_source", return_value={"commit": dover.DOVER_COMMIT}), \
             patch.object(dover, "inspect_runtime", return_value=runtime), \
             patch.object(dover.subprocess, "run", side_effect=fake_run) as process:
            result = dover.run_official(**self.kwargs, **kwargs)
        return result, process

    def test_help_needs_only_standard_library(self):
        process = subprocess.run([sys.executable, "-S", str(Path(dover.__file__)), "--help"],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--check-only", process.stdout)

    def test_only_explicit_official_fused_line_is_parsed(self):
        self.assertEqual(dover.parse_fused_score("2.1\n" + dover.SCORE_LABEL + " 7.5e-1\n"), 0.75)
        for text in ("0.75", dover.SCORE_LABEL + " nan", dover.SCORE_LABEL + " inf",
                     dover.SCORE_LABEL + " 1.1", dover.SCORE_LABEL + " -0.01",
                     dover.SCORE_LABEL + " tensor(0.75)", dover.SCORE_LABEL + " 0.75 extra",
                     (dover.SCORE_LABEL + " 0.75\n") * 2):
            with self.subTest(text=text), self.assertRaises(ValueError):
                dover.parse_fused_score(text)

    def test_success_preserves_log_and_records_hashes(self):
        result, process = self.run_fixture()
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["score"], 0.75)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(result["checkpoints"]["dover"]["sha256"],
                         hashlib.sha256(b"dover fixture").hexdigest())
        self.assertTrue((self.output / "dover.log").is_file())
        self.assertEqual(json.loads((self.output / "result.json").read_text())["score"], 0.75)

    def test_existing_output_is_never_reused(self):
        self.output.mkdir()
        (self.output / "dover.log").write_text(dover.SCORE_LABEL + " 0.99\n")
        with self.assertRaises(FileExistsError), patch.object(dover.subprocess, "run") as process:
            dover.run_official(**self.kwargs)
        process.assert_not_called()

    def test_check_only_does_not_start_official_process(self):
        result, process = self.run_fixture(check_only=True)
        self.assertEqual(result["status"], "preflight_passed_not_scored")
        self.assertIsNone(result["score"])
        self.assertFalse(result["official_inference_started"])
        process.assert_not_called()
        self.assertFalse((self.output / "dover.log").exists())

    def test_missing_convnext_does_not_trigger_download_or_inference(self):
        (self.torch_home / "hub" / "checkpoints" / dover.CONVNEXT_FILENAME).unlink()
        result, process = self.run_fixture()
        self.assertEqual(result["status"], "preflight_failed")
        self.assertIsNone(result["score"])
        self.assertIn("no download attempted", result["error"])
        process.assert_not_called()

    def test_checkpoint_mismatch_stops_before_inference(self):
        result, process = self.run_fixture(checkpoint_sha256="0" * 64)
        self.assertEqual(result["status"], "preflight_failed")
        self.assertIn("SHA-256 mismatch", result["error"])
        process.assert_not_called()

    def test_nonzero_exit_cannot_be_accepted_even_with_score_line(self):
        result, _ = self.run_fixture(returncode=1)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["score"])
        self.assertTrue((self.output / "dover.log").is_file())

    def test_unrecognized_output_is_log_only(self):
        result, _ = self.run_fixture(text="Official process finished without a fused-score line\n")
        self.assertEqual(result["status"], "official_finished_score_unparsed")
        self.assertIsNone(result["score"])
        self.assertEqual(result["returncode"], 0)
        self.assertTrue((self.output / "dover.log").is_file())

    def test_source_verification_rejects_edits_and_extra_python(self):
        (self.root / "dover").mkdir()
        source = self.root / "dover" / "__init__.py"
        source.write_bytes(b"# fixture\r\n")
        pinned = {"dover/__init__.py": hashlib.sha256(b"# fixture\n").hexdigest()}
        with patch.object(dover, "SOURCE_SHA256", pinned):
            dover.verify_source(self.root)
            source.write_bytes(b"# edited\n")
            with self.assertRaisesRegex(ValueError, "source differs"):
                dover.verify_source(self.root)
            source.write_bytes(b"# fixture\n")
            (self.root / "dover" / "extra.py").write_bytes(b"# extra\n")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                dover.verify_source(self.root)

    def test_runtime_requires_explicit_modern_torch_experiment(self):
        def version(name):
            return "2.10.0" if name == "torch" else "1.0"
        with patch.object(dover.importlib.metadata, "version", side_effect=version):
            self.assertTrue(dover.inspect_runtime("cpu", False)["problems"])
            self.assertFalse(dover.inspect_runtime("cpu", True)["problems"])


if __name__ == "__main__":
    unittest.main()
