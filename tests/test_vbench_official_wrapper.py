"""Offline contracts for the pinned, single-video VBench wrapper."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest
import uuid
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_vbench_official.py"
spec = importlib.util.spec_from_file_location("vbench_official_test_module", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VBenchOfficialTests(unittest.TestCase):
    def setUp(self):
        # Inherit the parent ACL: Windows sandbox tokens cannot access the
        # owner-only ACL created by Python's mode-0700 TemporaryDirectory.
        temp_parent = Path(tempfile.gettempdir()).resolve()
        self.root = temp_parent / ("vbench-wrapper-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.source = self.root / "source"
        (self.source / "vbench").mkdir(parents=True)
        (self.source / "evaluate.py").write_text("# fixture\n")
        (self.source / "vbench/__init__.py").write_text("# fixture\n")
        self.video = self.root / "input with spaces.mp4"
        self.video.write_bytes(b"video fixture")
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.output = self.root / "output"
        self.pin = mock.patch.object(module, "SOURCE_SHA256", module.source_fingerprint(self.source))
        self.pin.start()
        self.addCleanup(self.pin.stop)
        self.dino_pin = mock.patch.dict(
            module.KNOWN_SHA256,
            {"dino_model/dino_vitbase16_pretrain.pth": module.hashlib.sha256(b"test model bytes").hexdigest()},
            clear=False,
        )
        self.dino_pin.start()
        self.addCleanup(self.dino_pin.stop)

    def args(self, **overrides):
        values = dict(vbench_root=self.source, videos_path=self.video,
                      dimensions=["temporal_flickering"], output_dir=self.output,
                      cache_dir=self.cache, static_subset_ack=True)
        values.update(overrides)
        return values

    def result(self, dimension="temporal_flickering", aggregate=.8, raw=.8, video=None):
        return {dimension: [aggregate, [{"video_path": str(video or self.video), "video_results": raw}]]}

    def write_result(self, data):
        path = self.root / "result.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def provision(self, dimensions):
        manifest = {}
        required = set().union(*(set(module.REQUIRED_FILES[d]) for d in dimensions))
        for relative in required:
            path = self.cache / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test model bytes")
            manifest[relative] = module.sha256(path)
        if "subject_consistency" in dimensions:
            repo = self.cache / module.DINO_REPOSITORY_PREFIX
            for relative, payload in {
                "hubconf.py": b"def dino_vitb16(**kwargs): pass\n",
                "utils.py": b"# official dino utils fixture\n",
                "vision_transformer.py": b"# official dino vision fixture\n",
            }.items():
                target = repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                key = f"{module.DINO_REPOSITORY_PREFIX}/{relative}"
                manifest[key] = module.sha256(target)
            checkpoint = self.cache / module.DINO_CHECKPOINT_RELATIVE
            torch_checkpoint = self.cache / "torch" / "hub" / "checkpoints" / module.DINO_TORCH_CHECKPOINT_NAME
            torch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch_checkpoint.write_bytes(checkpoint.read_bytes())
        path = self.root / "weights.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_help_and_main_check_only_without_model_runtime(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with mock.patch.object(module.subprocess, "run") as runner:
            code = module.main(["--vbench-root", str(self.source), "--videos-path", str(self.video),
                                "--dimension", "temporal_flickering", "--cache-dir", str(self.cache),
                                "--output-dir", str(self.output), "--static-subset-ack", "--check-only"])
        self.assertEqual(code, 0)
        runner.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_source_pin_rejects_changed_extra_missing_files(self):
        module.verify_source(self.source)
        path = self.source / "vbench/__init__.py"
        path.write_text("# changed")
        with self.assertRaisesRegex(ValueError, "pinned"):
            module.verify_source(self.source)
        path.write_text("# fixture\n")
        extra = self.source / "vbench/extra.py"
        extra.write_text("# extra")
        with self.assertRaises(ValueError):
            module.verify_source(self.source)
        extra.unlink()
        path.unlink()
        with self.assertRaises(ValueError):
            module.verify_source(self.source)

    def test_source_fingerprint_normalizes_text_but_not_binary_bytes(self):
        text_file = self.source / "vbench" / "line_endings.md"
        binary_file = self.source / "vbench" / "opaque.bin"
        text_file.write_bytes(b"line 1\r\nline 2\r\n")
        binary_file.write_bytes(b"\x00\r\n\xff")
        first = module.source_fingerprint(self.source)
        text_file.write_bytes(b"line 1\nline 2\n")
        self.assertEqual(first, module.source_fingerprint(self.source))
        binary_file.write_bytes(b"\x00\n\xff")
        self.assertNotEqual(first, module.source_fingerprint(self.source))

    def test_input_mode_static_and_motion_gates(self):
        cases = [dict(videos_path=self.root), dict(mode="vbench_standard"),
                 dict(dimensions=["motion_smoothness"]), dict(static_subset_ack=False),
                 dict(dimensions=["overall_consistency"]), dict(prompt=" "),
                 dict(prompt="None"), dict(gpu="0,1"), dict(load_ckpt_from_local=False)]
        for values in cases:
            with self.subTest(values=values), mock.patch.object(module.subprocess, "run") as runner:
                with self.assertRaises(ValueError):
                    module.run_official(**self.args(**values))
                runner.assert_not_called()

    def test_prompt_file_exact_mapping_and_duplicate_keys(self):
        path = self.root / "prompts.json"
        path.write_text(json.dumps({self.video.name: "move the cup"}))
        self.assertEqual(module.resolve_prompt(self.video, None, path), "move the cup")
        for text in ('{"wrong.mp4":"x"}', '{"a":"x","a":"y"}',
                     '{"a":"x","b":"y"}', '[]'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.resolve_prompt(self.video, None, path)
        with self.assertRaises(ValueError):
            module.resolve_prompt(self.video, "x", path)

    def test_weight_missing_bad_hash_and_missing_tokenizer(self):
        with self.assertRaises(ValueError):
            module.verify_cache(self.cache, ["dynamic_degree"], None)
        manifest = self.provision(["overall_consistency"])
        module.verify_cache(self.cache, ["overall_consistency"], manifest)
        tokenizer = self.cache / "ViCLIP/bpe_simple_vocab_16e6.txt.gz"
        tokenizer.unlink()
        with self.assertRaises(FileNotFoundError):
            module.verify_cache(self.cache, ["overall_consistency"], manifest)
        tokenizer.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            module.verify_cache(self.cache, ["overall_consistency"], manifest)

    def test_dino_requires_source_and_matching_torch_url_cache(self):
        manifest = self.provision(["subject_consistency"])
        records = module.verify_cache(
            self.cache, ["subject_consistency"], manifest,
            torch_home=self.cache / "torch",
        )
        self.assertIn("torch_home/hub/checkpoints/dino_vitbase16_pretrain.pth", records)
        self.assertTrue(
            any(key.startswith(module.DINO_REPOSITORY_PREFIX + "/") for key in records)
        )
        torch_checkpoint = self.cache / "torch" / "hub" / "checkpoints" / module.DINO_TORCH_CHECKPOINT_NAME
        torch_checkpoint.write_bytes(b"wrong checkpoint")
        with self.assertRaisesRegex(ValueError, "URL-cache checkpoint"):
            module.verify_cache(
                self.cache, ["subject_consistency"], manifest,
                torch_home=self.cache / "torch",
            )

    def test_musiq_scale_dynamic_boolean_and_unbounded_regressors(self):
        cases = [("imaging_quality", .75, 75), ("dynamic_degree", 1., True),
                 ("overall_consistency", -.2, -.2), ("aesthetic_quality", 1.1, 1.1),
                 ("imaging_quality", 1.2, 120)]
        for dimension, aggregate, raw in cases:
            result = module.validate_result(self.write_result(self.result(dimension, aggregate, raw)), self.video, [dimension])
            self.assertEqual(result[dimension]["score"], aggregate)
        for dimension, aggregate, raw in [("imaging_quality", 75., 75.), ("dynamic_degree", 1., 1.),
                                           ("overall_consistency", 1.2, 1.2), ("temporal_flickering", True, True)]:
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                module.validate_result(self.write_result(self.result(dimension, aggregate, raw)), self.video, [dimension])

    def test_malformed_incomplete_wrong_video_and_nonfinite_results(self):
        invalid = [[], {}, {"temporal_flickering": [.8, []]},
                   {"temporal_flickering": [.8, self.result()["temporal_flickering"][1] * 2]},
                   self.result(video=self.root / "different.mp4"), self.result(aggregate=.7),
                   self.result(aggregate=float("nan"), raw=float("nan")),
                   self.result(aggregate=float("inf"), raw=float("inf")),
                   self.result(aggregate=2., raw=2.)]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                module.validate_result(self.write_result(data), self.video, ["temporal_flickering"])

    def test_success_official_output_creation_and_safe_environment(self):
        def fake_run(command, *, cwd, env, stdout, stderr):
            self.assertEqual(cwd, self.source)
            self.assertEqual(command[command.index("--load_ckpt_from_local") + 1], "True")
            self.assertEqual(env["WORLD_SIZE"], "1")
            self.assertEqual(env["TORCH_HOME"], str(self.cache / "torch"))
            self.assertNotIn("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
            self.output.mkdir(exist_ok=True)  # exactly what official VBench does
            (self.output / "results_test_eval_results.json").write_text(json.dumps(self.result()))
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}), \
             mock.patch.object(module.subprocess, "run", side_effect=fake_run):
            result = module.run_official(**self.args(gpu="2"))
        self.assertEqual(result["status"], "complete")
        self.assertTrue((self.output / "wrapper_metadata.json").is_file())
        with mock.patch.object(module.subprocess, "run") as runner:
            with self.assertRaises(FileExistsError):
                module.run_official(**self.args())
            runner.assert_not_called()

    def test_explicit_torch_home_is_forwarded_without_changing_model_cache(self):
        torch_home = self.root / "writable-torch-home"
        def fake_run(command, *, cwd, env, stdout, stderr):
            self.assertEqual(env["TORCH_HOME"], str(torch_home))
            self.output.mkdir(exist_ok=True)
            (self.output / "results_test_eval_results.json").write_text(json.dumps(self.result()))
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(module.subprocess, "run", side_effect=fake_run):
            result = module.run_official(**self.args(torch_home=torch_home))
        self.assertEqual(result["torch_home"], str(torch_home))

    def test_failures_keep_log_without_validated_metadata(self):
        for failure in ("exit", "missing", "changed", "invalid", "multiple"):
            output = self.root / failure
            def fake_run(command, **kwargs):
                if failure != "missing":
                    (output / "results_test_eval_results.json").write_text(json.dumps(
                        self.result(aggregate=.2) if failure == "invalid" else self.result()))
                if failure == "multiple":
                    (output / "results_second_eval_results.json").write_text("{}")
                if failure == "changed":
                    self.video.write_bytes(b"modified video")
                return subprocess.CompletedProcess(command, 1 if failure == "exit" else 0)
            with self.subTest(failure=failure), mock.patch.object(module.subprocess, "run", side_effect=fake_run):
                with self.assertRaises((ValueError, RuntimeError)):
                    module.run_official(**self.args(output_dir=output))
            self.assertTrue((output / "vbench.log").is_file())
            self.assertFalse((output / "wrapper_metadata.json").exists())

    def test_prompt_mapping_is_passed_as_prompt_and_pickle_requires_verified_files(self):
        manifest = self.provision(["overall_consistency"])
        prompt_file = self.root / "prompt.json"
        prompt_file.write_text(json.dumps({self.video.name: "move the red cup"}))
        def fake_run(command, *, env, **kwargs):
            self.assertNotIn("--prompt_file", command)
            self.assertEqual(command[command.index("--prompt") + 1], "move the red cup")
            self.assertEqual(env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
            self.assertNotIn("TORCH_FORCE_WEIGHTS_ONLY_LOAD", env)
            (self.output / "results_test_eval_results.json").write_text(json.dumps(
                self.result("overall_consistency", -.2, -.2)))
            return subprocess.CompletedProcess(command, 0)
        with mock.patch.dict(os.environ, {"TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}), \
             mock.patch.object(module.subprocess, "run", side_effect=fake_run):
            result = module.run_official(**self.args(dimensions=["overall_consistency"],
                        weights_manifest=manifest, prompt_file=prompt_file, allow_trusted_pickle=True))
        self.assertEqual(result["prompt"], "move the red cup")
        self.assertEqual(result["scores"]["overall_consistency"]["score"], -.2)


if __name__ == "__main__":
    unittest.main()
