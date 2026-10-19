import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from percept_harness.video_metrics import clipiqa, dover, external, motion, videophy_joint
from test_metrics_vbench import BASE_ENV, VENV_BIN, FakeRun, child_env, dumps

DOVER_FILE = Path(dover.__file__)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path.resolve()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(BASE_ENV))


# --- motion ---------------------------------------------------------------

def motion_result(video, score=0.9, mean=None, rows=None):
    rows = [{"video_path": str(video), "video_results": score}] if rows is None else rows
    return {"motion_smoothness": [score if mean is None else mean, rows]}


def write(tmp, payload, text=None):
    path = tmp / "r.json"
    path.write_text(text if text is not None else json.dumps(payload))
    return path


def test_motion_read_score_good(tmp):
    video = tmp / "v.mp4"
    assert motion.read_score(write(tmp, motion_result(video, 1)), video) == 1.0
    assert motion.read_score(write(tmp, motion_result(video, 0.25)), video) == 0.25


@pytest.mark.parametrize("payload, error, message", [
    (lambda v: motion_result(v, rows=[]), ValueError, "Expected exactly one result for the requested video"),
    (lambda v: motion_result(v, rows=[{"video_path": "/x.mp4", "video_results": 1}]), ValueError, "Expected exactly one result"),
    (lambda v: motion_result(v, True), ValueError, "Expected a numeric Motion Smoothness score"),
    (lambda v: motion_result(v, "1"), ValueError, "Expected a numeric Motion Smoothness score"),
    (lambda v: motion_result(v, 1.5), ValueError, "Invalid Motion Smoothness score"),
    (lambda v: motion_result(v, -0.1), ValueError, "Invalid Motion Smoothness score"),
    (lambda v: motion_result(v, 0.5, mean=0.6), ValueError, "Single-video score and aggregate disagree"),
    (lambda v: motion_result(v, 0.5, mean=True), ValueError, "Expected a numeric"),
    (lambda v: {"other": []}, KeyError, "motion_smoothness"),
    (lambda v: motion_result(v, rows=[{"video_results": 1}]), KeyError, "video_path"),
])
def test_motion_read_score_bad(tmp, payload, error, message):
    video = tmp / "v.mp4"
    with pytest.raises(error, match=message):
        motion.read_score(write(tmp, payload(video)), video)


def test_motion_read_score_nan(tmp):
    video = tmp / "v.mp4"
    text = '{"motion_smoothness": [NaN, [{"video_path": %s, "video_results": NaN}]]}' % json.dumps(str(video))
    with pytest.raises(ValueError, match="Invalid Motion Smoothness score"):
        motion.read_score(write(tmp, None, text), video)


@pytest.fixture
def motion_setup(tmp, monkeypatch):
    video = tmp / "v.mp4"
    video.write_bytes(b"v")
    cache = tmp / "cache"
    (cache / "amt_model").mkdir(parents=True)
    (cache / "amt_model" / "amt-s.pth").write_bytes(b"w")
    scripts = tmp / "bin"
    scripts.mkdir()
    (scripts / "vbench").write_text("")
    versions = {"vbench": "0.1.5", "torch": "2.7.0"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(motion.sysconfig, "get_path", lambda name: str(scripts))
    return video, cache, scripts, versions


def test_motion_preflight_errors(tmp, motion_setup, monkeypatch):
    video, cache, scripts, versions = motion_setup
    out = tmp / "out.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError, match="Output already exists"):
        motion.score_video(video, cache, out)
    other = tmp / "v.MP4"
    other.write_bytes(b"v")
    with pytest.raises(ValueError, match="lowercase .mp4"):
        motion.score_video(other, cache, tmp / "o.json")
    versions["vbench"] = "0.1.4"
    with pytest.raises(RuntimeError, match=r"Expected vbench==0.1.5; found 0.1.4"):
        motion.score_video(video, cache, tmp / "o.json")
    versions["vbench"] = "0.1.5"
    (scripts / "vbench").unlink()
    with pytest.raises(FileNotFoundError, match="Official VBench CLI not found"):
        motion.score_video(video, cache, tmp / "o.json")
    (cache / "amt_model" / "amt-s.pth").write_bytes(b"")
    with pytest.raises(ValueError, match="Missing checkpoint: .*amt-s.pth"):
        motion.score_video(video, cache, tmp / "o.json")
    assert not (tmp / "o.json").exists()


def motion_writer(video, payload=None):
    def write_result(command, kwargs):
        output = Path(command[command.index("--output_path") + 1])
        (output / "x_eval_results.json").write_text(json.dumps(payload or motion_result(video)))
        kwargs["stdout"].write("log\n")
    return write_result


def test_motion_full_run_env_and_golden(tmp, motion_setup, env, monkeypatch, capsys):
    video, cache, scripts, _ = motion_setup
    fake = FakeRun(motion_writer(video))
    monkeypatch.setattr(subprocess, "run", fake)
    out = tmp / "res" / "out.json"
    payload = motion.score_video(video, cache, out, gpu="2")
    (command, kwargs), = fake.calls
    run_dir = Path(command[command.index("--output_path") + 1])
    assert run_dir.parent == out.parent and run_dir.name.startswith("out.vbench-")
    assert command == [str(scripts / "vbench"), "evaluate", "--videos_path", str(video),
                       "--dimension", "motion_smoothness", "--mode", "custom_input", "--output_path", str(run_dir)]
    assert "cwd" not in kwargs or kwargs["cwd"] is None
    assert kwargs["stderr"] == subprocess.STDOUT
    expected_env = {**BASE_ENV, "PATH": VENV_BIN + os.pathsep + "/usr/bin", "VBENCH_CACHE_DIR": str(cache),
                    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "CUDA_VISIBLE_DEVICES": "2"}
    del expected_env["TORCH_FORCE_WEIGHTS_ONLY_LOAD"]
    assert child_env(kwargs) == expected_env
    expected = {"metric": "vbench_motion_smoothness", "score": 0.9, "higher_is_better": True, "video": str(video),
                "versions": {"vbench": "0.1.5", "torch": "2.7.0"},
                "checkpoint": {"path": str(cache / "amt_model" / "amt-s.pth")},
                "protocol": "Unmodified VBench 0.1.5 motion_smoothness (AMT-S), custom_input",
                "cuda_visible_devices": "2", "official_result": str(run_dir / "x_eval_results.json"),
                "log": str(run_dir / "vbench.log"), "command": command}
    assert payload == expected
    assert out.read_text() == dumps(expected)
    assert (run_dir / "vbench.log").read_text() == "log\n"
    assert capsys.readouterr().err == f"Official VBench output and log: {run_dir}\n"


def test_motion_no_weights_only_is_unconditional(tmp, motion_setup, monkeypatch):
    video, cache, _, _ = motion_setup
    monkeypatch.setattr(os, "environ", {"PATH": "/bin", "PYTORCH_ALLOC_CONF": "keep"})
    fake = FakeRun(motion_writer(video))
    monkeypatch.setattr(subprocess, "run", fake)
    motion.score_video(video, cache, tmp / "o.json")
    assert child_env(fake.calls[0][1]) == {"PATH": VENV_BIN + os.pathsep + "/bin", "PYTORCH_ALLOC_CONF": "keep",
                                           "VBENCH_CACHE_DIR": str(cache), "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}


def test_motion_run_failures(tmp, motion_setup, monkeypatch):
    video, cache, _, _ = motion_setup
    monkeypatch.setattr(subprocess, "run", FakeRun(returncode=4))
    with pytest.raises(RuntimeError, match=r"^VBench exited with 4; see .*vbench.log$"):
        motion.score_video(video, cache, tmp / "a.json")
    monkeypatch.setattr(subprocess, "run", FakeRun())
    with pytest.raises(RuntimeError, match="Expected one official result JSON; see "):
        motion.score_video(video, cache, tmp / "b.json")
    assert not (tmp / "a.json").exists() and not (tmp / "b.json").exists()


def test_motion_main(tmp, motion_setup, monkeypatch, capsys):
    video, cache, _, _ = motion_setup
    monkeypatch.setattr(subprocess, "run", FakeRun(motion_writer(video)))
    out = tmp / "o.json"
    argv = ["--video", str(video), "--cache-dir", str(cache), "--output", str(out)]
    assert motion.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out == dumps(json.loads(out.read_text()))
    assert motion.main(argv) == 1
    assert capsys.readouterr().err == f"Motion Smoothness failed: Output already exists: {out}\n"
    with pytest.raises(SystemExit) as exit_info:
        motion.main(["--video", str(video)])
    assert exit_info.value.code == 2


# --- dover ----------------------------------------------------------------

class FixedDatetime:
    @staticmethod
    def now(tz):
        from datetime import datetime
        return datetime(2026, 1, 2, 3, 4, 5, tzinfo=tz)


@pytest.fixture
def dover_setup(tmp, monkeypatch):
    root = tmp / "DOVER"
    (root / "dover").mkdir(parents=True)
    for name in ("evaluate_one_video.py", "dover.yml", "dover/__init__.py"):
        (root / name).write_text("")
    (root / "pretrained_weights").mkdir()
    (root / "pretrained_weights" / "DOVER.pth").write_bytes(b"12345")
    torch_home = tmp / "th"
    (torch_home / "hub" / "checkpoints").mkdir(parents=True)
    (torch_home / "hub" / "checkpoints" / dover.CONVNEXT_FILENAME).write_bytes(b"123")
    video = tmp / "v.mp4"
    video.write_bytes(b"vv")
    versions = {name: "1.0" for name in dover.RUNTIME_PACKAGES}
    versions.update(torch="1.13.1", torchvision="0.14.1")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(dover, "datetime", FixedDatetime)
    return root, torch_home, video, versions


def dover_expected(root, torch_home, video, out, **extra):
    result = {"metric": "dover_fused_official", "score": None, "higher_is_better": True,
              "status": "preflight_passed_not_scored", "official_inference_started": False, "check_only": True,
              "started_at": "2026-01-02T03:04:05+00:00", "source_commit": dover.DOVER_COMMIT, "device": "cpu",
              "log": str(out / "dover.log"),
              "protocol": "Unchanged evaluate_one_video.py -f with pinned dover.yml",
              "runtime_validation": "No GPU/CPU model inference has been validated by this preparation package",
              "source": {"origin": dover.DOVER_ORIGIN, "expected_commit": dover.DOVER_COMMIT},
              "dover_root": str(root), "video": {"path": str(video), "bytes": 2},
              "checkpoints": {"dover": {"path": str(root / "pretrained_weights" / "DOVER.pth"), "bytes": 5},
                              "convnext_initialization": {"path": str(torch_home / "hub" / "checkpoints" / dover.CONVNEXT_FILENAME), "bytes": 3}},
              "runtime": {"python": sys.version, "executable": sys.executable,
                          "packages": {name: ("1.13.1" if name == "torch" else "0.14.1" if name == "torchvision" else "1.0") for name in dover.RUNTIME_PACKAGES},
                          "torch_1_13": True, "allow_unverified_runtime": False, "cuda": None, "problems": [],
                          "model_loading_tested": False},
              "environment": {"TORCH_HOME": str(torch_home), "CUDA_VISIBLE_DEVICES": None, "PYTHONDONTWRITEBYTECODE": "1",
                              "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1"},
              "command": [sys.executable, "-B", "-u", str(root / "evaluate_one_video.py"), "-o", str(root / "dover.yml"),
                          "-v", str(video), "-d", "cpu", "-f"],
              "working_directory": str(root), "finished_at": "2026-01-02T03:04:05+00:00"}
    result.update(extra)
    return result


def test_dover_check_only_golden(tmp, dover_setup, env, capsys):
    root, torch_home, video, _ = dover_setup
    out = tmp / "out"
    argv = ["--dover-root", str(root), "--video", str(video), "--torch-home", str(torch_home),
            "--output-dir", str(out), "--check-only"]
    assert dover.main(argv) == 0
    expected = dover_expected(root, torch_home, video, out)
    assert (out / "result.json").read_text() == dumps(expected, allow_nan=False)
    assert capsys.readouterr().out == dumps(expected)
    assert not (out / "result.pending").exists()
    assert dover.main(argv) == 1
    assert capsys.readouterr().err == f"DOVER wrapper failed: [Errno 17] File exists: '{out}'\n"


def test_dover_full_run_env(tmp, dover_setup, env, monkeypatch):
    root, torch_home, video, _ = dover_setup
    fake = FakeRun(lambda command, kwargs: kwargs["stdout"].write(f"{dover.SCORE_LABEL} 0.625\n".encode()))
    monkeypatch.setattr(subprocess, "run", fake)
    out = tmp / "out"
    result = dover.run_official(dover_root=root, video=video, torch_home=torch_home, output_dir=out, gpu="3")
    (command, kwargs), = fake.calls
    assert kwargs["cwd"] == root and kwargs["stderr"] == subprocess.STDOUT
    assert child_env(kwargs) == {**BASE_ENV, "TORCH_HOME": str(torch_home), "PYTHONDONTWRITEBYTECODE": "1",
                                 "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "3"}
    expected = dover_expected(root, torch_home, video, out, check_only=False, status="scored", score=0.625,
                              official_inference_started=True, returncode=0)
    expected["environment"]["CUDA_VISIBLE_DEVICES"] = "3"
    assert result == expected
    assert json.loads((out / "result.json").read_text()) == expected


def test_dover_cuda_probe_env(tmp, dover_setup, env, monkeypatch):
    root, torch_home, video, _ = dover_setup
    def probe(command, kwargs):
        assert command[-1] == "0"
    fake = FakeRun(probe, returncode=1)
    monkeypatch.setattr(subprocess, "run", fake)
    result = dover.run_official(dover_root=root, video=video, torch_home=torch_home, output_dir=tmp / "o",
                                device="cuda", gpu="5", check_only=True)
    assert child_env(fake.calls[0][1]) == {**BASE_ENV, "CUDA_VISIBLE_DEVICES": "5"}
    assert result["status"] == "preflight_failed"
    assert result["error"] == "CUDA preflight failed: CUDA probe exited 1"


@pytest.mark.parametrize("mutate, message", [
    (lambda root, video, versions: (root / "dover.yml").unlink(), "Missing required source file: .*dover.yml"),
    (lambda root, video, versions: (root / "pretrained_weights" / "DOVER.pth").write_bytes(b""), "Required local checkpoint missing/empty"),
    (lambda root, video, versions: video.write_bytes(b""), "--video must be a nonempty single video file"),
    (lambda root, video, versions: versions.update(torch="2.7.0"), "Expected Torch 1.13.x"),
    (lambda root, video, versions: versions.update(torchvision="0.15.0"), "matching torchvision 0.14.x"),
    (lambda root, video, versions: versions.update(timm=None), "Missing package: timm"),
])
def test_dover_preflight_failures(tmp, dover_setup, monkeypatch, mutate, message):
    root, torch_home, video, versions = dover_setup
    mutate(root, video, versions)
    if versions.get("timm", "") is None:
        def version(name):
            if versions[name] is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        monkeypatch.setattr(importlib.metadata, "version", version)
    out = tmp / "out"
    code = dover.main(["--dover-root", str(root), "--video", str(video), "--torch-home", str(torch_home),
                       "--output-dir", str(out), "--check-only"])
    assert code == 1
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "preflight_failed" and result["score"] is None
    assert re.search(message, result["error"])


def test_dover_invalid_device(tmp, dover_setup):
    root, torch_home, video, _ = dover_setup
    result = dover.run_official(dover_root=root, video=video, torch_home=torch_home, output_dir=tmp / "o", device="gpu")
    assert result["error"] == "--device must be cpu, cuda, or cuda:N"
    result = dover.run_official(dover_root=root, video=video, torch_home=torch_home, output_dir=tmp / "p",
                                device="cuda:1", gpu="0")
    assert result["error"] == "With --gpu, use --device cuda or cuda:0 (the visible device is remapped)"


def test_dover_unparsed_and_failed_runs(tmp, dover_setup, monkeypatch):
    root, torch_home, video, _ = dover_setup
    monkeypatch.setattr(subprocess, "run", FakeRun(lambda c, k: k["stdout"].write(b"no score\n")))
    result = dover.run_official(dover_root=root, video=video, torch_home=torch_home, output_dir=tmp / "a")
    assert result["status"] == "official_finished_score_unparsed" and result["score"] is None
    monkeypatch.setattr(subprocess, "run", FakeRun(returncode=9))
    result = dover.run_official(dover_root=root, video=video, torch_home=torch_home, output_dir=tmp / "b")
    assert result["status"] == "failed" and result["returncode"] == 9
    assert result["error"] == f"Official DOVER exited with 9; inspect {tmp / 'b' / 'dover.log'}"


@pytest.mark.parametrize("text, expected", [
    (f"x\n{dover.SCORE_LABEL} 0.5\n", 0.5), (f"  {dover.SCORE_LABEL} 1\n", 1.0), (f"{dover.SCORE_LABEL} .25e0", 0.25)])
def test_dover_parse_score_good(text, expected):
    assert dover.parse_fused_score(text) == expected


@pytest.mark.parametrize("text, message", [
    ("", "exactly one"), (f"{dover.SCORE_LABEL} 0.5\n{dover.SCORE_LABEL} 0.5", "exactly one"),
    (f"pre {dover.SCORE_LABEL} 0.5", "exactly one"), (f"{dover.SCORE_LABEL} nan", "plain finite number"),
    (f"{dover.SCORE_LABEL} 1.5", "outside"), (f"{dover.SCORE_LABEL} 0.5 extra", "plain finite number")])
def test_dover_parse_score_bad(text, message):
    with pytest.raises(ValueError, match=message):
        dover.parse_fused_score(text)


def test_dover_runs_as_standalone_file(tmp):
    # Documented for older upstream Pythons: the file must run without the package installed.
    completed = subprocess.run([sys.executable, "-S", str(DOVER_FILE), "--help"], cwd=tmp,
                               capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout and "--dover-root" in completed.stdout


# --- external -------------------------------------------------------------

class GitRun(FakeRun):
    def __init__(self, head, returncode=0):
        super().__init__(returncode=returncode)
        self.head = head

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[:1] == ["git"]:
            return subprocess.CompletedProcess(command, 0, stdout=self.head + "\n", stderr="")
        return subprocess.CompletedProcess(command, self.returncode)


@pytest.fixture
def checkout(tmp):
    source = tmp / "src"
    (source / ".git").mkdir(parents=True)
    manifest = tmp / "m.json"
    manifest.write_text("[]")
    return source, manifest


def test_external_verify_revision(tmp, checkout):
    source, _ = checkout
    runner = GitRun("abc")
    assert external.verify_revision(source, "abc", runner=runner) == {"revision": "abc"}
    assert runner.calls == [(["git", "-C", str(source), "rev-parse", "HEAD"], {"check": True, "capture_output": True, "text": True})]
    assert external.verify_revision(source, None, runner=runner) == {"revision": None}
    with pytest.raises(ValueError, match="revision mismatch: expected abd, got abc"):
        external.verify_revision(source, "abd", runner=runner)
    with pytest.raises(ValueError, match="Source is not a git checkout"):
        external.verify_revision(tmp, "abc", runner=runner)


def test_external_preflight_and_run(tmp, checkout, env):
    source, manifest = checkout
    rev = external.SPECS["mechanics"].revision
    runner = GitRun(rev)
    args = dict(metric="mechanics", source_dir=source, checkpoint=None, manifest=manifest, generated=None, reference=None)
    expected = {"metric": "mechanics", "source": str(source), "source_url": "https://github.com/Vchitect/VBench",
                "revision": rev, "checkpoint": None, "manifest": str(manifest), "generated": None, "reference": None,
                "command_hint": "bash evaluate.sh --max_parallel_tasks 1"}
    assert external.preflight(**args, runner=runner) == expected
    assert external.run_official(**args, command=["bash", "x"], check_only=True, runner=runner) == {
        **expected, "command": ["bash", "x"], "status": "preflight_passed"}
    out = tmp / "logs" / "r.json"
    result = external.run_official(**args, command=["bash", "x"], output=out, runner=runner)
    assert result == {**expected, "command": ["bash", "x"], "status": "complete", "log": str(out) + ".log"}
    command, kwargs = runner.calls[-1]
    assert command == ["bash", "x"] and kwargs["cwd"] == str(source) and kwargs["check"] is False
    assert child_env(kwargs) == BASE_ENV
    assert (tmp / "logs" / "r.json.log").exists() and not out.exists()
    with pytest.raises(FileExistsError, match="output already exists"):
        external.run_official(**args, command=["bash"], output=out.with_suffix(".json.log"), runner=runner)
    failing = GitRun(rev, returncode=5)
    with pytest.raises(RuntimeError, match="official mechanics command exited with 5"):
        external.run_official(**args, command=["bash"], runner=failing)


@pytest.mark.parametrize("metric, overrides, message", [
    ("nope", {}, "unknown external metric: nope"),
    ("mechanics", {"source_dir": None}, "requires --source-dir"),
    ("mechanics", {"manifest": None}, "requires --manifest"),
    ("instruction_following", {}, "requires --checkpoint"),
    ("psnr", {"manifest": None}, "requires --generated"),
])
def test_external_preflight_errors(checkout, metric, overrides, message):
    source, manifest = checkout
    args = dict(metric=metric, source_dir=source, checkpoint=None, manifest=manifest, generated=None, reference=None)
    args.update(overrides)
    spec = external.SPECS.get(metric)
    with pytest.raises(ValueError, match=message):
        external.preflight(**args, runner=GitRun(spec.revision if spec else ""))


def test_external_preflight_path_kinds(tmp, checkout):
    source, manifest = checkout
    runner = GitRun(external.SPECS["psnr"].revision)
    with pytest.raises(ValueError, match="generated must be a directory"):
        external.preflight(metric="psnr", source_dir=source, checkpoint=None, manifest=None,
                           generated=manifest, reference=tmp, runner=runner)
    with pytest.raises(ValueError, match="manifest must be a file"):
        external.preflight(metric="mechanics", source_dir=source, checkpoint=None, manifest=tmp,
                           generated=None, reference=None, runner=GitRun(external.SPECS["mechanics"].revision))
    with pytest.raises(ValueError, match="requires --reference"):
        external.preflight(metric="psnr", source_dir=source, checkpoint=None, manifest=None,
                           generated=tmp, reference=None, runner=runner)


def test_external_main(tmp, checkout, capsys):
    source, manifest = checkout
    assert external.main(["--metric", "mechanics", "--source-dir", str(tmp), "--manifest", str(manifest), "--", "bash"]) == 1
    assert capsys.readouterr().err == f"Official external metric runner failed: Source is not a git checkout: {tmp}\n"
    assert external.main(["--metric", "mechanics", "--source-dir", str(source)]) == 1
    assert capsys.readouterr().err == "Official external metric runner failed: append the exact official command after --\n"
    with pytest.raises(SystemExit) as exit_info:
        external.main(["--metric", "bogus"])
    assert exit_info.value.code == 2


def test_external_specs_pin_vbench_revision():
    vbench = {name for name, spec in external.SPECS.items() if spec.source_url == "https://github.com/Vchitect/VBench"}
    assert vbench == {"motion_order_understanding", "motion_rationality", "mechanics", "thermotics", "material"}
    assert {external.SPECS[name].revision for name in vbench} == {"fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"}


# --- clipiqa (no torch: only argument parsing, preflight and output) ------

def test_clipiqa_score_video_preflight(tmp):
    with pytest.raises(FileNotFoundError):
        clipiqa.score_video(tmp / "missing.mp4")
    with pytest.raises(ValueError, match="Expected a video file"):
        clipiqa.score_video(tmp)
    video = tmp / "v.mp4"
    video.write_bytes(b"v")
    with pytest.raises(ValueError, match="sample_step and max_side must be positive"):
        clipiqa.score_video(video, sample_step=0)
    with pytest.raises(ValueError, match=r"Missing checkpoint: .*CLIP-IQA%2B_learned_prompts-603f3273.pth"):
        clipiqa.score_video(video, cache_dir=tmp / "cache")


def test_clipiqa_main_output(tmp, monkeypatch, capsys):
    calls = []
    def fake_score(*args):
        calls.append(args)
        print("model noise")
        return {"metric": "clipiqa+", "score": 0.5, "video": "猫.mp4"}
    monkeypatch.setattr(clipiqa, "score_video", fake_score)
    out = tmp / "new" / "o.json"
    assert clipiqa.main(["--video", "v.mp4", "--output", str(out), "--sample-step", "2"]) == 0
    assert calls == [(Path("v.mp4"), "auto", 2, 640, None)]
    text = dumps({"metric": "clipiqa+", "score": 0.5, "video": "猫.mp4"})
    assert out.read_text() == text
    captured = capsys.readouterr()
    assert captured.out == text and captured.err == "model noise\n"
    assert clipiqa.main(["--video", "v.mp4", "--output", str(out)]) == 1
    assert capsys.readouterr().err == f"CLIP-IQA+ failed: Output already exists; choose a new filename: {out}\n"
    assert len(calls) == 1
    assert clipiqa.main(["--video", "v.mp4"]) == 0
    assert capsys.readouterr().out == text


def test_clipiqa_main_rejects_non_finite(monkeypatch, capsys):
    monkeypatch.setattr(clipiqa, "score_video", lambda *a: {"score": float("nan")})
    assert clipiqa.main(["--video", "v.mp4"]) == 1
    assert capsys.readouterr().err.startswith("CLIP-IQA+ failed: Out of range float values")


@pytest.mark.parametrize("argv", [["--video", "v", "--sample-step", "0"], ["--video", "v", "--max-side", "-1"], []])
def test_clipiqa_argument_errors(argv):
    with pytest.raises(SystemExit) as exit_info:
        clipiqa.main(argv)
    assert exit_info.value.code == 2


# --- videophy_joint -------------------------------------------------------

def test_videophy_aggregate():
    pc = [{"videopath": "/v/a.mp4", "pc": "4"}, {"videopath": "/v/b.mp4", "pc": 5}]
    sa = [{"videopath": "/v/b.mp4", "sa": 3.0}, {"videopath": "/v/a.mp4", "sa": "5"}]
    assert videophy_joint.aggregate(pc, sa) == {
        "metric": "video_phy_joint_analysis", "rule": "SA >= 4 and PC >= 4", "higher_is_better": True,
        "joint_rate": 0.5, "rows": [{"video": "/v/a.mp4", "pc": 4, "sa": 5, "joint_pass": True},
                                    {"video": "/v/b.mp4", "pc": 5, "sa": 3, "joint_pass": False}]}


@pytest.mark.parametrize("pc, sa, message", [
    ([{"videopath": "/a", "pc": 4}], [{"videopath": "/b", "sa": 4}], "PC/SA video sets differ"),
    ([{"videopath": "/a", "pc": 4}, {"videopath": "/a", "pc": 4}], [{"videopath": "/a", "sa": 4}], "Duplicate PC video key"),
    ([{"videopath": "/a", "pc": 4.5}], [{"videopath": "/a", "sa": 4}], r"PC row 0 score must be a finite integer in \[1,5\]"),
    ([{"videopath": "/a", "pc": True}], [{"videopath": "/a", "sa": 4}], "PC row 0 score must be an integer"),
    ([{"videopath": "/a", "pc": "x"}], [{"videopath": "/a", "sa": 4}], "non-numeric score"),
    ([{"videopath": "/a"}], [{"videopath": "/a", "sa": 4}], "needs columns"),
    ([], [{"videopath": "/a", "sa": 4}], "at least one score row"),
    ("rows", [], "must be a collection"),
    ([{"videopath": " ", "pc": 4}], [{"videopath": "/a", "sa": 4}], "non-empty path string"),
])
def test_videophy_aggregate_errors(pc, sa, message):
    with pytest.raises(ValueError, match=message):
        videophy_joint.aggregate(pc, sa)


def test_videophy_main(tmp, capsys):
    (tmp / "pc.csv").write_text("videopath,pc\na.mp4,4\n")
    (tmp / "sa.json").write_text(json.dumps({"rows": [{"videopath": "a.mp4", "sa": 5}]}))
    out = tmp / "new" / "joint.json"
    argv = ["--pc", str(tmp / "pc.csv"), "--sa", str(tmp / "sa.json"), "--output", str(out), "--root", "/data"]
    assert videophy_joint.main(argv) == 0
    expected = {"metric": "video_phy_joint_analysis", "rule": "SA >= 4 and PC >= 4", "higher_is_better": True,
                "joint_rate": 1.0, "rows": [{"video": "a.mp4", "pc": 4, "sa": 5, "joint_pass": True}]}
    assert out.read_text() == dumps(expected)
    assert capsys.readouterr().out == dumps(expected)
    assert videophy_joint.main(argv) == 1
    assert capsys.readouterr().err == f"VideoPhy-2 Joint aggregation failed: Output already exists: {out}\n"
    (tmp / "bad.csv").write_text("videopath,pc\na.mp4,4,9\n")
    assert videophy_joint.main(["--pc", str(tmp / "bad.csv"), "--sa", str(tmp / "sa.json"), "--output", str(tmp / "x.json")]) == 1
    assert "CSV row width does not match" in capsys.readouterr().err
