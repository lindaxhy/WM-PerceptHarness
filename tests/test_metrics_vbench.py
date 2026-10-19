import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from percept_harness.video_metrics import vbench_backend as vb
from percept_harness.video_metrics import vbench_clip as vc

REV = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
BASE_ENV = {"PATH": "/usr/bin", "HOME": "/home/u", "PYTHONPATH": "/old",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}
VENV_BIN = str(Path(sys.executable).parent)


def child_env(kwargs):
    env = dict(kwargs["env"])
    env.pop("PYTEST_CURRENT_TEST", None)  # pytest writes into the patched os.environ
    return env


def dumps(value, newline=True, **kwargs):
    return json.dumps(value, ensure_ascii=False, indent=2, **kwargs) + ("\n" if newline else "")


class FakeRun:
    def __init__(self, on_call=None, returncode=0):
        self.calls, self.on_call, self.returncode = [], on_call, returncode

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.on_call:
            self.on_call(command, kwargs)
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr="")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(BASE_ENV))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))


@pytest.fixture
def vbench(tmp_path):
    tmp = tmp_path.resolve()
    root = tmp / "VBench"
    (root / "vbench").mkdir(parents=True)
    (root / "evaluate.py").write_text("")
    (root / "vbench" / "__init__.py").write_text("")
    (root / "competitions").mkdir()
    (root / "competitions" / "clip_score.py").write_text("")
    cache = tmp / "cache"
    checkpoint = cache / "pyiqa_model" / "musiq_spaq_ckpt-358bb6af.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    video = tmp / "clip.mp4"
    video.write_bytes(b"v")
    clip_home = tmp / "cliphome"
    model = clip_home / ".cache" / "clip" / "ViT-B-32.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"m")
    return SimpleNamespace(tmp=tmp, root=root, cache=cache, video=video, out=tmp / "out", clip_home=clip_home, model=model)


def official_writer(payload):
    def write(command, kwargs):
        output = Path(command[command.index("--output_path") + 1])
        (output / "clip_eval_results.json").write_text(json.dumps(payload))
        kwargs["stdout"].write("official log\n")
    return write


def iq_result(video, aggregate=0.5, raw=50.0):
    return {"imaging_quality": [aggregate, [{"video_path": str(video), "video_results": raw}]]}


# --- vbench_backend -------------------------------------------------------

def run_args(v, **overrides):
    args = dict(vbench_root=v.root, videos_path=v.video, dimensions=["imaging_quality"],
                output_dir=v.out, cache_dir=v.cache)
    args.update(overrides)
    return args


def test_backend_check_only_metadata(vbench, env):
    result = vb.run_official(**run_args(vbench), check_only=True)
    command = [sys.executable, str(vbench.root / "evaluate.py"), "--videos_path", str(vbench.video),
               "--dimension", "imaging_quality", "--mode", "custom_input", "--output_path", str(vbench.out),
               "--load_ckpt_from_local", "True", "--imaging_quality_preprocessing_mode", "longer"]
    assert result == {
        "wrapper": "score_vbench_official.py", "source": {"expected_revision": REV},
        "video": str(vbench.video), "dimensions": ["imaging_quality"], "mode": "custom_input",
        "prompt": None, "cache_dir": str(vbench.cache),
        "weights": {"pyiqa_model/musiq_spaq_ckpt-358bb6af.pth": str(vbench.cache / "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth")},
        "torch_home": str(vbench.cache / "torch"), "cuda_visible_devices": None,
        "static_subset_ack": False, "allow_trusted_pickle": False,
        "imaging_quality_preprocessing_mode": "longer", "command": command, "status": "preflight_passed"}
    assert not vbench.out.exists()


@pytest.mark.parametrize("overrides, error, message", [
    ({"dimensions": ["motion_smoothness"]}, ValueError, "Supported dimensions"),
    ({"dimensions": []}, ValueError, "Supported dimensions"),
    ({"imaging_quality_preprocessing_mode": "bogus"}, ValueError, "Invalid imaging-quality"),
    ({"dimensions": ["overall_consistency"]}, ValueError, "overall_consistency requires --prompt"),
    ({"dimensions": ["temporal_flickering"]}, ValueError, "temporal_flickering requires --static-subset-ack"),
    ({"gpu": "0,1"}, ValueError, "Select one GPU"),
    ({"gpu": " "}, ValueError, "Select one GPU"),
    ({"cache_dir": None}, ValueError, "Specify an explicit --cache-dir"),
    ({"prompt": "None"}, ValueError, "reserved literal"),
    ({"prompt": "a", "prompt_file": Path("x.json")}, ValueError, "mutually exclusive"),
])
def test_backend_preflight_errors(vbench, env, overrides, error, message):
    with pytest.raises(error, match=message):
        vb.run_official(**run_args(vbench, **overrides), check_only=True)


def test_backend_preflight_path_errors(vbench, env):
    vbench.out.mkdir()
    with pytest.raises(FileExistsError, match="Output directory already exists"):
        vb.run_official(**run_args(vbench), check_only=True)
    with pytest.raises(ValueError, match="outside the pinned source tree"):
        vb.run_official(**run_args(vbench, output_dir=vbench.root / "out"), check_only=True)
    with pytest.raises(ValueError, match="Torch hub cache outside"):
        vb.run_official(**run_args(vbench, output_dir=vbench.tmp / "o2", torch_home=vbench.root / "t"), check_only=True)
    other = vbench.tmp / "clip.MP4"
    other.write_bytes(b"v")
    with pytest.raises(ValueError, match="lowercase .mp4"):
        vb.run_official(**run_args(vbench, output_dir=vbench.tmp / "o3", videos_path=other), check_only=True)
    (vbench.root / "evaluate.py").unlink()
    with pytest.raises(FileNotFoundError, match="Missing required source file: .*evaluate.py"):
        vb.run_official(**run_args(vbench, output_dir=vbench.tmp / "o4"), check_only=True)


def test_backend_cache_errors(vbench, env):
    with pytest.raises(FileNotFoundError, match="Pre-provision model/cache file"):
        vb.run_official(**run_args(vbench, dimensions=["aesthetic_quality"]), check_only=True)
    manifest = vbench.tmp / "m.json"
    manifest.write_text('["../escape"]')
    with pytest.raises(ValueError, match="cache-relative"):
        vb.run_official(**run_args(vbench, weights_manifest=manifest), check_only=True)
    manifest.write_text('[1]')
    with pytest.raises(ValueError, match="JSON list of cache-relative"):
        vb.run_official(**run_args(vbench, weights_manifest=manifest), check_only=True)
    with pytest.raises(ValueError, match="--allow-trusted-pickle requires local model files"):
        vb.run_official(**run_args(vbench, dimensions=["temporal_flickering"], static_subset_ack=True,
                                   allow_trusted_pickle=True), check_only=True)


def test_backend_subject_consistency_cache(vbench):
    dino = vbench.cache / "dino_model" / "dino_vitbase16_pretrain.pth"
    dino.parent.mkdir(parents=True)
    dino.write_bytes(b"d")
    with pytest.raises(ValueError, match="explicit Torch hub cache"):
        vb.verify_cache(vbench.cache, ["subject_consistency"], None)
    torch_home = vbench.tmp / "torch"
    with pytest.raises(FileNotFoundError, match="DINO source checkout"):
        vb.verify_cache(vbench.cache, ["subject_consistency"], None, torch_home=torch_home)
    repo = vbench.cache / vb.DINO_REPOSITORY_PREFIX
    repo.mkdir(parents=True)
    (repo / "hubconf.py").write_text("")
    with pytest.raises(FileNotFoundError, match="DINO Torch hub checkpoint"):
        vb.verify_cache(vbench.cache, ["subject_consistency"], None, torch_home=torch_home)
    checkpoint = torch_home / "hub" / "checkpoints" / vb.DINO_TORCH_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"c")
    assert vb.verify_cache(vbench.cache, ["subject_consistency"], None, torch_home=torch_home) == {
        "dino_model/dino_vitbase16_pretrain.pth": str(dino), vb.DINO_REPOSITORY_PREFIX: str(repo),
        "torch_home/hub/checkpoints/dino_vitbase16_pretrain.pth": str(checkpoint)}


def test_backend_prompt_file(vbench):
    mapping = vbench.tmp / "prompts.json"
    mapping.write_text(json.dumps({"clip.mp4": "a cat"}))
    assert vb.resolve_prompt(vbench.video, None, mapping) == "a cat"
    mapping.write_text(json.dumps({"other.mp4": "a cat"}))
    with pytest.raises(ValueError, match="does not match"):
        vb.resolve_prompt(vbench.video, None, mapping)
    mapping.write_text('{"clip.mp4": "a", "clip.mp4": "b"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        vb.resolve_prompt(vbench.video, None, mapping)


def expected_backend_env(v, **extra):
    env = dict(BASE_ENV)
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    env.update({"PATH": VENV_BIN + os.pathsep + "/usr/bin", "VBENCH_CACHE_DIR": str(v.cache),
                "TORCH_HOME": str(v.cache / "torch"), "PYTHONPATH": str(v.root), "PYTHONNOUSERSITE": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1"})
    env.update(extra)
    return env


def test_backend_full_run_env_and_golden(vbench, env, monkeypatch, capsys):
    fake = FakeRun(official_writer(iq_result(vbench.video)))
    monkeypatch.setattr(subprocess, "run", fake)
    result = vb.run_official(**run_args(vbench), gpu="1")
    (command, kwargs), = fake.calls
    assert kwargs["cwd"] == vbench.root
    assert kwargs["stderr"] == subprocess.STDOUT
    assert child_env(kwargs) == expected_backend_env(vbench, CUDA_VISIBLE_DEVICES="1")
    request = {"wrapper": "score_vbench_official.py", "source": {"expected_revision": REV},
               "video": str(vbench.video), "dimensions": ["imaging_quality"], "mode": "custom_input",
               "prompt": None, "cache_dir": str(vbench.cache),
               "weights": {"pyiqa_model/musiq_spaq_ckpt-358bb6af.pth": str(vbench.cache / "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth")},
               "torch_home": str(vbench.cache / "torch"), "cuda_visible_devices": "1",
               "static_subset_ack": False, "allow_trusted_pickle": False,
               "imaging_quality_preprocessing_mode": "longer", "command": command, "status": "preflight_passed"}
    assert (vbench.out / "wrapper_request.json").read_text() == dumps(request, newline=False)
    metadata = {**request, "status": "complete",
                "scores": {"imaging_quality": {"score": 0.5, "official_video_score": 50.0, "higher_is_better": True}},
                "versions": {name: None for name in ("torch", "torchvision", "pyiqa", "openai-clip", "decord", "numpy")},
                "official_result": str(vbench.out / "clip_eval_results.json"), "log": str(vbench.out / "vbench.log")}
    assert result == metadata
    assert (vbench.out / "wrapper_metadata.json").read_text() == dumps(metadata)
    assert (vbench.out / "vbench.log").read_text() == "official log\n"
    assert capsys.readouterr().err == f"Official VBench output and log: {vbench.out}\n"


def test_backend_env_trusted_pickle_and_inherited_alloc(vbench, env, monkeypatch):
    os.environ["PYTORCH_ALLOC_CONF"] = "custom"
    fake = FakeRun(official_writer(iq_result(vbench.video)))
    monkeypatch.setattr(subprocess, "run", fake)
    vb.run_official(**run_args(vbench), allow_trusted_pickle=True)
    env = expected_backend_env(vbench, PYTORCH_ALLOC_CONF="custom", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
    assert child_env(fake.calls[0][1]) == env


def test_backend_run_failures(vbench, env, monkeypatch):
    monkeypatch.setattr(subprocess, "run", FakeRun(returncode=3))
    with pytest.raises(RuntimeError, match=r"Official VBench exited with 3; see .*vbench.log"):
        vb.run_official(**run_args(vbench))
    monkeypatch.setattr(subprocess, "run", FakeRun())
    with pytest.raises(ValueError, match="Expected exactly one official evaluation JSON"):
        vb.run_official(**run_args(vbench, output_dir=vbench.tmp / "o2"))


def test_backend_metric_main(vbench, env, capsys):
    argv = ["--vbench-root", str(vbench.root), "--videos-path", str(vbench.video),
            "--cache-dir", str(vbench.cache), "--output-dir", str(vbench.out), "--check-only"]
    assert vb.metric_main("imaging_quality", argv) == 0
    out = capsys.readouterr().out
    assert out == dumps(vb.run_official(**run_args(vbench), check_only=True))
    assert vb.metric_main("imaging_quality", argv + ["--gpu", "0,1"]) == 1
    assert capsys.readouterr().err == "imaging-quality failed: Select one GPU index or UUID for this single-process wrapper\n"
    for removed in (["--mode", "custom_input"], ["--load-ckpt-from-local"]):
        with pytest.raises(SystemExit) as exit_info:
            vb.metric_main("imaging_quality", argv + removed)
        assert exit_info.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err
    with pytest.raises(TypeError):
        vb.run_official(**run_args(vbench), mode="custom_input", check_only=True)
    with pytest.raises(ValueError, match="Unsupported metric"):
        vb.metric_main("motion_smoothness", [])
    with pytest.raises(SystemExit):
        vb.metric_main("imaging_quality", ["--help"])
    assert "percept imaging-quality" in capsys.readouterr().out


def write_result(tmp, payload, text=None):
    path = tmp / "r_eval_results.json"
    path.write_text(text if text is not None else json.dumps(payload))
    return path


def test_backend_validate_result_good(vbench):
    v = str(vbench.video)
    payload = {"imaging_quality": [0.5, [{"video_path": v, "video_results": 50.0}]],
               "dynamic_degree": [1.0, [{"video_path": v, "video_results": True}]],
               "overall_consistency": [-0.2, [{"video_path": v, "video_results": -0.2}]],
               "aesthetic_quality": [7, [{"video_path": v, "video_results": 7}]]}
    assert vb.validate_result(write_result(vbench.tmp, payload), vbench.video, list(payload)) == {
        "imaging_quality": {"score": 0.5, "official_video_score": 50.0, "higher_is_better": True},
        "dynamic_degree": {"score": 1.0, "official_video_score": True, "higher_is_better": None},
        "overall_consistency": {"score": -0.2, "official_video_score": -0.2, "higher_is_better": True},
        "aesthetic_quality": {"score": 7.0, "official_video_score": 7, "higher_is_better": True}}


@pytest.mark.parametrize("payload, dimensions, message", [
    ({"imaging_quality": [0.5, []]}, ["imaging_quality", "dynamic_degree"], "do not exactly match"),
    ({"imaging_quality": [0.5]}, ["imaging_quality"], "Invalid official result structure: imaging_quality"),
    ({"imaging_quality": [0.5, []]}, ["imaging_quality"], "Expected exactly one video result: imaging_quality"),
    ({"imaging_quality": [0.5, [1]]}, ["imaging_quality"], "Expected exactly one video result"),
    ({"imaging_quality": [0.5, [{"video_path": "/elsewhere.mp4", "video_results": 50}]]}, ["imaging_quality"], "different video: imaging_quality"),
    ({"imaging_quality": [0.5, [{"video_path": 3, "video_results": 50}]]}, ["imaging_quality"], "different video"),
    ({"imaging_quality": [True, [{"video_path": "V", "video_results": 50}]]}, ["imaging_quality"], "Expected a finite numeric imaging_quality aggregate"),
    ({"imaging_quality": [0.5, [{"video_path": "V", "video_results": "50"}]]}, ["imaging_quality"], "Expected a finite numeric imaging_quality video score"),
    ({"imaging_quality": [0.6, [{"video_path": "V", "video_results": 50}]]}, ["imaging_quality"], "ratio mismatch: imaging_quality"),
    ({"dynamic_degree": [1.0, [{"video_path": "V", "video_results": 1.0}]]}, ["dynamic_degree"], "must be Boolean"),
    ({"subject_consistency": [1.5, [{"video_path": "V", "video_results": 1.5}]]}, ["subject_consistency"], "Invalid normalized score: subject_consistency"),
    ({"overall_consistency": [-1.5, [{"video_path": "V", "video_results": -1.5}]]}, ["overall_consistency"], r"must lie in \[-1, 1\]"),
])
def test_backend_validate_result_bad(vbench, payload, dimensions, message):
    text = json.dumps(payload).replace('"V"', json.dumps(str(vbench.video)))
    with pytest.raises(ValueError, match=message):
        vb.validate_result(write_result(vbench.tmp, None, text), vbench.video, dimensions)


def test_backend_validate_result_rejects_nan_and_duplicates(vbench):
    v = json.dumps(str(vbench.video))
    with pytest.raises(ValueError, match="finite numeric"):
        vb.validate_result(write_result(vbench.tmp, None, '{"imaging_quality": [NaN, [{"video_path": %s, "video_results": 1}]]}' % v), vbench.video, ["imaging_quality"])
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        vb.validate_result(write_result(vbench.tmp, None, '{"a": 1, "a": 2}'), vbench.video, ["a"])


# --- vbench_clip ----------------------------------------------------------

PROMPT = "一只猫 jumps"


def clip_args(v, **overrides):
    args = dict(vbench_root=v.root, videos_path=v.video, prompt=PROMPT, clip_home=v.clip_home, output_dir=v.out)
    args.update(overrides)
    return args


def clip_preflight(v):
    return {"wrapper": "score_vbench_clip_score.py", "source": {"expected_revision": REV},
            "video": str(v.video), "prompt": PROMPT, "clip_model": str(v.model), "status": "preflight_passed"}


def test_clip_check_only(vbench):
    info = [{"prompt_en": PROMPT, "dimension": ["clip_score"], "video_list": [str(vbench.video)]}]
    assert vc.run_official(**clip_args(vbench), check_only=True) == {**clip_preflight(vbench), "official_full_info": info}
    assert not vbench.out.exists()


@pytest.mark.parametrize("overrides, error, message", [
    ({"prompt": " "}, ValueError, "original non-empty generation instruction"),
    ({"prompt": "None"}, ValueError, "original non-empty generation instruction"),
])
def test_clip_preflight_errors(vbench, overrides, error, message):
    with pytest.raises(error, match=message):
        vc.run_official(**clip_args(vbench, **overrides), check_only=True)


def test_clip_preflight_path_errors(vbench):
    with pytest.raises(ValueError, match="Keep output outside the pinned source tree"):
        vc.run_official(**clip_args(vbench, output_dir=vbench.root / "o"), check_only=True)
    vbench.model.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Official CLIP cache file is missing"):
        vc.run_official(**clip_args(vbench), check_only=True)
    vbench.model.write_bytes(b"m")
    other = vbench.tmp / "clip.avi"
    other.write_bytes(b"v")
    with pytest.raises(ValueError, match="lowercase .mp4"):
        vc.run_official(**clip_args(vbench, videos_path=other), check_only=True)
    with pytest.raises(ValueError, match="Expected a video file"):
        vc.run_official(**clip_args(vbench, videos_path=vbench.tmp), check_only=True)
    (vbench.root / "competitions" / "clip_score.py").unlink()
    with pytest.raises(FileNotFoundError, match="Missing required source file: .*clip_score.py"):
        vc.run_official(**clip_args(vbench), check_only=True)
    vbench.out.mkdir()
    with pytest.raises(FileExistsError, match="Output directory already exists"):
        vc.run_official(**clip_args(vbench), check_only=True)


def test_clip_full_run_env_and_golden(vbench, env, monkeypatch):
    def write(command, kwargs):
        Path(command[-1]).write_text(json.dumps([0.25, [{"video_path": str(vbench.video), "video_results": 0.25}]]))
    fake = FakeRun(write)
    monkeypatch.setattr(subprocess, "run", fake)
    result = vc.run_official(**clip_args(vbench))
    (command, kwargs), = fake.calls
    info_path, result_path = vbench.out / "official_full_info.json", vbench.out / "clip_score_result.json"
    assert command[:2] == [sys.executable, "-c"] and command[3:] == [str(info_path), str(result_path)]
    assert "from competitions.clip_score import compute_clip_score" in command[2]
    assert kwargs["cwd"] == vbench.root
    assert child_env(kwargs) == {**BASE_ENV, "PYTHONPATH": str(vbench.root) + os.pathsep + "/old",
                                 "HOME": str(vbench.clip_home), "PYTHONNOUSERSITE": "1"}
    info = [{"prompt_en": PROMPT, "dimension": ["clip_score"], "video_list": [str(vbench.video)]}]
    assert info_path.read_text() == dumps(info)
    metadata = {**clip_preflight(vbench), "status": "complete",
                "score": {"score": 0.25, "official_video_score": 0.25, "higher_is_better": True},
                "official_result": str(result_path), "log": str(vbench.out / "vbench_clip_score.log"), "command": command}
    assert result == metadata
    assert (vbench.out / "wrapper_metadata.json").read_text() == dumps(metadata)


def test_clip_env_without_pythonpath(vbench, env, monkeypatch):
    del os.environ["PYTHONPATH"]
    fake = FakeRun(returncode=2)
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError, match=r"Official CLIP Score exited with 2; see .*vbench_clip_score.log"):
        vc.run_official(**clip_args(vbench))
    assert child_env(fake.calls[0][1])["PYTHONPATH"] == str(vbench.root) + os.pathsep


def test_clip_main(vbench, capsys):
    argv = ["--vbench-root", str(vbench.root), "--videos-path", str(vbench.video), "--prompt", PROMPT,
            "--clip-home", str(vbench.clip_home), "--output-dir", str(vbench.out)]
    assert vc.main(argv + ["--check-only"]) == 0
    assert capsys.readouterr().out == dumps(vc.run_official(**clip_args(vbench), check_only=True))
    assert vc.main(argv[:-2] + ["--output-dir", str(vbench.root / "o"), "--check-only"]) == 1
    assert capsys.readouterr().err == "VBench CLIP Score wrapper failed: Keep output outside the pinned source tree\n"
    with pytest.raises(SystemExit) as exit_info:
        vc.main(argv[:4])
    assert exit_info.value.code == 2


def test_clip_validate_result_good(vbench):
    path = write_result(vbench.tmp, [1, [{"video_path": str(vbench.video), "video_results": 1}]])
    assert vc.validate_result(path, vbench.video) == {"score": 1.0, "official_video_score": 1, "higher_is_better": True}


@pytest.mark.parametrize("text, message", [
    ('{"a": 1}', r"must be \[aggregate, video_results\]"),
    ('[1, [], 3]', r"must be \[aggregate, video_results\]"),
    ('[true, []]', "aggregate is not finite"),
    ('[NaN, []]', "aggregate is not finite"),
    ('["1", []]', "aggregate is not finite"),
    ('[1, []]', "Expected one official per-video CLIP Score result"),
    ('[1, [1]]', "Expected one official per-video CLIP Score result"),
    ('[1, [{"video_results": 1}]]', "Official result refers to a different video"),
    ('[1, [{"video_path": "/x.mp4", "video_results": 1}]]', "Official result refers to a different video"),
    ('[1, [{"video_path": V, "video_results": false}]]', "video result is not finite"),
    ('[1, [{"video_path": V, "video_results": Infinity}]]', "video result is not finite"),
    ('[1, [{"video_path": V, "video_results": 2}]]', "Official aggregate and single-video score differ"),
])
def test_clip_validate_result_bad(vbench, text, message):
    path = write_result(vbench.tmp, None, text.replace("V", json.dumps(str(vbench.video))))
    with pytest.raises(ValueError, match=message):
        vc.validate_result(path, vbench.video)


def test_clip_validate_result_non_string_path_is_type_error(vbench):
    path = write_result(vbench.tmp, [1, [{"video_path": 5, "video_results": 1}]])
    with pytest.raises(TypeError):
        vc.validate_result(path, vbench.video)
