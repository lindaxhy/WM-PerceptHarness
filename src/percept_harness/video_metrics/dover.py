#!/usr/bin/env python3
"""Run one video through the pinned, unchanged DOVER CLI with -f fusion.

Only Python's standard library is needed for --help and file preflight.  Model
dependencies and both checkpoints must be prepared separately.  This wrapper
does not download files, sample frames, run a replacement model, or fuse scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys

try:
    from ._runtime import package_versions, subprocess_env, verify_source
except ImportError:  # Run directly as a file by an older upstream Python.
    from _runtime import package_versions, subprocess_env, verify_source

DOVER_COMMIT = "f1ddc96215bc7fbcf8f315c65d47905f339c3419"
DOVER_ORIGIN = "https://github.com/VQAssessment/DOVER"
CONVNEXT_FILENAME = "convnext_tiny_1k_224_ema.pth"
SCORE_LABEL = "Normalized fused overall score (scale in [0,1]):"
RUNTIME_PACKAGES = (
    "torch", "torchvision", "numpy", "PyYAML", "decord", "opencv-python",
    "timm", "einops", "scikit-video", "scipy", "tqdm",
)
SOURCE_FILES = ("evaluate_one_video.py", "dover.yml", "dover/__init__.py")


def checkpoint_record(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required local checkpoint missing/empty; no download attempted: {path}")
    return {"path": str(path), "bytes": path.stat().st_size}


def inspect_runtime(device: str, allow_unverified_runtime: bool, gpu: str | None = None) -> dict:
    versions = package_versions(RUNTIME_PACKAGES)
    problems = [f"Missing package: {name}" for name, value in versions.items() if value is None]
    original_torch = bool(re.match(r"^1\.13\.", versions["torch"] or ""))
    if not original_torch and not allow_unverified_runtime:
        problems.append("Expected Torch 1.13.x; a newer environment needs --allow-unverified-runtime")
    if original_torch and not (versions["torchvision"] or "").startswith("0.14."):
        problems.append("Torch 1.13.x requires the matching torchvision 0.14.x series")
    # Probe in a fresh child so CUDA_VISIBLE_DEVICES remapping is effective
    # even if the wrapper is called repeatedly from a process importing Torch.
    # CPU preflight is metadata-only, not a model-loading validation.
    cuda = None
    if device.startswith("cuda") and versions["torch"]:
        try:
            probe = subprocess.run(
                [sys.executable, "-c", "import json,sys,torch; "
                 "i=int(sys.argv[1]); "
                 "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                 "print(json.dumps({'name':torch.cuda.get_device_name(i), "
                 "'capability':list(torch.cuda.get_device_capability(i)), "
                 "'compiled_arches':torch.cuda.get_arch_list()}))",
                 device.partition(":")[2] or "0"],
                env=subprocess_env(gpu=gpu), capture_output=True, text=True, timeout=30,
            )
            if probe.returncode:
                raise RuntimeError(probe.stderr[-2000:] or f"CUDA probe exited {probe.returncode}")
            cuda = json.loads(probe.stdout)
            if cuda["capability"][0] >= 12 and original_torch:
                problems.append("Torch 1.13 CUDA wheels do not support this Blackwell GPU; use an isolated CPU check or a separately validated modern runtime")
        except Exception as error:
            problems.append(f"CUDA preflight failed: {error}")
    return {"python": sys.version, "executable": sys.executable, "packages": versions,
            "torch_1_13": original_torch, "allow_unverified_runtime": allow_unverified_runtime,
            "cuda": cuda, "problems": problems, "model_loading_tested": False}


def parse_fused_score(log_text: str) -> float:
    lines = [line.strip() for line in log_text.splitlines() if SCORE_LABEL in line]
    if len(lines) != 1 or not lines[0].startswith(SCORE_LABEL):
        raise ValueError("Expected exactly one official normalized fused score line")
    value_text = lines[0][len(SCORE_LABEL):].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value_text):
        raise ValueError("Official fused score is not a plain finite number")
    value = float(value_text)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Official fused score is outside [0, 1]")
    return value


def build_command(root: Path, video: Path, device: str) -> list[str]:
    return [sys.executable, "-B", "-u", str(root / "evaluate_one_video.py"),
            "-o", str(root / "dover.yml"), "-v", str(video), "-d", device, "-f"]


def _save(path: Path, payload: dict) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def run_official(*, dover_root: Path, video: Path, torch_home: Path, output_dir: Path,
                 device: str = "cpu", gpu: str | None = None, check_only: bool = False,
                 allow_unverified_runtime: bool = False) -> dict:
    output_dir = output_dir.expanduser().absolute()
    # Exclusive directory ownership prevents reading stale logs or results.
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    log_path = output_dir / "dover.log"
    result = {"metric": "dover_fused_official", "score": None, "higher_is_better": True,
              "status": "preflight", "official_inference_started": False,
              "check_only": check_only, "started_at": datetime.now(timezone.utc).isoformat(),
              "source_commit": DOVER_COMMIT, "device": device, "log": str(log_path),
              "protocol": "Unchanged evaluate_one_video.py -f with pinned dover.yml",
              "runtime_validation": "No GPU/CPU model inference has been validated by this preparation package"}
    _save(result_path, result)
    try:
        root = dover_root.expanduser().resolve(strict=True)
        video = video.expanduser().resolve(strict=True)
        torch_home = torch_home.expanduser().resolve(strict=True)
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError("--video must be a nonempty single video file")
        if not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise ValueError("--device must be cpu, cuda, or cuda:N")
        result["source"] = verify_source(root, SOURCE_FILES, {"origin": DOVER_ORIGIN, "expected_commit": DOVER_COMMIT})
        result["dover_root"] = str(root)
        result["video"] = {"path": str(video), "bytes": video.stat().st_size}
        # dover.yml intentionally remains unchanged.  Use a local symlink at
        # this exact location if the full checkpoint lives on a model volume.
        checkpoints = {
            "dover": checkpoint_record(root / "pretrained_weights" / "DOVER.pth"),
            "convnext_initialization": checkpoint_record(torch_home / "hub" / "checkpoints" / CONVNEXT_FILENAME),
        }
        result["checkpoints"] = checkpoints
        if gpu is not None and device.startswith("cuda:") and device != "cuda:0":
            raise ValueError("With --gpu, use --device cuda or cuda:0 (the visible device is remapped)")
        result["runtime"] = inspect_runtime(device, allow_unverified_runtime, gpu)
        if result["runtime"]["problems"]:
            raise RuntimeError("; ".join(result["runtime"]["problems"]))
        env = subprocess_env(gpu=gpu, values={"TORCH_HOME": str(torch_home), "PYTHONDONTWRITEBYTECODE": "1",
                                              "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1"})
        result["environment"] = {key: env.get(key) for key in ("TORCH_HOME", "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "WANDB_MODE", "HF_HUB_OFFLINE")}
        command = build_command(root, video, device)
        result["command"] = command
        result["working_directory"] = str(root)
        if check_only:
            result["status"] = "preflight_passed_not_scored"
            return result
        result["official_inference_started"] = True
        result["status"] = "running"
        _save(result_path, result)
        with log_path.open("xb") as stream:
            completed = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"Official DOVER exited with {completed.returncode}; inspect {log_path}")
        try:
            result["score"] = parse_fused_score(log_path.read_text(encoding="utf-8", errors="replace"))
        except ValueError as error:
            result["status"] = "official_finished_score_unparsed"
            result["error"] = str(error)
            return result
        result["status"] = "scored" if result["runtime"]["torch_1_13"] else "scored_unverified_runtime"
        return result
    except Exception as error:
        result["status"] = "failed" if result["official_inference_started"] else "preflight_failed"
        result["score"] = None
        result["error"] = str(error)
        return result
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save(result_path, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dover-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True,
                        help="Existing TORCH_HOME containing hub/checkpoints/convnext_tiny_1k_224_ema.pth")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, or cuda:N")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for the official process")
    parser.add_argument("--check-only", action="store_true", help="Validate files and environment; never start inference")
    parser.add_argument("--allow-unverified-runtime", action="store_true",
                        help="Explicitly permit a non-Torch-1.13 compatibility experiment; label its output separately")
    args = parser.parse_args(argv)
    try:
        result = run_official(**vars(args))
    except Exception as error:
        print(f"DOVER wrapper failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] in {"scored", "scored_unverified_runtime", "preflight_passed_not_scored"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
