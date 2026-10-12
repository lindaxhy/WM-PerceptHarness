#!/usr/bin/env python3
"""Run one video through the pinned, unchanged DOVER CLI with -f fusion.

Only Python's standard library is needed for --help and file preflight.  Model
dependencies and both checkpoints must be prepared separately.  This wrapper
does not download files, sample frames, run a replacement model, or fuse scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

DOVER_COMMIT = "f1ddc96215bc7fbcf8f315c65d47905f339c3419"
DOVER_ORIGIN = "https://github.com/VQAssessment/DOVER"
CONVNEXT_FILENAME = "convnext_tiny_1k_224_ema.pth"
SCORE_LABEL = "Normalized fused overall score (scale in [0,1]):"
# SHA-256 of LF-normalized files from the commit above.  This also verifies
# official tar exports without .git and does not trust a user-supplied revision.
SOURCE_SHA256 = {
    "evaluate_one_video.py": "76226b3509772a4c2159bb0cb3526f5c2c52c841feb09d5cb1cc502427e5b564",
    "dover.yml": "48f35676852ac7fd535338b0e0d45b833eabe8c6b4be9c6a7a4bfdf671f964fe",
    "requirements.txt": "b2288e81d50782082b998e208cedb7dcfe03a9fe429ee187b62850344e1b3787",
    "setup.py": "847929587a1e91297d42f6a9b96cc2427231f3d40031fabda5e17a0f54e1d11b",
    "dover/__init__.py": "7bff0dcd5bc9afc65d4393449aae2d0ec8181b6e9028341c2bd5509c16d7444d",
    "dover/datasets/__init__.py": "118fec1d52cfe3cff8491e7eec4628211c25cff8cc7b456d20b77c2e3f33e2f4",
    "dover/datasets/basic_datasets.py": "cf61875d05ab17dd239f90fa62a60f408260f1248b2ec83ac39b49e0267fd6b5",
    "dover/datasets/dover_datasets.py": "5f017410154117ca1ff93afef2ede1e6360be382c259fe722c34b33c8bc1ea8a",
    "dover/models/__init__.py": "a2f5133edafdae33ef9dcea6e082d3577f1d0a3728fa337d2514703620346a07",
    "dover/models/backbone_get_attention.py": "ad710791bee65fc73c1df5b0c66d09a7adcad26bfe009e62418f4b6b1ceb7d45",
    "dover/models/backbone_v0_1.py": "cc4e99fb65ee591af4ddb3dc333744c536c18b2fec22982eae4f3a7bb1662a68",
    "dover/models/conv_backbone.py": "fa50ec66714ed1191123aeff4572da02cb95f3de8884349194ca6828c81c6e90",
    "dover/models/evaluator.py": "c589b03d6fd5b48d15d7da609c0feddf48d79fbb670d047674afb1314a12bc64",
    "dover/models/head.py": "4f66f792c018c5aa69cf4305a5b6e6e82d2f47cc9aa725c51042f1c29dcccad1",
    "dover/models/swin_backbone.py": "18045b20bf6e97e5dc5cae56ec07746e57b904a04ad4aeeeab28b9ec4f57d990",
    "dover/models/xclip_backbone.py": "0e6b4c29d0a89c89bd52cfce745892b43aeed1d7263501f2890775c36cb8d226",
    "dover/version.py": "c65276beb52b51602a72f6cba52fa1a6212c33b97fee994ea2962097a6ec0473",
}
RUNTIME_PACKAGES = (
    "torch", "torchvision", "numpy", "PyYAML", "decord", "opencv-python",
    "timm", "einops", "scikit-video", "scipy", "tqdm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(root: Path) -> dict:
    expected_python = {name for name in SOURCE_SHA256 if name.startswith("dover/")}
    actual_python = {path.relative_to(root).as_posix() for path in (root / "dover").rglob("*.py")}
    if actual_python != expected_python:
        raise ValueError("DOVER Python source inventory differs from the pinned commit")
    for name, expected in SOURCE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned DOVER source: {path}")
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"DOVER source differs from {DOVER_COMMIT}: {name}")
    return {"origin": DOVER_ORIGIN, "commit": DOVER_COMMIT,
            "verification": "LF-normalized SHA-256 of all runtime Python files, config and requirements",
            "files": SOURCE_SHA256.copy()}


def checkpoint_record(path: Path, expected: str | None = None) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required local checkpoint missing/empty; no download attempted: {path}")
    digest = sha256(path)
    if expected is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError("Expected checkpoint SHA-256 must contain exactly 64 hexadecimal characters")
        if digest != expected.lower():
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
            "expected_sha256_checked": expected is not None}


def inspect_runtime(device: str, allow_unverified_runtime: bool, gpu: str | None = None) -> dict:
    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
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
            probe_env = os.environ.copy()
            if gpu is not None:
                probe_env["CUDA_VISIBLE_DEVICES"] = gpu
            probe = subprocess.run(
                [sys.executable, "-c", "import json,sys,torch; "
                 "i=int(sys.argv[1]); "
                 "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                 "print(json.dumps({'name':torch.cuda.get_device_name(i), "
                 "'capability':list(torch.cuda.get_device_capability(i)), "
                 "'compiled_arches':torch.cuda.get_arch_list()}))",
                 device.partition(":")[2] or "0"],
                env=probe_env, capture_output=True, text=True, timeout=30,
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
                 checkpoint_sha256: str | None = None, convnext_sha256: str | None = None,
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
        result["source"] = verify_source(root)
        result["dover_root"] = str(root)
        result["video"] = {"path": str(video), "sha256": sha256(video), "bytes": video.stat().st_size}
        # dover.yml intentionally remains unchanged.  Use a local symlink at
        # this exact location if the full checkpoint lives on a model volume.
        checkpoints = {
            "dover": checkpoint_record(root / "pretrained_weights" / "DOVER.pth", checkpoint_sha256),
            "convnext_initialization": checkpoint_record(torch_home / "hub" / "checkpoints" / CONVNEXT_FILENAME, convnext_sha256),
        }
        result["checkpoints"] = checkpoints
        if gpu is not None and device.startswith("cuda:") and device != "cuda:0":
            raise ValueError("With --gpu, use --device cuda or cuda:0 (the visible device is remapped)")
        result["runtime"] = inspect_runtime(device, allow_unverified_runtime, gpu)
        if result["runtime"]["problems"]:
            raise RuntimeError("; ".join(result["runtime"]["problems"]))
        env = os.environ.copy()
        env["TORCH_HOME"] = str(torch_home)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["WANDB_MODE"] = "disabled"
        env["HF_HUB_OFFLINE"] = "1"
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
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
        result["log_sha256"] = sha256(log_path)
        if completed.returncode:
            raise RuntimeError(f"Official DOVER exited with {completed.returncode}; inspect {log_path}")
        verify_source(root)
        if sha256(video) != result["video"]["sha256"]:
            raise RuntimeError("Video changed during inference; no score accepted")
        for item in checkpoints.values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError("Checkpoint changed during inference; no score accepted")
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
    parser.add_argument("--checkpoint-sha256", help="Expected SHA-256 from your verified DOVER.pth acquisition")
    parser.add_argument("--convnext-sha256", help="Expected SHA-256 from your verified ConvNeXt acquisition")
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
