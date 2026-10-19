#!/usr/bin/env python3
"""Score one MP4 through the unmodified VBench 0.1.5 Motion Smoothness CLI.

This wrapper performs no frame sampling, resizing, interpolation or rounding.
CUDA and the trusted AMT-S checkpoint documented in docs/metrics.md are required.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import sysconfig
import tempfile

from percept_harness.video_metrics._runtime import run_vbench, subprocess_env

VBENCH_VERSION = "0.1.5"


def read_score(result_path: Path, video: Path) -> float:
    """Reject incomplete results or results for a different video."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mean, rows = result["motion_smoothness"]
    if len(rows) != 1 or Path(rows[0]["video_path"]).resolve() != video:
        raise ValueError("Expected exactly one result for the requested video")
    score = rows[0]["video_results"]
    for value in (score, mean):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Expected a numeric Motion Smoothness score")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid Motion Smoothness score")
    if not math.isclose(score, mean, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Single-video score and aggregate disagree")
    return float(score)


def score_video(video: Path, cache: Path, output: Path, gpu: str | None = None) -> dict:
    video = video.expanduser().resolve(strict=True)
    cache = cache.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not video.is_file() or video.suffix != ".mp4":
        raise ValueError("VBench Motion Smoothness requires a lowercase .mp4 file")
    version = importlib.metadata.version("vbench")
    if version != VBENCH_VERSION:
        raise RuntimeError(f"Expected vbench=={VBENCH_VERSION}; found {version}")
    checkpoint = cache / "amt_model" / "amt-s.pth"
    from percept_harness.video_metrics import check_weights
    check_weights("motion", cache)
    executable = Path(sysconfig.get_path("scripts")) / ("vbench.exe" if os.name == "nt" else "vbench")
    if not executable.is_file():
        raise FileNotFoundError(f"Official VBench CLI not found: {executable}")

    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep official output and logs; a fresh directory prevents stale-result reuse.
    run_dir = Path(tempfile.mkdtemp(prefix=output.stem + ".vbench-", dir=output.parent))
    log = run_dir / "vbench.log"
    env = subprocess_env(venv_path=True, alloc_conf=True, gpu=gpu, unset=("TORCH_FORCE_WEIGHTS_ONLY_LOAD",),
                         values={"VBENCH_CACHE_DIR": str(cache), "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    command = [str(executable), "evaluate", "--videos_path", str(video),
               "--dimension", "motion_smoothness", "--mode", "custom_input",
               "--output_path", str(run_dir)]
    print(f"Official VBench output and log: {run_dir}", file=sys.stderr)
    result_path = run_vbench(command, log=log, output=run_dir, label="VBench", mode="w", env=env,
                             missing=RuntimeError(f"Expected one official result JSON; see {run_dir}"))
    score = read_score(result_path, video)
    payload = {
        "metric": "vbench_motion_smoothness", "score": score,
        "higher_is_better": True, "video": str(video),
        "versions": {name: importlib.metadata.version(name) for name in ("vbench", "torch")},
        "checkpoint": {"path": str(checkpoint)},
        "protocol": "Unmodified VBench 0.1.5 motion_smoothness (AMT-S), custom_input",
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "official_result": str(result_path), "log": str(log), "command": command,
    }
    # Exclusive creation also protects against another process creating the file.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="VBench cache containing amt_model/amt-s.pth")
    parser.add_argument("--output", type=Path, required=True, help="New single-video JSON file; never overwritten")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value; default inherits the environment")
    args = parser.parse_args(argv)
    try:
        result = score_video(args.video, args.cache_dir, args.output, args.gpu)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f"Motion Smoothness failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
