#!/usr/bin/env python3
"""Run the pinned VBench competition CLIP Score implementation.

The scoring function is imported from the official ``competitions/clip_score.py``
file.  This script only builds the one-video metadata JSON, checks the
official CLIP checkpoint, and validates the returned structure.  It does not
implement or rescale CLIP Score.

The upstream competition implementation calls ``clip.load('ViT-B/32')`` and
therefore expects the official OpenAI CLIP cache layout::

    <clip-home>/.cache/clip/ViT-B-32.pt

Pass ``--clip-home`` as the directory that contains ``.cache/clip``.  No
download is attempted by this wrapper.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from percept_harness.video_metrics._runtime import VBENCH_REVISION, run_logged, subprocess_env, verify_source

CLIP_FILENAME = "ViT-B-32.pt"
SOURCE_FILES = ("competitions/clip_score.py", "vbench/__init__.py")


def verify_clip_cache(clip_home: Path) -> Path:
    home = clip_home.expanduser().resolve(strict=True)
    model = home / ".cache" / "clip" / CLIP_FILENAME
    if not model.is_file() or model.stat().st_size == 0:
        raise FileNotFoundError(
            f"Official CLIP cache file is missing (no automatic download): {model}"
        )
    return model


def build_full_info(video: Path, prompt: str) -> list[dict]:
    if video.suffix != ".mp4":
        raise ValueError("This wrapper requires one lowercase .mp4 file")
    if not isinstance(prompt, str) or not prompt.strip() or prompt.strip() == "None":
        raise ValueError("Prompt must be the original non-empty generation instruction")
    return [{"prompt_en": prompt, "dimension": ["clip_score"],
             "video_list": [str(video)]}]


def validate_result(path: Path, video: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 2:
        raise ValueError("Official CLIP Score result must be [aggregate, video_results]")
    aggregate, rows = data
    if isinstance(aggregate, bool) or not isinstance(aggregate, (int, float)) or not math.isfinite(float(aggregate)):
        raise ValueError("Official CLIP Score aggregate is not finite")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("Expected one official per-video CLIP Score result")
    row = rows[0]
    if Path(row.get("video_path", "")).resolve() != video:
        raise ValueError("Official result refers to a different video")
    raw = row.get("video_results")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError("Official CLIP Score video result is not finite")
    if not math.isclose(float(aggregate), float(raw), rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Official aggregate and single-video score differ")
    return {"score": float(aggregate), "official_video_score": raw, "higher_is_better": True}


def run_official(*, vbench_root: Path, videos_path: Path, prompt: str,
                 clip_home: Path, output_dir: Path, check_only: bool = False) -> dict:
    root = vbench_root.expanduser().resolve(strict=True)
    video = videos_path.expanduser().resolve(strict=True)
    if not video.is_file():
        raise ValueError(f"Expected a video file: {video}")
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if output == root or root in output.parents:
        raise ValueError("Keep output outside the pinned source tree")
    source = verify_source(root, SOURCE_FILES, {"expected_revision": VBENCH_REVISION})
    model = verify_clip_cache(clip_home)
    info = build_full_info(video, prompt)
    metadata = {
        "wrapper": "score_vbench_clip_score.py", "source": source,
        "video": str(video),
        "prompt": prompt, "clip_model": str(model),
        "status": "preflight_passed",
    }
    if check_only:
        metadata["official_full_info"] = info
        return metadata

    output.mkdir(parents=True, exist_ok=False)
    info_path = output / "official_full_info.json"
    result_path = output / "clip_score_result.json"
    log_path = output / "vbench_clip_score.log"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Call the unmodified official competition function.  The tiny runner is
    # orchestration only; all frame sampling, CLIP encoding and averaging stay
    # in competitions/clip_score.py.
    runner = (
        "import json,sys,torch; "
        "from competitions.clip_score import compute_clip_score; "
        "result=compute_clip_score(sys.argv[1], torch.device('cuda'), []); "
        "json.dump(result, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)"
    )
    env = subprocess_env(prepend={"PYTHONPATH": str(root)},
                         values={"HOME": str(clip_home.expanduser().resolve()), "PYTHONNOUSERSITE": "1"})
    command = [sys.executable, "-c", runner, str(info_path), str(result_path)]
    run_logged(command, log=log_path, label="Official CLIP Score", cwd=root, env=env)
    score = validate_result(result_path, video)
    metadata.update(status="complete", score=score,
                    official_result=str(result_path), log=str(log_path),
                    command=command)
    (output / "wrapper_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--videos-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--clip-home", type=Path, required=True,
                        help="Directory containing .cache/clip/ViT-B-32.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_official(**vars(args)), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"VBench CLIP Score wrapper failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
