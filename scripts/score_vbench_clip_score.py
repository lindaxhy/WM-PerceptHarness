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
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

VBENCH_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
SOURCE_SHA256 = "e2d235a360ac7fb322404c4a6db9e62e39f6986d93564eb2eb31e2e9ca5004f1"
CLIP_FILENAME = "ViT-B-32.pt"
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
TEXT_SOURCE_SUFFIXES = frozenset({
    ".c", ".cpp", ".cu", ".h", ".ipynb", ".json", ".md", ".py", ".sh",
    ".txt", ".yaml", ".yml",
})
TEXT_SOURCE_NAMES = frozenset({"Dockerfile", "LICENSE", "Makefile"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    files = []
    for directory in (root / "competitions", root / "vbench", root / "vbench2_beta_long"):
        files.extend(path for path in directory.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix not in (".pyc", ".pyo"))
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        payload = path.read_bytes()
        if path.suffix.lower() in TEXT_SOURCE_SUFFIXES or path.name in TEXT_SOURCE_NAMES:
            payload = payload.replace(b"\r\n", b"\n")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def verify_source(root: Path) -> dict:
    actual = source_fingerprint(root)
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"VBench competition source differs from pinned {VBENCH_REVISION}; "
            f"expected {SOURCE_SHA256}, got {actual}"
        )
    return {"revision": VBENCH_REVISION, "source_sha256": actual}


def verify_clip_cache(clip_home: Path, expected_sha256: str | None = None) -> tuple[Path, str]:
    home = clip_home.expanduser().resolve(strict=True)
    model = home / ".cache" / "clip" / CLIP_FILENAME
    if not model.is_file() or model.stat().st_size == 0:
        raise FileNotFoundError(
            f"Official CLIP cache file is missing (no automatic download): {model}"
        )
    actual = sha256(model)
    expected = expected_sha256 or CLIP_SHA256
    if actual != expected:
        raise ValueError(f"CLIP checkpoint SHA256 mismatch: expected {expected}, got {actual}")
    return model, actual


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
                 clip_home: Path, output_dir: Path, check_only: bool = False,
                 clip_sha256: str | None = None) -> dict:
    root = vbench_root.expanduser().resolve(strict=True)
    video = videos_path.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if output == root or root in output.parents:
        raise ValueError("Keep output outside the pinned source tree")
    source = verify_source(root)
    model, model_hash = verify_clip_cache(clip_home, clip_sha256)
    info = build_full_info(video, prompt)
    metadata = {
        "wrapper": "score_vbench_clip_score.py", "source": source,
        "video": str(video), "video_sha256": sha256(video),
        "prompt": prompt, "clip_model": str(model),
        "clip_sha256": model_hash, "status": "preflight_passed",
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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(clip_home.expanduser().resolve())
    env["PYTHONNOUSERSITE"] = "1"
    command = [sys.executable, "-c", runner, str(info_path), str(result_path)]
    with log_path.open("x", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=root, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Official CLIP Score exited with {completed.returncode}; see {log_path}")
    score = validate_result(result_path, video)
    if sha256(video) != metadata["video_sha256"] or sha256(model) != model_hash:
        raise ValueError("Video or CLIP checkpoint changed during scoring")
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
    parser.add_argument("--clip-sha256", help="Override only when matching the official checkpoint release")
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
