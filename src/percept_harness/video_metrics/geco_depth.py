"""Score one video with the depth column of the official GeCo-Eval evaluator."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import math
import os
from pathlib import Path
import subprocess
import tempfile

from ._runtime import run_cli, run_logged, verify_source, write_json_exclusive

REVISION = "b66d1c68647ba75b5e09d05755cedac79a435bd5"
EVALUATOR = "GeCo-Eval_evaluation.py"
# A custom model key makes upstream use the frame rate we pass instead of its model lookup.
MODEL = "Gen_PerceptHarness"
CATEGORY = "indoor_prompts"  # Upstream directory label only, independent of scoring.
CLIP = "clip"
WINDOW_SECONDS = 3.0  # upstream --win_sec default
EVAL_FPS = 8.0  # upstream --eval_fps default
COLUMNS = ("motion", "depth", "fused")


def window_starts(duration: float) -> list[float]:
    """Start times of the 3 s windows scored by the default "two" protocol.

    Upstream covers a clip with ceil(T / 3) windows and, beyond four, stretches each to T / 4, so a
    window of a long clip holds hundreds of frames that are all paired with each other. Here windows
    stay 3 s: clips up to 6 s keep the upstream windows (they cover the clip), longer clips use two
    windows starting at 1/3 and 2/3 of T - 3 s, away from the conditioning frame and the final frames.
    """
    needed = math.ceil(duration / WINDOW_SECONDS)
    if needed <= 1:
        return [0.0]
    span = duration - WINDOW_SECONDS
    return [0.0, span] if needed == 2 else [span * (k / 3) for k in (1, 2)]


def window_frames(frames: int, fps: float, start: float, eval_fps: float) -> list[int]:
    """Frame indices of one window, as upstream utils.sample_indices_for_window selects them."""
    duration = frames / fps
    count = max(2, math.floor(min(WINDOW_SECONDS, duration) * eval_fps))
    last = max(0.0, duration - 1.0 / fps)
    times = (min(max(start + j / eval_fps, 0.0), last) for j in range(count))
    return sorted({min(frames - 1, max(0, round(t * fps))) for t in times})


def upstream_keeps_all(count: int, eval_fps: float) -> bool:
    """Whether upstream scores a directory of `count` frames at `eval_fps` as one window of all frames."""
    return (count >= 2 and math.ceil(count / eval_fps / WINDOW_SECONDS) <= 1
            and window_frames(count, eval_fps, 0.0, eval_fps) == list(range(count)))


def read_scores(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if any(row["category"] != CATEGORY for row in rows):
        raise ValueError("Official GeCo result refers to a different clip")
    return {row["clip_id"]: {name: float(row[name]) for name in COLUMNS} for row in rows}


def combine(windows: list[dict], scores: dict[str, dict[str, float]], path: Path) -> float:
    """Upstream rule: frame-weighted mean over windows whose three official scores are all finite."""
    if set(scores) != {window["clip"] for window in windows}:
        raise ValueError("Official GeCo result refers to a different clip")
    total = weight = 0.0
    for window in windows:
        row = scores[window.pop("clip")]
        valid = all(math.isfinite(row[name]) for name in COLUMNS) and row["depth"] >= 0
        window["depth"] = row["depth"] if valid else None
        if valid:
            total += row["depth"] * window["frames"]
            weight += window["frames"]
    if not weight:
        raise ValueError(f"No valid GeCo depth error in {path}. See geco.log")
    return total / weight


def frame_count(video: Path, fps: Fraction) -> int:
    done = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-progress", "pipe:1", "-i", str(video), "-map", "0:v:0",
         "-vf", f"fps={fps}", "-f", "null", "-"], check=True, capture_output=True, text=True)
    counts = [line.partition("=")[2] for line in done.stdout.splitlines() if line.startswith("frame=")]
    return int(counts[-1]) if counts else 0


def extract_windows(video: Path, fps: Fraction, root: Path, log: Path) -> tuple[list[dict], float]:
    """Decode only the frames of the scored windows, one upstream clip directory per window."""
    frames = frame_count(video, fps)
    if frames < 2:
        raise ValueError("GeCo needs at least two decoded frames")
    eval_fps = min(EVAL_FPS, float(fps))
    plan = [(start, window_frames(frames, float(fps), start, eval_fps))
            for start in window_starts(frames / float(fps))]
    if not all(upstream_keeps_all(len(indices), eval_fps) for _, indices in plan):
        raise ValueError("GeCo would not score every selected frame of a window")
    wanted = sorted({index for _, indices in plan for index in indices})
    pool = root.parent / "selected"
    pool.mkdir()
    # Renumber the kept frames so that the image writer neither drops nor repeats any of them.
    select = "+".join(f"eq(n\\,{index})" for index in wanted)
    run_logged(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf",
                f"fps={fps},select='{select}',setpts=N/FRAME_RATE/TB", str(pool / "frame_%09d.png")],
               log=log, label="Frame extraction")
    files = sorted(pool.glob("frame_*.png"))
    if len(files) != len(wanted):
        raise ValueError(f"Decoded {len(files)} of {len(wanted)} selected frames")
    decoded = dict(zip(wanted, files))
    windows = []
    for number, (start, indices) in enumerate(plan):
        clip = f"window_{number}"
        target = root / MODEL / CATEGORY / clip
        target.mkdir(parents=True)
        for position, index in enumerate(indices):
            os.symlink(decoded[index], target / f"frame_{position:09d}.png")
        windows.append({"clip": clip, "start_seconds": start, "frames": len(indices)})
    return windows, eval_fps


def run_official(*, geco_root: Path, geco_python: Path, videos_path: Path, output_dir: Path,
                 windows: str = "two", ufm_checkpoint: Path | None = None) -> dict:
    root = geco_root.expanduser().resolve(strict=True)
    source = verify_source(root, (EVALUATOR, "geco/metric.py"), {}, REVISION)
    # Preserve the interpreter's virtualenv path even when it is a symlink.
    python = geco_python.expanduser().absolute()
    video = videos_path.expanduser().resolve(strict=True)
    if not python.is_file() or not video.is_file():
        raise ValueError("Expected a GeCo Python interpreter and a video file")
    output = output_dir.expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("Keep output outside the GeCo checkout")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    )
    fps = Fraction(probe.stdout.strip())
    if fps <= 0:
        raise ValueError(f"Invalid average video frame rate: {fps}")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        frames_root = Path(scratch) / "frames"
        if windows == "two":
            plan, eval_fps = extract_windows(video, fps, frames_root, output / "decode.log")
            # Each directory is exactly one window, sampled at the rate upstream would have used.
            rates = ["--default_fps", str(eval_fps), "--eval_fps", str(eval_fps)]
        else:
            frames = frames_root / MODEL / CATEGORY / CLIP
            frames.mkdir(parents=True)
            # Normalize variable-rate timestamps onto the CFR grid expected by GeCo.
            run_logged(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0",
                        "-vf", f"fps={fps}", str(frames / "frame_%09d.png")],
                       log=output / "decode.log", label="Frame extraction")
            count = len(list(frames.glob("frame_*.png")))
            if count < 2:
                raise ValueError("GeCo needs at least two decoded frames")
            plan, rates = [{"clip": CLIP, "start_seconds": None, "frames": count}], ["--default_fps", str(float(fps))]
        checkpoint = [] if ufm_checkpoint is None else ["--flow_ckpt", str(ufm_checkpoint.expanduser().resolve(strict=True))]
        run_logged([str(python), EVALUATOR, "--frames_root", str(frames_root), "--models", MODEL,
                    "--categories", CATEGORY, *rates, "--geometry", "vggt", "--flow", "ufm",
                    "--flow_model", "base", *checkpoint, "--csv_dir", str(output)],
                   log=output / "geco.log", label="Official GeCo-Eval", cwd=root)
    csv_path = output / f"{MODEL}.csv"
    score = combine(plan, read_scores(csv_path), csv_path)
    result = {"metric": "geco_depth_error", "score": score, "higher_is_better": False,
              "window_protocol": windows, "windows": plan, "source": source, "video": str(video),
              "frame_rate": str(fps), "geometry": "vggt", "flow": "ufm/base",
              "official_result": str(csv_path)}
    write_json_exclusive(output / "wrapper_metadata.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geco-root", type=Path, required=True)
    parser.add_argument("--geco-python", type=Path, required=True,
                        help="Python executable in the upstream GeCo environment")
    parser.add_argument("--videos-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows", choices=("two", "official"), default="two",
                        help="two: at most two fixed 3 s windows (default); "
                             "official: upstream selection, which stretches windows on clips over 12 s")
    parser.add_argument("--ufm-checkpoint", type=Path,
                        help="Local UFM-Base directory or checkpoint, for machines without Hugging Face access")
    args = parser.parse_args(argv)
    return run_cli("GeCo depth error", lambda: run_official(**vars(args)))
