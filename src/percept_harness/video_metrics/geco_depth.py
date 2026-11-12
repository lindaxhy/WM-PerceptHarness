"""Score one video with the depth column of the official GeCo-Eval evaluator."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import tempfile

from ._runtime import run_cli, run_logged, single_row, verify_source, write_json_exclusive

REVISION = "b66d1c68647ba75b5e09d05755cedac79a435bd5"
EVALUATOR = "GeCo-Eval_evaluation.py"
# A custom model key makes upstream use our video's FPS instead of its model lookup.
MODEL = "Gen_PerceptHarness"
CATEGORY = "indoor_prompts"  # Upstream directory label only, independent of scoring.
CLIP = "clip"


def read_depth(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        row = single_row(list(csv.DictReader(stream)), "Expected one official GeCo result")
    if (row["clip_id"], row["category"]) != (CLIP, CATEGORY):
        raise ValueError("Official GeCo result refers to a different clip")
    score = float(row["depth"])
    if not math.isfinite(score) or score < 0:
        raise ValueError(f"No valid GeCo depth error in {path}. See geco.log")
    return score


def run_official(*, geco_root: Path, geco_python: Path, videos_path: Path,
                 output_dir: Path) -> dict:
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
        frames = Path(scratch) / MODEL / CATEGORY / CLIP
        frames.mkdir(parents=True)
        # Normalize variable-rate timestamps onto the CFR grid expected by GeCo.
        run_logged(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0",
                    "-vf", f"fps={fps}", str(frames / "frame_%09d.png")],
                   log=output / "decode.log", label="Frame extraction")
        if not (frames / "frame_000000002.png").is_file():
            raise ValueError("GeCo needs at least two decoded frames")
        run_logged([str(python), EVALUATOR, "--frames_root", scratch, "--models", MODEL,
                    "--categories", CATEGORY, "--default_fps", str(float(fps)),
                    "--geometry", "vggt", "--flow", "ufm", "--flow_model", "base",
                    "--csv_dir", str(output)],
                   log=output / "geco.log", label="Official GeCo-Eval", cwd=root)
    csv_path = output / f"{MODEL}.csv"
    result = {"metric": "geco_depth_error", "score": read_depth(csv_path),
              "higher_is_better": False, "source": source, "video": str(video),
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
    args = parser.parse_args(argv)
    return run_cli("GeCo depth error", lambda: run_official(**vars(args)))
