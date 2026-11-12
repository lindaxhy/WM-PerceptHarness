import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from percept_harness.cli import main
from percept_harness.video_metrics import geco_depth as geco

FFMPEG = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                            reason="requires FFmpeg and ffprobe")

# Stands in for the upstream neural evaluator: checks the frame layout and rates it is handed
# and writes one CSV row per clip directory, as upstream does.
EVALUATOR = """import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ("frames_root", "models", "categories", "default_fps", "geometry", "flow", "flow_model", "csv_dir"):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--eval_fps")
parser.add_argument("--flow_ckpt")
args = parser.parse_args()
expected = json.loads(Path(__file__).with_name("expected.json").read_text())
clips = sorted(path for path in (Path(args.frames_root) / args.models / args.categories).iterdir())
assert [path.name for path in clips] == sorted(expected["depth"])
assert [len(list(path.glob("*.png"))) for path in clips] == expected["frames"]
assert all(Path(frame).read_bytes()[:4] == b"\\x89PNG" for path in clips for frame in path.glob("*.png"))
assert abs(float(args.default_fps) - expected["default_fps"]) < 1e-8
assert args.eval_fps == expected["eval_fps"] and args.flow_ckpt == expected["flow_ckpt"]
assert (args.geometry, args.flow, args.flow_model) == ("vggt", "ufm", "base")
with (Path(args.csv_dir) / (args.models + ".csv")).open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(["clip_id", "category", "motion", "depth", "fused"])
    for clip, depth in expected["depth"].items():
        writer.writerow([clip, args.categories, 0.6, depth, 0.7])
"""


def test_two_window_plan_matches_the_upstream_sampler():
    # Up to 6 s: the upstream windows, which cover the clip.
    assert geco.window_starts(2.0) == [0.0]
    assert geco.window_starts(5.0) == [0.0, 2.0]
    # Longer: two 3 s windows at 1/3 and 2/3 of T - 3 s, however long the clip is.
    assert geco.window_starts(30.0) == pytest.approx([9.0, 18.0])
    assert geco.window_starts(150.0) == pytest.approx([49.0, 98.0])
    frames = geco.window_frames(720, 24.0, 9.0, 8.0)
    assert frames == list(range(216, 288, 3)) and geco.upstream_keeps_all(len(frames), 8.0)
    # A 6 fps video is sampled at its own rate; the last window ends on the last frame.
    assert geco.window_frames(180, 6.0, 27.0, 6.0) == list(range(162, 180))
    assert geco.upstream_keeps_all(18, 6.0) and not geco.upstream_keeps_all(25, 8.0)


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_depth_is_not_a_score(tmp_path, value):
    path = tmp_path / "result.csv"
    path.write_text(f"clip_id,category,motion,depth,fused\nclip,indoor_prompts,1,{value},2\n")
    with pytest.raises(ValueError, match="No valid GeCo depth"):
        geco.combine([{"clip": "clip", "start_seconds": None, "frames": 8}], geco.read_scores(path), path)


def test_windows_are_weighted_by_frames_and_invalid_ones_are_left_out(tmp_path):
    path = tmp_path / "result.csv"
    path.write_text("clip_id,category,motion,depth,fused\nwindow_0,indoor_prompts,1,0.1,2\n"
                    "window_1,indoor_prompts,1,0.4,2\nwindow_2,indoor_prompts,nan,0.9,2\n")
    windows = [{"clip": f"window_{k}", "start_seconds": float(k), "frames": frames} for k, frames in enumerate((24, 12, 24))]
    assert geco.combine(windows, geco.read_scores(path), path) == pytest.approx(0.2)
    assert [window["depth"] for window in windows] == [0.1, 0.4, None]


def upstream_fixture(tmp_path, monkeypatch, expected):
    root = tmp_path / "upstream"
    (root / "geco").mkdir(parents=True)
    (root / "geco/metric.py").touch()
    (root / geco.EVALUATOR).write_text(EVALUATOR)
    (root / "expected.json").write_text(json.dumps(expected))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-qm", "fixture"], cwd=root, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setattr(geco, "REVISION", revision)
    return root


def make_video(path, frames):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=24000/1001",
                    "-frames:v", str(frames), "-c:v", "mpeg4", str(path)], check=True)


@FFMPEG
def test_long_video_is_scored_on_two_fixed_windows(tmp_path, monkeypatch):
    # Exercise the real decoder and CLI. Only the upstream neural evaluator is a fixture.
    checkpoint = tmp_path / "ufm-base"
    checkpoint.mkdir()
    root = upstream_fixture(tmp_path, monkeypatch, {
        "depth": {"window_0": 0.10, "window_1": 0.14}, "frames": [24, 24], "default_fps": 8.0,
        "eval_fps": "8.0", "flow_ckpt": str(checkpoint.resolve())})
    video = tmp_path / "long clip.mp4"
    make_video(video, 480)  # about 20 s
    output = tmp_path / "scores"
    argv = ["score", "geco-depth", "--geco-root", str(root), "--videos-path", str(video),
            "--geco-python", sys.executable, "--output-dir", str(output), "--ufm-checkpoint", str(checkpoint)]
    assert main(argv) == 0
    result = json.loads((output / "wrapper_metadata.json").read_text())
    assert result["score"] == pytest.approx(0.12) and result["higher_is_better"] is False
    assert result["window_protocol"] == "two" and result["frame_rate"] == "24000/1001"
    assert [window["frames"] for window in result["windows"]] == [24, 24]
    duration = 480 * 1001 / 24000
    assert [window["start_seconds"] for window in result["windows"]] == pytest.approx(
        [(duration - 3) / 3, 2 * (duration - 3) / 3])
    assert Path(result["official_result"]).is_file()
    assert main(argv) == 1  # an existing result must not be overwritten


@FFMPEG
def test_official_selection_hands_every_frame_to_upstream(tmp_path, monkeypatch):
    root = upstream_fixture(tmp_path, monkeypatch, {
        "depth": {"clip": 0.12}, "frames": [8], "default_fps": 24000 / 1001, "eval_fps": None, "flow_ckpt": None})
    video = tmp_path / "short clip.mp4"
    make_video(video, 8)
    output = tmp_path / "scores"
    assert main(["score", "geco-depth", "--geco-root", str(root), "--videos-path", str(video),
                 "--geco-python", sys.executable, "--output-dir", str(output), "--windows", "official"]) == 0
    result = json.loads((output / "wrapper_metadata.json").read_text())
    assert result["score"] == 0.12 and result["window_protocol"] == "official"
