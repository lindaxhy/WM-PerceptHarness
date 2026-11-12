import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from percept_harness.cli import main
from percept_harness.video_metrics import geco_depth as geco


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_depth_is_not_a_score(tmp_path, value):
    path = tmp_path / "result.csv"
    path.write_text(f"clip_id,category,motion,depth,fused\nclip,indoor_prompts,1,{value},2\n")
    with pytest.raises(ValueError, match="No valid GeCo depth"):
        geco.read_depth(path)


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                    reason="requires FFmpeg and ffprobe")
def test_video_to_official_csv_contract(tmp_path, monkeypatch):
    # Exercise the real decoder and CLI. Only the upstream neural evaluator is a fixture.
    root = tmp_path / "upstream"
    (root / "geco").mkdir(parents=True)
    (root / "geco/metric.py").touch()
    (root / geco.EVALUATOR).write_text(
        """import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ("frames_root", "models", "categories", "default_fps", "geometry",
             "flow", "flow_model", "csv_dir"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
frames = Path(args.frames_root) / args.models / args.categories / "clip"
assert len(list(frames.glob("*.png"))) == 8
assert abs(float(args.default_fps) - 24000 / 1001) < 1e-8
assert (args.geometry, args.flow, args.flow_model) == ("vggt", "ufm", "base")
with (Path(args.csv_dir) / (args.models + ".csv")).open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(["clip_id", "category", "motion", "depth", "fused"])
    writer.writerow(["clip", args.categories, 0.6, 0.12, 0.7])
"""
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-qm", "fixture"], cwd=root, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setattr(geco, "REVISION", revision)

    video = tmp_path / "short clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=size=32x32:rate=24000/1001", "-frames:v", "8",
                    "-c:v", "mpeg4", str(video)], check=True)
    output = tmp_path / "scores"
    argv = ["score", "geco-depth", "--geco-root", str(root), "--videos-path", str(video),
            "--geco-python", sys.executable, "--output-dir", str(output)]
    assert main(argv) == 0
    result = json.loads((output / "wrapper_metadata.json").read_text())
    assert result["score"] == 0.12
    assert result["higher_is_better"] is False
    assert result["frame_rate"] == "24000/1001"
    assert Path(result["official_result"]).is_file()
    assert main(argv) == 1  # an existing result must not be overwritten
