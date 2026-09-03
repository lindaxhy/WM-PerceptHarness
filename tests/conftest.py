from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def short_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a deterministic two-second, ten-FPS video without audio."""
    destination = tmp_path_factory.mktemp("media") / "fixture.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-r",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-t",
            "2",
            str(destination),
        ],
        check=True,
        shell=False,
    )
    return destination


@pytest.fixture(scope="session")
def longer_silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a deterministic 2.2-second video with no audio stream."""
    destination = tmp_path_factory.mktemp("longer-media") / "fixture-2.2s.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:d=1.1:r=10",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:d=1.1:r=10",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-r",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-t",
            "2.2",
            str(destination),
        ],
        check=True,
        shell=False,
    )
    return destination
