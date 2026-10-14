"""Shared video metadata, playback timing and timestamped frame extraction.

``probe_video`` reads duration, size and frame rate with FFprobe.
``extract_frames`` writes frames with FFmpeg for model requests.
Model inference lives in ``models``; these helpers preserve video timing.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

_MAX_TIME_BASE_CHARS = 64


class MediaError(RuntimeError):
    """Base class for stable media-processing failures."""


class MediaProbeError(MediaError):
    """FFprobe could not return usable video metadata."""


class FrameExtractionError(MediaError):
    """FFmpeg could not extract a requested frame."""


@dataclass(frozen=True)
class VideoMetadata:
    duration: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class TimeSpan:
    start: float
    end: float

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (self.start, self.end)):
            raise ValueError("time span bounds must be finite numbers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("time span must satisfy 0 <= start < end")


@dataclass(frozen=True)
class FrameRef:
    path: Path
    timestamp: float


def probe_video(path: Path) -> VideoMetadata:
    """Read stable video metadata from FFprobe JSON output."""
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-select_streams",
                "v:0",
                "-show_streams",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        duration = _video_duration(video, payload.get("format", {}))
        width = int(video["width"])
        height = int(video["height"])
        fps = _video_frame_rate(video)
    except (
        OSError,
        subprocess.SubprocessError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        ZeroDivisionError,
        InvalidOperation,
    ):
        raise MediaProbeError("unable to probe video") from None
    if (
        duration is None
        or not math.isfinite(duration)
        or duration <= 0
        or width <= 0
        or height <= 0
    ):
        raise MediaProbeError("video metadata is invalid")
    return VideoMetadata(duration, width, height, fps)


def playback_origin(container: Any) -> Decimal:
    """FFmpeg input seeking is relative to the container start, not raw PTS."""
    value = container.get("start_time") if isinstance(container, dict) else None
    if value in (None, "N/A"):
        return Decimal(0)
    if isinstance(value, bool):
        raise ValueError("invalid playback origin")
    origin = Decimal(str(value))
    if not origin.is_finite():
        raise ValueError("invalid playback origin")
    return origin


def _video_duration(video: dict[str, Any], container: Any) -> float | None:
    duration = _timestamp_duration(video)
    if duration is None:
        duration = _positive_finite_float(video.get("duration"))
    if duration is not None:
        origin = playback_origin(container)
        start = video.get("start_time")
        video_start = origin if start in (None, "N/A") else Decimal(str(start))
        if not video_start.is_finite() or video_start < origin:
            raise ValueError("invalid video stream start")
        # Duration is an interval length. The annotation end also includes a
        # possible initial video delay relative to audio/container playback.
        return float(Decimal(str(duration)) + video_start - origin)
    if isinstance(container, dict):
        return _positive_finite_float(container.get("duration"))
    return None


def _timestamp_duration(video: dict[str, Any]) -> float | None:
    duration_ts = video.get("duration_ts")
    if (
        isinstance(duration_ts, bool)
        or not isinstance(duration_ts, int)
        or duration_ts <= 0
    ):
        return None

    time_base = video.get("time_base")
    if not isinstance(time_base, str) or len(time_base) >= _MAX_TIME_BASE_CHARS:
        return None
    if re.fullmatch(r"[0-9]+/[0-9]+", time_base) is None:
        return None
    try:
        tick_seconds = Fraction(time_base)
        if tick_seconds <= 0:
            return None
        duration = float(duration_ts * tick_seconds)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return _positive_finite_float(duration)


def _positive_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def extract_frames(
    path: Path,
    span: TimeSpan,
    fps: float,
    output_dir: Path,
) -> list[FrameRef]:
    """Decode once; keep the first source frame in each sampling interval.

    Intervals are anchored at span.start. Empty intervals produce no frame;
    timestamps are selected source PTS on the container playback clock.
    Mixed-resolution inputs use FFmpeg's first-frame output canvas.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(tempfile.mkdtemp(prefix="frames-", dir=output_dir))
    # The epsilon stabilizes bin boundaries against floating-point rounding.
    current_bin = f"floor((t-{span.start})*{fps}+1e-9)"
    previous_bin = f"floor((prev_selected_t-{span.start})*{fps}+1e-9)"
    selection = (f"gte(t,{span.start})*lt(t,{span.end})*"
                 f"if(isnan(prev_selected_t),1,gt({current_bin},{previous_bin}))")
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
             "-copyts", "-start_at_zero", "-reinit_filter", "0",
             "-t", str(span.end), "-i", str(path),
             "-map", "0:v:0", "-an", "-sn", "-dn",
             "-vf", f"select='{selection}',showinfo=checksum=0",
             "-fps_mode", "passthrough", "-q:v", "2", "-y",
             str(destination / "frame_%06d.jpg")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        # Integer PTS + rational time base avoids showinfo's rounded pts_time.
        bases = set(re.findall(r"config in time_base:\s*(\d+/\d+)", completed.stderr))
        if len(bases) != 1:
            raise ValueError("missing frame time base")
        tick = Fraction(bases.pop())
        if tick <= 0:
            raise ValueError("invalid frame time base")
        frames = []
        previous = -math.inf
        for number, pts in re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)", completed.stderr):
            timestamp = float(int(pts) * tick)
            image = destination / f"frame_{int(number) + 1:06d}.jpg"
            if (int(number) != len(frames)
                    or not span.start <= timestamp < span.end or timestamp <= previous
                    or not image.is_file() or image.stat().st_size == 0):
                raise ValueError("invalid extracted frame")
            frames.append(FrameRef(image, timestamp))
            previous = timestamp
        if not frames or len(list(destination.glob("frame_*.jpg"))) != len(frames):
            raise ValueError("no source frames in requested span")
        return frames
    except (OSError, subprocess.SubprocessError, ValueError, ZeroDivisionError):
        shutil.rmtree(destination, ignore_errors=True)
        raise FrameExtractionError("unable to extract video frames") from None


def _video_frame_rate(video: dict[str, Any]) -> float:
    for field in ("avg_frame_rate", "r_frame_rate"):
        value = video.get(field)
        if value is None:
            continue
        try:
            rate = float(Fraction(value))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    raise ValueError("video has no usable frame rate")
