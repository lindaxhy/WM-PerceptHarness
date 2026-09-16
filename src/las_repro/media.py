"""Local media probing, segmentation, and frame extraction."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import Any

_TIMESTAMP_QUANTUM = Decimal("0.000001")
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
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("time span bounds must be finite")
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
            shell=False,
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
        json.JSONDecodeError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        raise MediaProbeError("unable to probe video") from None
    if (
        duration is None
        or not math.isfinite(duration)
        or duration <= 0
        or width <= 0
        or height <= 0
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise MediaProbeError("video metadata is invalid")
    return VideoMetadata(duration, width, height, fps)


def _video_duration(video: dict[str, Any], container: Any) -> float | None:
    timestamp_duration = _timestamp_duration(video)
    if timestamp_duration is not None:
        return timestamp_duration

    direct = _positive_finite_float(video.get("duration"))
    if direct is not None:
        return direct

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
    separator = time_base.find("/")
    if (
        separator <= 0
        or separator != time_base.rfind("/")
        or separator == len(time_base) - 1
    ):
        return None
    numerator_text = time_base[:separator]
    denominator_text = time_base[separator + 1 :]
    if not all(
        part.isascii() and part.isdigit()
        for part in (numerator_text, denominator_text)
    ):
        return None
    try:
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except ValueError:
        return None
    if numerator <= 0 or denominator <= 0:
        return None
    try:
        duration = float(Fraction(duration_ts) * Fraction(numerator, denominator))
    except OverflowError:
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


def plan_segments(
    duration: float,
    max_seconds: float,
    overlap_seconds: float,
) -> list[TimeSpan]:
    """Plan deterministic overlapping spans ending exactly at ``duration``."""
    if not all(math.isfinite(value) for value in (duration, max_seconds, overlap_seconds)):
        raise ValueError("segment values must be finite")
    if duration <= 0 or max_seconds <= 0 or overlap_seconds < 0:
        raise ValueError("duration and max_seconds must be positive; overlap cannot be negative")
    if overlap_seconds >= max_seconds:
        raise ValueError("overlap must be smaller than max_seconds")

    duration_decimal = Decimal(str(duration))
    maximum = Decimal(str(max_seconds))
    overlap = Decimal(str(overlap_seconds))
    start = Decimal("0")
    spans: list[TimeSpan] = []
    while start < duration_decimal:
        end = min(start + maximum, duration_decimal)
        spans.append(TimeSpan(float(start), duration if end == duration_decimal else float(end)))
        if end == duration_decimal:
            break
        next_start = (end - overlap).quantize(_TIMESTAMP_QUANTUM)
        if next_start <= start:
            raise ValueError("rounded segment step does not make progress")
        start = next_start
    return spans


def extract_frames(
    path: Path,
    span: TimeSpan,
    fps: float,
    output_dir: Path,
) -> list[FrameRef]:
    """Extract frames at an absolute timeline cadence within ``span``."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.resolve(strict=True).is_dir():
        raise FrameExtractionError("frame output is not a directory")

    start = Decimal(str(span.start))
    end = Decimal(str(span.end))
    fps_decimal = Decimal(str(fps))
    frames: list[FrameRef] = []
    timestamp = start
    index = 0
    while timestamp < end:
        absolute = float(timestamp)
        milliseconds = int((timestamp * 1000).quantize(Decimal("1"), ROUND_HALF_UP))
        destination = output_dir / f"frame_{index:06d}_{milliseconds:012d}.jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    _decimal_text(timestamp),
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-dn",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-y",
                    str(destination),
                ],
                check=True,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            destination.unlink(missing_ok=True)
            raise FrameExtractionError("unable to extract video frame") from None
        frames.append(FrameRef(destination, absolute))
        index += 1
        # Exact per-index division: accumulating a pre-rounded interval can
        # land a timestamp fractionally below ``end`` at what is really the
        # video's end, producing one out-of-range ffmpeg seek.
        timestamp = start + Decimal(index) / fps_decimal
    return frames


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


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")
