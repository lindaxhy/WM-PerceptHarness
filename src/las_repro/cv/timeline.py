"""Exact decoder-order video timelines and deterministic frame sampling."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Callable

from .contracts import FrameTimeline, FrameTimestamp, SamplingPolicy


_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_MAX_MJPEG_STREAM_BYTES = 512 * 1024 * 1024
_MAX_JPEG_FRAME_BYTES = 64 * 1024 * 1024
_STREAM_READ_BYTES = 64 * 1024


class TimelineError(RuntimeError):
    """A sanitized timeline or sampled-frame boundary failure."""


@dataclass(frozen=True, slots=True)
class SampledFrame:
    """One SAM-local JPEG and its immutable source-frame alignment."""

    sam_index: int
    source_frame_index: int
    source_timestamp_seconds: float
    path: Path


@dataclass(frozen=True, slots=True)
class SampledFrameSet:
    """SAM-local frames, ordered so each tuple offset is its SAM frame index."""

    frames: tuple[SampledFrame, ...]


def probe_frame_timeline(
    path: Path,
    *,
    run: Callable[..., object] = subprocess.run,
) -> FrameTimeline:
    """Return decoder-order source PTS without deriving timestamps from FPS."""
    descriptor, source_identity = _open_pinned_regular_file(path)
    try:
        completed = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-show_frames",
                _descriptor_path(descriptor),
            ],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            pass_fds=(descriptor,),
        )
        payload = json.loads(completed.stdout)  # type: ignore[attr-defined]
        frames = payload["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError
        points = tuple(
            FrameTimestamp(
                frame_index=index,
                timestamp_seconds=_timestamp_seconds(frame),
            )
            for index, frame in enumerate(frames)
        )
        timeline = FrameTimeline(frames=points)
        if _regular_file_identity(path) != source_identity:
            raise ValueError
        return timeline
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        InvalidOperation,
    ):
        raise TimelineError("unable to probe video frame timeline") from None
    finally:
        os.close(descriptor)


def _regular_file_identity(path: Path) -> tuple[int, int]:
    try:
        status = path.lstat()
    except OSError:
        raise TimelineError("unable to probe video frame timeline") from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise TimelineError("unable to probe video frame timeline")
    return status.st_dev, status.st_ino


def _open_pinned_regular_file(path: Path) -> tuple[int, tuple[int, int]]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise TimelineError("unable to probe video frame timeline")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError:
        raise TimelineError("unable to probe video frame timeline") from None
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        identity = opened.st_dev, opened.st_ino
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise ValueError
        return descriptor, identity
    except (OSError, ValueError):
        os.close(descriptor)
        raise TimelineError("unable to probe video frame timeline") from None


def _descriptor_path(descriptor: int) -> str:
    return f"/dev/fd/{descriptor}"


def _timestamp_seconds(frame: object) -> float:
    if not isinstance(frame, dict):
        raise ValueError
    value = frame.get("best_effort_timestamp_time")
    if not isinstance(value, str) or not value:
        raise ValueError
    timestamp = Decimal(value)
    if not timestamp.is_finite() or timestamp < 0:
        raise ValueError
    seconds = float(timestamp)
    if not math.isfinite(seconds):
        raise ValueError
    return seconds


def initial_sample_indices(
    timeline: FrameTimeline, policy: SamplingPolicy
) -> tuple[int, ...]:
    """Select bounded, decoder-order source frames for the initial SAM pass."""
    try:
        start, end = _timeline_bounds(timeline)
        if end - start <= Fraction(str(policy.short_video_seconds)):
            source_fps = _source_fps(timeline, start, end)
            if source_fps is None or source_fps <= Fraction(str(policy.max_fps)):
                return tuple(point.frame_index for point in timeline.frames)
            rate = Fraction(str(policy.max_fps))
        else:
            rate = Fraction(str(policy.scan_fps))
        return _select_pts_at_or_after(timeline, start, end, rate)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        raise TimelineError("unable to sample video frames") from None


def refinement_sample_indices(
    timeline: FrameTimeline,
    change_indices: tuple[int, ...],
    policy: SamplingPolicy,
) -> tuple[int, ...]:
    """Densely resample bounded source-PTS windows around source-frame changes."""
    try:
        start, end = _timeline_bounds(timeline)
        points_by_index = {point.frame_index: point for point in timeline.frames}
        radius = Fraction(str(policy.refinement_radius_seconds))
        rate = Fraction(str(policy.max_fps))
        windows: list[tuple[Fraction, Fraction]] = []
        for index in change_indices:
            if isinstance(index, bool) or index not in points_by_index:
                raise ValueError
            change_time = Fraction(str(points_by_index[index].timestamp_seconds))
            windows.append(
                (
                    max(start, change_time - radius),
                    min(end, change_time + radius),
                )
            )
        selected: set[int] = set()
        for window_start, window_end in _merge_overlapping_windows(windows):
            selected.update(
                _select_pts_at_or_after(
                    timeline,
                    window_start,
                    window_end,
                    rate,
                    retain_timeline_bounds=False,
                )
            )
        return _cap_selected_pts(timeline, selected, rate)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        raise TimelineError("unable to sample video frames") from None


def _merge_overlapping_windows(
    windows: list[tuple[Fraction, Fraction]],
) -> tuple[tuple[Fraction, Fraction], ...]:
    merged: list[tuple[Fraction, Fraction]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def _cap_selected_pts(
    timeline: FrameTimeline, selected: set[int], rate: Fraction
) -> tuple[int, ...]:
    points_by_index = {point.frame_index: point for point in timeline.frames}
    interval = Fraction(1, 1) / rate
    capped: list[int] = []
    previous: Fraction | None = None
    for index in sorted(selected):
        timestamp = Fraction(str(points_by_index[index].timestamp_seconds))
        if previous is None or timestamp - previous >= interval:
            capped.append(index)
            previous = timestamp
    return tuple(capped)


def _timeline_bounds(timeline: FrameTimeline) -> tuple[Fraction, Fraction]:
    frames = timeline.frames
    if not frames:
        raise ValueError
    start = Fraction(str(frames[0].timestamp_seconds))
    end = Fraction(str(frames[-1].timestamp_seconds))
    if end < start:
        raise ValueError
    return start, end


def _source_fps(
    timeline: FrameTimeline, start: Fraction, end: Fraction
) -> Fraction | None:
    if len(timeline.frames) == 1 or end == start:
        return None
    return Fraction(len(timeline.frames) - 1, 1) / (end - start)


def _select_pts_at_or_after(
    timeline: FrameTimeline,
    start: Fraction,
    end: Fraction,
    rate: Fraction,
    *,
    retain_timeline_bounds: bool = True,
) -> tuple[int, ...]:
    if rate <= 0:
        raise ValueError
    points = timeline.frames
    timestamps = tuple(Fraction(str(point.timestamp_seconds)) for point in points)
    selected: list[int] = [points[0].frame_index] if retain_timeline_bounds else []
    interval = Fraction(1, 1) / rate
    target = start
    cursor = 0
    while target <= end:
        while cursor < len(timestamps) and timestamps[cursor] < target:
            cursor += 1
        if cursor == len(timestamps) or timestamps[cursor] > end:
            break
        selected.append(points[cursor].frame_index)
        target += interval
    if retain_timeline_bounds:
        selected.append(points[-1].frame_index)
    return tuple(sorted(set(selected)))


def materialize_sampled_frames(
    video_path: Path,
    timeline: FrameTimeline,
    indices: tuple[int, ...],
    destination: Path,
    *,
    run: Callable[..., object] = subprocess.run,
) -> SampledFrameSet:
    """Extract selected decoder indices as zero-based SAM-local JPEG files."""
    try:
        descriptor, source_identity = _open_pinned_regular_file(video_path)
        requested = tuple(indices)
        source_by_index = {point.frame_index: point for point in timeline.frames}
        if not requested or any(
            isinstance(index, bool) or index not in source_by_index for index in requested
        ):
            raise ValueError
        if requested != tuple(sorted(set(requested))):
            raise ValueError
        destination.mkdir(parents=True, exist_ok=False)
        destination_descriptor, destination_identity = _open_pinned_directory(destination)
        _extract_pinned_jpegs(
            descriptor,
            destination_descriptor,
            requested,
            run,
        )
        if _regular_file_identity(video_path) != source_identity:
            raise ValueError
        if _directory_identity(destination) != destination_identity:
            raise ValueError
        expected_paths = tuple(destination / f"{number:06d}.jpg" for number in range(len(requested)))
        actual_paths = tuple(sorted(destination.glob("*.jpg")))
        if actual_paths != expected_paths or any(
            not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink()
            for path in actual_paths
        ):
            raise ValueError
        return SampledFrameSet(
            frames=tuple(
                SampledFrame(
                    sam_index=sam_index,
                    source_frame_index=source_by_index[index].frame_index,
                    source_timestamp_seconds=source_by_index[index].timestamp_seconds,
                    path=expected_paths[sam_index],
                )
                for sam_index, index in enumerate(requested)
            )
        )
    except (
        TimelineError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ):
        raise TimelineError("unable to materialize sampled frames") from None
    finally:
        if "destination_descriptor" in locals():
            os.close(destination_descriptor)
        if "descriptor" in locals():
            os.close(descriptor)


def _extract_pinned_jpegs(
    source_descriptor: int,
    destination_descriptor: int,
    source_indices: tuple[int, ...],
    run: Callable[..., object],
) -> None:
    stream_name, stream_descriptor, stream_identity = _open_private_stream(
        destination_descriptor
    )
    try:
        expression = "+".join(f"eq(n\\,{index})" for index in source_indices)
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                _descriptor_path(source_descriptor),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                f"select='{expression}'",
                "-frames:v",
                str(len(source_indices)),
                "-fps_mode:v",
                "passthrough",
                "-c:v",
                "mjpeg",
                "-q:v",
                "2",
                "-f",
                "image2pipe",
                f"pipe:{stream_descriptor}",
            ],
            check=True,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(source_descriptor, stream_descriptor),
        )
        stream_size = os.fstat(stream_descriptor).st_size
        if stream_size <= 0 or stream_size > _MAX_MJPEG_STREAM_BYTES:
            raise ValueError
        os.lseek(stream_descriptor, 0, os.SEEK_SET)
        _split_mjpeg_stream(
            stream_descriptor,
            destination_descriptor,
            len(source_indices),
        )
    finally:
        os.close(stream_descriptor)
        _unlink_owned_stream(destination_descriptor, stream_name, stream_identity)


def _open_private_stream(destination_descriptor: int) -> tuple[str, int, tuple[int, int]]:
    for _ in range(16):
        name = f".mjpeg-{os.urandom(16).hex()}.tmp"
        try:
            descriptor = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=destination_descriptor,
            )
        except FileExistsError:
            continue
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            os.close(descriptor)
            raise ValueError
        return name, descriptor, (status.st_dev, status.st_ino)
    raise ValueError


def _unlink_owned_stream(
    destination_descriptor: int, name: str, identity: tuple[int, int]
) -> None:
    try:
        status = os.stat(name, dir_fd=destination_descriptor, follow_symlinks=False)
        if (
            stat.S_ISREG(status.st_mode)
            and not stat.S_ISLNK(status.st_mode)
            and (status.st_dev, status.st_ino) == identity
        ):
            os.unlink(name, dir_fd=destination_descriptor)
    except OSError:
        pass


def _split_mjpeg_stream(
    stream_descriptor: int,
    destination_descriptor: int,
    expected_count: int,
) -> None:
    buffer = bytearray()
    current = bytearray()
    frame_count = 0
    in_frame = False
    while chunk := os.read(stream_descriptor, _STREAM_READ_BYTES):
        buffer.extend(chunk)
        while True:
            if not in_frame:
                if len(buffer) < len(_JPEG_SOI):
                    break
                if not buffer.startswith(_JPEG_SOI):
                    raise ValueError
                current = bytearray(_JPEG_SOI)
                del buffer[: len(_JPEG_SOI)]
                in_frame = True
            marker = buffer.find(_JPEG_EOI)
            if marker < 0:
                trailing_marker = 1 if buffer.endswith(b"\xff") else 0
                current.extend(buffer[:-trailing_marker] if trailing_marker else buffer)
                if len(current) > _MAX_JPEG_FRAME_BYTES:
                    raise ValueError
                if trailing_marker:
                    buffer[:] = buffer[-trailing_marker:]
                else:
                    buffer.clear()
                break
            current.extend(buffer[: marker + len(_JPEG_EOI)])
            del buffer[: marker + len(_JPEG_EOI)]
            if len(current) > _MAX_JPEG_FRAME_BYTES or frame_count >= expected_count:
                raise ValueError
            _write_pinned_jpeg(destination_descriptor, frame_count, current)
            frame_count += 1
            current.clear()
            in_frame = False
    if in_frame or buffer or frame_count != expected_count:
        raise ValueError


def _write_pinned_jpeg(
    destination_descriptor: int, sam_index: int, data: bytearray
) -> None:
    output_descriptor = os.open(
        f"{sam_index:06d}.jpg",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=destination_descriptor,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(output_descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
    finally:
        os.close(output_descriptor)


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        status = path.lstat()
    except OSError:
        raise ValueError from None
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError
    return status.st_dev, status.st_ino


def _open_pinned_directory(path: Path) -> tuple[int, tuple[int, int]]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ValueError
    try:
        descriptor = os.open(path, os.O_RDONLY | directory | nofollow)
    except OSError:
        raise ValueError from None
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        identity = opened.st_dev, opened.st_ino
        if (
            not stat.S_ISDIR(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise ValueError
        return descriptor, identity
    except (OSError, ValueError):
        os.close(descriptor)
        raise ValueError from None
