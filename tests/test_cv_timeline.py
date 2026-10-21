import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from percept_harness.cv import timeline
from percept_harness.cv.contracts import FrameTimeline, SamplingPolicy
from percept_harness.cv.timeline import (
    TimelineError, initial_sample_indices, materialize_sampled_frames, probe_frame_timeline,
)


def _timeline(*timestamps):
    return FrameTimeline.model_validate(
        {"frames": [{"frame_index": i, "timestamp_seconds": t} for i, t in enumerate(timestamps)]})


def _sample(timestamps, fps):
    return initial_sample_indices(_timeline(*timestamps), SamplingPolicy(max_fps=fps))


def test_initial_sample_indices_boundaries():
    assert _sample((0.0,), 0.5) == (0,)
    # Exactly at the rate, and one microsecond short of it, keep every frame.
    assert _sample((0.0, 0.1, 0.2, 0.3), 10.0) == (0, 1, 2, 3)
    assert _sample((0.0, 0.1, 0.2, 0.299999), 10.0) == (0, 1, 2, 3)
    assert _sample((0.0, 0.1, 0.2, 0.299998), 10.0) == (0, 1, 2, 3)
    assert _sample((0.0, 0.1, 0.15, 0.2), 10.0) == (0, 1, 3)
    thirty = tuple(round(i / 30, 6) for i in range(31))
    assert _sample(thirty, 10.0) == tuple(range(0, 31, 3))
    assert _sample(thirty, 4.0) == (0, 8, 15, 23, 30)
    # A timestamp one microsecond before the target still counts as reaching it.
    assert _sample((0.0, 0.05, 0.499999, 0.6, 1.0), 2.0) == (0, 2, 4)
    assert _sample((0.0, 0.05, 0.499998, 0.6, 1.0), 2.0) == (0, 3, 4)
    assert _sample((0.0, 0.05, 0.099999, 0.15, 0.2), 10.0) == (0, 2, 4)
    assert _sample((0.0, 0.05, 0.099998, 0.15, 0.2), 10.0) == (0, 3, 4)


def _split(tmp_path, stream, expected_count, read_bytes, monkeypatch):
    monkeypatch.setattr(timeline, "_STREAM_READ_BYTES", read_bytes)
    case = Path(tempfile.mkdtemp(dir=tmp_path))
    source = case / "stream"
    source.write_bytes(stream)
    output = case / "out"
    output.mkdir()
    source_fd = os.open(source, os.O_RDONLY)
    output_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        timeline._split_mjpeg_stream(source_fd, output_fd, expected_count)
    finally:
        os.close(source_fd)
        os.close(output_fd)
    return [path.read_bytes() for path in sorted(output.iterdir())]


FIRST = b"\xff\xd8\x01\xff\x00\xff\xff\x02\xff\xd9"
SECOND = b"\xff\xd8\xff\xd9"


@pytest.mark.parametrize("read_bytes", range(1, 16))
def test_split_mjpeg_stream_across_chunks(tmp_path, monkeypatch, read_bytes):
    assert _split(tmp_path, FIRST + SECOND, 2, read_bytes, monkeypatch) == [FIRST, SECOND]


@pytest.mark.parametrize("stream, count", [
    (FIRST + SECOND, 1), (FIRST + SECOND, 3), (b"\x00" + FIRST, 1),
    (FIRST + b"\xff", 1), (FIRST + b"\xff\xd8\x01", 1), (b"", 1),
])
def test_split_mjpeg_stream_rejects(tmp_path, monkeypatch, stream, count):
    with pytest.raises(ValueError):
        _split(tmp_path, stream, count, 3, monkeypatch)


def test_split_mjpeg_stream_frame_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(timeline, "_MAX_JPEG_FRAME_BYTES", 8)
    assert _split(tmp_path, SECOND + b"\xff\xd8" + b"\x00" * 4 + b"\xff\xd9", 2, 4, monkeypatch)
    with pytest.raises(ValueError):
        _split(tmp_path, b"\xff\xd8" + b"\x00" * 5 + b"\xff\xd9", 1, 4, monkeypatch)
    with pytest.raises(ValueError):
        _split(tmp_path, b"\xff\xd8" + b"\x00" * 10, 1, 4, monkeypatch)


def test_probe_frame_timeline(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[-1] == f"/dev/fd/{kwargs['pass_fds'][0]}"
        return SimpleNamespace(stdout=json.dumps({
            "format": {"start_time": "1.5"},
            "frames": [{"best_effort_timestamp_time": value} for value in ("1.5", "1.540000", "1.58")]}))

    result = probe_frame_timeline(video, run=run)
    assert [(p.frame_index, p.timestamp_seconds) for p in result.frames] == [(0, 0.0), (1, 0.04), (2, 0.08)]
    assert calls[0][0][:2] == ["ffprobe", "-v"] and calls[0][1]["shell"] is False

    for stdout in ("{}", json.dumps({"frames": []}),
                   json.dumps({"frames": [{"best_effort_timestamp_time": "0.1"}], "format": {"start_time": "0.2"}})):
        with pytest.raises(TimelineError, match="unable to probe video frame timeline"):
            probe_frame_timeline(video, run=lambda *a, stdout=stdout, **k: SimpleNamespace(stdout=stdout))
    link = tmp_path / "link.mp4"
    link.symlink_to(video)
    for path in (link, tmp_path, tmp_path / "missing.mp4"):
        with pytest.raises(TimelineError, match="unable to probe video frame timeline"):
            probe_frame_timeline(path, run=run)


def test_materialize_sampled_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    source = _timeline(0.0, 0.5, 1.0, 1.5)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        os.write(kwargs["pass_fds"][1], FIRST + SECOND)

    frames = materialize_sampled_frames(video, source, (1, 3), tmp_path / "frames", run=run)
    assert [(f.sam_index, f.source_frame_index, f.source_timestamp_seconds, f.path.name)
            for f in frames.frames] == [(0, 1, 0.5, "000000.jpg"), (1, 3, 1.5, "000001.jpg")]
    assert frames.frames[0].path.read_bytes() == FIRST
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["000000.jpg", "000001.jpg"]
    assert "select='eq(n\\,1)+eq(n\\,3)'" in commands[0]
    for number, indices in enumerate(((), (3, 1), (1, 1), (7,), (True,))):
        with pytest.raises(TimelineError, match="unable to materialize sampled frames"):
            materialize_sampled_frames(video, source, indices, tmp_path / f"bad{number}", run=run)
        assert not (tmp_path / f"bad{number}").exists()
    with pytest.raises(TimelineError):
        materialize_sampled_frames(video, source, (1,), tmp_path / "short", run=run)
    with pytest.raises(TimelineError):
        materialize_sampled_frames(video, source, (1, 3), tmp_path / "frames", run=run)
