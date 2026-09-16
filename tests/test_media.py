from __future__ import annotations

import json
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path

import pytest

import las_repro.media as media_module
from las_repro.media import (
    FrameRef,
    MediaProbeError,
    TimeSpan,
    extract_frames,
    plan_segments,
    probe_video,
)


class _SplitForbiddenTimeBase(str):
    def split(self, *args: object, **kwargs: object) -> list[str]:
        raise AssertionError("time_base must be bounded before splitting")

def _mock_ffprobe(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]
) -> None:
    monkeypatch.setattr(
        media_module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout=json.dumps(payload)),
    )


def _video_stream(**metadata: object) -> dict[str, object]:
    return {
        "codec_type": "video",
        "width": 320,
        "height": 180,
        "avg_frame_rate": "30/1",
        **metadata,
    }


def test_probe_video_prefers_selected_video_stream_duration_over_longer_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing format duration would annotate past the final video frame."""
    payload = {
        "format": {"duration": "10.934"},
        "streams": [
            _video_stream(duration="10.933333333333334"),
            {"codec_type": "audio", "duration": "10.934"},
        ],
    }
    _mock_ffprobe(monkeypatch, payload)

    metadata = probe_video(tmp_path / "video.mp4")

    assert metadata.duration == 10.933333333333334
    assert (metadata.width, metadata.height, metadata.fps) == (320, 180, 30.0)


def test_probe_video_derives_duration_from_video_timestamp_time_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignoring duration_ts/time_base loses a valid video-specific endpoint."""
    payload = {
        "format": {"duration": "10.934"},
        "streams": [
            _video_stream(duration="N/A", duration_ts=328, time_base="1/30"),
        ],
    }
    _mock_ffprobe(monkeypatch, payload)

    assert probe_video(tmp_path / "video.mp4").duration == float(Fraction(328, 30))


def test_probe_video_prefers_exact_timestamp_duration_over_truncated_direct_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Choosing FFprobe's rounded direct value loses the exact stream endpoint."""
    payload = {
        "format": {"duration": "10.934"},
        "streams": [
            _video_stream(
                duration="10.933333",
                duration_ts=328,
                time_base="1/30",
            ),
        ],
    }
    _mock_ffprobe(monkeypatch, payload)

    assert probe_video(tmp_path / "video.mp4").duration == float(Fraction(328, 30))


def test_probe_video_falls_back_to_container_duration_without_usable_video_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rejecting every fallback would break videos with container-only duration."""
    payload = {
        "format": {"duration": "8.5"},
        "streams": [_video_stream(duration="N/A")],
    }
    _mock_ffprobe(monkeypatch, payload)

    assert probe_video(tmp_path / "video.mp4").duration == 8.5


@pytest.mark.parametrize(
    "video_metadata",
    [
        {},
        {"duration": "N/A"},
        {"duration": "invalid"},
        {"duration": float("nan")},
        {"duration": float("inf")},
        {"duration": 10**400},
        {"duration": True},
        {"duration": False},
        {"duration": 0},
        {"duration": -1},
        {"duration": "N/A", "duration_ts": True, "time_base": "1/30"},
        {"duration": "N/A", "duration_ts": 1.5, "time_base": "1/30"},
        {"duration": "N/A", "duration_ts": "328", "time_base": "1/30"},
        {"duration": "N/A", "duration_ts": 10**400, "time_base": "1/1"},
        {"duration": "N/A", "duration_ts": 1, "time_base": "1/" + "1" * 4301},
        {"duration": "N/A", "duration_ts": 328, "time_base": "invalid"},
        {"duration": "N/A", "duration_ts": 328, "time_base": "1/0"},
        {"duration": "N/A", "duration_ts": 328, "time_base": "0/30"},
        {"duration": "N/A", "duration_ts": 328, "time_base": "1/-30"},
        {"duration": "N/A", "duration_ts": 328, "time_base": "1/30/2"},
    ],
)
def test_probe_video_skips_invalid_video_duration_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, video_metadata: dict[str, object]
) -> None:
    """Accepting an invalid video candidate would hide a valid container fallback."""
    payload = {
        "format": {"duration": "8.5"},
        "streams": [_video_stream(**video_metadata)],
    }
    _mock_ffprobe(monkeypatch, payload)

    assert probe_video(tmp_path / "video.mp4").duration == 8.5


@pytest.mark.parametrize(
    "format_duration",
    [None, "N/A", "invalid", float("nan"), float("inf"), True, False, 0, -1],
)
def test_probe_video_raises_stable_error_when_all_duration_candidates_are_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format_duration: object
) -> None:
    """Returning a nonpositive or nonfinite duration would corrupt segment planning."""
    payload = {
        "format": {"duration": format_duration},
        "streams": [
            _video_stream(duration="N/A", duration_ts="328", time_base="1/30"),
        ],
    }
    _mock_ffprobe(monkeypatch, payload)

    with pytest.raises(MediaProbeError):
        probe_video(tmp_path / "video.mp4")


def test_probe_video_rejects_slash_dense_time_base_before_splitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting a rejected slash-dense rational would allocate once per slash."""
    payload = {
        "format": {"duration": "8.5"},
        "streams": [
            _video_stream(
                duration="N/A",
                duration_ts=1,
                time_base=_SplitForbiddenTimeBase("/" * 64),
            ),
        ],
    }
    _mock_ffprobe(monkeypatch, payload)
    monkeypatch.setattr(media_module.json, "loads", lambda _: payload)

    assert probe_video(tmp_path / "video.mp4").duration == 8.5

    payload["format"] = {"duration": "N/A"}
    with pytest.raises(MediaProbeError):
        probe_video(tmp_path / "video.mp4")


def test_probe_video_reports_duration_dimensions_and_fps(short_video: Path):
    """Ignoring FFprobe stream metadata must make preprocessing lose video facts."""
    metadata = probe_video(short_video)

    assert metadata.duration == pytest.approx(2.0, abs=0.05)
    assert (metadata.width, metadata.height) == (320, 180)
    assert metadata.fps == pytest.approx(10.0, rel=0.01)


def test_probe_video_falls_back_when_average_frame_rate_is_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A valid real frame rate must survive an unknown 0/0 average rate."""
    payload = {
        "format": {"duration": "2.0"},
        "streams": [
            {
                "codec_type": "video",
                "width": 320,
                "height": 180,
                "avg_frame_rate": "0/0",
                "r_frame_rate": "10/1",
            }
        ],
    }
    monkeypatch.setattr(
        media_module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout=json.dumps(payload)),
    )

    assert probe_video(tmp_path / "video.mp4").fps == 10.0


def test_extract_frames_keeps_absolute_timestamps_in_refs_and_names(
    short_video: Path, tmp_path: Path
):
    """Resetting a clipped span to time zero must be observable in frame references."""
    frames = extract_frames(
        short_video,
        TimeSpan(start=0.5, end=1.5),
        fps=2.0,
        output_dir=tmp_path / "frames",
    )

    assert frames == [
        FrameRef(path=tmp_path / "frames" / "frame_000000_000000000500.jpg", timestamp=0.5),
        FrameRef(path=tmp_path / "frames" / "frame_000001_000000001000.jpg", timestamp=1.0),
    ]
    assert all(frame.path.is_file() for frame in frames)


def test_extract_frames_explicitly_disables_all_nonvideo_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"jpeg")
        return object()

    monkeypatch.setattr(media_module.subprocess, "run", fake_run)
    video = tmp_path / "silent clip.mp4"
    video.write_bytes(b"video")

    extract_frames(video, TimeSpan(2.0, 2.5), 2.0, tmp_path / "frames")

    assert len(calls) == 1
    command, options = calls[0]
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "-an" in command
    assert "-sn" in command
    assert "-dn" in command
    assert options["stdout"] is subprocess.DEVNULL
    assert options["stderr"] is subprocess.DEVNULL


def test_plan_segments_has_exact_terminal_end_and_overlap():
    """Floating-point drift must not omit or extend the final video boundary."""
    assert plan_segments(65.0, 30.0, 2.0) == [
        TimeSpan(0.0, 30.0),
        TimeSpan(28.0, 58.0),
        TimeSpan(56.0, 65.0),
    ]


@pytest.mark.parametrize("overlap", [1.0, 1.1])
def test_plan_segments_rejects_nonprogressing_overlap(overlap: float):
    """A segment plan must never loop when its overlap consumes the step."""
    with pytest.raises(ValueError):
        plan_segments(2.0, 1.0, overlap)


def test_plan_segments_rejects_a_step_lost_when_timestamps_are_rounded():
    """Sub-precision segment progress must fail instead of looping forever."""
    with pytest.raises(ValueError):
        plan_segments(2.0, 1.0000004, 1.0000003)


