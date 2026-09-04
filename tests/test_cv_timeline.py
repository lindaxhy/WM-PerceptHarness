from __future__ import annotations

import json
import os
import subprocess
import types
from pathlib import Path

import pytest

from las_repro.cv.contracts import FrameTimeline, FrameTimestamp, SamplingPolicy
from las_repro.cv.timeline import (
    TimelineError,
    initial_sample_indices,
    materialize_sampled_frames,
    probe_frame_timeline,
    refinement_sample_indices,
)


@pytest.fixture
def fake_run():
    """Return a successful FFprobe-shaped process result."""

    def run(*args: object, **kwargs: object) -> types.SimpleNamespace:
        run.calls.append((args, kwargs))
        return types.SimpleNamespace(stdout=run.stdout)

    run.stdout = json.dumps({"frames": []})
    run.calls = []
    return run


def test_probe_frame_timeline_preserves_nonuniform_pts(
    tmp_path: Path, fake_run
) -> None:
    """Replacing decoder PTS with an average frame rate loses VFR alignment."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"video")
    fake_run.stdout = json.dumps(
        {
            "frames": [
                {"best_effort_timestamp_time": "0.000000"},
                {"best_effort_timestamp_time": "0.033000"},
                {"best_effort_timestamp_time": "0.071000"},
            ]
        }
    )

    timeline = probe_frame_timeline(video, run=fake_run)

    assert [point.frame_index for point in timeline.frames] == [0, 1, 2]
    assert [point.timestamp_seconds for point in timeline.frames] == [0.0, 0.033, 0.071]
    arguments, keywords = fake_run.calls[0]
    assert arguments[0][:-1] == [
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
    ]
    assert arguments[0][-1].startswith("/dev/fd/")
    assert keywords["shell"] is False
    assert len(keywords["pass_fds"]) == 1


@pytest.mark.parametrize(
    "stdout",
    [
        "not json",
        json.dumps({}),
        json.dumps({"frames": []}),
        json.dumps({"frames": [{}]}),
        json.dumps({"frames": [{"best_effort_timestamp_time": "invalid"}]}),
        json.dumps({"frames": [{"best_effort_timestamp_time": "NaN"}]}),
        json.dumps({"frames": [{"best_effort_timestamp_time": "Infinity"}]}),
        json.dumps({"frames": [{"best_effort_timestamp_time": "1e100000"}]}),
        json.dumps({"frames": [{"best_effort_timestamp_time": "-0.001"}]}),
        json.dumps(
            {
                "frames": [
                    {"best_effort_timestamp_time": "0.0"},
                    {"best_effort_timestamp_time": "0.0"},
                ]
            }
        ),
        json.dumps(
            {
                "frames": [
                    {"best_effort_timestamp_time": "0.1"},
                    {"best_effort_timestamp_time": "0.05"},
                ]
            }
        ),
    ],
)
def test_probe_frame_timeline_rejects_invalid_pts_mappings(
    tmp_path: Path, fake_run, stdout: str
) -> None:
    """Accepting malformed PTS would make source-frame evidence ambiguous."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"video")
    fake_run.stdout = stdout

    with pytest.raises(TimelineError, match="unable to probe video frame timeline"):
        probe_frame_timeline(video, run=fake_run)


def test_probe_frame_timeline_rejects_symlink_and_inode_swap(
    tmp_path: Path, fake_run
) -> None:
    """A pathname redirected during probing must not be accepted as one source."""
    target = tmp_path / "target.mp4"
    target.write_bytes(b"original")
    symlink = tmp_path / "link.mp4"
    symlink.symlink_to(target)

    with pytest.raises(TimelineError, match="unable to probe video frame timeline"):
        probe_frame_timeline(symlink, run=fake_run)
    assert fake_run.calls == []

    fake_run.stdout = json.dumps({"frames": [{"best_effort_timestamp_time": "0"}]})

    def replace_source(*args: object, **kwargs: object) -> types.SimpleNamespace:
        target.unlink()
        target.write_bytes(b"replacement")
        return types.SimpleNamespace(stdout=fake_run.stdout)

    with pytest.raises(TimelineError, match="unable to probe video frame timeline"):
        probe_frame_timeline(target, run=replace_source)


def test_probe_frame_timeline_pins_the_validated_inode_through_ffprobe(
    tmp_path: Path,
) -> None:
    """A swap restored before post-checks must not alter the probed source bytes."""
    video = tmp_path / "source.mp4"
    video.write_bytes(b"original")
    held = tmp_path / "held.mp4"

    def swap_and_restore(*args: object, **kwargs: object) -> types.SimpleNamespace:
        command = args[0]
        assert isinstance(command, list)
        os.replace(video, held)
        video.write_bytes(b"attacker")
        if command[-1] == str(video):
            payload = {"frames": [{"best_effort_timestamp_time": "0.5"}]}
        else:
            descriptors = kwargs.get("pass_fds")
            assert isinstance(descriptors, tuple)
            descriptor = descriptors[0]
            assert os.read(descriptor, 8) == b"original"
            payload = {"frames": [{"best_effort_timestamp_time": "0.0"}]}
        video.unlink()
        os.replace(held, video)
        return types.SimpleNamespace(stdout=json.dumps(payload))

    timeline = probe_frame_timeline(video, run=swap_and_restore)

    assert [point.timestamp_seconds for point in timeline.frames] == [0.0]


def test_probe_frame_timeline_sanitizes_subprocess_failure(
    tmp_path: Path,
) -> None:
    """FFprobe diagnostics can contain paths and must not cross the API boundary."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"video")

    def fail(*args: object, **kwargs: object) -> object:
        raise subprocess.CalledProcessError(1, ["ffprobe"], stderr="secret path")

    with pytest.raises(TimelineError, match="unable to probe video frame timeline"):
        probe_frame_timeline(video, run=fail)


def _timeline(fps: int, duration_seconds: int) -> FrameTimeline:
    return FrameTimeline(
        frames=tuple(
            FrameTimestamp(frame_index=index, timestamp_seconds=index / fps)
            for index in range(fps * duration_seconds + 1)
        )
    )


def _policy() -> SamplingPolicy:
    return SamplingPolicy(
        short_video_seconds=30.0,
        scan_fps=8.0,
        max_fps=30.0,
        refinement_radius_seconds=1.0,
    )


def test_initial_sampling_keeps_all_short_video_frames_at_or_below_max_fps() -> None:
    """Dropping source frames below the cap loses available short-video evidence."""
    timeline = _timeline(10, 2)

    indices = initial_sample_indices(timeline, _policy())

    assert indices == tuple(range(21))


def test_initial_sampling_uses_first_eligible_pts_above_short_video_cap() -> None:
    """Rounding sample times can select a decoded frame before its requested PTS."""
    timeline = FrameTimeline(
        frames=(
            FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
            FrameTimestamp(frame_index=1, timestamp_seconds=0.02),
            FrameTimestamp(frame_index=2, timestamp_seconds=0.04),
            FrameTimestamp(frame_index=3, timestamp_seconds=0.068),
            FrameTimestamp(frame_index=4, timestamp_seconds=0.1),
        )
    )

    indices = initial_sample_indices(timeline, _policy())

    assert indices == (0, 2, 3, 4)


def test_long_video_scans_at_eight_fps_and_refines_one_second_windows() -> None:
    """A long-video scan must bound cost while refinement stays local to a change."""
    timeline = _timeline(30, 60)
    policy = _policy()

    scan = initial_sample_indices(timeline, policy)
    refined = refinement_sample_indices(timeline, (300,), policy)

    assert scan[0] == 0
    assert scan[-1] == 1800
    assert scan == tuple(sorted(set(scan)))
    assert len(scan) <= 60 * 8 + 1
    assert refined == tuple(sorted(set(refined)))
    assert len(refined) <= 2 * 30 + 1
    assert all(9.0 <= timeline.frames[index].timestamp_seconds <= 11.0 for index in refined)


def test_refinement_rejects_change_indices_absent_from_source_timeline() -> None:
    """A stale local/SAM index must not be mistaken for an original source index."""
    with pytest.raises(TimelineError, match="unable to sample video frames"):
        refinement_sample_indices(_timeline(30, 60), (9999,), _policy())


def test_overlapping_refinement_windows_do_not_exceed_max_fps() -> None:
    """Independently anchored overlap grids can interleave at twice the allowed rate."""
    timeline = _timeline(60, 60)

    refined = refinement_sample_indices(timeline, (600, 601), _policy())

    timestamps = [timeline.frames[index].timestamp_seconds for index in refined]
    assert all(
        later - earlier >= (1 / 30) - 1e-12
        for earlier, later in zip(timestamps, timestamps[1:])
    )


def test_materialize_sampled_frames_uses_source_indices_and_returns_sam_mapping(
    tmp_path: Path,
) -> None:
    """Renumbering before extraction would associate SAM masks with wrong source PTS."""
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "sam-frames"
    timeline = FrameTimeline(
        frames=(
            FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
            FrameTimestamp(frame_index=2, timestamp_seconds=0.071),
            FrameTimestamp(frame_index=5, timestamp_seconds=0.2),
        )
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def extract(*args: object, **kwargs: object) -> types.SimpleNamespace:
        calls.append((args, kwargs))
        command = args[0]
        assert isinstance(command, list)
        output = command[-1]
        assert isinstance(output, str) and output.startswith("pipe:")
        os.write(int(output.removeprefix("pipe:")), b"jpeg")
        return types.SimpleNamespace(stdout="")

    sampled = materialize_sampled_frames(
        video, timeline, (0, 2, 5), destination, run=extract
    )

    assert [frame.sam_index for frame in sampled.frames] == [0, 1, 2]
    assert [frame.source_frame_index for frame in sampled.frames] == [0, 2, 5]
    assert [frame.source_timestamp_seconds for frame in sampled.frames] == [0.0, 0.071, 0.2]
    assert [frame.path.name for frame in sampled.frames] == [
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
    ]
    assert len(calls) == 3
    assert [arguments[0][12] for arguments, _ in calls] == [
        "select='eq(n\\,0)'",
        "select='eq(n\\,2)'",
        "select='eq(n\\,5)'",
    ]
    assert all(arguments[0][0] == "ffmpeg" for arguments, _ in calls)
    assert all(keywords["shell"] is False for _, keywords in calls)


def test_materialize_sampled_frames_rejects_missing_output_or_unknown_source_index(
    tmp_path: Path,
) -> None:
    """Returning a partial directory or unprobed index would corrupt SAM alignment."""
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    timeline = FrameTimeline(
        frames=(
            FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
            FrameTimestamp(frame_index=2, timestamp_seconds=0.1),
        )
    )

    with pytest.raises(TimelineError, match="unable to materialize sampled frames"):
        materialize_sampled_frames(video, timeline, (8,), tmp_path / "unknown")

    def write_one(*args: object, **kwargs: object) -> types.SimpleNamespace:
        (tmp_path / "partial" / "000000.jpg").write_bytes(b"jpeg")
        return types.SimpleNamespace(stdout="")

    with pytest.raises(TimelineError, match="unable to materialize sampled frames"):
        materialize_sampled_frames(video, timeline, (0, 2), tmp_path / "partial", run=write_one)


def test_materialize_sampled_frames_pins_destination_during_swap_and_restore(
    tmp_path: Path,
) -> None:
    """A swapped output pathname must not redirect FFmpeg writes outside its directory."""
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    destination = tmp_path / "sam-frames"
    held = tmp_path / "held-frames"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    timeline = FrameTimeline(
        frames=(
            FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
            FrameTimestamp(frame_index=2, timestamp_seconds=0.1),
        )
    )

    def swap_and_write(*args: object, **kwargs: object) -> types.SimpleNamespace:
        command = args[0]
        assert isinstance(command, list)
        output = command[-1]
        if not isinstance(output, str) or not output.startswith("pipe:"):
            raise OSError("FFmpeg did not receive a pinned output descriptor")
        os.replace(destination, held)
        destination.symlink_to(attacker, target_is_directory=True)
        os.write(int(output.removeprefix("pipe:")), b"jpeg")
        destination.unlink()
        os.replace(held, destination)
        return types.SimpleNamespace(stdout="")

    sampled = materialize_sampled_frames(
        video, timeline, (0, 2), destination, run=swap_and_write
    )

    assert [frame.path.name for frame in sampled.frames] == ["000000.jpg", "000001.jpg"]
    assert sorted(path.name for path in destination.glob("*.jpg")) == [
        "000000.jpg",
        "000001.jpg",
    ]
    assert list(attacker.iterdir()) == []


def test_materialize_sampled_frames_extracts_real_zero_based_jpegs(
    short_video: Path, tmp_path: Path
) -> None:
    """An FFmpeg filter mismatch would leave SAM with missing or misnumbered JPEGs."""
    timeline = probe_frame_timeline(short_video)
    selected = (0, 5, len(timeline.frames) - 1)

    sampled = materialize_sampled_frames(
        short_video, timeline, selected, tmp_path / "sam-frames"
    )

    assert [frame.source_frame_index for frame in sampled.frames] == list(selected)
    assert [frame.path.name for frame in sampled.frames] == [
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
    ]
    assert all(frame.path.read_bytes().startswith(b"\xff\xd8") for frame in sampled.frames)
