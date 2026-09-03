from __future__ import annotations

import json
import logging
import os
import socket
import ssl
import subprocess
import sys
import threading
import types
from fractions import Fraction
from pathlib import Path

import httpx
import pytest

import las_repro.media as media_module
from las_repro.config import Settings
from las_repro.media import (
    DownloadTooLarge,
    FrameRef,
    MediaDownloadError,
    MediaProbeError,
    MediaResolver,
    MediaSourceRejected,
    TimeSpan,
    TosAdapter,
    TosNotConfigured,
    TosUri,
    extract_frames,
    plan_segments,
    probe_video,
)


def _public_dns(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, *chunks: bytes):
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks


class _SplitForbiddenTimeBase(str):
    def split(self, *args: object, **kwargs: object) -> list[str]:
        raise AssertionError("time_base must be bounded before splitting")


def _http_resolver(tmp_path: Path, handler, *, max_bytes: int = 8) -> MediaResolver:
    settings = Settings(
        allowed_media_roots=(tmp_path,),
        max_download_bytes=max_bytes,
    )
    return MediaResolver(
        settings,
        http_transport=httpx.MockTransport(handler),
        dns_resolver=_public_dns,
    )


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


def test_local_resolution_accepts_regular_file_inside_allowed_root(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    media = allowed / "video.mp4"
    media.write_bytes(b"video")
    resolver = MediaResolver(Settings(allowed_media_roots=(allowed,)))

    assert resolver.resolve(str(media), tmp_path / "task") == media.resolve()


def test_local_resolution_rejects_symlink_escaping_allowed_root(tmp_path: Path):
    """Checking the lexical path instead of the resolved target enables root escape."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    link = allowed / "escape.mp4"
    link.symlink_to(outside)
    resolver = MediaResolver(Settings(allowed_media_roots=(allowed,)))

    with pytest.raises(MediaSourceRejected):
        resolver.resolve(str(link), tmp_path / "task")


def test_local_resolution_rejects_non_regular_fifo(tmp_path: Path):
    """Reading a FIFO as uploaded media can block a worker indefinitely."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    fifo = allowed / "stream"
    os.mkfifo(fifo)
    resolver = MediaResolver(Settings(allowed_media_roots=(allowed,)))

    with pytest.raises(MediaSourceRejected):
        resolver.resolve(str(fifo), tmp_path / "task")


def test_local_resolution_does_not_follow_target_swapped_after_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A resolved in-root file replaced by a symlink must not authorize its target."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    media = allowed / "video.mp4"
    media.write_bytes(b"inside")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    original_resolve = Path.resolve
    swapped = False

    def resolving_then_swapping(self: Path, *args, **kwargs):
        nonlocal swapped
        result = original_resolve(self, *args, **kwargs)
        if self == media and not swapped:
            swapped = True
            media.unlink()
            media.symlink_to(outside)
        return result

    monkeypatch.setattr(Path, "resolve", resolving_then_swapping)
    resolver = MediaResolver(Settings(allowed_media_roots=(allowed,)))

    with pytest.raises(MediaSourceRejected):
        resolver.resolve(str(media), tmp_path / "task")


def test_http_rejects_literal_loopback_before_network_access(tmp_path: Path):
    called = False

    def handler(request: httpx.Request):
        nonlocal called
        called = True
        return httpx.Response(200, content=b"data")

    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,)),
        http_transport=httpx.MockTransport(handler),
    )

    with pytest.raises(MediaSourceRejected):
        resolver.resolve("http://127.0.0.1/video.mp4", tmp_path / "task")
    assert called is False


def test_http_rejects_hostname_resolving_to_private_address(tmp_path: Path):
    def private_dns(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 80))]

    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,)),
        http_transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"data")),
        dns_resolver=private_dns,
    )

    with pytest.raises(MediaSourceRejected):
        resolver.resolve("http://media.example/video.mp4", tmp_path / "task")


@pytest.mark.parametrize("address", ["224.0.0.1", "ff02::1"])
def test_http_rejects_multicast_dns_answers(tmp_path: Path, address: str):
    def multicast_dns(*args, **kwargs):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        socket_address = (address, 80, 0, 0) if family == socket.AF_INET6 else (address, 80)
        return [(family, socket.SOCK_STREAM, 6, "", socket_address)]

    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,)),
        http_transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"data")),
        dns_resolver=multicast_dns,
    )

    with pytest.raises(MediaSourceRejected):
        resolver.resolve("http://media.example/video.mp4", tmp_path / "task")


def test_http_connection_uses_only_the_vetted_numeric_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The socket connection must not resolve the approved hostname a second time."""
    client_socket, server_socket = socket.socketpair()
    attempted_addresses: list[tuple[str, int]] = []
    received_requests: list[bytes] = []

    def fake_create_connection(address, timeout=None, source_address=None):
        attempted_addresses.append(address)
        return client_socket

    def serve_response() -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += server_socket.recv(4096)
            received_requests.append(request)
            server_socket.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndata"
            )
        finally:
            server_socket.close()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    server = threading.Thread(target=serve_response, daemon=True)
    server.start()
    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,), max_download_bytes=8),
        dns_resolver=_public_dns,
    )

    result = resolver.resolve("http://media.example/video.mp4", tmp_path / "task")
    server.join(timeout=2)

    assert result.read_bytes() == b"data"
    assert attempted_addresses == [("93.184.216.34", 80)]
    assert b"Host: media.example\r\n" in received_requests[0]


def test_http_rejects_truncated_declared_body_without_publishing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client_socket, server_socket = socket.socketpair()

    def fake_create_connection(address, timeout=None, source_address=None):
        return client_socket

    def serve_truncated_response() -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += server_socket.recv(4096)
            server_socket.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\n1234"
            )
        finally:
            server_socket.close()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    server = threading.Thread(target=serve_truncated_response, daemon=True)
    server.start()
    task_dir = tmp_path / "task"
    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,), max_download_bytes=8),
        dns_resolver=_public_dns,
    )

    with pytest.raises(MediaDownloadError):
        resolver.resolve("http://media.example/video.mp4", task_dir)
    server.join(timeout=2)

    assert list(task_dir.glob("media*")) == []
    assert list(task_dir.glob("*.part")) == []


def test_http_normalizes_body_timeout_and_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client_socket, server_socket = socket.socketpair()

    def fake_create_connection(address, timeout=None, source_address=None):
        return client_socket

    def serve_headers() -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += server_socket.recv(4096)
            server_socket.sendall(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
        finally:
            server_socket.close()

    def timing_out_reader(response):
        yield b"1234"
        raise TimeoutError("socket read timed out")

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    monkeypatch.setattr(media_module, "_read_http_response", timing_out_reader)
    server = threading.Thread(target=serve_headers, daemon=True)
    server.start()
    task_dir = tmp_path / "task"
    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,), max_download_bytes=8),
        dns_resolver=_public_dns,
    )

    with pytest.raises(MediaDownloadError):
        resolver.resolve("http://media.example/video.mp4", task_dir)
    server.join(timeout=2)

    assert list(task_dir.glob("media*")) == []
    assert list(task_dir.glob("*.part")) == []


def test_https_keeps_connect_timeout_through_tls_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client_socket, server_socket = socket.socketpair()
    handshake_timeouts: list[float | None] = []

    class FakeTlsContext:
        def wrap_socket(self, sock, server_hostname=None):
            handshake_timeouts.append(sock.gettimeout())
            return sock

    def fake_create_connection(address, timeout=None, source_address=None):
        client_socket.settimeout(timeout)
        return client_socket

    def serve_response() -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += server_socket.recv(4096)
            server_socket.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndata"
            )
        finally:
            server_socket.close()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    monkeypatch.setattr(ssl, "create_default_context", lambda: FakeTlsContext())
    server = threading.Thread(target=serve_response, daemon=True)
    server.start()
    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,), max_download_bytes=8),
        dns_resolver=_public_dns,
    )

    result = resolver.resolve("https://media.example/video.mp4", tmp_path / "task")
    server.join(timeout=2)

    assert result.read_bytes() == b"data"
    assert handshake_timeouts == [5.0]


def test_http_rejects_malformed_authority_with_stable_media_error(tmp_path: Path):
    resolver = MediaResolver(Settings(allowed_media_roots=(tmp_path,)))

    with pytest.raises(MediaSourceRejected):
        resolver.resolve("http://[malformed/video.mp4", tmp_path / "task")


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:not-a-port/video.mp4",
        "http://8.8.8.8:65536/video.mp4",
        "http://[2606:4700:4700::1111]:not-a-port/video.mp4",
        "http://[2606:4700:4700::1111]:65536/video.mp4",
    ],
)
def test_http_rejects_malformed_literal_port_before_network_access(
    tmp_path: Path, url: str
):
    called = False

    def handler(request: httpx.Request):
        nonlocal called
        called = True
        return httpx.Response(200, content=b"data")

    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,)),
        http_transport=httpx.MockTransport(handler),
    )

    with pytest.raises(MediaSourceRejected):
        resolver.resolve(url, tmp_path / "task")
    assert called is False


def test_http_rejects_malformed_literal_port_redirect_before_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client_socket, server_socket = socket.socketpair()
    attempted_addresses: list[tuple[str, int]] = []

    def fake_create_connection(address, timeout=None, source_address=None):
        attempted_addresses.append(address)
        return client_socket

    def serve_redirect() -> None:
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += server_socket.recv(4096)
            server_socket.sendall(
                b"HTTP/1.1 302 Found\r\n"
                b"Location: http://8.8.8.8:not-a-port/video.mp4\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
        finally:
            server_socket.close()

    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    server = threading.Thread(target=serve_redirect, daemon=True)
    server.start()
    resolver = MediaResolver(
        Settings(allowed_media_roots=(tmp_path,)),
        dns_resolver=_public_dns,
    )

    with pytest.raises(MediaSourceRejected):
        resolver.resolve("http://media.example/video.mp4", tmp_path / "task")
    server.join(timeout=2)

    assert attempted_addresses == [("93.184.216.34", 80)]


def test_http_revalidates_each_redirect_target(tmp_path: Path):
    requested_hosts: list[str] = []

    def handler(request: httpx.Request):
        requested_hosts.append(request.url.host)
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/secret"})

    resolver = _http_resolver(tmp_path, handler)

    with pytest.raises(MediaSourceRejected):
        resolver.resolve("https://media.example/video.mp4", tmp_path / "task")
    assert requested_hosts == ["media.example"]


def test_http_rejects_oversize_content_length_without_part_file(tmp_path: Path):
    resolver = _http_resolver(
        tmp_path,
        lambda request: httpx.Response(200, headers={"Content-Length": "9"}, content=b""),
    )
    task_dir = tmp_path / "task"

    with pytest.raises(DownloadTooLarge):
        resolver.resolve("https://media.example/video.mp4", task_dir)
    assert list(task_dir.glob("*.part")) == []


def test_http_stops_streamed_body_at_limit_and_removes_part_file(tmp_path: Path):
    resolver = _http_resolver(
        tmp_path,
        lambda request: httpx.Response(200, stream=_ChunkStream(b"12345", b"6789")),
    )
    task_dir = tmp_path / "task"

    with pytest.raises(DownloadTooLarge):
        resolver.resolve("https://media.example/video.mp4", task_dir)
    assert list(task_dir.glob("*.part")) == []
    assert list(task_dir.glob("media*")) == []


def test_http_download_streams_to_task_local_atomic_destination(tmp_path: Path):
    resolver = _http_resolver(
        tmp_path,
        lambda request: httpx.Response(200, stream=_ChunkStream(b"1234", b"5678")),
    )

    resolved = resolver.resolve("https://media.example/video.mp4", tmp_path / "task")

    assert resolved == tmp_path / "task" / "media.mp4"
    assert resolved.read_bytes() == b"12345678"
    assert list(resolved.parent.glob("*.part")) == []


def test_tos_without_injected_adapter_is_explicitly_rejected(tmp_path: Path):
    resolver = MediaResolver(Settings(allowed_media_roots=(tmp_path,)))

    with pytest.raises(TosNotConfigured):
        resolver.resolve("tos://bucket/video.mp4", tmp_path / "task")


@pytest.mark.parametrize(
    "uri",
    [
        "tos:///key",
        "tos://bucket",
        "tos://bucket/",
        "tos://bucket/key?query=forbidden",
        "tos://bucket:not-a-port/key",
        "tos://[malformed/key",
    ],
)
def test_tos_rejects_missing_or_ambiguous_object_identity(tmp_path: Path, uri: str):
    resolver = MediaResolver(Settings(allowed_media_roots=(tmp_path,)), tos_adapter=object())

    with pytest.raises(MediaSourceRejected):
        resolver.resolve(uri, tmp_path / "task")


def test_injected_tos_receives_parsed_unescaped_key_without_logging_credentials(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    class FakeTosAdapter:
        def __init__(self):
            self.received = None

        def download(self, uri: TosUri, destination: Path, max_bytes: int) -> Path:
            self.received = (uri, destination, max_bytes)
            destination.write_bytes(b"tos-video")
            return destination

    fake = FakeTosAdapter()
    access_key = "AKIA-MUST-NOT-BE-LOGGED"
    secret_key = "SECRET-MUST-NOT-BE-LOGGED"
    settings = Settings(
        allowed_media_roots=(tmp_path,),
        max_download_bytes=123,
        tos_endpoint="tos.example.invalid",
        tos_access_key=access_key,
        tos_secret_key=secret_key,
    )
    resolver = MediaResolver(settings, tos_adapter=fake)

    with caplog.at_level(logging.DEBUG):
        resolved = resolver.resolve("tos://videos/folder%2Fclip.mp4", tmp_path / "task")

    assert fake.received == (
        TosUri(bucket="videos", key="folder%2Fclip.mp4"),
        tmp_path / "task" / "media.mp4",
        123,
    )
    assert resolved.read_bytes() == b"tos-video"
    assert access_key not in caplog.text
    assert secret_key not in caplog.text


def test_tos_adapter_requires_complete_configuration_before_sdk_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Missing credentials must not trigger optional SDK loading or network work."""
    monkeypatch.delitem(sys.modules, "tos", raising=False)
    adapter = TosAdapter(Settings(tos_endpoint="tos.example.invalid"))

    with pytest.raises(TosNotConfigured):
        adapter.download(TosUri("videos", "clip.mp4"), tmp_path / "clip.mp4", 8)

    assert "tos" not in sys.modules


def test_tos_adapter_lazily_streams_with_config_credentials_and_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A TOS client must receive configured secrets while logs and partial files do not."""
    calls: dict[str, object] = {}

    class FakeResponse:
        content_length = 8

        def __init__(self):
            self._chunks = iter((b"1234", b"5678", b""))

        def read(self, size: int) -> bytes:
            calls.setdefault("read_sizes", []).append(size)
            return next(self._chunks)

        def close(self) -> None:
            calls["closed"] = True

    class FakeClient:
        def __init__(self, access_key, secret_key, endpoint, region):
            calls["client"] = (access_key, secret_key, endpoint, region)

        def get_object(self, bucket: str, key: str) -> FakeResponse:
            calls["object"] = (bucket, key)
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "tos", types.SimpleNamespace(TosClientV2=FakeClient))
    access_key = "TOS-ACCESS-SECRET"
    secret_key = "TOS-SECRET-SECRET"
    adapter = TosAdapter(
        Settings(
            tos_endpoint="tos.example.invalid",
            tos_region="cn-beijing",
            tos_access_key=access_key,
            tos_secret_key=secret_key,
        )
    )

    with caplog.at_level(logging.DEBUG):
        result = adapter.download(
            TosUri("videos", "folder%2Fclip.mp4"),
            tmp_path / "task" / "media.mp4",
            8,
        )

    assert result.read_bytes() == b"12345678"
    assert calls["client"] == (
        access_key,
        secret_key,
        "tos.example.invalid",
        "cn-beijing",
    )
    assert calls["object"] == ("videos", "folder%2Fclip.mp4")
    assert calls["closed"] is True
    assert list(result.parent.glob("*.part")) == []
    assert access_key not in caplog.text
    assert secret_key not in caplog.text


def test_tos_adapter_enforces_streamed_limit_and_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeResponse:
        content_length = None

        def __init__(self):
            self._chunks = iter((b"12345", b"6789", b""))

        def read(self, size: int) -> bytes:
            return next(self._chunks)

        def close(self) -> None:
            pass

    class FakeClient:
        def __init__(self, *args):
            pass

        def get_object(self, bucket: str, key: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "tos", types.SimpleNamespace(TosClientV2=FakeClient))
    adapter = TosAdapter(
        Settings(
            tos_endpoint="tos.example.invalid",
            tos_region="cn-beijing",
            tos_access_key="access",
            tos_secret_key="secret",
        )
    )
    destination = tmp_path / "task" / "media.mp4"

    with pytest.raises(DownloadTooLarge):
        adapter.download(TosUri("videos", "clip.mp4"), destination, 8)

    assert destination.exists() is False
    assert list(destination.parent.glob("*.part")) == []
