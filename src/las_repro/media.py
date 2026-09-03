"""Safe media resolution, download, probing, segmentation, and frame extraction."""

from __future__ import annotations

import importlib
import http.client
import ipaddress
import json
import math
import os
import socket
import ssl
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import SplitResult, urljoin, urlsplit

import httpx

from .config import Settings


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_REDIRECTS = 3
_IO_CHUNK_BYTES = 1024 * 1024
_TIMESTAMP_QUANTUM = Decimal("0.000001")
_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 30.0
_MAX_TIME_BASE_CHARS = 64


class MediaError(RuntimeError):
    """Base class for stable media-processing failures."""


class MediaSourceRejected(MediaError):
    """The supplied source violates the configured media policy."""


class DownloadTooLarge(MediaError):
    """A remote object exceeds the configured byte limit."""


class MediaDownloadError(MediaError):
    """A remote object could not be downloaded safely."""


class TosNotConfigured(MediaError):
    """TOS access was requested without a configured adapter or credentials."""


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


@dataclass(frozen=True)
class TosUri:
    bucket: str
    key: str


class TosDownloader(Protocol):
    def download(self, uri: TosUri, destination: Path, max_bytes: int) -> Path:
        """Download one parsed TOS object to ``destination``."""


DnsResolver = Callable[..., list[tuple[Any, ...]]]


class MediaResolver:
    """Resolve configured local, HTTP(S), or injected TOS media sources.

    Allowed local roots are an operator-managed trust boundary: untrusted
    requesters must not have filesystem write access to them.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        tos_adapter: TosDownloader | None = None,
        http_transport: httpx.BaseTransport | None = None,
        dns_resolver: DnsResolver = socket.getaddrinfo,
    ) -> None:
        self._settings = settings
        self._tos_adapter = tos_adapter
        self._http_transport = http_transport
        self._dns_resolver = dns_resolver

    def resolve(self, video_url: str, task_dir: Path) -> Path:
        """Return a policy-approved local path for ``video_url``."""
        if not isinstance(video_url, str) or not video_url:
            raise MediaSourceRejected("media source must be a non-empty string")

        parsed = _safe_urlsplit(video_url)
        scheme = parsed.scheme.lower()
        if scheme in {"http", "https"}:
            destination = self._remote_destination(task_dir, parsed.path)
            return self._download_http(video_url, destination)
        if scheme == "tos":
            uri = _parse_tos_uri(parsed)
            if self._tos_adapter is None:
                raise TosNotConfigured("TOS adapter is not configured")
            destination = self._remote_destination(task_dir, parsed.path)
            try:
                result = self._tos_adapter.download(
                    uri,
                    destination,
                    self._settings.max_download_bytes,
                )
            except MediaError:
                raise
            except Exception:
                raise MediaDownloadError("TOS download failed") from None
            return _require_expected_download(result, destination)
        if scheme:
            raise MediaSourceRejected("unsupported media source scheme")
        return self._resolve_local(Path(video_url))

    def _resolve_local(self, source: Path) -> Path:
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError):
            raise MediaSourceRejected("local media does not exist") from None

        allowed = False
        for configured_root in self._settings.allowed_media_roots:
            try:
                root = configured_root.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if not root.is_dir():
                continue
            if resolved.is_relative_to(root):
                allowed = True
                break
        if not allowed:
            raise MediaSourceRejected("local media is outside allowed roots")

        try:
            mode = resolved.lstat().st_mode
        except OSError:
            raise MediaSourceRejected("local media cannot be inspected") from None
        if not stat.S_ISREG(mode):
            raise MediaSourceRejected("local media must be a regular file")
        return resolved

    def _remote_destination(self, task_dir: Path, remote_path: str) -> Path:
        try:
            task_dir.mkdir(parents=True, exist_ok=True)
            resolved_task_dir = task_dir.resolve(strict=True)
        except (OSError, RuntimeError):
            raise MediaSourceRejected("task directory cannot be prepared") from None
        if not resolved_task_dir.is_dir():
            raise MediaSourceRejected("task destination must be a directory")
        suffix = PurePosixPath(remote_path).suffix
        if not suffix or len(suffix) > 16 or not suffix[1:].isalnum():
            suffix = ".bin"
        return resolved_task_dir / f"media{suffix.lower()}"

    def _download_http(self, source: str, destination: Path) -> Path:
        if self._http_transport is None:
            return self._download_pinned_http(source, destination)
        timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT_SECONDS,
            read=_READ_TIMEOUT_SECONDS,
            write=_READ_TIMEOUT_SECONDS,
            pool=_CONNECT_TIMEOUT_SECONDS,
        )
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=timeout,
                transport=self._http_transport,
                trust_env=False,
            ) as client:
                current = source
                for redirects in range(_MAX_REDIRECTS + 1):
                    _validate_http_target(current, self._dns_resolver)
                    with client.stream("GET", current) as response:
                        if response.status_code in _REDIRECT_STATUSES:
                            if redirects == _MAX_REDIRECTS:
                                raise MediaSourceRejected("HTTP redirect limit exceeded")
                            location = response.headers.get("Location")
                            if not location:
                                raise MediaDownloadError("HTTP redirect omitted Location")
                            try:
                                current = urljoin(current, location)
                            except ValueError:
                                raise MediaSourceRejected("invalid HTTP redirect target") from None
                            continue
                        response.raise_for_status()
                        length = _content_length(response.headers.get("Content-Length"))
                        return _write_atomic_stream(
                            destination,
                            response.iter_bytes(_IO_CHUNK_BYTES),
                            self._settings.max_download_bytes,
                            content_length=length,
                        )
        except MediaError:
            raise
        except (httpx.HTTPError, OSError):
            raise MediaDownloadError("HTTP download failed") from None
        raise MediaDownloadError("HTTP download failed")

    def _download_pinned_http(self, source: str, destination: Path) -> Path:
        current = source
        for redirects in range(_MAX_REDIRECTS + 1):
            parsed, addresses, port = _validate_http_target(current, self._dns_resolver)
            with _open_pinned_response(parsed, addresses, port) as response:
                if response.status in _REDIRECT_STATUSES:
                    if redirects == _MAX_REDIRECTS:
                        raise MediaSourceRejected("HTTP redirect limit exceeded")
                    location = response.getheader("Location")
                    if not location:
                        raise MediaDownloadError("HTTP redirect omitted Location")
                    try:
                        current = urljoin(current, location)
                    except ValueError:
                        raise MediaSourceRejected("invalid HTTP redirect target") from None
                    continue
                if response.status < 200 or response.status >= 300:
                    raise MediaDownloadError("HTTP download returned an error status")
                length = _content_length(response.getheader("Content-Length"))
                try:
                    return _write_atomic_stream(
                        destination,
                        _read_http_response(response),
                        self._settings.max_download_bytes,
                        content_length=length,
                    )
                except MediaError:
                    raise
                except (OSError, http.client.HTTPException):
                    raise MediaDownloadError("HTTP response body read failed") from None
        raise MediaDownloadError("HTTP download failed")


class TosAdapter:
    """Lazy Volcengine TOS SDK adapter using only service configuration."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    def download(self, uri: TosUri, destination: Path, max_bytes: int) -> Path:
        if not isinstance(uri, TosUri) or not uri.bucket or not uri.key:
            raise MediaSourceRejected("invalid TOS object identity")
        client = self._get_client()
        response: Any | None = None
        try:
            response = client.get_object(uri.bucket, uri.key)
            length = _optional_nonnegative_int(getattr(response, "content_length", None))
            return _write_atomic_stream(
                destination,
                _read_tos_response(response),
                max_bytes,
                content_length=length,
            )
        except MediaError:
            raise
        except Exception:
            raise MediaDownloadError("TOS download failed") from None
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        endpoint = _secret_value(self._settings.tos_endpoint)
        region = _secret_value(self._settings.tos_region)
        access_key = _secret_value(self._settings.tos_access_key)
        secret_key = _secret_value(self._settings.tos_secret_key)
        if not all((endpoint, region, access_key, secret_key)):
            raise TosNotConfigured("TOS credentials are not configured")
        try:
            tos = importlib.import_module("tos")
            self._client = tos.TosClientV2(access_key, secret_key, endpoint, region)
        except ImportError:
            raise TosNotConfigured("TOS SDK is not installed") from None
        except Exception:
            raise TosNotConfigured("TOS client could not be configured") from None
        return self._client


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
    interval = Decimal(1) / Decimal(str(fps))
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
        timestamp = start + interval * index
    return frames


def _parse_tos_uri(parsed: SplitResult) -> TosUri:
    try:
        port = parsed.port
    except ValueError:
        raise MediaSourceRejected("invalid TOS URI") from None
    if (
        parsed.scheme.lower() != "tos"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or parsed.path == "/"
    ):
        raise MediaSourceRejected("invalid TOS URI")
    key = parsed.path[1:]
    if not key:
        raise MediaSourceRejected("TOS object key is required")
    return TosUri(bucket=parsed.netloc, key=key)


def _validate_http_target(
    url: str,
    resolver: DnsResolver,
) -> tuple[SplitResult, tuple[str, ...], int]:
    parsed = _safe_urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise MediaSourceRejected("HTTP redirect used an unsupported scheme")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise MediaSourceRejected("invalid HTTP media URL")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError:
        raise MediaSourceRejected("HTTP target port is invalid") from None
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None:
        _require_public_address(literal)
        return parsed, (str(literal),), port
    try:
        answers = resolver(parsed.hostname, port, type=socket.SOCK_STREAM)
    except (OSError, ValueError, UnicodeError):
        raise MediaSourceRejected("HTTP target cannot be resolved safely") from None
    if not answers:
        raise MediaSourceRejected("HTTP target has no addresses")
    addresses: list[str] = []
    for answer in answers:
        try:
            address = ipaddress.ip_address(answer[4][0])
        except (IndexError, TypeError, ValueError):
            raise MediaSourceRejected("HTTP target returned an invalid address") from None
        _require_public_address(address)
        normalized = str(address)
        if normalized not in addresses:
            addresses.append(normalized)
    return parsed, tuple(addresses), port


def _require_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if (
        address.is_loopback
        or address.is_link_local
        or address.is_private
        or address.is_multicast
        or address.is_unspecified
        or not address.is_global
    ):
        raise MediaSourceRejected("HTTP target address is not public")


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError:
        raise MediaDownloadError("HTTP Content-Length is invalid") from None
    if length < 0:
        raise MediaDownloadError("HTTP Content-Length is invalid")
    return length


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise MediaDownloadError("remote object length is invalid") from None
    if result < 0:
        raise MediaDownloadError("remote object length is invalid")
    return result


def _write_atomic_stream(
    destination: Path,
    chunks: Iterable[bytes],
    max_bytes: int,
    *,
    content_length: int | None,
) -> Path:
    if max_bytes < 0:
        raise ValueError("max_bytes cannot be negative")
    if content_length is not None and content_length > max_bytes:
        raise DownloadTooLarge("remote media exceeds byte limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise MediaDownloadError("remote media stream returned invalid data")
                total += len(chunk)
                if total > max_bytes:
                    raise DownloadTooLarge("remote media exceeds byte limit")
                output.write(chunk)
            if content_length is not None and total != content_length:
                raise MediaDownloadError("remote media body was incomplete")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        return destination
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _require_expected_download(result: Path, destination: Path) -> Path:
    try:
        resolved_result = Path(result).resolve(strict=True)
        resolved_destination = destination.resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        raise MediaDownloadError("TOS adapter did not create the destination") from None
    if resolved_result != resolved_destination or not resolved_result.is_file():
        raise MediaDownloadError("TOS adapter returned an unexpected destination")
    return resolved_destination


def _read_tos_response(response: Any) -> Iterator[bytes]:
    while True:
        chunk = response.read(_IO_CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


def _read_http_response(response: http.client.HTTPResponse) -> Iterator[bytes]:
    while True:
        chunk = response.read(_IO_CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, address: str, port: int) -> None:
        super().__init__(host, port, timeout=_CONNECT_TIMEOUT_SECONDS)
        self._address = address

    def connect(self) -> None:
        self._connect_tcp()
        self.sock.settimeout(_READ_TIMEOUT_SECONDS)

    def _connect_tcp(self) -> None:
        self.sock = socket.create_connection(
            (self._address, self.port),
            timeout=self.timeout,
        )
        self.sock.settimeout(_CONNECT_TIMEOUT_SECONDS)


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    def __init__(self, host: str, address: str, port: int) -> None:
        super().__init__(host, address, port)
        self._context = ssl.create_default_context()

    def connect(self) -> None:
        self._connect_tcp()
        try:
            self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)
            self.sock.settimeout(_READ_TIMEOUT_SECONDS)
        except BaseException:
            self.close()
            raise


@contextmanager
def _open_pinned_response(
    parsed: SplitResult,
    addresses: tuple[str, ...],
    port: int,
) -> Iterator[http.client.HTTPResponse]:
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    for address in addresses:
        connection_type = (
            _PinnedHTTPSConnection if parsed.scheme.lower() == "https" else _PinnedHTTPConnection
        )
        connection = connection_type(parsed.hostname or "", address, port)
        try:
            connection.request(
                "GET",
                target,
                headers={"Accept-Encoding": "identity", "Connection": "close"},
            )
            response = connection.getresponse()
        except (OSError, http.client.HTTPException):
            connection.close()
            continue
        try:
            yield response
        finally:
            connection.close()
        return
    raise MediaDownloadError("HTTP connection to approved addresses failed") from None


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


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    getter = getattr(value, "get_secret_value", None)
    return getter() if callable(getter) else str(value)


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _safe_urlsplit(value: str) -> SplitResult:
    try:
        return urlsplit(value)
    except ValueError:
        raise MediaSourceRejected("media source URL is malformed") from None
