#!/usr/bin/env python3
"""Run one isolated Qwen3-VL video smoke process per requested CUDA device."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
import queue as queue_module
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


_FAILURE_CODES = frozenset(
    {
        "CUDA_UNAVAILABLE",
        "DEVICE_OUT_OF_RANGE",
        "WRONG_DEVICE",
        "WORKER_FAILED",
        "PROCESS_START_FAILED",
        "PROCESS_EXITED",
        "PROCESS_TIMEOUT",
        "PROCESS_UNREAPABLE",
        "NO_REPORT",
        "INVALID_REPORT",
    }
)
_REPORT_STRINGS = (
    "gpu_name",
    "model_class",
    "model_type",
    "torch_version",
    "transformers_version",
    "qwen_vl_utils_version",
    "device_scope",
)
_SENSITIVE_MARKERS = ("key", "token", "authorization", "secret", "password")
_OUTPUT_CAPTURE_LIMIT = 64 * 1024
_LIFETIME_OUTPUT_RESOURCES: list[Any] = []


class SmokeSafetyError(ValueError):
    """Smoke-test arguments are malformed or unsafe."""


class SmokeProcessError(RuntimeError):
    """A spawned process could not be safely reaped."""


class _ProbeFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def parse_devices(value: str) -> tuple[int, ...]:
    """Parse a nonempty unique comma-separated list of CUDA ordinals."""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SmokeSafetyError("devices must be comma-separated CUDA ordinals")
    parts = value.split(",")
    if any(not part.isdecimal() for part in parts):
        raise SmokeSafetyError("devices must be comma-separated CUDA ordinals")
    devices = tuple(int(part) for part in parts)
    if not devices or len(set(devices)) != len(devices):
        raise SmokeSafetyError("devices must be unique")
    return devices


def run_gpu_smoke(
    model_dir: str | Path,
    video: str | Path,
    devices: Sequence[int],
    *,
    timeout_seconds: float = 1_800.0,
    mp_context: Any = None,
    worker_target: Callable[..., None] | None = None,
    require_unmasked: bool = False,
) -> list[dict[str, Any]]:
    """Spawn one process per post-CUDA_VISIBLE_DEVICES logical ordinal."""
    local_model_dir = _existing_directory(model_dir, "model directory")
    local_video = _existing_file(video, "video")
    assigned_devices = _validated_devices(devices)
    timeout = _positive_finite(timeout_seconds, "timeout_seconds")
    if not isinstance(require_unmasked, bool):
        raise SmokeSafetyError("require_unmasked must be a boolean")
    if require_unmasked and "CUDA_VISIBLE_DEVICES" in os.environ:
        raise SmokeSafetyError("CUDA visibility mask is not allowed in acceptance mode")
    context = mp_context or multiprocessing.get_context("spawn")
    target = worker_target or _device_worker
    processes: dict[int, tuple[Any, Any]] = {}
    reports: dict[int, dict[str, Any]] = {}

    try:
        for device in assigned_devices:
            result_queue = context.Queue()
            process = context.Process(
                target=_isolated_worker,
                args=(
                    target,
                    result_queue,
                    str(local_model_dir),
                    str(local_video),
                    device,
                ),
            )
            processes[device] = (process, result_queue)
            try:
                _start_process_with_deferred_sigint(process)
            except Exception:
                reports[device] = _failure_report(device, "PROCESS_START_FAILED")

        deadline = time.monotonic() + timeout
        for device in assigned_devices:
            process, result_queue = processes[device]
            if device in reports:
                _close_queue(result_queue)
                continue
            remaining = max(0.0, deadline - time.monotonic())
            process.join(timeout=remaining)
            if process.is_alive():
                reaped = _reap_process(process)
                reports[device] = _failure_report(
                    device,
                    "PROCESS_TIMEOUT" if reaped else "PROCESS_UNREAPABLE",
                )
                _close_queue(result_queue)
                continue
            if process.exitcode not in (0, None):
                reports[device] = _failure_report(device, "PROCESS_EXITED")
                _close_queue(result_queue)
                continue
            try:
                raw_report = result_queue.get(timeout=0.5)
            except (queue_module.Empty, EOFError, OSError):
                code = (
                    "PROCESS_EXITED"
                    if process.exitcode not in (0, None)
                    else "NO_REPORT"
                )
                reports[device] = _failure_report(device, code)
            else:
                reports[device] = _sanitize_report(raw_report, device)
            finally:
                _close_queue(result_queue)

        return [reports[device] for device in sorted(assigned_devices)]
    finally:
        unreapable = False
        for process, result_queue in processes.values():
            try:
                if not _reap_process(process):
                    unreapable = True
            except Exception:
                unreapable = True
            _close_queue(result_queue)
        if unreapable:
            raise SmokeProcessError("spawned GPU smoke process could not be reaped")


def _probe_device(
    model_dir: str | Path,
    video: str | Path,
    device: int,
    *,
    torch_module: Any = None,
    backend_loader: Callable[[Path, str, str], Any] | None = None,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    clock: Callable[[], float] = time.perf_counter,
    visible_devices: str | None | object = ...,
) -> dict[str, Any]:
    """Load and exercise one backend in a child process without logging paths."""
    local_model_dir = _existing_directory(model_dir, "model directory")
    local_video = _existing_file(video, "video")
    assigned = _validated_devices((device,))[0]
    torch = torch_module or importlib.import_module("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise _ProbeFailure("CUDA_UNAVAILABLE")
    if assigned >= cuda.device_count():
        raise _ProbeFailure("DEVICE_OUT_OF_RANGE")

    cuda.set_device(assigned)
    if cuda.current_device() != assigned:
        raise _ProbeFailure("WRONG_DEVICE")
    cuda.reset_peak_memory_stats(assigned)
    loader = backend_loader or _qwen_loader()
    started = _finite_clock(clock())
    backend = loader(local_model_dir, f"cuda:{assigned}", "auto")
    if not backend.loaded_on_assigned_device():
        raise _ProbeFailure("WRONG_DEVICE")

    from las_repro.media import TimeSpan
    from las_repro.models.base import ModelRequest

    result = backend.generate(
        ModelRequest(
            stage="general_segment",
            video_path=local_video,
            span=TimeSpan(0.0, 1.0),
            fps=1.0,
            prompt='Return exactly this JSON object: {"ok":true}',
            schema_name="gpu_smoke",
            video_session_id=None,
            media_resolution="low",
            reasoning_effort="low",
            clip_context="low",
        )
    )
    finished = _finite_clock(clock())
    if result != {"ok": True}:
        raise _ProbeFailure("WORKER_FAILED")
    observed = cuda.current_device()
    if observed != assigned:
        raise _ProbeFailure("WRONG_DEVICE")
    peak = cuda.max_memory_allocated(assigned)
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise _ProbeFailure("WORKER_FAILED")

    model = getattr(backend, "model", None)
    config = getattr(model, "config", None)
    return {
        "status": "passed",
        "assigned_device": assigned,
        "observed_device": observed,
        "gpu_name": str(cuda.get_device_name(assigned)),
        "model_class": type(model).__name__,
        "model_type": str(getattr(config, "model_type", "unknown")),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "transformers_version": str(version_lookup("transformers")),
        "qwen_vl_utils_version": str(version_lookup("qwen-vl-utils")),
        "device_scope": _device_scope(visible_devices),
        "latency_seconds": finished - started,
        "peak_allocated_bytes": peak,
    }


def _device_worker(
    result_queue: Any,
    model_dir: str,
    video: str,
    device: int,
) -> None:
    try:
        report = _probe_device(model_dir, video, device)
    except _ProbeFailure as error:
        report = _failure_report(device, error.code)
    except BaseException:
        report = _failure_report(device, "WORKER_FAILED")
    try:
        result_queue.put(report)
    except BaseException:
        pass


def _isolated_worker(
    target: Callable[..., None],
    result_queue: Any,
    model_dir: str,
    video: str,
    device: int,
) -> None:
    if multiprocessing.current_process().name != "MainProcess":
        try:
            _install_lifetime_output_sink()
        except BaseException:
            return
        target(result_queue, model_dir, video, device)
        return
    with _suppress_native_output():
        target(result_queue, model_dir, video, device)


class _FdText(io.TextIOBase):
    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8", errors="replace")
        offset = 0
        while offset < len(encoded):
            offset += os.write(self._descriptor, encoded[offset:])
        return len(value)

    def flush(self) -> None:
        return None


@contextlib.contextmanager
def _suppress_native_output() -> Any:
    """Capture bounded Python/native child streams, then discard them."""
    saved: list[tuple[int, int]] = []
    temporary_files = [
        tempfile.TemporaryFile(prefix="las-gpu-smoke-output-") for _ in range(2)
    ]
    pipes = [os.pipe() for _ in range(2)]
    threads = [
        threading.Thread(
            target=_drain_bounded,
            args=(read_fd, temporary),
            daemon=True,
        )
        for (read_fd, _), temporary in zip(pipes, temporary_files, strict=True)
    ]
    try:
        for thread in threads:
            thread.start()
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for target, (_, write_fd) in zip((1, 2), pipes, strict=True):
            saved.append((target, os.dup(target)))
            os.dup2(write_fd, target)
            os.close(write_fd)
        with contextlib.redirect_stdout(_FdText(1)), contextlib.redirect_stderr(
            _FdText(2)
        ):
            yield
    finally:
        for target, descriptor in reversed(saved):
            try:
                os.dup2(descriptor, target)
            finally:
                os.close(descriptor)
        for thread in threads:
            thread.join(timeout=1.0)
        for temporary in temporary_files:
            temporary.close()


def _drain_bounded(read_fd: int, destination: Any) -> None:
    remaining = _OUTPUT_CAPTURE_LIMIT
    try:
        with os.fdopen(read_fd, "rb", closefd=True) as source:
            while True:
                chunk = source.read(8192)
                if not chunk:
                    return
                if remaining > 0:
                    retained = chunk[:remaining]
                    destination.write(retained)
                    remaining -= len(retained)
    except (OSError, ValueError):
        return


def _install_lifetime_output_sink() -> None:
    """Permanently redirect a spawned child's streams through process exit."""
    temporary_files = [
        tempfile.TemporaryFile(prefix="las-gpu-smoke-process-output-")
        for _ in range(2)
    ]
    pipes = [os.pipe() for _ in range(2)]
    threads = [
        threading.Thread(
            target=_drain_bounded,
            args=(read_fd, temporary),
            daemon=True,
        )
        for (read_fd, _), temporary in zip(pipes, temporary_files, strict=True)
    ]
    for thread in threads:
        thread.start()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    for target, (_, write_fd) in zip((1, 2), pipes, strict=True):
        os.dup2(write_fd, target)
        os.close(write_fd)
    sys.stdout = _FdText(1)
    sys.stderr = _FdText(2)
    _LIFETIME_OUTPUT_RESOURCES.extend(temporary_files)
    _LIFETIME_OUTPUT_RESOURCES.extend(threads)


def _reap_process(process: Any) -> bool:
    if not process.is_alive():
        return True
    process.terminate()
    process.join(timeout=5.0)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if not callable(kill):
            return False
        kill()
        process.join(timeout=5.0)
    return not process.is_alive()


def _start_process_with_deferred_sigint(process: Any) -> None:
    if threading.current_thread() is not threading.main_thread():
        process.start()
        return
    previous = signal.getsignal(signal.SIGINT)
    received: list[tuple[int, Any]] = []

    def defer(signum: int, frame: Any) -> None:
        if not received:
            received.append((signum, frame))

    signal.signal(signal.SIGINT, defer)
    try:
        process.start()
    finally:
        signal.signal(signal.SIGINT, previous)
        if received:
            signum, frame = received[0]
            if previous is signal.SIG_IGN:
                pass
            elif callable(previous):
                previous(signum, frame)
            else:
                raise KeyboardInterrupt


def _qwen_loader() -> Callable[[Path, str, str], Any]:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    return Qwen3VLModel.load


def _sanitize_report(report: Any, expected_device: int) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return _failure_report(expected_device, "INVALID_REPORT")
    assigned = report.get("assigned_device")
    observed = report.get("observed_device")
    if assigned != expected_device or (
        observed is not None and observed != expected_device
    ):
        return _failure_report(expected_device, "WRONG_DEVICE")
    if report.get("status") != "passed":
        code = report.get("error_code")
        return _failure_report(
            expected_device,
            code if isinstance(code, str) and code in _FAILURE_CODES else "WORKER_FAILED",
        )

    strings: dict[str, str] = {}
    for name in _REPORT_STRINGS:
        value = _safe_report_string(report.get(name))
        if value is None:
            return _failure_report(expected_device, "INVALID_REPORT")
        strings[name] = value
    latency = report.get("latency_seconds")
    peak = report.get("peak_allocated_bytes")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or latency < 0
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak < 0
    ):
        return _failure_report(expected_device, "INVALID_REPORT")
    return {
        "status": "passed",
        "assigned_device": expected_device,
        "observed_device": expected_device,
        **strings,
        "latency_seconds": float(latency),
        "peak_allocated_bytes": peak,
    }


def _safe_report_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 160:
        return None
    if any(not character.isprintable() for character in value):
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return None
    return value


def _device_scope(visible_devices: str | None | object = ...) -> str:
    value = (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is ...
        else visible_devices
    )
    if value is None:
        return "unmasked-logical-ordinal"
    if value == "":
        return "empty-mask-no-visible-devices"
    if not isinstance(value, str):
        return "masked-logical-ordinal-count-unknown"
    return f"masked-logical-ordinal-count-{len(value.split(','))}"


def _failure_report(device: int, code: str) -> dict[str, Any]:
    safe_code = code if code in _FAILURE_CODES else "WORKER_FAILED"
    return {
        "status": "failed",
        "assigned_device": device,
        "error_code": safe_code,
    }


def _validated_devices(devices: Sequence[int]) -> tuple[int, ...]:
    if isinstance(devices, (str, bytes)):
        raise SmokeSafetyError("devices must be CUDA ordinals")
    values = tuple(devices)
    if (
        not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise SmokeSafetyError("devices must be unique non-negative CUDA ordinals")
    return values


def _existing_directory(path: str | Path, name: str) -> Path:
    return _existing_path(path, name, directory=True)


def _existing_file(path: str | Path, name: str) -> Path:
    return _existing_path(path, name, directory=False)


def _existing_path(path: str | Path, name: str, *, directory: bool) -> Path:
    if not isinstance(path, (str, Path)) or not str(path):
        raise SmokeSafetyError(f"{name} must be a local path")
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise SmokeSafetyError(f"{name} is unavailable") from None
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise SmokeSafetyError(f"{name} has the wrong type")
    return resolved


def _positive_finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise SmokeSafetyError(f"{name} must be finite and positive")
    return float(value)


def _finite_clock(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _ProbeFailure("WORKER_FAILED")
    return float(value)


def _close_queue(result_queue: Any) -> None:
    try:
        close = getattr(result_queue, "close", None)
        if callable(close):
            close()
        join_thread = getattr(result_queue, "join_thread", None)
        if callable(join_thread):
            join_thread()
    except Exception:
        pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, "invalid command arguments\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Run an offline strict-JSON Qwen3-VL smoke on each CUDA device."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--devices", default="0,1,2,3", type=parse_devices)
    parser.add_argument("--timeout", default=1_800.0, type=float)
    parser.add_argument(
        "--require-unmasked",
        action="store_true",
        help=(
            "reject CUDA_VISIBLE_DEVICES; --devices values are always logical "
            "ordinals after CUDA visibility filtering"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        reports = run_gpu_smoke(
            args.model_dir,
            args.video,
            args.devices,
            timeout_seconds=args.timeout,
            require_unmasked=args.require_unmasked,
        )
    except SmokeSafetyError:
        print("GPU smoke arguments were rejected", file=sys.stderr)
        return 2
    except SmokeProcessError:
        print(
            json.dumps(
                {"error_code": "PROCESS_UNREAPABLE", "status": "failed"},
                sort_keys=True,
            )
        )
        return 1
    except BaseException:
        print("GPU smoke orchestration failed", file=sys.stderr)
        return 1
    passed = all(report["status"] == "passed" for report in reports)
    print(
        json.dumps(
            {"status": "passed" if passed else "failed", "devices": reports},
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
