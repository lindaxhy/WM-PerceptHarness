#!/usr/bin/env python3
"""Run one closed, local-only SAM3.1 production-boundary smoke."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

from percept_harness.cv.artifacts import CvArtifactError, CvArtifactStore, cv_cache_key
from percept_harness.cv.contracts import (
    CvEvidenceRequest,
    EntityPrompt,
    EntityRole,
    EvidenceStatus,
    EvidenceThresholds,
    FrameTimeline,
    SamplingPolicy,
)
from percept_harness.cv.timeline import probe_frame_timeline
from percept_harness.media import probe_video

PINNED_REPOSITORY_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
PHYSICAL_CV_DEVICE = 3


class SmokeDependencies:
    """Inject only the expensive runtime boundary in deterministic tests."""

    def __init__(
        self,
        *,
        load_provider: Callable[..., Any],
        probe_timeline: Callable[[Path], FrameTimeline],
        probe_duration: Callable[[Path], float],
        probe_gpu: Callable[[], tuple[str, int]],
        monotonic: Callable[[], float],
    ) -> None:
        self.load_provider = load_provider
        self.probe_timeline = probe_timeline
        self.probe_duration = probe_duration
        self.probe_gpu = probe_gpu
        self.monotonic = monotonic


def _load_provider(**kwargs: Any) -> Any:
    from percept_harness.cv.sam31 import Sam31EvidenceProvider

    return Sam31EvidenceProvider.load(**kwargs)


def _probe_gpu() -> tuple[str, int]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError
    logical_index = torch.cuda.current_device()
    return str(torch.cuda.get_device_name(logical_index)), logical_index


DEFAULT_DEPENDENCIES = SmokeDependencies(
    load_provider=_load_provider,
    probe_timeline=probe_frame_timeline,
    probe_duration=lambda path: probe_video(path).duration,
    probe_gpu=_probe_gpu,
    monotonic=time.monotonic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one local SAM3.1 artifact publish/reload smoke."
    )
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--allowed-media-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    return parser


def _canonical_record(record: dict[str, object]) -> None:
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))


@contextmanager
def _suppress_runtime_stdout() -> Iterator[None]:
    """Discard Python, native-extension, and child-process stdout."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        saved_stdout = os.dup(1)
        try:
            sys.stdout.flush()
            os.dup2(sink.fileno(), 1)
            with redirect_stdout(sink):
                yield
        finally:
            flush_result: int | None = None
            try:
                sink.flush()
                libc = ctypes.CDLL(None)
                libc.fflush.argtypes = [ctypes.c_void_p]
                libc.fflush.restype = ctypes.c_int
                flush_result = libc.fflush(None)
            finally:
                os.dup2(saved_stdout, 1)
                os.close(saved_stdout)
            if flush_result != 0:
                raise OSError


def _regular_file(path: Path, category: str) -> tuple[Path, tuple[int, int]]:
    try:
        absolute = Path(os.path.abspath(path))
        status = absolute.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise ValueError
        return absolute, (status.st_dev, status.st_ino)
    except (OSError, ValueError):
        raise ValueError(category) from None


def _local_directory(path: Path, category: str) -> Path:
    try:
        absolute = Path(os.path.abspath(path))
        status = absolute.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise ValueError
        return absolute
    except (OSError, ValueError):
        raise ValueError(category) from None


def _contained_video(path: Path, allowed_root: Path) -> Path:
    root = _local_directory(allowed_root, "video").resolve(strict=True)
    video, _identity = _regular_file(path, "video")
    try:
        resolved = video.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise ValueError("video") from None
    return resolved


def _sha256(path: Path, category: str) -> str:
    file_path, identity = _regular_file(path, category)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError(category)
    descriptor: int | None = None
    try:
        descriptor = os.open(file_path, os.O_RDONLY | nofollow)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != identity:
            raise ValueError
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        current = file_path.lstat()
        if (current.st_dev, current.st_ino) != identity:
            raise ValueError
        return digest
    except (OSError, ValueError):
        raise ValueError(category) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _request(
    video: Path,
    timeline: FrameTimeline,
    duration_seconds: float,
    checkpoint_sha256: str,
) -> CvEvidenceRequest:
    if duration_seconds <= 0.0:
        raise ValueError("video")
    return CvEvidenceRequest(
        schema_version="cv_request_v1",
        provider="sam31",
        model_identity="sam3.1",
        video_path=video,
        video_sha256=_sha256(video, "video"),
        duration_seconds=duration_seconds,
        frame_count=len(timeline.frames),
        checkpoint_sha256=checkpoint_sha256,
        timeline=timeline,
        entities=(
            EntityPrompt(
                entity_id="person",
                canonical_label="person",
                aliases=("human",),
                role=EntityRole.ACTOR,
            ),
            EntityPrompt(
                entity_id="handled_object",
                canonical_label="handled object",
                aliases=("object",),
                role=EntityRole.MANIPULATED_OBJECT,
            ),
        ),
        sampling=SamplingPolicy(
            short_video_seconds=30.0,
            scan_fps=8.0,
            max_fps=30.0,
            refinement_radius_seconds=1.0,
        ),
        thresholds=EvidenceThresholds(
            min_confidence=0.5,
            min_area_fraction=0.01,
            occlusion_visibility_drop=0.5,
        ),
    )


def _run(
    arguments: argparse.Namespace, dependencies: SmokeDependencies
) -> dict[str, object]:
    if arguments.device != PHYSICAL_CV_DEVICE:
        raise ValueError("device")
    repository = _local_directory(arguments.repository, "repository")
    checkpoint, _identity = _regular_file(arguments.checkpoint, "checkpoint")
    expected_digest = arguments.checkpoint_sha256
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
        or _sha256(checkpoint, "checkpoint") != expected_digest
    ):
        raise ValueError("checkpoint")
    video = _contained_video(arguments.video, arguments.allowed_media_root)

    # This must precede the provider loader and its optional Torch/SAM imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CV_DEVICE)
    provider: Any = None
    started = dependencies.monotonic()
    try:
        gpu_name, logical_index = dependencies.probe_gpu()
        if logical_index != 0 or not gpu_name:
            raise RuntimeError
        timeline = dependencies.probe_timeline(video)
        duration_seconds = dependencies.probe_duration(video)
        request = _request(video, timeline, duration_seconds, expected_digest)
        provider = dependencies.load_provider(
            repository_path=repository,
            checkpoint_path=checkpoint,
            checkpoint_sha256=expected_digest,
        )
        with CvArtifactStore(arguments.cache_root) as store:
            with store.staging(cv_cache_key(request)) as staging:
                artifact = provider.analyze(request, staging)
                try:
                    handle = store.publish(request, staging, artifact)
                except CvArtifactError:
                    raise ValueError("artifact") from None
            try:
                reloaded = store.load(handle)
            except CvArtifactError:
                raise ValueError("artifact") from None
        observations = sum(len(track.observations) for track in reloaded.tracks)
        if (
            reloaded.status is not EvidenceStatus.AVAILABLE
            or not reloaded.tracks
            or observations < 1
        ):
            raise ValueError("artifact")
        metrics = provider.request_metrics()
        return {
            "checkpoint_sha256": expected_digest,
            "elapsed_seconds": round(dependencies.monotonic() - started, 3),
            "frame_count": len(timeline.frames),
            "gpu_index": PHYSICAL_CV_DEVICE,
            "gpu_name": gpu_name,
            "observation_count": observations,
            "pass": True,
            "peak_allocated_bytes": int(metrics["peak_allocated_bytes"]),
            "source_revision": PINNED_REPOSITORY_REVISION,
            "track_count": len(reloaded.tracks),
        }
    finally:
        if provider is not None:
            provider.close()


def main(
    argv: list[str] | None = None,
    *,
    dependencies: SmokeDependencies = DEFAULT_DEPENDENCIES,
) -> int:
    arguments = _parser().parse_args(argv)
    try:
        with _suppress_runtime_stdout():
            record = _run(arguments, dependencies)
    except ValueError as error:
        category = str(error)
        if category not in {"artifact", "checkpoint", "device", "repository", "video"}:
            category = "runtime"
        _canonical_record({"error": category, "pass": False})
        return 1
    except BaseException:  # noqa: BLE001 - the public smoke record is sanitized.
        _canonical_record({"error": "runtime", "pass": False})
        return 1
    _canonical_record(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
