"""Lazy, local-only SAM3.1 Object Multiplex evidence adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from typing import Any, Iterator
import zipfile
import zlib

from .base import CvOutOfMemoryError, CvProviderError
from .contracts import (
    ArtifactFile,
    CvEvidenceArtifact,
    CvEvidenceRequest,
    CvTrack,
    EvidenceStatus,
    TrackObservation,
)
from .timeline import (
    SampledFrameSet,
    initial_sample_indices,
    materialize_sampled_frames,
    refinement_sample_indices,
)


_PINNED_REPOSITORY_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
_LOAD_FAILURE = "Unable to load local SAM3.1 runtime"
_INFERENCE_FAILURE = "SAM3.1 CV evidence inference failed"
_OOM_FAILURE = "SAM3.1 CV evidence inference ran out of memory"
_HASH_READ_BYTES = 1024 * 1024
_MAX_OVERLAYS = 24
_NETWORK_PREFIX = re.compile(
    r"(?i)^(?:https?|ssh|git|ftp|s3|gs|hf)://|^[^/\\\s]+@[^/\\\s]+:"
)


@dataclass(frozen=True, slots=True)
class _Detection:
    local_frame_index: int
    source_frame_index: int
    timestamp_seconds: float
    object_id: int
    probability: float
    box_xywh: tuple[float, float, float, float]
    mask: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class _PromptRun:
    entity_id: str
    detections: tuple[_Detection, ...]
    present_ids_by_frame: tuple[frozenset[int], ...]
    mask_height: int
    mask_width: int


@dataclass(frozen=True, slots=True)
class _SamModuleSnapshot:
    modules: dict[str, Any]
    namespaces: dict[str, dict[str, Any]]


class Sam31EvidenceProvider:
    """Adapt one official-compatible SAM3.1 predictor to the CV contract."""

    def __init__(
        self,
        *,
        predictor: Any,
        torch_module: Any,
        execution_chunk_frames: int = 8,
        materialize_frames: Callable[..., SampledFrameSet] = materialize_sampled_frames,
        initial_sampler: Callable[..., tuple[int, ...]] = initial_sample_indices,
        refinement_sampler: Callable[..., tuple[int, ...]] = refinement_sample_indices,
        checkpoint_sha256: str | None = None,
        repository_revision: str = _PINNED_REPOSITORY_REVISION,
    ) -> None:
        _require_predictor_interface(predictor)
        self._predictor = predictor
        self._torch = torch_module
        self._materialize_frames = materialize_frames
        self._initial_sampler = initial_sampler
        self._refinement_sampler = refinement_sampler
        self._execution_chunk_frames = _positive_integer(
            execution_chunk_frames, "execution_chunk_frames"
        )
        self.checkpoint_sha256 = checkpoint_sha256
        self.repository_revision = repository_revision
        self._closed = False
        self._metrics = {
            "processed_frames": 0,
            "entity_prompts": 0,
            "track_count": 0,
            "peak_allocated_bytes": 0,
        }

    @classmethod
    def load(
        cls,
        repository_path: Path | str,
        checkpoint_path: Path | str,
        checkpoint_sha256: str,
        *,
        compile_model: bool = False,
        predictor_factory: Callable[..., Any] | None = None,
    ) -> Sam31EvidenceProvider:
        """Validate and load the pinned local runtime without any download path."""
        imported_snapshot: _SamModuleSnapshot | None = None
        predictor: Any = None
        try:
            if not isinstance(compile_model, bool):
                raise ValueError
            repository = _validated_repository(repository_path)
            checkpoint, digest = _validated_checkpoint(
                checkpoint_path, checkpoint_sha256
            )
            _verify_repository_revision(repository)
            imported_snapshot = _sam_module_snapshot()
            if any(
                not _module_is_beneath(module, repository)
                for module in imported_snapshot.modules.values()
            ):
                raise ValueError
            builder = predictor_factory
            if builder is not None and not callable(builder):
                raise TypeError
            if builder is None:
                builder = _import_local_builder(repository, imported_snapshot)
            torch_module = importlib.import_module("torch")
            predictor = builder(
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                multiplex_count=16,
                gpus_to_use=[0],
                compile=compile_model,
            )
            _require_predictor_interface(predictor)
            if predictor_factory is None or _sam_module_snapshot().modules:
                _verify_loaded_sam_modules(repository)
            return cls(
                predictor=predictor,
                torch_module=torch_module,
                checkpoint_sha256=digest,
                repository_revision=_PINNED_REPOSITORY_REVISION,
            )
        except BaseException as error:
            _shutdown_predictor(predictor)
            if imported_snapshot is not None:
                _restore_sam_modules(imported_snapshot)
            if isinstance(error, Exception):
                raise CvProviderError(_LOAD_FAILURE) from None
            raise

    @property
    def execution_chunk_frames(self) -> int:
        return self._execution_chunk_frames

    @execution_chunk_frames.setter
    def execution_chunk_frames(self, value: int) -> None:
        self._execution_chunk_frames = _positive_integer(
            value, "execution_chunk_frames"
        )

    def set_execution_chunk_frames(self, value: int) -> None:
        """Expose Task-5's positive worker-owned OOM batching interface."""
        self.execution_chunk_frames = value

    def analyze(
        self, request: CvEvidenceRequest, staging_dir: Path
    ) -> CvEvidenceArtifact:
        """Track normalized prompts over one immutable sampled-frame mapping."""
        work_directory: Path | None = None
        output_directories: list[Path] = []
        completed = False
        try:
            if self._closed:
                raise RuntimeError
            validated_request = CvEvidenceRequest.model_validate(request, strict=True)
            if validated_request.provider != "sam31":
                raise ValueError
            if (
                self.checkpoint_sha256 is not None
                and validated_request.checkpoint_sha256 != self.checkpoint_sha256
            ):
                raise ValueError
            staging = _prepare_staging_directory(staging_dir)
            _reset_peak_memory(self._torch)
            self._metrics = {
                "processed_frames": 0,
                "entity_prompts": 0,
                "track_count": 0,
                "peak_allocated_bytes": 0,
            }
            if not validated_request.entities:
                artifact = _empty_artifact(validated_request)
                self._metrics["peak_allocated_bytes"] = _peak_memory(self._torch)
                completed = True
                return artifact

            work_directory = Path(
                tempfile.mkdtemp(prefix=".sam31-frames-", dir=staging)
            )
            initial_indices = tuple(
                self._initial_sampler(
                    validated_request.timeline, validated_request.sampling
                )
            )
            _validate_source_indices(validated_request, initial_indices)
            if (
                validated_request.duration_seconds
                <= validated_request.sampling.short_video_seconds
            ):
                final_frames = self._materialize(
                    validated_request,
                    initial_indices,
                    work_directory / "final",
                )
            else:
                scan_frames = self._materialize(
                    validated_request,
                    initial_indices,
                    work_directory / "scan",
                )
                preliminary = self._run_all_prompts(
                    validated_request, scan_frames
                )
                change_indices = _visibility_change_source_indices(
                    preliminary, scan_frames
                )
                refined = tuple(
                    self._refinement_sampler(
                        validated_request.timeline,
                        change_indices,
                        validated_request.sampling,
                    )
                )
                _validate_source_indices(
                    validated_request, refined, allow_empty=True
                )
                final_indices = tuple(sorted(set(initial_indices) | set(refined)))
                final_frames = self._materialize(
                    validated_request,
                    final_indices,
                    work_directory / "final",
                )

            prompt_runs = self._run_all_prompts(validated_request, final_frames)
            artifact = _build_artifact(
                validated_request,
                staging,
                prompt_runs,
                output_directories,
            )
            self._metrics = {
                "processed_frames": len(final_frames.frames),
                "entity_prompts": len(validated_request.entities),
                "track_count": len(artifact.tracks),
                "peak_allocated_bytes": _peak_memory(self._torch),
            }
            completed = True
            return artifact
        except Exception as error:
            if _is_cuda_oom(self._torch, error):
                _empty_cuda_cache(self._torch)
                raise CvOutOfMemoryError(_OOM_FAILURE) from None
            raise CvProviderError(_INFERENCE_FAILURE) from None
        finally:
            if work_directory is not None:
                _remove_tree(work_directory)
            if not completed:
                for directory in reversed(output_directories):
                    _remove_tree(directory)

    def request_metrics(self) -> dict[str, int]:
        """Return bounded metrics for the most recent completed request."""
        return dict(self._metrics)

    def close(self) -> None:
        """Release predictor-owned process state once."""
        if self._closed:
            return
        self._closed = True
        predictor = self._predictor
        self._predictor = None
        shutdown = getattr(predictor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                raise CvProviderError("Unable to close SAM3.1 runtime") from None

    def _materialize(
        self,
        request: CvEvidenceRequest,
        indices: tuple[int, ...],
        destination: Path,
    ) -> SampledFrameSet:
        sampled = self._materialize_frames(
            request.video_path,
            request.timeline,
            indices,
            destination,
        )
        _validate_sampled_frames(sampled, request, indices, destination)
        return sampled

    def _run_all_prompts(
        self,
        request: CvEvidenceRequest,
        sampled: SampledFrameSet,
    ) -> tuple[_PromptRun, ...]:
        with _predictor_session(self._predictor, sampled) as session_id:
            return tuple(
                self._run_prompt(request, sampled, session_id, entity)
                for entity in request.entities
            )

    def _run_prompt(
        self,
        request: CvEvidenceRequest,
        sampled: SampledFrameSet,
        session_id: str,
        entity: Any,
    ) -> _PromptRun:
        self._predictor.handle_request(
            {"type": "reset_session", "session_id": session_id}
        )
        self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": entity.canonical_label,
                "output_prob_thresh": request.thresholds.min_confidence,
            }
        )
        frame_count = len(sampled.frames)
        chunk_size = self.execution_chunk_frames
        detections: list[_Detection] = []
        present: list[frozenset[int] | None] = [None] * frame_count
        mask_shape: tuple[int, int] | None = None
        for start in range(0, frame_count, chunk_size):
            count = min(chunk_size, frame_count - start)
            stream_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": start,
                "max_frame_num_to_track": count - 1,
                "output_prob_thresh": request.thresholds.min_confidence,
            }
            stream = self._predictor.handle_stream_request(stream_request)
            stream_failed = False
            next_expected_index = start
            try:
                for response in stream:
                    local_index, parsed, response_shape = _parse_frame_response(
                        response,
                        sampled,
                        allowed_indices=range(start, start + count),
                    )
                    if local_index != next_expected_index:
                        raise ValueError
                    next_expected_index += 1
                    if present[local_index] is not None:
                        raise ValueError
                    if mask_shape is None:
                        mask_shape = response_shape
                    elif mask_shape != response_shape:
                        raise ValueError
                    present[local_index] = frozenset(
                        detection.object_id for detection in parsed
                    )
                    detections.extend(parsed)
            except BaseException:
                stream_failed = True
                raise
            finally:
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    try:
                        close_stream()
                    except Exception:
                        if not stream_failed:
                            raise
            if next_expected_index != start + count:
                raise ValueError
        if any(value is None for value in present) or mask_shape is None:
            raise ValueError
        if len({
            (detection.local_frame_index, detection.object_id)
            for detection in detections
        }) != len(detections):
            raise ValueError
        return _PromptRun(
            entity_id=entity.entity_id,
            detections=tuple(
                sorted(
                    detections,
                    key=lambda item: (item.local_frame_index, item.object_id),
                )
            ),
            present_ids_by_frame=tuple(value for value in present if value is not None),
            mask_height=mask_shape[0],
            mask_width=mask_shape[1],
        )


def _empty_artifact(request: CvEvidenceRequest) -> CvEvidenceArtifact:
    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status=EvidenceStatus.AVAILABLE,
        provider="sam31",
        model_identity=request.model_identity,
        video_sha256=request.video_sha256,
        checkpoint_sha256=request.checkpoint_sha256,
        entities=request.entities,
        tracks=(),
        files=(),
    )


def _require_predictor_interface(predictor: Any) -> None:
    if not callable(getattr(predictor, "handle_request", None)):
        raise TypeError
    if not callable(getattr(predictor, "handle_stream_request", None)):
        raise TypeError


def _shutdown_predictor(predictor: Any) -> None:
    shutdown = getattr(predictor, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass


def _validated_repository(value: Path | str) -> Path:
    raw = _local_path(value)
    try:
        status = raw.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise ValueError
        resolved = raw.resolve(strict=True)
        if resolved != raw.absolute():
            raise ValueError
        package = resolved / "sam3" / "__init__.py"
        package_status = package.lstat()
        if stat.S_ISLNK(package_status.st_mode) or not stat.S_ISREG(
            package_status.st_mode
        ):
            raise ValueError
        return resolved
    except (OSError, RuntimeError, ValueError):
        raise ValueError from None


def _validated_checkpoint(
    value: Path | str, expected_sha256: str
) -> tuple[Path, str]:
    if not isinstance(expected_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ) is None:
        raise ValueError
    path = _local_path(value)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError
        resolved = path.resolve(strict=True)
        if resolved != path.absolute():
            raise ValueError
        descriptor = os.open(resolved, os.O_RDONLY | nofollow)
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size <= 0
            or identity != (before.st_dev, before.st_ino)
        ):
            raise ValueError
        digest = hashlib.sha256()
        while payload := os.read(descriptor, _HASH_READ_BYTES):
            digest.update(payload)
        after_open = os.fstat(descriptor)
        after_path = resolved.lstat()
        if (
            (after_open.st_dev, after_open.st_ino) != identity
            or (after_path.st_dev, after_path.st_ino) != identity
            or after_open.st_size != opened.st_size
            or after_open.st_mtime_ns != opened.st_mtime_ns
            or digest.hexdigest() != expected_sha256
        ):
            raise ValueError
        return resolved, expected_sha256
    except (OSError, RuntimeError, ValueError):
        raise ValueError from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _local_path(value: Path | str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError:
        raise ValueError from None
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError
    if raw.startswith("//") or _NETWORK_PREFIX.search(raw):
        raise ValueError
    return Path(raw)


def _verify_repository_revision(repository: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    if not isinstance(completed.stdout, str):
        raise ValueError
    if completed.stdout.strip() != _PINNED_REPOSITORY_REVISION:
        raise ValueError


def _sam_module_snapshot() -> _SamModuleSnapshot:
    modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "sam3" or name.startswith("sam3.")
    }
    return _SamModuleSnapshot(
        modules=modules,
        namespaces={
            name: dict(module.__dict__)
            for name, module in modules.items()
            if isinstance(getattr(module, "__dict__", None), dict)
        },
    )


def _import_local_builder(
    repository: Path, snapshot: _SamModuleSnapshot
) -> Callable[..., Any]:
    for module in snapshot.modules.values():
        if not _module_is_beneath(module, repository):
            raise ValueError
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(repository))
        importlib.invalidate_caches()
        package = importlib.import_module("sam3")
        builder_module = importlib.import_module("sam3.model_builder")
        if not _module_is_beneath(package, repository):
            raise ValueError
        if not _module_is_beneath(builder_module, repository):
            raise ValueError
        builder = getattr(
            builder_module, "build_sam3_multiplex_video_predictor", None
        )
        if not callable(builder):
            raise ValueError
        return builder
    except Exception:
        _restore_sam_modules(snapshot)
        raise
    finally:
        sys.path[:] = original_path
        importlib.invalidate_caches()


def _verify_loaded_sam_modules(repository: Path) -> None:
    modules = [
        module
        for name, module in sys.modules.items()
        if name == "sam3" or name.startswith("sam3.")
    ]
    if not modules or any(
        not _module_is_beneath(module, repository) for module in modules
    ):
        raise ValueError


def _module_is_beneath(module: Any, repository: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        return False
    try:
        return Path(origin).resolve(strict=True).is_relative_to(repository)
    except (OSError, RuntimeError):
        return False


def _restore_sam_modules(snapshot: _SamModuleSnapshot) -> None:
    for name in tuple(sys.modules):
        if name == "sam3" or name.startswith("sam3."):
            sys.modules.pop(name, None)
    for name, namespace in snapshot.namespaces.items():
        module_namespace = snapshot.modules[name].__dict__
        module_namespace.clear()
        module_namespace.update(namespace)
    sys.modules.update(snapshot.modules)


def _prepare_staging_directory(value: Path) -> Path:
    path = _local_path(value)
    try:
        if path.exists() or path.is_symlink():
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise ValueError
        else:
            path.mkdir(mode=0o700, parents=True, exist_ok=False)
        resolved = path.resolve(strict=True)
        status = resolved.stat()
        if (
            resolved != path.absolute()
            or status.st_uid != os.getuid()
            or stat.S_IMODE(status.st_mode) & 0o077
        ):
            raise ValueError
        return resolved
    except (OSError, RuntimeError, ValueError):
        raise ValueError from None


def _validate_source_indices(
    request: CvEvidenceRequest,
    indices: tuple[int, ...],
    *,
    allow_empty: bool = False,
) -> None:
    known = {point.frame_index for point in request.timeline.frames}
    if not indices and not allow_empty:
        raise ValueError
    if indices != tuple(sorted(set(indices))):
        raise ValueError
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError
    if not set(indices) <= known:
        raise ValueError


def _validate_sampled_frames(
    sampled: SampledFrameSet,
    request: CvEvidenceRequest,
    indices: tuple[int, ...],
    destination: Path,
) -> None:
    if not isinstance(sampled, SampledFrameSet):
        raise TypeError
    if len(sampled.frames) != len(indices):
        raise ValueError
    directory_status = destination.lstat()
    if stat.S_ISLNK(directory_status.st_mode) or not stat.S_ISDIR(
        directory_status.st_mode
    ):
        raise ValueError
    expected_names = {f"{index:06d}.jpg" for index in range(len(indices))}
    actual_names = {entry.name for entry in destination.iterdir()}
    if actual_names != expected_names:
        raise ValueError
    by_index = {point.frame_index: point for point in request.timeline.frames}
    for local_index, (frame, source_index) in enumerate(zip(sampled.frames, indices)):
        expected_path = destination / f"{local_index:06d}.jpg"
        if (
            frame.sam_index != local_index
            or frame.source_frame_index != source_index
            or frame.source_timestamp_seconds
            != by_index[source_index].timestamp_seconds
            or frame.path != expected_path
        ):
            raise ValueError
        status = frame.path.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise ValueError


@contextmanager
def _predictor_session(
    predictor: Any, sampled: SampledFrameSet
) -> Iterator[str]:
    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(sampled.frames[0].path.parent),
        }
    )
    if not isinstance(response, Mapping):
        raise ValueError
    session_id = response.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError
    body_failed = False
    try:
        yield session_id
    except BaseException:
        body_failed = True
        raise
    finally:
        try:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        except Exception:
            if not body_failed:
                raise


def _parse_frame_response(
    response: Any,
    sampled: SampledFrameSet,
    *,
    allowed_indices: range,
) -> tuple[int, tuple[_Detection, ...], tuple[int, int]]:
    if not isinstance(response, Mapping):
        raise TypeError
    local_index = response.get("frame_index")
    if (
        isinstance(local_index, bool)
        or not isinstance(local_index, Integral)
        or int(local_index) not in allowed_indices
        or int(local_index) >= len(sampled.frames)
    ):
        raise ValueError
    local_index = int(local_index)
    outputs = response.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError
    required = (
        "out_obj_ids",
        "out_probs",
        "out_boxes_xywh",
        "out_binary_masks",
    )
    if any(name not in outputs for name in required):
        raise ValueError
    ids_shape, object_ids = _array_values(outputs[required[0]], "integer")
    probs_shape, probabilities = _array_values(outputs[required[1]], "float")
    boxes_shape, boxes = _array_values(outputs[required[2]], "float")
    masks_shape, masks = _array_values(outputs[required[3]], "boolean")
    count = ids_shape[0] if len(ids_shape) == 1 else -1
    if (
        ids_shape != (count,)
        or probs_shape != (count,)
        or boxes_shape != (count, 4)
        or len(masks_shape) != 3
        or masks_shape[0] != count
        or masks_shape[1] <= 0
        or masks_shape[2] <= 0
    ):
        raise ValueError
    _require_nested_shape(object_ids, ids_shape)
    _require_nested_shape(probabilities, probs_shape)
    _require_nested_shape(boxes, boxes_shape)
    _require_nested_shape(masks, masks_shape)
    parsed_ids = tuple(_strict_integer(value) for value in object_ids)
    if len(set(parsed_ids)) != len(parsed_ids):
        raise ValueError
    frame = sampled.frames[local_index]
    detections: list[_Detection] = []
    for object_id, probability_value, box_value, mask_value in zip(
        parsed_ids, probabilities, boxes, masks
    ):
        probability = _bounded_float(probability_value, lower=0.0, upper=1.0)
        box = tuple(
            _bounded_float(value, lower=0.0, upper=1.0) for value in box_value
        )
        x, y, width, height = box
        if width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError
        mask = tuple(
            tuple(_strict_boolean(value) for value in row) for row in mask_value
        )
        if not any(value for row in mask for value in row):
            raise ValueError
        detections.append(
            _Detection(
                local_frame_index=local_index,
                source_frame_index=frame.source_frame_index,
                timestamp_seconds=frame.source_timestamp_seconds,
                object_id=object_id,
                probability=probability,
                box_xywh=(x, y, width, height),
                mask=mask,
            )
        )
    return local_index, tuple(detections), (masks_shape[1], masks_shape[2])


def _array_values(value: Any, expected_kind: str) -> tuple[tuple[int, ...], Any]:
    shape_value = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    convert = getattr(value, "tolist", None)
    if not callable(convert) or not isinstance(shape_value, Sequence):
        raise TypeError
    shape = tuple(_nonnegative_dimension(item) for item in shape_value)
    kind = getattr(dtype, "kind", None)
    accepted = {
        "integer": {"i", "u"},
        "float": {"f"},
        "boolean": {"b"},
    }[expected_kind]
    if kind not in accepted:
        raise TypeError
    return shape, convert()


def _nonnegative_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError
    converted = int(value)
    if converted < 0:
        raise ValueError
    return converted


def _require_nested_shape(value: Any, shape: tuple[int, ...]) -> None:
    if not shape:
        if isinstance(value, (list, tuple)):
            raise ValueError
        return
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        raise ValueError
    for item in value:
        _require_nested_shape(item, shape[1:])


def _strict_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError
    converted = int(value)
    if converted < 0:
        raise ValueError
    return converted


def _strict_boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError
    return value


def _bounded_float(value: Any, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError
    converted = float(value)
    if not math.isfinite(converted) or not lower <= converted <= upper:
        raise ValueError
    return converted


def _visibility_change_source_indices(
    prompt_runs: tuple[_PromptRun, ...], sampled: SampledFrameSet
) -> tuple[int, ...]:
    changed: set[int] = set()
    for prompt_run in prompt_runs:
        object_ids = sorted(set().union(*prompt_run.present_ids_by_frame))
        for object_id in object_ids:
            previous = object_id in prompt_run.present_ids_by_frame[0]
            for local_index, present_ids in enumerate(
                prompt_run.present_ids_by_frame[1:], start=1
            ):
                current = object_id in present_ids
                if current != previous:
                    changed.add(sampled.frames[local_index].source_frame_index)
                previous = current
    return tuple(sorted(changed))


def _build_artifact(
    request: CvEvidenceRequest,
    staging: Path,
    prompt_runs: tuple[_PromptRun, ...],
    output_directories: list[Path],
) -> CvEvidenceArtifact:
    masks_directory = _create_private_directory(staging / "masks")
    output_directories.append(masks_directory)
    artifact_files: list[ArtifactFile] = []
    tracks: list[CvTrack] = []
    for prompt_run in prompt_runs:
        mask_relative = f"masks/{prompt_run.entity_id}.npz"
        mask_path = staging / mask_relative
        ordered_detections = tuple(
            sorted(
                prompt_run.detections,
                key=lambda item: (item.local_frame_index, item.object_id),
            )
        )
        _write_mask_npz(
            mask_path,
            ordered_detections,
            prompt_run.mask_height,
            prompt_run.mask_width,
        )
        artifact_files.append(_artifact_file(staging, mask_path))
        object_ids = sorted({item.object_id for item in ordered_detections})
        for object_id in object_ids:
            observations = tuple(
                _observation(detection, mask_relative)
                for detection in ordered_detections
                if detection.object_id == object_id
            )
            tracks.append(
                CvTrack(
                    track_id=f"{prompt_run.entity_id}_{object_id}",
                    entity_id=prompt_run.entity_id,
                    observations=observations,
                )
            )

    overlays = _select_overlays(prompt_runs)
    if overlays:
        overlays_directory = _create_private_directory(staging / "overlays")
        output_directories.append(overlays_directory)
        for entity_id, object_id, detection in overlays[:_MAX_OVERLAYS]:
            relative = (
                f"overlays/{entity_id}-{object_id}-"
                f"{detection.source_frame_index:08d}.png"
            )
            overlay_path = staging / relative
            _write_mask_png(overlay_path, detection.mask)
            artifact_files.append(_artifact_file(staging, overlay_path))

    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status=EvidenceStatus.AVAILABLE,
        provider="sam31",
        model_identity=request.model_identity,
        video_sha256=request.video_sha256,
        checkpoint_sha256=request.checkpoint_sha256,
        entities=request.entities,
        tracks=tuple(tracks),
        files=tuple(sorted(artifact_files, key=lambda item: item.path)),
    )


def _observation(detection: _Detection, mask_relative: str) -> TrackObservation:
    x, y, width, height = detection.box_xywh
    true_pixels = [
        (column, row)
        for row, values in enumerate(detection.mask)
        for column, present in enumerate(values)
        if present
    ]
    mask_height = len(detection.mask)
    mask_width = len(detection.mask[0])
    return TrackObservation(
        frame_index=detection.source_frame_index,
        timestamp_seconds=detection.timestamp_seconds,
        bbox_xyxy=(x, y, x + width, y + height),
        mask_ref=mask_relative,
        visible=True,
        confidence=detection.probability,
        area_fraction=len(true_pixels) / (mask_height * mask_width),
        center_xy=(
            sum(column for column, _ in true_pixels) / len(true_pixels) / mask_width,
            sum(row for _, row in true_pixels) / len(true_pixels) / mask_height,
        ),
    )


def _select_overlays(
    prompt_runs: tuple[_PromptRun, ...],
) -> tuple[tuple[str, int, _Detection], ...]:
    selected: dict[tuple[str, int, int], _Detection] = {}
    for prompt_run in prompt_runs:
        by_frame_and_object = {
            (item.local_frame_index, item.object_id): item
            for item in prompt_run.detections
        }
        object_ids = sorted(set().union(*prompt_run.present_ids_by_frame))
        for object_id in object_ids:
            previous = object_id in prompt_run.present_ids_by_frame[0]
            for local_index, present_ids in enumerate(
                prompt_run.present_ids_by_frame[1:], start=1
            ):
                current = object_id in present_ids
                if current != previous:
                    candidates = (
                        by_frame_and_object.get((local_index - 1, object_id)),
                        by_frame_and_object.get((local_index, object_id)),
                    )
                    for detection in candidates:
                        if detection is not None:
                            key = (
                                prompt_run.entity_id,
                                object_id,
                                detection.source_frame_index,
                            )
                            selected.setdefault(key, detection)
                previous = current
    return tuple(
        (entity_id, object_id, selected[(entity_id, object_id, frame_index)])
        for entity_id, object_id, frame_index in sorted(selected)
    )


def _create_private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, exist_ok=False)
    return path


def _write_mask_npz(
    path: Path,
    detections: tuple[_Detection, ...],
    height: int,
    width: int,
) -> None:
    raw = _npy_boolean_payload(
        tuple(detection.mask for detection in detections), height, width
    )
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w+b", closefd=False) as output:
            with zipfile.ZipFile(
                output,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                info = zipfile.ZipInfo("masks.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)


def _npy_boolean_payload(
    masks: tuple[tuple[tuple[bool, ...], ...], ...],
    height: int,
    width: int,
) -> bytes:
    shape = (len(masks), height, width)
    header_text = (
        "{'descr': '|b1', 'fortran_order': False, "
        f"'shape': {shape!r}, }}"
    )
    prefix_length = 6 + 2 + 2
    padding = (-((prefix_length + len(header_text) + 1) % 16)) % 16
    header = (header_text + (" " * padding) + "\n").encode("latin1")
    data = bytes(
        int(value)
        for mask in masks
        for row in mask
        for value in row
    )
    return b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(header)) + header + data


def _write_mask_png(path: Path, mask: tuple[tuple[bool, ...], ...]) -> None:
    height = len(mask)
    width = len(mask[0])
    rows = b"".join(
        b"\x00"
        + b"".join(
            (b"\xff\x20\x20" if value else b"\x00\x00\x00")
            for value in row
        )
        for row in mask
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(rows, level=9))
        + _png_chunk(b"IEND", b"")
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _artifact_file(staging: Path, path: Path) -> ArtifactFile:
    payload_hash = hashlib.sha256()
    size = 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError
        while payload := os.read(descriptor, _HASH_READ_BYTES):
            payload_hash.update(payload)
            size += len(payload)
        if size != status.st_size:
            raise ValueError
    finally:
        os.close(descriptor)
    return ArtifactFile(
        path=path.relative_to(staging).as_posix(),
        sha256=payload_hash.hexdigest(),
        size_bytes=size,
    )


def _reset_peak_memory(torch_module: Any) -> None:
    reset = getattr(getattr(torch_module, "cuda", None), "reset_peak_memory_stats", None)
    if callable(reset):
        reset()


def _peak_memory(torch_module: Any) -> int:
    maximum = getattr(getattr(torch_module, "cuda", None), "max_memory_allocated", None)
    if not callable(maximum):
        return 0
    value = maximum()
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError
    return int(value)


def _empty_cuda_cache(torch_module: Any) -> None:
    empty = getattr(getattr(torch_module, "cuda", None), "empty_cache", None)
    if callable(empty):
        try:
            empty()
        except Exception:
            pass


def _is_cuda_oom(torch_module: Any, error: Exception) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
    return isinstance(oom_type, type) and isinstance(error, oom_type)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _remove_tree(path: Path) -> None:
    try:
        if path.exists() and not path.is_symlink():
            shutil.rmtree(path)
    except OSError:
        pass
