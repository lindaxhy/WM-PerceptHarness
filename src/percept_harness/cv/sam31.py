"""Lazy, local-only SAM3.1 Object Multiplex evidence adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import io
from numbers import Integral
import os
from pathlib import Path
import logging
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator
import zipfile

from .base import CvOutOfMemoryError, CvProviderError
from .contracts import (
    ArtifactFile,
    MAX_ARTIFACT_BYTES,
    MAX_OVERLAYS,
    CvEvidenceArtifact,
    CvEvidenceRequest,
    CvTrack,
    EvidenceStatus,
    FrameTimeline,
    FrameTimestamp,
    OverlayRecord,
    TrackObservation,
)
from .timeline import (
    SampledFrameSet,
    initial_sample_indices,
    materialize_sampled_frames,
)


_PINNED_REPOSITORY_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
_LOAD_FAILURE = "Unable to load local SAM3.1 runtime"
_INFERENCE_FAILURE = "SAM3.1 CV evidence inference failed"
_OOM_FAILURE = "SAM3.1 CV evidence inference ran out of memory"
_CLOSE_FAILURE = "Unable to close SAM3.1 runtime"
_logger = logging.getLogger(__name__)
_HASH_READ_BYTES = 1024 * 1024
# Maximum candidates returned by the pinned upstream builder default.
_MAX_OBJECTS = 16
_ZIP_ENTRY_ALLOWANCE = 1024
_BPE_REPOSITORY_PATH = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
# Pinned sam3_multiplex_base.py uses this score for removed objects.
_REMOVED_OBJECT_SCORE = -10000.0
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
    area_fraction: float
    center_xy: tuple[float, float]
    mask_index: int


@dataclass(frozen=True, slots=True)
class _PromptRun:
    entity_id: str
    detections: tuple[_Detection, ...]
    present_ids_by_frame: tuple[frozenset[int], ...]
    mask_height: int
    mask_width: int
    mask_relative: str


class _CleanupFailure(Exception):
    """Mark a cleanup-only BaseException for boundary sanitization."""


@dataclass(slots=True)
class _ArtifactBudget:
    projected_bytes: int = 0
    actual_bytes: int = 0

    def reserve(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError
        projected = self.projected_bytes + byte_count
        if projected > MAX_ARTIFACT_BYTES:
            raise ValueError
        self.projected_bytes = projected

    def record_file(self, size: int) -> None:
        if size < 0 or self.actual_bytes + size > MAX_ARTIFACT_BYTES:
            raise ValueError
        self.actual_bytes += size


class _MaskArchiveWriter:
    """Incrementally deflate one prompt's masks without retaining mask tensors."""

    def __init__(self, path: Path, budget: _ArtifactBudget) -> None:
        budget.reserve(_ZIP_ENTRY_ALLOWANCE)
        self.path = path
        self.budget = budget
        self.mask_count = 0
        self._closed = False
        self._output = os.fdopen(os.open(
            path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
        ), "w+b")
        try:
            self._archive = zipfile.ZipFile(
                self._output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            )
        except BaseException:
            self._output.close()
            raise

    def write_mask(self, mask: Any) -> int:
        """Archive one visible mask without retaining masks from other frames."""
        if self._closed:
            raise RuntimeError
        self.budget.reserve(mask.nbytes + _ZIP_ENTRY_ALLOWANCE)
        mask_index = self.mask_count
        _write_npy(self._archive, f"masks/{mask_index:08d}.npy", mask)
        self.mask_count += 1
        return mask_index

    def finish(
        self, detections: tuple[_Detection, ...], sampled: SampledFrameSet
    ) -> None:
        if self._closed or len(detections) != self.mask_count:
            raise ValueError
        arrays = [
            (name, [getattr(item, attribute) for item in detections])
            for name, attribute in (
                ("local_frame_indices.npy", "local_frame_index"),
                ("frame_indices.npy", "source_frame_index"),
                ("object_ids.npy", "object_id"), ("mask_indices.npy", "mask_index"),
            )
        ]
        arrays.append((
            "sampled_frame_indices.npy",
            [item.source_frame_index for item in sampled.frames],
        ))
        try:
            numpy = importlib.import_module("numpy")
            for name, values in arrays:
                self.budget.reserve(
                    len(values) * 8 + _ZIP_ENTRY_ALLOWANCE
                )
                _write_npy(self._archive, name, numpy.asarray(values, dtype="<i8"))
            self._archive.close()
            self._output.flush()
            os.fsync(self._output.fileno())
            self._output.close()
            self._closed = True
            self.budget.record_file(self.path.stat().st_size)
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        for cleanup in (
            self._archive.close, self._output.close,
            lambda: self.path.unlink(missing_ok=True),
        ):
            with suppress(BaseException):
                cleanup()


class Sam31EvidenceProvider:
    """Adapt one official-compatible SAM3.1 predictor to the CV contract."""

    def __init__(
        self,
        *,
        predictor: Any,
        torch_module: Any,
        materialize_frames: Callable[..., SampledFrameSet] = materialize_sampled_frames,
        initial_sampler: Callable[..., tuple[int, ...]] = initial_sample_indices,
        repository_revision: str = _PINNED_REPOSITORY_REVISION,
        overlay_renderer: Callable[..., None] | None = None,
        import_snapshot: dict[str, Any] | None = None,
        sam_source_root: Path | None = None,
    ) -> None:
        _require_predictor_interface(predictor)
        _install_pinned_multiplex_init_compatibility(predictor)
        self._predictor = predictor
        self._torch = torch_module
        self._materialize_frames = materialize_frames
        self._initial_sampler = initial_sampler
        self._execution_chunk_frames = 8
        self._apply_execution_chunk_frames()
        self.repository_revision = repository_revision
        self._overlay_renderer = overlay_renderer or _render_context_overlay
        self._import_snapshot = import_snapshot
        self._sam_source_root = sam_source_root
        self._closed = False
        self._metrics: dict[str, int] = {}

    @classmethod
    def load(
        cls,
        repository_path: Path | str,
        checkpoint_path: Path | str,
        *,
        predictor_factory: Callable[..., Any] | None = None,
        bpe_path: Path | str | None = None,
    ) -> Sam31EvidenceProvider:
        """Load a clean pinned checkout and local assets without downloading.

        These trusted local assets must remain unchanged while the provider runs.
        """
        imported_snapshot: dict[str, Any] | None = None
        predictor: Any = None
        try:
            repository = _validated_repository(repository_path)
            _verify_repository_revision(repository)
            checkpoint = _validated_local_asset(checkpoint_path)
            bpe = _validated_local_asset(
                bpe_path if bpe_path is not None else repository / _BPE_REPOSITORY_PATH
            )
            if _git_blob_digest(bpe) != _pinned_blob_object_id(
                repository, _BPE_REPOSITORY_PATH
            ):
                raise ValueError
            imported_snapshot = _sam_modules()
            if any(
                not _module_is_beneath(module, repository)
                for module in imported_snapshot.values()
            ):
                raise ValueError
            builder = (
                predictor_factory if predictor_factory is not None
                else _import_local_builder(repository)
            )
            if not callable(builder):
                raise TypeError
            torch_module = importlib.import_module("torch")
            predictor = builder(
                checkpoint_path=str(checkpoint),
                bpe_path=str(bpe),
                use_fa3=False,
            )
            if predictor_factory is None or _sam_modules():
                _verify_loaded_sam_modules(repository)
            return cls(
                predictor=predictor,
                torch_module=torch_module,
                repository_revision=_PINNED_REPOSITORY_REVISION,
                import_snapshot=imported_snapshot,
                sam_source_root=repository if _sam_modules() else None,
            )
        except BaseException as error:
            _shutdown_predictor(predictor)
            if imported_snapshot is not None:
                _restore_sam_modules(imported_snapshot)
            if isinstance(error, Exception):
                _logger.debug("SAM3.1 runtime load failed", exc_info=error)
                raise CvProviderError(_LOAD_FAILURE) from error
            raise

    @property
    def execution_chunk_frames(self) -> int:
        return self._execution_chunk_frames

    @execution_chunk_frames.setter
    def execution_chunk_frames(self, value: int) -> None:
        self._execution_chunk_frames = value
        self._apply_execution_chunk_frames()

    def _apply_execution_chunk_frames(self) -> None:
        model = getattr(self._predictor, "model", None)
        if model is None:
            return
        for name in ("postprocess_batch_size", "batched_grounding_batch_size"):
            if hasattr(model, name):
                setattr(model, name, self._execution_chunk_frames)
                if getattr(model, name) != self._execution_chunk_frames:
                    raise ValueError

    def analyze(self, request: CvEvidenceRequest, staging_dir: Path) -> CvEvidenceArtifact:
        """Probe or track one nominated entity within its selected source window."""
        outputs: list[Path] = []
        try:
            if self._closed:
                raise ValueError("SAM provider is closed")
            self._metrics = {}
            staging = _prepare_staging_directory(staging_dir)
            _reset_peak_memory(self._torch)
            reference = request.spatial_prompts[0].frame_index
            indices = (reference,) if request.mode == "probe" else tuple(sorted(
                set(self._initial_sampler(request.timeline, request.sampling)) | {reference}
            ))
            budget = _ArtifactBudget()
            with tempfile.TemporaryDirectory(prefix=".sam31-frames-", dir=staging) as work:
                destination = Path(work) / "frames"
                sampled = self._materialize_frames(
                    request.video_path, request.timeline, indices, destination
                )
                _validate_sampled_frames(sampled, request, indices, destination)
                masks_directory = _create_private_directory(staging / "masks")
                outputs.append(masks_directory)
                with _predictor_session(self._predictor, sampled) as session_id:
                    run = self._run_prompt(
                        request, sampled, session_id,
                        masks_directory / f"{request.entities[0].entity_id}.npz", budget,
                    )
                artifact = _build_artifact(
                    request, staging, sampled, run, outputs, budget, self._overlay_renderer
                )
                if self._sam_source_root is not None:
                    _verify_loaded_sam_modules(self._sam_source_root)
                self._metrics = {
                    "processed_frames": len(sampled.frames),
                    "execution_chunk_frames": self.execution_chunk_frames,
                    "entity_prompts": 1,
                    "track_count": len(artifact.tracks),
                    "peak_allocated_bytes": _peak_memory(self._torch),
                }
            return artifact
        except BaseException as error:
            for directory in reversed(outputs):
                with suppress(BaseException):
                    shutil.rmtree(directory)
            if _is_cuda_oom(self._torch, error):
                _empty_cuda_cache(self._torch)
                raise CvOutOfMemoryError(_OOM_FAILURE) from None
            if isinstance(error, Exception):
                raise CvProviderError(_INFERENCE_FAILURE) from None
            raise

    def request_metrics(self) -> dict[str, int]:
        """Return bounded metrics for the most recent completed request."""
        return dict(self._metrics)

    def close(self) -> None:
        """Release predictor and imported modules, including after shutdown failure."""
        if self._closed:
            return
        self._closed = True
        try:
            try:
                shutdown = getattr(self._predictor, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            finally:
                if self._import_snapshot is not None:
                    _restore_sam_modules(self._import_snapshot)
        except BaseException:
            raise CvProviderError(_CLOSE_FAILURE) from None
        finally:
            self._predictor = None
            self._import_snapshot = None
            self._sam_source_root = None

    def _run_prompt(
        self, request: CvEvidenceRequest, sampled: SampledFrameSet,
        session_id: str, mask_path: Path, budget: _ArtifactBudget,
    ) -> _PromptRun:
        entity = request.entities[0]
        spatial = request.spatial_prompts[0]
        dimensions = _sampled_frame_dimensions(sampled)
        reference_index = next(
            index for index, frame in enumerate(sampled.frames)
            if frame.source_frame_index == spatial.frame_index
        )
        left, top, right, bottom = spatial.bbox_xyxy
        prompt_response = self._predictor.handle_request({
            "type": "add_prompt", "session_id": session_id,
            "frame_index": reference_index, "text": entity.canonical_label,
            "bounding_boxes": [[left, top, right - left, bottom - top]],
            "bounding_box_labels": [1],
        })
        frame_count = len(sampled.frames)
        present: list[frozenset[int]] = [frozenset()] * frame_count
        detections: list[_Detection] = []
        writer = _MaskArchiveWriter(mask_path, budget)
        expected_indices = (
            (reference_index,) if request.mode == "probe" else
            tuple(range(reference_index, frame_count))
            + tuple(range(reference_index - 1, -1, -1))
        )
        try:
            stream = (
                iter((prompt_response,)) if request.mode == "probe" else
                self._predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": session_id,
                    "propagation_direction": "both", "start_frame_index": reference_index,
                })
            )
            with _cleanup(lambda: getattr(stream, "close", lambda: None)()):
                for expected, response in zip(expected_indices, stream, strict=True):
                    local_index, parsed = _parse_frame_response(
                        response, sampled,
                        expected_dimensions=dimensions, mask_writer=writer,
                    )
                    if local_index != expected:
                        raise ValueError
                    present[local_index] = frozenset(item.object_id for item in parsed)
                    detections.extend(parsed)
            ordered = tuple(sorted(
                detections, key=lambda item: (item.local_frame_index, item.object_id)
            ))
            writer.finish(ordered, sampled)
            return _PromptRun(
                entity.entity_id, ordered, tuple(present), *dimensions,
                f"masks/{entity.entity_id}.npz",
            )
        except BaseException:
            writer.abort()
            raise


def _require_predictor_interface(predictor: Any) -> None:
    if not callable(getattr(predictor, "handle_request", None)):
        raise TypeError
    if not callable(getattr(predictor, "handle_stream_request", None)):
        raise TypeError


def _install_pinned_multiplex_init_compatibility(predictor: Any) -> None:
    """Bridge the one known BasePredictor/multiplex init signature mismatch."""
    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    if not callable(init_state):
        return
    parameters = inspect.signature(init_state).parameters
    if "offload_state_to_cpu" in parameters:
        return

    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        if "offload_state_to_cpu" in kwargs:
            if kwargs["offload_state_to_cpu"] is not False:
                raise ValueError
            kwargs.pop("offload_state_to_cpu")
        return init_state(*args, **kwargs)

    setattr(model, "init_state", compatible_init_state)


def _shutdown_predictor(predictor: Any) -> None:
    try:
        shutdown = getattr(predictor, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except BaseException:
        pass


def _validated_repository(value: Path | str) -> Path:
    raw = _local_path(value)
    resolved = raw.resolve(strict=True)
    if (
        not stat.S_ISDIR(raw.lstat().st_mode) or resolved != raw.absolute()
        or not stat.S_ISREG((resolved / "sam3" / "__init__.py").lstat().st_mode)
    ):
        raise ValueError
    return resolved


def _validated_local_asset(value: Path | str) -> Path:
    path = _local_path(value)
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise ValueError
    return path.resolve(strict=True)


def _git_blob_digest(path: Path) -> str:
    """Check the tokenizer against the pinned Git blob without loading it all."""
    with path.open("rb") as source:
        digest = hashlib.sha1(f"blob {os.fstat(source.fileno()).st_size}\0".encode("ascii"))
        while payload := source.read(_HASH_READ_BYTES):
            digest.update(payload)
    return digest.hexdigest()


def _local_path(value: Path | str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError
    if raw.startswith("//") or _NETWORK_PREFIX.search(raw):
        raise ValueError
    return Path(raw)


def _verify_repository_revision(repository: Path) -> None:
    if _git(repository, "rev-parse", "--verify", "HEAD^{commit}") != _PINNED_REPOSITORY_REVISION:
        raise ValueError
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all", "--", "sam3"):
        raise ValueError


def _pinned_blob_object_id(repository: Path, path: str) -> str:
    return _git(repository, "rev-parse", "--verify", f"{_PINNED_REPOSITORY_REVISION}:{path}")


def _git(repository: Path, *arguments: str) -> str:
    # The checkout is trusted and local; Git must not redirect to another repo.
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    completed = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repository), *arguments],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env=environment,
    )
    return completed.stdout.strip()


def _sam_modules() -> dict[str, Any]:
    return {
        name: module for name, module in sys.modules.items()
        if name == "sam3" or name.startswith("sam3.")
    }


def _import_local_builder(repository: Path) -> Callable[..., Any]:
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(repository))
        importlib.invalidate_caches()
        builder_module = importlib.import_module("sam3.model_builder")
        _verify_loaded_sam_modules(repository)
        builder = getattr(builder_module, "build_sam3_multiplex_video_predictor", None)
        if not callable(builder):
            raise ValueError
        return builder
    finally:
        sys.path[:] = original_path


def _verify_loaded_sam_modules(repository: Path) -> None:
    modules = _sam_modules()
    if not modules or any(
        not _module_is_beneath(module, repository) for module in modules.values()
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


def _restore_sam_modules(snapshot: dict[str, Any]) -> None:
    for name in _sam_modules():
        sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _prepare_staging_directory(value: Path) -> Path:
    path = _local_path(value)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    status = path.lstat()
    if (
        resolved != path.absolute() or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError
    return resolved


def _validate_sampled_frames(
    sampled: SampledFrameSet, request: CvEvidenceRequest,
    indices: tuple[int, ...], destination: Path,
) -> None:
    clock = {point.frame_index: point.timestamp_seconds for point in request.timeline.frames}
    if (
        not indices or any(type(index) is not int for index in indices)
        or indices != tuple(sorted(set(indices)))
        or not set(indices) <= clock.keys()
        or not isinstance(sampled, SampledFrameSet) or len(sampled.frames) != len(indices)
        or destination.is_symlink() or not destination.is_dir()
        or {path.name for path in destination.iterdir()}
        != {f"{index:06d}.jpg" for index in range(len(indices))}
    ):
        raise ValueError
    for index, (frame, source) in enumerate(zip(sampled.frames, indices)):
        if (
            frame.sam_index != index or frame.source_frame_index != source
            or frame.source_timestamp_seconds != clock[source]
            or frame.path != destination / f"{index:06d}.jpg"
            or not stat.S_ISREG(frame.path.lstat().st_mode)
        ):
            raise ValueError


def _sampled_frame_dimensions(sampled: SampledFrameSet) -> tuple[int, int]:
    image_module = importlib.import_module("PIL.Image")
    dimensions = set()
    for frame in sampled.frames:
        with image_module.open(frame.path) as image:
            dimensions.add((image.height, image.width))
    if len(dimensions) != 1:
        raise ValueError
    return dimensions.pop()


@contextmanager
def _cleanup(action: Callable[[], Any]) -> Iterator[None]:
    """Close a resource while preserving an exception from its body."""
    try:
        yield
    finally:
        body_failed = sys.exc_info()[0] is not None
        try:
            action()
        except BaseException:
            if not body_failed:
                raise _CleanupFailure from None


@contextmanager
def _predictor_session(predictor: Any, sampled: SampledFrameSet) -> Iterator[str]:
    response = predictor.handle_request({
        "type": "start_session", "resource_path": str(sampled.frames[0].path.parent),
    })
    session_id = response.get("session_id") if isinstance(response, Mapping) else None
    if not isinstance(session_id, str) or not session_id:
        raise ValueError
    with _cleanup(lambda: predictor.handle_request({
        "type": "close_session", "session_id": session_id,
    })):
        yield session_id


def _parse_frame_response(
    response: Any,
    sampled: SampledFrameSet,
    *,
    expected_dimensions: tuple[int, int],
    mask_writer: _MaskArchiveWriter,
) -> tuple[int, tuple[_Detection, ...]]:
    if not isinstance(response, Mapping):
        raise TypeError
    local_index = response.get("frame_index")
    if (
        isinstance(local_index, bool)
        or not isinstance(local_index, Integral)
        or not 0 <= local_index < len(sampled.frames)
    ):
        raise ValueError
    local_index = int(local_index)
    outputs = response.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError
    numpy = importlib.import_module("numpy")
    object_ids, probabilities, boxes, masks = (
        numpy.asarray(outputs[name])
        for name in (
            "out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks"
        )
    )
    if (
        object_ids.dtype.kind not in "iu"
        or probabilities.dtype.kind != "f"
        or boxes.dtype.kind != "f"
        or masks.dtype.kind != "b"
    ):
        raise TypeError
    count = object_ids.size
    if (
        count > _MAX_OBJECTS
        or object_ids.shape != (count,)
        or probabilities.shape != (count,)
        or boxes.shape != (count, 4)
        or masks.shape != (count, *expected_dimensions)
        or min(expected_dimensions) <= 0
        or numpy.any(object_ids < 0)
        or numpy.unique(object_ids).size != count
    ):
        raise ValueError
    # Use Python-equivalent precision when validating normalized coordinates.
    boxes = boxes.astype(numpy.float64, copy=False)
    removed = probabilities == _REMOVED_OBJECT_SCORE
    if (
        not numpy.all(numpy.isfinite(probabilities))
        or numpy.any((~removed) & ((probabilities < 0) | (probabilities > 1)))
        or not numpy.all(numpy.isfinite(boxes))
        or numpy.any((boxes < 0) | (boxes > 1))
        or numpy.any(boxes[:, 2:] <= 0)
        or numpy.any(boxes[:, :2] + boxes[:, 2:] > 1)
    ):
        raise ValueError
    frame = sampled.frames[local_index]
    detections: list[_Detection] = []
    for offset in numpy.argsort(object_ids):
        if removed[offset]:
            continue
        mask = masks[offset]
        statistics = _mask_statistics(mask)
        # Empty masks from stale duplicate tracks mean the object is invisible.
        if statistics is None:
            continue
        area_fraction, center_xy = statistics
        mask_index = mask_writer.write_mask(mask)
        detections.append(
            _Detection(
                local_frame_index=local_index,
                source_frame_index=frame.source_frame_index,
                timestamp_seconds=frame.source_timestamp_seconds,
                object_id=int(object_ids[offset]),
                probability=float(probabilities[offset]),
                box_xywh=tuple(float(value) for value in boxes[offset]),
                area_fraction=area_fraction,
                center_xy=center_xy,
                mask_index=mask_index,
            )
        )
    return local_index, tuple(detections)


def _mask_statistics(mask: Any) -> tuple[float, tuple[float, float]] | None:
    """Compute pixel-center statistics using only row/column sized temporaries."""
    numpy = importlib.import_module("numpy")
    count = int(numpy.count_nonzero(mask))
    if not count:
        return None
    height, width = mask.shape
    column_sum = numpy.dot(mask.sum(axis=0), numpy.arange(width, dtype=numpy.float64) + 0.5)
    row_sum = numpy.dot(mask.sum(axis=1), numpy.arange(height, dtype=numpy.float64) + 0.5)
    return count / mask.size, (
        float(column_sum / count / width), float(row_sum / count / height)
    )


def _write_npy(archive: zipfile.ZipFile, name: str, array: Any) -> None:
    numpy = importlib.import_module("numpy")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    with archive.open(info, mode="w", force_zip64=True) as member:
        numpy.save(member, array, allow_pickle=False)


def _build_artifact(
    request: CvEvidenceRequest, staging: Path, sampled: SampledFrameSet,
    run: _PromptRun, output_directories: list[Path], budget: _ArtifactBudget,
    overlay_renderer: Callable[..., None],
) -> CvEvidenceArtifact:
    files = [_artifact_file(staging, staging / run.mask_relative)]
    records: list[OverlayRecord] = []
    tracks = tuple(
        CvTrack(
            track_id=f"{run.entity_id}_{object_id}", entity_id=run.entity_id,
            observations=tuple(
                _observation(item, run.mask_relative) for item in run.detections
                if item.object_id == object_id
            ),
        )
        for object_id in sorted({item.object_id for item in run.detections})
    )
    references = tuple(
        item for item in run.detections
        if item.source_frame_index == request.spatial_prompts[0].frame_index
    )
    reference_keys = {(item.object_id, item.source_frame_index) for item in references}
    overlays = references + tuple(
        item for item in _select_overlays(run)
        if (item.object_id, item.source_frame_index) not in reference_keys
    )
    if overlays:
        output_directories.append(_create_private_directory(staging / "overlays"))
        for item in overlays[:MAX_OVERLAYS]:
            budget.reserve(
                run.mask_height * run.mask_width * 3 + _ZIP_ENTRY_ALLOWANCE
            )
            relative = (
                f"overlays/{run.entity_id}-{item.object_id}-{item.source_frame_index:08d}.png"
            )
            x, y, width, height = item.box_xywh
            overlay_renderer(
                frame_path=sampled.frames[item.local_frame_index].path,
                archive_path=staging / run.mask_relative,
                mask_member=f"masks/{item.mask_index:08d}.npy",
                bbox_xyxy=(x, y, x + width, y + height), destination=staging / relative,
            )
            overlay_file = _artifact_file(staging, staging / relative)
            budget.record_file(overlay_file.size_bytes)
            files.append(overlay_file)
            records.append(OverlayRecord(
                path=relative, track_id=f"{run.entity_id}_{item.object_id}",
                frame_index=item.source_frame_index,
            ))
    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1", status=EvidenceStatus.AVAILABLE,
        provider="sam31", model_identity=request.model_identity,
        video_identity=request.video_identity, checkpoint_identity=request.checkpoint_identity,
        processed_timeline=FrameTimeline(frames=tuple(
            FrameTimestamp(
                frame_index=frame.source_frame_index,
                timestamp_seconds=frame.source_timestamp_seconds,
            )
            for frame in sampled.frames
        )),
        entities=request.entities, tracks=tracks,
        files=tuple(sorted(files, key=lambda item: item.path)),
        overlay_records=tuple(sorted(records, key=lambda item: item.path)),
    )


def _observation(detection: _Detection, mask_relative: str) -> TrackObservation:
    x, y, width, height = detection.box_xywh
    return TrackObservation(
        frame_index=detection.source_frame_index,
        timestamp_seconds=detection.timestamp_seconds,
        bbox_xyxy=(x, y, x + width, y + height),
        mask_ref=mask_relative,
        visible=True,
        confidence=detection.probability,
        area_fraction=detection.area_fraction,
        center_xy=detection.center_xy,
    )


def _select_overlays(run: _PromptRun) -> tuple[_Detection, ...]:
    detections = {(item.local_frame_index, item.object_id): item for item in run.detections}
    selected: dict[tuple[int, int], _Detection] = {}
    for index, (before, after) in enumerate(zip(
        run.present_ids_by_frame, run.present_ids_by_frame[1:]
    ), start=1):
        for object_id in before ^ after:
            for frame_index in (index - 1, index):
                item = detections.get((frame_index, object_id))
                if item is not None:
                    selected[(object_id, item.source_frame_index)] = item
    return tuple(selected[key] for key in sorted(selected))


def _create_private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, exist_ok=False)
    return path


def _render_context_overlay(
    *,
    frame_path: Path,
    archive_path: Path,
    mask_member: str,
    bbox_xyxy: tuple[float, float, float, float],
    destination: Path,
) -> None:
    numpy = importlib.import_module("numpy")
    cv2 = importlib.import_module("cv2")
    with zipfile.ZipFile(archive_path) as archive:
        mask = numpy.load(io.BytesIO(archive.read(mask_member)), allow_pickle=False)
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None or tuple(mask.shape) != tuple(image.shape[:2]):
        raise ValueError
    if getattr(mask.dtype, "kind", None) != "b":
        raise TypeError
    canvas = image.copy()
    canvas[mask] = (
        (canvas[mask].astype(numpy.float32) * 0.4)
        + numpy.asarray((20, 20, 255), dtype=numpy.float32) * 0.6
    ).astype(numpy.uint8)
    height, width = image.shape[:2]
    left, top, right, bottom = bbox_xyxy
    cv2.rectangle(
        canvas,
        (max(0, round(left * width)), max(0, round(top * height))),
        (min(width - 1, round(right * width)), min(height - 1, round(bottom * height))),
        (0, 255, 255),
        max(1, min(width, height) // 200),
    )
    success, encoded = cv2.imencode(
        ".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9]
    )
    if not success:
        raise ValueError
    _write_exclusive_bytes(destination, memoryview(encoded))


def _write_exclusive_bytes(path: Path, payload: memoryview) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _artifact_file(staging: Path, path: Path) -> ArtifactFile:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode):
        raise ValueError
    return ArtifactFile(path=path.relative_to(staging).as_posix(), size_bytes=status.st_size)


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
        except BaseException:
            pass


def _is_cuda_oom(torch_module: Any, error: Exception) -> bool:
    oom_type = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
    return isinstance(oom_type, type) and isinstance(error, oom_type)
