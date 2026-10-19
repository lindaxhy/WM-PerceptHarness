"""Lazy, local-only SAM3.1 Object Multiplex evidence adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
import importlib
import logging
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Iterator

from .base import CvOutOfMemoryError, CvProviderError
from .contracts import CvEvidenceArtifact, CvEvidenceRequest
from .masks import ArtifactBudget, MaskArchiveWriter, mask_archive_name
from .sam31_artifacts import (
    Detection,
    PromptRun,
    build_artifact,
    create_private_directory,
    parse_frame_response,
    render_context_overlay,
)
from .sam31_runtime import (
    BPE_REPOSITORY_PATH,
    empty_cuda_cache,
    git_blob_digest,
    import_local_builder,
    install_pinned_multiplex_init_compatibility,
    is_cuda_oom,
    local_path,
    module_is_beneath,
    peak_memory,
    pinned_blob_object_id,
    require_predictor_interface,
    reset_peak_memory,
    restore_sam_modules,
    sam_modules,
    shutdown_predictor,
    validated_local_asset,
    validated_repository,
    verify_loaded_sam_modules,
    verify_repository_revision,
)
from .timeline import (
    SampledFrameSet,
    initial_sample_indices,
    materialize_sampled_frames,
)


_LOAD_FAILURE = "Unable to load local SAM3.1 runtime"
_INFERENCE_FAILURE = "SAM3.1 CV evidence inference failed"
_OOM_FAILURE = "SAM3.1 CV evidence inference ran out of memory"
_CLOSE_FAILURE = "Unable to close SAM3.1 runtime"
_logger = logging.getLogger(__name__)


class _CleanupFailure(Exception):
    """Mark a cleanup-only BaseException for boundary sanitization."""


class Sam31EvidenceProvider:
    """Adapt one official-compatible SAM3.1 predictor to the CV contract."""

    def __init__(
        self,
        *,
        predictor: Any,
        torch_module: Any,
        materialize_frames: Callable[..., SampledFrameSet] = materialize_sampled_frames,
        initial_sampler: Callable[..., tuple[int, ...]] = initial_sample_indices,
        overlay_renderer: Callable[..., None] | None = None,
        import_snapshot: dict[str, Any] | None = None,
        sam_source_root: Path | None = None,
    ) -> None:
        require_predictor_interface(predictor)
        install_pinned_multiplex_init_compatibility(predictor)
        self._predictor = predictor
        self._torch = torch_module
        self._materialize_frames = materialize_frames
        self._initial_sampler = initial_sampler
        self._execution_chunk_frames = 8
        self._apply_execution_chunk_frames()
        self._overlay_renderer = overlay_renderer or render_context_overlay
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
            repository = validated_repository(repository_path)
            verify_repository_revision(repository)
            checkpoint = validated_local_asset(checkpoint_path)
            bpe = validated_local_asset(
                bpe_path if bpe_path is not None else repository / BPE_REPOSITORY_PATH
            )
            if git_blob_digest(bpe) != pinned_blob_object_id(
                repository, BPE_REPOSITORY_PATH
            ):
                raise ValueError
            imported_snapshot = sam_modules()
            if any(
                not module_is_beneath(module, repository)
                for module in imported_snapshot.values()
            ):
                raise ValueError
            builder = (
                predictor_factory if predictor_factory is not None
                else import_local_builder(repository)
            )
            if not callable(builder):
                raise TypeError
            torch_module = importlib.import_module("torch")
            predictor = builder(
                checkpoint_path=str(checkpoint),
                bpe_path=str(bpe),
                use_fa3=False,
            )
            if predictor_factory is None or sam_modules():
                verify_loaded_sam_modules(repository)
            return cls(
                predictor=predictor,
                torch_module=torch_module,
                import_snapshot=imported_snapshot,
                sam_source_root=repository if sam_modules() else None,
            )
        except BaseException as error:
            with suppress(BaseException):
                shutdown_predictor(predictor)
            if imported_snapshot is not None:
                restore_sam_modules(imported_snapshot)
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
            reset_peak_memory(self._torch)
            reference = request.spatial_prompts[0].frame_index
            indices = (reference,) if request.mode == "probe" else tuple(sorted(
                set(self._initial_sampler(request.timeline, request.sampling)) | {reference}
            ))
            budget = ArtifactBudget()
            with tempfile.TemporaryDirectory(prefix=".sam31-frames-", dir=staging) as work:
                destination = Path(work) / "frames"
                sampled = self._materialize_frames(
                    request.video_path, request.timeline, indices, destination
                )
                _validate_sampled_frames(sampled, request, indices, destination)
                outputs.append(create_private_directory(staging / "masks"))
                with _predictor_session(self._predictor, sampled) as session_id:
                    run = self._run_prompt(
                        request, sampled, session_id,
                        staging / mask_archive_name(request.entities[0].entity_id), budget,
                    )
                artifact = build_artifact(
                    request, staging, sampled, run, outputs, budget, self._overlay_renderer
                )
                if self._sam_source_root is not None:
                    verify_loaded_sam_modules(self._sam_source_root)
                self._metrics = {
                    "processed_frames": len(sampled.frames),
                    "execution_chunk_frames": self.execution_chunk_frames,
                    "peak_allocated_bytes": peak_memory(self._torch),
                }
            return artifact
        except BaseException as error:
            for directory in reversed(outputs):
                with suppress(BaseException):
                    shutil.rmtree(directory)
            if is_cuda_oom(self._torch, error):
                empty_cuda_cache(self._torch)
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
                shutdown_predictor(self._predictor)
            finally:
                if self._import_snapshot is not None:
                    restore_sam_modules(self._import_snapshot)
        except BaseException:
            raise CvProviderError(_CLOSE_FAILURE) from None
        finally:
            self._predictor = None
            self._import_snapshot = None
            self._sam_source_root = None

    def _run_prompt(
        self, request: CvEvidenceRequest, sampled: SampledFrameSet,
        session_id: str, mask_path: Path, budget: ArtifactBudget,
    ) -> PromptRun:
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
        detections: list[Detection] = []
        writer = MaskArchiveWriter(mask_path, budget)
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
                    local_index, parsed = parse_frame_response(
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
            return PromptRun(
                entity.entity_id, ordered, tuple(present), *dimensions,
                mask_archive_name(entity.entity_id),
            )
        except BaseException:
            writer.abort()
            raise


def _prepare_staging_directory(value: Path) -> Path:
    path = local_path(value)
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


