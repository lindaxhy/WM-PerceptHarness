"""Lazy, local-only SAM3.1 Object Multiplex evidence adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import io
import itertools
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

from .base import CvOutOfMemoryError, CvProviderError
from .contracts import (
    ArtifactFile,
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
    refinement_sample_indices,
)


_PINNED_REPOSITORY_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
_LOAD_FAILURE = "Unable to load local SAM3.1 runtime"
_INFERENCE_FAILURE = "SAM3.1 CV evidence inference failed"
_OOM_FAILURE = "SAM3.1 CV evidence inference ran out of memory"
_CLOSE_FAILURE = "Unable to close SAM3.1 runtime"
_HASH_READ_BYTES = 1024 * 1024
_MAX_OVERLAYS = 24
_MAX_OBJECTS = 16
_DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
_DEFAULT_MAX_ARTIFACT_FILES = 10_000
_ZIP_ENTRY_ALLOWANCE = 1024
_MAX_GIT_TREE_BYTES = 16 * 1024 * 1024
_MAX_GIT_SOURCE_FILES = 100_000
_MAX_GIT_SOURCE_BYTES = 1024 * 1024 * 1024
_BPE_REPOSITORY_PATH = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
# Pinned sam3_multiplex_base.py uses this score for removed objects.
_REMOVED_OBJECT_SCORE = -10000.0
# Alias fallback tuning: a canonical-label run covering at least this fraction
# of the sampled frames is trusted outright; below it, aliases are probed.
_ALIAS_RETRY_MAX_COVERAGE = 0.5
# An alias run replaces the canonical one only when it covers at least this
# much more of the video, so near-ties never churn the track.
_ALIAS_MIN_COVERAGE_GAIN = 0.15
# Where both runs observe the entity, their per-frame union boxes must agree
# at least this well; an alias that matched a different object is rejected.
_ALIAS_CONSISTENCY_MIN_IOU = 0.4
_ALIAS_CONSISTENCY_MIN_FRAMES = 1
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
    mask_relative: str | None


@dataclass(frozen=True, slots=True)
class _SamModuleSnapshot:
    modules: dict[str, Any]
    namespaces: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _PinnedBlob:
    path: str
    object_id: str
    size: int


class _CleanupFailure(Exception):
    """Mark a cleanup-only BaseException for boundary sanitization."""


@dataclass(slots=True)
class _ArtifactBudget:
    max_bytes: int
    max_files: int
    projected_bytes: int = 0
    actual_bytes: int = 0
    files: int = 0

    def reserve(self, byte_count: int, *, file_count: int = 0) -> None:
        if byte_count < 0 or file_count < 0:
            raise ValueError
        projected = self.projected_bytes + byte_count
        files = self.files + file_count
        if projected > self.max_bytes or files > self.max_files:
            raise ValueError
        self.projected_bytes = projected
        self.files = files

    def record_file(self, size: int) -> None:
        if size < 0 or self.actual_bytes + size > self.max_bytes:
            raise ValueError
        self.actual_bytes += size


class _MaskArchiveWriter:
    """Incrementally deflate one prompt's masks without retaining mask tensors."""

    def __init__(self, path: Path, budget: _ArtifactBudget) -> None:
        budget.reserve(_ZIP_ENTRY_ALLOWANCE, file_count=1)
        self.path = path
        self.budget = budget
        self.mask_count = 0
        self._closed = False
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            self._output = os.fdopen(descriptor, "w+b")
            descriptor = -1
            self._archive = zipfile.ZipFile(
                self._output,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def write_mask(
        self,
        masks: Any,
        object_offset: int,
        height: int,
        width: int,
    ) -> tuple[int, float, tuple[float, float]] | None:
        """Archive one mask, or return None for an all-empty (invisible) mask.

        A stale duplicate track can keep a live box after its object left the
        scene while its mask is already empty; that object is invisible in
        this frame and must not become an archive entry or reject the run.
        Rows are buffered so emptiness is known before the entry is created:
        a zip member cannot be deleted once opened.
        """
        if self._closed:
            raise RuntimeError
        self.budget.reserve(height * width + _ZIP_ENTRY_ALLOWANCE)
        rows: list[bytes] = []
        true_count = 0
        weighted_columns = 0
        weighted_rows = 0
        for row_index in range(height):
            row = _mask_row_bytes(masks, object_offset, row_index, width)
            rows.append(row)
            row_count = row.count(1)
            true_count += row_count
            weighted_rows += row_index * row_count
            weighted_columns += sum(itertools.compress(range(width), row))
        if true_count <= 0:
            return None
        mask_index = self.mask_count
        member_name = f"masks/{mask_index:08d}.npy"
        info = _zip_info(member_name)
        with self._archive.open(info, mode="w", force_zip64=True) as member:
            member.write(_npy_header("|b1", (height, width)))
            for row in rows:
                member.write(row)
        self.mask_count += 1
        pixels = height * width
        return (
            mask_index,
            true_count / pixels,
            (
                (weighted_columns + (0.5 * true_count)) / true_count / width,
                (weighted_rows + (0.5 * true_count)) / true_count / height,
            ),
        )

    def finish(
        self, detections: tuple[_Detection, ...], sampled: SampledFrameSet
    ) -> None:
        if self._closed or len(detections) != self.mask_count:
            raise ValueError
        arrays = (
            (
                "local_frame_indices.npy",
                tuple(item.local_frame_index for item in detections),
            ),
            (
                "frame_indices.npy",
                tuple(item.source_frame_index for item in detections),
            ),
            ("object_ids.npy", tuple(item.object_id for item in detections)),
            ("mask_indices.npy", tuple(item.mask_index for item in detections)),
            (
                "sampled_frame_indices.npy",
                tuple(item.source_frame_index for item in sampled.frames),
            ),
        )
        try:
            for name, values in arrays:
                self.budget.reserve(
                    len(values) * 8 + _ZIP_ENTRY_ALLOWANCE
                )
                _write_int64_npy(self._archive, name, values)
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
        try:
            self._archive.close()
        except BaseException:
            pass
        try:
            self._output.close()
        except BaseException:
            pass
        try:
            self.path.unlink(missing_ok=True)
        except BaseException:
            pass


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
        max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
        max_artifact_files: int = _DEFAULT_MAX_ARTIFACT_FILES,
        overlay_renderer: Callable[..., None] | None = None,
        runtime_directory: Path | None = None,
        import_snapshot: _SamModuleSnapshot | None = None,
        sam_source_root: Path | None = None,
    ) -> None:
        _require_predictor_interface(predictor)
        _install_pinned_multiplex_init_compatibility(predictor)
        self._predictor = predictor
        self._torch = torch_module
        self._materialize_frames = materialize_frames
        self._initial_sampler = initial_sampler
        self._refinement_sampler = refinement_sampler
        self._execution_chunk_frames = _positive_integer(
            execution_chunk_frames, "execution_chunk_frames"
        )
        self._apply_execution_chunk_frames()
        self.checkpoint_sha256 = checkpoint_sha256
        self.repository_revision = repository_revision
        self._max_artifact_bytes = _positive_integer(
            max_artifact_bytes, "max_artifact_bytes"
        )
        self._max_artifact_files = _positive_integer(
            max_artifact_files, "max_artifact_files"
        )
        self._overlay_renderer = overlay_renderer or _render_context_overlay
        self._runtime_directory = runtime_directory
        self._import_snapshot = import_snapshot
        self._sam_source_root = sam_source_root
        self._retry_request: CvEvidenceRequest | None = None
        self._retry_final_indices: tuple[int, ...] | None = None
        self._closed = False
        self._metrics = {
            "processed_frames": 0,
            "execution_chunk_frames": self.execution_chunk_frames,
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
        bpe_path: Path | str | None = None,
        max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
        max_artifact_files: int = _DEFAULT_MAX_ARTIFACT_FILES,
    ) -> Sam31EvidenceProvider:
        """Validate and load the pinned local runtime without any download path."""
        imported_snapshot: _SamModuleSnapshot | None = None
        runtime_directory: Path | None = None
        source_root: Path | None = None
        predictor: Any = None
        try:
            if not isinstance(compile_model, bool):
                raise ValueError
            repository = _validated_repository(repository_path)
            _verify_repository_revision(repository)
            pinned_bpe_object_id = _pinned_blob_object_id(
                repository, _BPE_REPOSITORY_PATH
            )
            runtime_directory = _private_runtime_directory()
            checkpoint, digest = _snapshot_local_asset(
                checkpoint_path,
                runtime_directory / "checkpoint.pt",
                expected_sha256=checkpoint_sha256,
            )
            bpe = _snapshot_local_asset(
                bpe_path
                if bpe_path is not None
                else repository / _BPE_REPOSITORY_PATH,
                runtime_directory / "bpe_simple_vocab_16e6.txt.gz",
                expected_git_object_id=pinned_bpe_object_id,
            )[0]
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
                source_root = _snapshot_repository_source(
                    repository, runtime_directory / "source"
                )
                _verify_repository_revision(repository)
                builder = _import_local_builder(source_root, imported_snapshot)
            torch_module = importlib.import_module("torch")
            predictor = builder(
                checkpoint_path=str(checkpoint),
                bpe_path=str(bpe),
                max_num_objects=16,
                multiplex_count=16,
                compile=compile_model,
                use_fa3=False,
            )
            _require_predictor_interface(predictor)
            if predictor_factory is None:
                if source_root is None:
                    raise ValueError
                _verify_loaded_sam_modules(source_root)
            elif _sam_module_snapshot().modules:
                _verify_loaded_sam_modules(repository)
            return cls(
                predictor=predictor,
                torch_module=torch_module,
                checkpoint_sha256=digest,
                repository_revision=_PINNED_REPOSITORY_REVISION,
                max_artifact_bytes=max_artifact_bytes,
                max_artifact_files=max_artifact_files,
                runtime_directory=runtime_directory,
                import_snapshot=(
                    imported_snapshot if predictor_factory is None else None
                ),
                sam_source_root=source_root,
            )
        except BaseException as error:
            _shutdown_predictor(predictor)
            if imported_snapshot is not None:
                try:
                    _restore_sam_modules(imported_snapshot)
                except BaseException:
                    pass
            if runtime_directory is not None:
                try:
                    _remove_tree(runtime_directory)
                except BaseException:
                    pass
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
        self._apply_execution_chunk_frames()

    def set_execution_chunk_frames(self, value: int) -> None:
        """Expose Task-5's positive worker-owned OOM batching interface."""
        self.execution_chunk_frames = value

    def _apply_execution_chunk_frames(self) -> None:
        model = getattr(self._predictor, "model", None)
        if model is None:
            return
        for name in ("postprocess_batch_size", "batched_grounding_batch_size"):
            if hasattr(model, name):
                setattr(model, name, self._execution_chunk_frames)
                if getattr(model, name) != self._execution_chunk_frames:
                    raise ValueError

    def analyze(
        self, request: CvEvidenceRequest, staging_dir: Path
    ) -> CvEvidenceArtifact:
        """Track normalized prompts over one immutable sampled-frame mapping."""
        work_directory: Path | None = None
        output_directories: list[Path] = []
        completed = False
        primary_error: BaseException | None = None
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
                "execution_chunk_frames": self.execution_chunk_frames,
                "entity_prompts": 0,
                "track_count": 0,
                "peak_allocated_bytes": 0,
            }
            if not validated_request.entities:
                artifact = _empty_artifact(validated_request)
                self._metrics["peak_allocated_bytes"] = _peak_memory(self._torch)
                self._clear_retry_plan()
                completed = True
                return artifact

            work_directory = Path(
                tempfile.mkdtemp(prefix=".sam31-frames-", dir=staging)
            )
            budget = _ArtifactBudget(
                max_bytes=self._max_artifact_bytes,
                max_files=self._max_artifact_files,
            )
            retry_indices = (
                self._retry_final_indices
                if self._retry_request == validated_request
                else None
            )
            if retry_indices is not None:
                final_frames = self._materialize(
                    validated_request,
                    retry_indices,
                    work_directory / "final",
                )
            else:
                self._clear_retry_plan()
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
                    final_indices = initial_indices
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
                    final_indices = tuple(
                        sorted(set(initial_indices) | set(refined))
                    )
                    del preliminary, scan_frames
                self._retry_request = validated_request
                self._retry_final_indices = final_indices
                final_frames = self._materialize(
                    validated_request,
                    final_indices,
                    work_directory / "final",
                )

            masks_directory = _create_private_directory(staging / "masks")
            output_directories.append(masks_directory)
            prompt_runs = self._run_all_prompts(
                validated_request,
                final_frames,
                masks_directory=masks_directory,
                budget=budget,
            )
            artifact = _build_artifact(
                validated_request,
                staging,
                final_frames,
                prompt_runs,
                output_directories,
                budget,
                self._overlay_renderer,
            )
            if self._sam_source_root is not None:
                _verify_loaded_sam_modules(self._sam_source_root)
            self._metrics = {
                "processed_frames": len(final_frames.frames),
                "execution_chunk_frames": self.execution_chunk_frames,
                "entity_prompts": len(validated_request.entities),
                "track_count": len(artifact.tracks),
                "peak_allocated_bytes": _peak_memory(self._torch),
            }
            self._clear_retry_plan()
            completed = True
            return artifact
        except Exception as error:
            primary_error = error
            if _is_cuda_oom(self._torch, error):
                _empty_cuda_cache(self._torch)
                raise CvOutOfMemoryError(_OOM_FAILURE) from None
            self._clear_retry_plan()
            raise CvProviderError(_INFERENCE_FAILURE) from None
        except BaseException as error:
            primary_error = error
            raise
        finally:
            cleanup_error: BaseException | None = None
            if work_directory is not None:
                try:
                    _remove_tree(work_directory)
                except BaseException as error:
                    cleanup_error = error
            if not completed or cleanup_error is not None:
                for directory in reversed(output_directories):
                    try:
                        _remove_tree(directory)
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
            if cleanup_error is not None and primary_error is None:
                raise CvProviderError(_INFERENCE_FAILURE) from None

    def request_metrics(self) -> dict[str, int]:
        """Return bounded metrics for the most recent completed request."""
        return dict(self._metrics)

    def _clear_retry_plan(self) -> None:
        self._retry_request = None
        self._retry_final_indices = None

    def close(self) -> None:
        """Release predictor-owned process state once."""
        if self._closed:
            return
        self._closed = True
        predictor = self._predictor
        self._predictor = None
        cleanup_error: BaseException | None = None
        try:
            shutdown = getattr(predictor, "shutdown", None)
            if callable(shutdown):
                shutdown()
        except BaseException as error:
            cleanup_error = error
        try:
            if self._import_snapshot is not None:
                _restore_sam_modules(self._import_snapshot)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        finally:
            self._import_snapshot = None
        try:
            if self._runtime_directory is not None:
                _remove_tree(self._runtime_directory)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        finally:
            self._runtime_directory = None
            self._sam_source_root = None
        if cleanup_error is not None:
            raise CvProviderError(_CLOSE_FAILURE) from None

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
        *,
        masks_directory: Path | None = None,
        budget: _ArtifactBudget | None = None,
    ) -> tuple[_PromptRun, ...]:
        dimensions = _sampled_frame_dimensions(sampled)

        def run_with_alias_fallback(entity: Any) -> _PromptRun:
            """Prefer the canonical label; adopt an alias with clearly better coverage.

            The canonical label is a VLM-chosen name; a wrong attribute in it
            (for example a misjudged color) can push open-vocabulary matching
            under threshold even though the entity is plainly visible —
            sometimes only during part of the video. Aliases from the same
            nomination frequently still match. An alias run replaces the
            canonical one only when it covers clearly more sampled frames
            AND, on frames where both runs see the entity, their detections
            agree on where it is, so an alias that matched some other object
            can never hijack the track.
            """
            mask_path = (
                masks_directory / f"{entity.entity_id}.npz"
                if masks_directory is not None
                else None
            )
            labels = (entity.canonical_label, *tuple(entity.aliases)[:2])

            def execute(label: str, destination: Path | None) -> _PromptRun:
                return self._run_prompt(
                    request,
                    sampled,
                    session_id,
                    entity,
                    dimensions,
                    mask_path=destination,
                    budget=budget,
                    prompt_text=label,
                )

            canonical = execute(labels[0], mask_path)
            coverage = _prompt_coverage(canonical)
            if len(labels) == 1 or coverage >= _ALIAS_RETRY_MAX_COVERAGE:
                return canonical
            if coverage == 0.0:
                # Nothing to protect: the first alias that sees the entity wins.
                last = canonical
                for label in labels[1:]:
                    if mask_path is not None:
                        mask_path.unlink(missing_ok=True)
                    run = execute(label, mask_path)
                    if _prompt_coverage(run) > 0.0:
                        return run
                    last = run
                return last
            # The canonical label sees the entity in only part of the video.
            # Probe the aliases without masks and keep the best clear improver.
            best: tuple[str, float, _PromptRun] | None = None
            for label in labels[1:]:
                probe = execute(label, None)
                probe_coverage = _prompt_coverage(probe)
                if probe_coverage < coverage + _ALIAS_MIN_COVERAGE_GAIN:
                    continue
                if best is None or probe_coverage > best[1]:
                    best = (label, probe_coverage, probe)
            if best is None:
                return canonical
            if mask_path is None:
                return best[2]
            mask_path.unlink(missing_ok=True)
            replacement = execute(best[0], mask_path)
            if _alias_replacement_consistent(canonical, replacement):
                return replacement
            # The alias matched something else; restore the canonical masks.
            mask_path.unlink(missing_ok=True)
            return execute(labels[0], mask_path)

        with _predictor_session(self._predictor, sampled) as session_id:
            return tuple(
                run_with_alias_fallback(entity) for entity in request.entities
            )

    def _run_prompt(
        self,
        request: CvEvidenceRequest,
        sampled: SampledFrameSet,
        session_id: str,
        entity: Any,
        frame_dimensions: tuple[int, int],
        *,
        mask_path: Path | None,
        budget: _ArtifactBudget | None,
        prompt_text: str | None = None,
    ) -> _PromptRun:
        self._predictor.handle_request(
            {"type": "reset_session", "session_id": session_id}
        )
        self._predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt_text or entity.canonical_label,
                "output_prob_thresh": request.thresholds.min_confidence,
            }
        )
        frame_count = len(sampled.frames)
        detections: list[_Detection] = []
        present: list[frozenset[int] | None] = [None] * frame_count
        mask_shape: tuple[int, int] | None = None
        writer = (
            _MaskArchiveWriter(mask_path, budget)
            if mask_path is not None and budget is not None
            else None
        )
        stream_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": frame_count,
            "output_prob_thresh": request.thresholds.min_confidence,
        }
        try:
            stream = self._predictor.handle_stream_request(stream_request)
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        stream_failed = False
        next_expected_index = 0
        try:
            for response in stream:
                local_index, parsed, response_shape = _parse_frame_response(
                    response,
                    sampled,
                    allowed_indices=range(frame_count),
                    expected_dimensions=frame_dimensions,
                    mask_writer=writer,
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
                if writer is not None:
                    detections.extend(parsed)
        except BaseException:
            stream_failed = True
            if writer is not None:
                writer.abort()
            raise
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                try:
                    close_stream()
                except BaseException:
                    if not stream_failed:
                        if writer is not None:
                            writer.abort()
                        raise _CleanupFailure from None
        try:
            if next_expected_index != frame_count:
                raise ValueError
            if any(value is None for value in present) or mask_shape is None:
                raise ValueError
            if len({
                (detection.local_frame_index, detection.object_id)
                for detection in detections
            }) != len(detections):
                raise ValueError
            ordered = tuple(
                sorted(
                    detections,
                    key=lambda item: (item.local_frame_index, item.object_id),
                )
            )
            if writer is not None:
                writer.finish(ordered, sampled)
            return _PromptRun(
                entity_id=entity.entity_id,
                detections=ordered,
                present_ids_by_frame=tuple(
                    value for value in present if value is not None
                ),
                mask_height=mask_shape[0],
                mask_width=mask_shape[1],
                mask_relative=(
                    f"masks/{entity.entity_id}.npz" if writer is not None else None
                ),
            )
        except BaseException:
            if writer is not None:
                writer.abort()
            raise


def _empty_artifact(request: CvEvidenceRequest) -> CvEvidenceArtifact:
    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status=EvidenceStatus.DISABLED,
        provider="sam31",
        model_identity=request.model_identity,
        video_sha256=request.video_sha256,
        checkpoint_sha256=request.checkpoint_sha256,
        processed_timeline=None,
        entities=request.entities,
        tracks=(),
        files=(),
        overlay_records=(),
    )


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
    try:
        parameters = inspect.signature(init_state).parameters
    except (TypeError, ValueError):
        raise TypeError from None
    if "offload_state_to_cpu" in parameters:
        return

    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        if "offload_state_to_cpu" in kwargs:
            if kwargs["offload_state_to_cpu"] is not False:
                raise ValueError
            kwargs = dict(kwargs)
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
    try:
        status = raw.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise ValueError
        resolved = raw.resolve(strict=True)
        if resolved != raw.absolute():
            raise ValueError
        _validate_trusted_ancestry(resolved)
        package = resolved / "sam3" / "__init__.py"
        package_status = package.lstat()
        if stat.S_ISLNK(package_status.st_mode) or not stat.S_ISREG(
            package_status.st_mode
        ):
            raise ValueError
        return resolved
    except (OSError, RuntimeError, ValueError):
        raise ValueError from None


def _private_runtime_directory() -> Path:
    raw_path = tempfile.mkdtemp(prefix=".las-sam31-runtime-")
    path: Path | None = None
    try:
        path = Path(raw_path)
        path.chmod(0o700)
        status = path.stat()
        if status.st_uid != os.getuid() or stat.S_IMODE(status.st_mode) != 0o700:
            raise ValueError
        return path
    except BaseException:
        if path is not None:
            try:
                _remove_tree(path)
            except BaseException:
                pass
        try:
            shutil.rmtree(raw_path)
        except BaseException:
            pass
        raise


def _snapshot_local_asset(
    value: Path | str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_git_object_id: str | None = None,
) -> tuple[Path, str]:
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError
    if expected_git_object_id is not None and (
        not isinstance(expected_git_object_id, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected_git_object_id) is None
    ):
        raise ValueError
    path = _local_path(value)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError
    descriptor: int | None = None
    try:
        before = path.lstat()
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_nlink != 1
            or before.st_uid not in {0, os.getuid()}
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise ValueError
        resolved = path.resolve(strict=True)
        if resolved != path.absolute():
            raise ValueError
        _validate_trusted_ancestry(resolved.parent)
        descriptor = os.open(resolved, os.O_RDONLY | nofollow)
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size <= 0
            or opened.st_nlink != 1
            or identity != (before.st_dev, before.st_ino)
        ):
            raise ValueError
        digest = hashlib.sha256()
        git_digest = hashlib.sha1()
        git_digest.update(f"blob {opened.st_size}\0".encode("ascii"))
        output_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
        )
        try:
            while payload := os.read(descriptor, _HASH_READ_BYTES):
                digest.update(payload)
                git_digest.update(payload)
                view = memoryview(payload)
                while view:
                    written = os.write(output_descriptor, view)
                    if written <= 0:
                        raise OSError
                    view = view[written:]
            os.fsync(output_descriptor)
        finally:
            os.close(output_descriptor)
        after_open = os.fstat(descriptor)
        after_path = resolved.lstat()
        actual_digest = digest.hexdigest()
        if (
            (after_open.st_dev, after_open.st_ino) != identity
            or (after_path.st_dev, after_path.st_ino) != identity
            or after_open.st_size != opened.st_size
            or after_open.st_mtime_ns != opened.st_mtime_ns
            or after_path.st_size != opened.st_size
            or after_path.st_mtime_ns != opened.st_mtime_ns
            or (expected_sha256 is not None and actual_digest != expected_sha256)
            or (
                expected_git_object_id is not None
                and git_digest.hexdigest() != expected_git_object_id
            )
        ):
            raise ValueError
        copied = destination.stat()
        if (
            not stat.S_ISREG(copied.st_mode)
            or copied.st_nlink != 1
            or copied.st_uid != os.getuid()
            or stat.S_IMODE(copied.st_mode) != 0o600
            or copied.st_size != opened.st_size
        ):
            raise ValueError
        return destination.resolve(strict=True), actual_digest
    except (OSError, RuntimeError, ValueError):
        raise ValueError from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_trusted_ancestry(path: Path) -> None:
    current = path
    while True:
        status = current.lstat()
        mode = stat.S_IMODE(status.st_mode)
        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISDIR(status.st_mode)
            or status.st_uid not in {0, os.getuid()}
            or (mode & 0o022 and not (status.st_uid == 0 and mode & stat.S_ISVTX))
        ):
            raise ValueError
        if current.parent == current:
            break
        current = current.parent


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
        _git_command(
            repository,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ),
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        env=_git_environment(),
    )
    if not isinstance(completed.stdout, str):
        raise ValueError
    if completed.stdout.strip() != _PINNED_REPOSITORY_REVISION:
        raise ValueError
    status = subprocess.run(
        _git_command(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "sam3",
        ),
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        env=_git_environment(),
    )
    if not isinstance(status.stdout, str) or status.stdout:
        raise ValueError


def _pinned_blob_object_id(repository: Path, path: str) -> str:
    if (
        not isinstance(path, str)
        or not path.startswith("sam3/")
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in Path(path).parts)
    ):
        raise ValueError
    completed = subprocess.run(
        _git_command(
            repository,
            "rev-parse",
            "--verify",
            f"{_PINNED_REPOSITORY_REVISION}:{path}",
        ),
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        env=_git_environment(),
    )
    if (
        not isinstance(completed.stdout, str)
        or re.fullmatch(r"[0-9a-f]{40}\n?", completed.stdout) is None
    ):
        raise ValueError
    return completed.stdout.strip()


def _snapshot_repository_source(repository: Path, destination: Path) -> Path:
    blobs = _pinned_sam_blobs(repository)
    names = {blob.path for blob in blobs}
    if "sam3/__init__.py" not in names or "sam3/model_builder.py" not in names:
        raise ValueError
    destination.mkdir(mode=0o700, exist_ok=False)
    try:
        for blob in blobs:
            relative = Path(blob.path)
            target = destination / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _copy_pinned_blob(repository, blob, target)
        _verify_pinned_snapshot(destination, blobs)
        for directory in sorted(
            (path for path in destination.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        destination.chmod(0o500)
        return destination.resolve(strict=True)
    except BaseException:
        try:
            _remove_tree(destination)
        except BaseException:
            pass
        raise


def _pinned_sam_blobs(repository: Path) -> tuple[_PinnedBlob, ...]:
    payload = _bounded_git_output(
        repository,
        (
            "ls-tree",
            "-r",
            "-z",
            "-l",
            "--full-tree",
            _PINNED_REPOSITORY_REVISION,
            "--",
            "sam3",
        ),
        _MAX_GIT_TREE_BYTES,
    )
    if not payload or not payload.endswith(b"\0"):
        raise ValueError
    blobs: list[_PinnedBlob] = []
    names: set[str] = set()
    total_bytes = 0
    pattern = re.compile(
        rb"(100644|100755) blob ([0-9a-f]{40}) +([0-9]+)\t([^\0]+)"
    )
    for record in payload[:-1].split(b"\0"):
        match = pattern.fullmatch(record)
        if match is None:
            raise ValueError
        try:
            name = match.group(4).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise ValueError from None
        path = Path(name)
        if (
            path.is_absolute()
            or path.parts[:1] != ("sam3",)
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != name
            or "\\" in name
            or len(name.encode("utf-8")) > 4096
            or any(
                ord(character) < 32 or 127 <= ord(character) <= 159
                for character in name
            )
            or name in names
        ):
            raise ValueError
        size = int(match.group(3))
        total_bytes += size
        if (
            len(blobs) >= _MAX_GIT_SOURCE_FILES
            or total_bytes > _MAX_GIT_SOURCE_BYTES
        ):
            raise ValueError
        names.add(name)
        blobs.append(
            _PinnedBlob(
                path=name,
                object_id=match.group(2).decode("ascii"),
                size=size,
            )
        )
    if not blobs:
        raise ValueError
    return tuple(sorted(blobs, key=lambda item: item.path))


def _bounded_git_output(
    repository: Path, arguments: tuple[str, ...], max_bytes: int
) -> bytes:
    if max_bytes <= 0:
        raise ValueError
    process = subprocess.Popen(
        _git_command(repository, *arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=False,
        env=_git_environment(),
    )
    output = bytearray()
    try:
        if process.stdout is None:
            raise ValueError
        while True:
            payload = process.stdout.read(
                min(_HASH_READ_BYTES, max_bytes - len(output) + 1)
            )
            if not payload:
                break
            output.extend(payload)
            if len(output) > max_bytes:
                raise ValueError
        if process.wait() != 0:
            raise ValueError
        return bytes(output)
    except BaseException:
        _stop_process(process)
        raise
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except BaseException:
                pass


def _copy_pinned_blob(
    repository: Path, blob: _PinnedBlob, destination: Path
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ValueError
    destination_descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
        0o400,
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            _git_command(repository, "cat-file", "blob", blob.object_id),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            env=_git_environment(),
        )
        if process.stdout is None:
            raise ValueError
        digest = hashlib.sha1()
        digest.update(f"blob {blob.size}\0".encode("ascii"))
        copied = 0
        while copied < blob.size:
            payload = process.stdout.read(
                min(_HASH_READ_BYTES, blob.size - copied)
            )
            if not payload:
                raise ValueError
            copied += len(payload)
            digest.update(payload)
            view = memoryview(payload)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
        if process.stdout.read(1) or process.wait() != 0:
            raise ValueError
        os.fsync(destination_descriptor)
        opened = os.fstat(destination_descriptor)
        after_path = destination.lstat()
        if (
            digest.hexdigest() != blob.object_id
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o400
            or opened.st_size != blob.size
            or (after_path.st_dev, after_path.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError
    except BaseException:
        if process is not None:
            _stop_process(process)
        raise
    finally:
        os.close(destination_descriptor)
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except BaseException:
                pass


def _verify_pinned_snapshot(
    destination: Path, blobs: tuple[_PinnedBlob, ...]
) -> None:
    expected = {blob.path: blob for blob in blobs}
    actual = {
        path.relative_to(destination).as_posix(): path
        for path in destination.rglob("*")
        if not path.is_dir()
    }
    if set(actual) != set(expected):
        raise ValueError
    for name, blob in expected.items():
        path = actual[name]
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or status.st_uid != os.getuid()
                or stat.S_IMODE(status.st_mode) != 0o400
                or status.st_size != blob.size
            ):
                raise ValueError
            digest = hashlib.sha1()
            digest.update(f"blob {blob.size}\0".encode("ascii"))
            while payload := os.read(descriptor, _HASH_READ_BYTES):
                digest.update(payload)
            if digest.hexdigest() != blob.object_id:
                raise ValueError
        finally:
            os.close(descriptor)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is None:
            process.kill()
    except BaseException:
        pass
    try:
        process.wait()
    except BaseException:
        pass


def _git_command(repository: Path, *arguments: str) -> list[str]:
    return [
        "git",
        "--no-replace-objects",
        "-C",
        str(repository),
        *arguments,
    ]


def _git_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
        }
    )
    return environment


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
    original_path = list(sys.path)
    try:
        for name in tuple(sys.modules):
            if name == "sam3" or name.startswith("sam3."):
                sys.modules.pop(name, None)
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
    except BaseException:
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


def _sampled_frame_dimensions(sampled: SampledFrameSet) -> tuple[int, int]:
    dimensions = tuple(_jpeg_dimensions(frame.path) for frame in sampled.frames)
    if not dimensions or len(set(dimensions)) != 1:
        raise ValueError
    return dimensions[0]


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            if source.read(2) != b"\xff\xd8":
                raise ValueError
            while True:
                prefix = source.read(1)
                if not prefix:
                    raise ValueError
                if prefix != b"\xff":
                    continue
                marker = source.read(1)
                while marker == b"\xff":
                    marker = source.read(1)
                if not marker or marker in {b"\xd8", b"\x01"}:
                    continue
                if marker == b"\xd9":
                    raise ValueError
                length_payload = source.read(2)
                if len(length_payload) != 2:
                    raise ValueError
                segment_length = struct.unpack(">H", length_payload)[0]
                if segment_length < 2:
                    raise ValueError
                if marker[0] in {
                    0xC0,
                    0xC1,
                    0xC2,
                    0xC3,
                    0xC5,
                    0xC6,
                    0xC7,
                    0xC9,
                    0xCA,
                    0xCB,
                    0xCD,
                    0xCE,
                    0xCF,
                }:
                    payload = source.read(segment_length - 2)
                    if len(payload) < 5:
                        raise ValueError
                    height, width = struct.unpack(">HH", payload[1:5])
                    if height <= 0 or width <= 0:
                        raise ValueError
                    return height, width
                source.seek(segment_length - 2, os.SEEK_CUR)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


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
        except BaseException:
            if not body_failed:
                raise _CleanupFailure from None


def _parse_frame_response(
    response: Any,
    sampled: SampledFrameSet,
    *,
    allowed_indices: range,
    expected_dimensions: tuple[int, int],
    mask_writer: _MaskArchiveWriter | None,
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
    object_ids = outputs[required[0]]
    probabilities = outputs[required[1]]
    boxes = outputs[required[2]]
    masks = outputs[required[3]]
    ids_shape = _array_metadata(object_ids, "integer")
    probs_shape = _array_metadata(probabilities, "float")
    boxes_shape = _array_metadata(boxes, "float")
    masks_shape = _array_metadata(masks, "boolean")
    count = ids_shape[0] if len(ids_shape) == 1 else -1
    if (
        ids_shape != (count,)
        or count > _MAX_OBJECTS
        or probs_shape != (count,)
        or boxes_shape != (count, 4)
        or len(masks_shape) != 3
        or masks_shape[0] != count
        or masks_shape[1] <= 0
        or masks_shape[2] <= 0
        or (masks_shape[1], masks_shape[2]) != expected_dimensions
    ):
        raise ValueError
    parsed_ids = tuple(
        _strict_integer(_array_item(object_ids, offset)) for offset in range(count)
    )
    if len(set(parsed_ids)) != len(parsed_ids):
        raise ValueError
    frame = sampled.frames[local_index]
    detections: list[_Detection] = []
    for object_offset in sorted(range(count), key=lambda offset: parsed_ids[offset]):
        object_id = parsed_ids[object_offset]
        raw_probability = _array_item(probabilities, object_offset)
        removed = (
            not isinstance(raw_probability, bool)
            and isinstance(raw_probability, Real)
            and float(raw_probability) == _REMOVED_OBJECT_SCORE
        )
        probability = (
            None
            if removed
            else _bounded_float(raw_probability, lower=0.0, upper=1.0)
        )
        box = tuple(
            _bounded_float(
                _array_item(boxes, object_offset, coordinate),
                lower=0.0,
                upper=1.0,
            )
            for coordinate in range(4)
        )
        x, y, width, height = box
        if width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError
        if removed:
            _consume_mask_rows(
                masks,
                object_offset,
                masks_shape[1],
                masks_shape[2],
            )
            continue
        if mask_writer is None:
            consumed = _consume_mask_rows(
                masks,
                object_offset,
                masks_shape[1],
                masks_shape[2],
            )
            if consumed is None:
                continue
            area_fraction, center_xy = consumed
            mask_index = -1
        else:
            written = mask_writer.write_mask(
                masks,
                object_offset,
                masks_shape[1],
                masks_shape[2],
            )
            if written is None:
                continue
            mask_index, area_fraction, center_xy = written
        detections.append(
            _Detection(
                local_frame_index=local_index,
                source_frame_index=frame.source_frame_index,
                timestamp_seconds=frame.source_timestamp_seconds,
                object_id=object_id,
                probability=probability,
                box_xywh=(x, y, width, height),
                area_fraction=area_fraction,
                center_xy=center_xy,
                mask_index=mask_index,
            )
        )
    return local_index, tuple(detections), (masks_shape[1], masks_shape[2])


def _array_metadata(value: Any, expected_kind: str) -> tuple[int, ...]:
    shape_value = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if not isinstance(shape_value, Sequence) or isinstance(shape_value, (str, bytes)):
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
    return shape


def _array_item(value: Any, *indices: int) -> Any:
    try:
        item = value[indices if len(indices) > 1 else indices[0]]
    except (IndexError, KeyError, TypeError):
        try:
            item = value
            for index in indices:
                item = item[index]
        except (IndexError, KeyError, TypeError):
            raise ValueError from None
    scalar = getattr(item, "item", None)
    item_shape = getattr(item, "shape", None)
    if callable(scalar) and (item_shape is None or tuple(item_shape) == ()):
        try:
            item = scalar()
        except (TypeError, ValueError):
            raise ValueError from None
    return item


def _mask_row_bytes(
    masks: Any, object_offset: int, row_index: int, width: int
) -> bytes:
    row = _array_item(masks, object_offset, row_index)
    if isinstance(row, (bytes, bytearray, memoryview)):
        payload = row
    else:
        convert = getattr(row, "tobytes", None)
        if callable(convert):
            try:
                payload = convert(order="C")
            except TypeError:
                payload = convert()
        else:
            try:
                if len(row) != width:
                    raise ValueError
                payload = bytes(
                    1 if _strict_mask_boolean(row[column]) else 0
                    for column in range(width)
                )
            except (IndexError, KeyError, TypeError):
                raise ValueError from None
    if type(payload) is not bytes:
        try:
            payload = memoryview(payload).tobytes()
        except TypeError:
            payload = bytes(payload)
    if len(payload) != width or payload.count(0) + payload.count(1) != width:
        raise ValueError
    return payload


def _strict_mask_boolean(value: Any) -> bool:
    scalar = getattr(value, "item", None)
    if callable(scalar):
        value = scalar()
    if not isinstance(value, bool):
        raise TypeError
    return value


def _consume_mask_rows(
    masks: Any, object_offset: int, height: int, width: int
) -> tuple[float, tuple[float, float]] | None:
    """Scan one mask; None means all-empty (the object is invisible)."""
    true_count = 0
    weighted_columns = 0
    weighted_rows = 0
    for row_index in range(height):
        row = _mask_row_bytes(masks, object_offset, row_index, width)
        row_count = row.count(1)
        true_count += row_count
        weighted_rows += row_index * row_count
        weighted_columns += sum(itertools.compress(range(width), row))
    if true_count <= 0:
        return None
    pixels = height * width
    return (
        true_count / pixels,
        (
            (weighted_columns + (0.5 * true_count)) / true_count / width,
            (weighted_rows + (0.5 * true_count)) / true_count / height,
        ),
    )


def _npy_header(descriptor: str, shape: tuple[int, ...]) -> bytes:
    shape_text = repr(shape)
    header_text = (
        "{'descr': "
        + repr(descriptor)
        + ", 'fortran_order': False, 'shape': "
        + shape_text
        + ", }"
    )
    prefix_length = 10
    padding = (-((prefix_length + len(header_text) + 1) % 16)) % 16
    header = (header_text + (" " * padding) + "\n").encode("latin1")
    if len(header) > 65_535:
        raise ValueError
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def _write_int64_npy(
    archive: zipfile.ZipFile, name: str, values: tuple[int, ...]
) -> None:
    with archive.open(_zip_info(name), mode="w", force_zip64=True) as member:
        member.write(_npy_header("<i8", (len(values),)))
        buffer = bytearray()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError
            buffer.extend(struct.pack("<q", value))
            if len(buffer) >= _HASH_READ_BYTES:
                member.write(buffer)
                buffer.clear()
        if buffer:
            member.write(buffer)


def _nonnegative_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError
    converted = int(value)
    if converted < 0:
        raise ValueError
    return converted


def _strict_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError
    converted = int(value)
    if converted < 0:
        raise ValueError
    return converted


def _bounded_float(value: Any, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError
    converted = float(value)
    if not math.isfinite(converted) or not lower <= converted <= upper:
        raise ValueError
    return converted


def _prompt_coverage(run: _PromptRun) -> float:
    """Fraction of sampled frames where the prompt detected the entity."""
    if not run.present_ids_by_frame:
        return 0.0
    detected = sum(1 for identifiers in run.present_ids_by_frame if identifiers)
    return detected / len(run.present_ids_by_frame)


def _frame_union_boxes(
    run: _PromptRun,
) -> dict[int, tuple[float, float, float, float]]:
    """Per-frame union of the run's detection boxes as (x0, y0, x1, y1)."""
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for detection in run.detections:
        x, y, width, height = detection.box_xywh
        candidate = (x, y, x + width, y + height)
        existing = boxes.get(detection.local_frame_index)
        if existing is not None:
            candidate = (
                min(existing[0], candidate[0]),
                min(existing[1], candidate[1]),
                max(existing[2], candidate[2]),
                max(existing[3], candidate[3]),
            )
        boxes[detection.local_frame_index] = candidate
    return boxes


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _alias_replacement_consistent(
    canonical: _PromptRun, replacement: _PromptRun
) -> bool:
    """Do both runs point at the same physical entity where both see it?"""
    reference = _frame_union_boxes(canonical)
    candidate = _frame_union_boxes(replacement)
    if not reference:
        return True
    common = sorted(set(reference) & set(candidate))
    if len(common) < _ALIAS_CONSISTENCY_MIN_FRAMES:
        return False
    overlaps = sorted(
        _box_iou(reference[index], candidate[index]) for index in common
    )
    return overlaps[len(overlaps) // 2] >= _ALIAS_CONSISTENCY_MIN_IOU


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
    sampled: SampledFrameSet,
    prompt_runs: tuple[_PromptRun, ...],
    output_directories: list[Path],
    budget: _ArtifactBudget,
    overlay_renderer: Callable[..., None],
) -> CvEvidenceArtifact:
    artifact_files: list[ArtifactFile] = []
    overlay_records: list[OverlayRecord] = []
    tracks: list[CvTrack] = []
    for prompt_run in prompt_runs:
        mask_relative = prompt_run.mask_relative
        if mask_relative is None:
            raise ValueError
        mask_path = staging / mask_relative
        ordered_detections = tuple(
            sorted(
                prompt_run.detections,
                key=lambda item: (item.local_frame_index, item.object_id),
            )
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
            prompt_run = next(
                item for item in prompt_runs if item.entity_id == entity_id
            )
            if prompt_run.mask_relative is None or detection.mask_index < 0:
                raise ValueError
            budget.reserve(
                (prompt_run.mask_height * prompt_run.mask_width * 3)
                + _ZIP_ENTRY_ALLOWANCE,
                file_count=1,
            )
            relative = (
                f"overlays/{entity_id}-{object_id}-"
                f"{detection.source_frame_index:08d}.png"
            )
            overlay_path = staging / relative
            x, y, width, height = detection.box_xywh
            overlay_renderer(
                frame_path=sampled.frames[detection.local_frame_index].path,
                archive_path=staging / prompt_run.mask_relative,
                mask_member=f"masks/{detection.mask_index:08d}.npy",
                bbox_xyxy=(x, y, x + width, y + height),
                destination=overlay_path,
            )
            overlay_file = _artifact_file(staging, overlay_path)
            budget.record_file(overlay_file.size_bytes)
            artifact_files.append(overlay_file)
            overlay_records.append(
                OverlayRecord(
                    path=relative,
                    track_id=f"{entity_id}_{object_id}",
                    frame_index=detection.source_frame_index,
                )
            )

    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status=EvidenceStatus.AVAILABLE,
        provider="sam31",
        model_identity=request.model_identity,
        video_sha256=request.video_sha256,
        checkpoint_sha256=request.checkpoint_sha256,
        processed_timeline=FrameTimeline(
            frames=tuple(
                FrameTimestamp(
                    frame_index=frame.source_frame_index,
                    timestamp_seconds=frame.source_timestamp_seconds,
                )
                for frame in sampled.frames
            )
        ),
        entities=request.entities,
        tracks=tuple(tracks),
        files=tuple(sorted(artifact_files, key=lambda item: item.path)),
        overlay_records=tuple(sorted(overlay_records, key=lambda item: item.path)),
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
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = payload
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        except BaseException:
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
        status = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise OSError("refusing to remove a replaced cleanup directory")
    for directory in [path, *path.rglob("*")]:
        try:
            if directory.is_dir() and not directory.is_symlink():
                directory.chmod(0o700)
        except OSError:
            pass
    shutil.rmtree(path)
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise OSError("cleanup directory still exists")
