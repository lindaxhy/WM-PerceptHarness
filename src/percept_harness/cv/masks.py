"""Byte-budgeted, incremental mask archives for SAM evidence."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
import zipfile

from .contracts import MAX_ARTIFACT_BYTES
from .timeline import SampledFrameSet

if TYPE_CHECKING:
    from .sam31_artifacts import Detection


ZIP_ENTRY_ALLOWANCE = 1024


@dataclass(slots=True)
class ArtifactBudget:
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


class MaskArchiveWriter:
    """Incrementally deflate one prompt's masks without retaining mask tensors."""

    def __init__(self, path: Path, budget: ArtifactBudget) -> None:
        budget.reserve(ZIP_ENTRY_ALLOWANCE)
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
        self.budget.reserve(mask.nbytes + ZIP_ENTRY_ALLOWANCE)
        mask_index = self.mask_count
        _write_npy(self._archive, mask_member_name(mask_index), mask)
        self.mask_count += 1
        return mask_index

    def finish(
        self, detections: tuple[Detection, ...], sampled: SampledFrameSet
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
                    len(values) * 8 + ZIP_ENTRY_ALLOWANCE
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


def _write_npy(archive: zipfile.ZipFile, name: str, array: Any) -> None:
    numpy = importlib.import_module("numpy")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    with archive.open(info, mode="w", force_zip64=True) as member:
        numpy.save(member, array, allow_pickle=False)


def mask_archive_name(entity_id: str) -> str:
    return f"masks/{entity_id}.npz"


def mask_member_name(mask_index: int) -> str:
    return f"masks/{mask_index:08d}.npy"
