"""Parse SAM3.1 predictor responses and assemble CV evidence artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import io
from numbers import Integral
import os
from pathlib import Path
import stat
from typing import Any
import zipfile

from .contracts import (
    ArtifactFile,
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
from .masks import (
    ZIP_ENTRY_ALLOWANCE,
    ArtifactBudget,
    MaskArchiveWriter,
    mask_member_name,
)
from .timeline import SampledFrameSet


# Maximum candidates returned by the pinned upstream builder default.
_MAX_OBJECTS = 16
# Pinned sam3_multiplex_base.py uses this score for removed objects.
_REMOVED_OBJECT_SCORE = -10000.0


@dataclass(frozen=True, slots=True)
class Detection:
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
class PromptRun:
    entity_id: str
    detections: tuple[Detection, ...]
    present_ids_by_frame: tuple[frozenset[int], ...]
    mask_height: int
    mask_width: int
    mask_relative: str


def parse_frame_response(
    response: Any,
    sampled: SampledFrameSet,
    *,
    expected_dimensions: tuple[int, int],
    mask_writer: MaskArchiveWriter,
) -> tuple[int, tuple[Detection, ...]]:
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
    detections: list[Detection] = []
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
            Detection(
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


def build_artifact(
    request: CvEvidenceRequest, staging: Path, sampled: SampledFrameSet,
    run: PromptRun, output_directories: list[Path], budget: ArtifactBudget,
    overlay_renderer: Callable[..., None],
) -> CvEvidenceArtifact:
    files = [_artifact_file(staging, staging / run.mask_relative)]
    records: list[OverlayRecord] = []
    tracks = tuple(
        CvTrack(
            track_id=_track_id(run.entity_id, object_id), entity_id=run.entity_id,
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
        output_directories.append(create_private_directory(staging / "overlays"))
        for item in overlays[:MAX_OVERLAYS]:
            budget.reserve(
                run.mask_height * run.mask_width * 3 + ZIP_ENTRY_ALLOWANCE
            )
            relative = (
                f"overlays/{run.entity_id}-{item.object_id}-{item.source_frame_index:08d}.png"
            )
            overlay_renderer(
                frame_path=sampled.frames[item.local_frame_index].path,
                archive_path=staging / run.mask_relative,
                mask_member=mask_member_name(item.mask_index),
                bbox_xyxy=_xyxy(item.box_xywh), destination=staging / relative,
            )
            overlay_file = _artifact_file(staging, staging / relative)
            budget.record_file(overlay_file.size_bytes)
            files.append(overlay_file)
            records.append(OverlayRecord(
                path=relative, track_id=_track_id(run.entity_id, item.object_id),
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


def _observation(detection: Detection, mask_relative: str) -> TrackObservation:
    return TrackObservation(
        frame_index=detection.source_frame_index,
        timestamp_seconds=detection.timestamp_seconds,
        bbox_xyxy=_xyxy(detection.box_xywh),
        mask_ref=mask_relative,
        visible=True,
        confidence=detection.probability,
        area_fraction=detection.area_fraction,
        center_xy=detection.center_xy,
    )


def _xyxy(box_xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, width, height = box_xywh
    return x, y, x + width, y + height


def _track_id(entity_id: str, object_id: int) -> str:
    return f"{entity_id}_{object_id}"


def _select_overlays(run: PromptRun) -> tuple[Detection, ...]:
    detections = {(item.local_frame_index, item.object_id): item for item in run.detections}
    selected: dict[tuple[int, int], Detection] = {}
    for index, (before, after) in enumerate(zip(
        run.present_ids_by_frame, run.present_ids_by_frame[1:]
    ), start=1):
        for object_id in before ^ after:
            for frame_index in (index - 1, index):
                item = detections.get((frame_index, object_id))
                if item is not None:
                    selected[(object_id, item.source_frame_index)] = item
    return tuple(selected[key] for key in sorted(selected))


def create_private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, exist_ok=False)
    return path


def render_context_overlay(
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
