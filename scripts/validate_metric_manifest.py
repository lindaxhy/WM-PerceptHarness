#!/usr/bin/env python3
"""Validate frozen video/prompt/reference metadata before metric runs.

The manifest is a JSON array.  Each record needs a unique ``video_id`` and a
``video`` path.  Optional fields are ``prompt``, ``dimensions``,
``reference_frames`` and ``auxiliary_info``.  This tool only validates data;
it never invents prompts, captions, questions, or reference frames.
This is a preparation manifest, not VBench-2.0's official full-info file.
Question dimensions use the official non-empty list-of-strings auxiliary_info
shape. Validation does not establish that questions or reference frames are
correct for the video, nor make unsupported custom_input dimensions runnable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from percept_harness.video_metrics._runtime import run_cli
from percept_harness.video_metrics.external import VBENCH2_DIMENSIONS, vbench2_auxiliary_problem


SEMANTIC_DIMENSIONS = frozenset(
    {
        "clip_score",
        "overall_consistency",
        "instruction_following",
        "action_binding",
        "motion_binding",
        "motion_order_understanding",
        "motion_rationality",
        "object_interactions",
        "physical_adherence",
        "mechanics",
        "thermotics",
        "material",
        "video_phy_sa",
        "phygen_eval_pca",
    }
)
REFERENCE_DIMENSIONS = frozenset({"psnr", "ssim", "lpips"})
AUXILIARY_DIMENSIONS = VBENCH2_DIMENSIONS
KNOWN_DIMENSIONS = SEMANTIC_DIMENSIONS | REFERENCE_DIMENSIONS | frozenset({
    "clipiqa+", "clipiqa_plus", "imaging_quality", "aesthetic_quality", "dover",
    "motion_smoothness", "temporal_flickering", "dynamic_degree", "video_phy_pc",
    "subject_consistency", "background_consistency", "long_subject_consistency",
    "long_background_consistency", "fid", "fvd", "object_class", "multiple_objects",
    "color", "spatial_relationship", "scene", "temporal_style", "human_action",
    "appearance_style",
})


def _string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _resolve_path(value: str, root: Path | None) -> Path:
    path = Path(value.strip()).expanduser()
    if root is not None and not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _check_payload(payload: Any) -> None:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Manifest must be a non-empty JSON array")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Every manifest item must be a JSON object")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _check_payload(payload)
    return payload


def validate_manifest(
    payload: list[dict[str, Any]], *, root: Path | None = None, dimensions: set[str] | None = None, check_files: bool = False
) -> dict[str, Any]:
    _check_payload(payload)
    if dimensions is not None and (
        not isinstance(dimensions, (set, frozenset, list, tuple)) or not all(_string(value) for value in dimensions)
    ):
        raise ValueError("dimensions must be a collection of non-empty strings")
    dimensions = set(dimensions or ())
    if dimensions - KNOWN_DIMENSIONS:
        raise ValueError(f"Unknown dimensions: {sorted(dimensions - KNOWN_DIMENSIONS)}")
    ids: set[str] = set()
    video_paths: set[str] = set()
    checked_dimensions: set[str] = set(dimensions)
    errors: list[str] = []
    normalized_root = root.expanduser().resolve() if root else None
    for index, item in enumerate(payload):
        prefix = f"records[{index}]"
        video_id = item.get("video_id")
        video = item.get("video")
        if not _string(video_id):
            errors.append(f"{prefix}.video_id must be a non-empty string")
        elif video_id.strip() in ids:
            errors.append(f"duplicate video_id: {video_id}")
        else:
            ids.add(video_id.strip())
        video_path = None
        if not _string(video):
            errors.append(f"{prefix}.video must be a non-empty path")
        else:
            video_path = _resolve_path(video, normalized_root)
            # Do not lowercase: a.mp4 and A.mp4 may be different on Linux.
            path_key = str(video_path)
            if path_key in video_paths:
                errors.append(f"duplicate video path: {video}")
            video_paths.add(path_key)
            if check_files and not video_path.is_file():
                errors.append(f"{prefix}.video does not exist: {video_path}")
        record_dimensions = item.get("dimensions", [])
        if not isinstance(record_dimensions, list) or not all(_string(value) for value in record_dimensions):
            errors.append(f"{prefix}.dimensions must be a list of non-empty strings")
            record_dimensions = []
        elif len(record_dimensions) != len(set(record_dimensions)):
            errors.append(f"{prefix}.dimensions contains duplicate dimensions")
        if set(record_dimensions) - KNOWN_DIMENSIONS:
            errors.append(f"{prefix}.dimensions has unknown names: {sorted(set(record_dimensions) - KNOWN_DIMENSIONS)}")
        effective_dimensions = dimensions or set(record_dimensions)
        checked_dimensions.update(effective_dimensions)
        if "prompt" in item and not _string(item["prompt"]):
            errors.append(f"{prefix}.prompt must be a non-empty string when supplied")
        if effective_dimensions & SEMANTIC_DIMENSIONS:
            if not _string(item.get("prompt")):
                errors.append(f"{prefix}.prompt is required for semantic dimensions")
        if "reference_frames" in item and not _string(item["reference_frames"]):
            errors.append(f"{prefix}.reference_frames must be a non-empty path when supplied")
        if effective_dimensions & REFERENCE_DIMENSIONS:
            reference = item.get("reference_frames")
            if not _string(reference):
                errors.append(f"{prefix}.reference_frames is required for PSNR/SSIM/LPIPS")
        if _string(item.get("reference_frames")):
            reference_path = _resolve_path(item["reference_frames"], normalized_root)
            if video_path is not None and reference_path == video_path:
                errors.append(f"{prefix}.reference_frames cannot be the generated video path")
            if check_files and not reference_path.is_dir():
                errors.append(f"{prefix}.reference_frames does not exist: {reference_path}")
        if "auxiliary_info" in item or effective_dimensions & AUXILIARY_DIMENSIONS:
            problem = vbench2_auxiliary_problem(item.get("auxiliary_info"),
                                                ordered_pair="motion_order_understanding" in effective_dimensions)
            if problem == "shape":
                errors.append(f"{prefix}.auxiliary_info must be a non-empty list of strings for VBench-2.0 question dimensions")
            elif problem == "count":
                errors.append(f"{prefix}.auxiliary_info must contain exactly two ordered actions for Motion Order Understanding")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "records": len(payload),
        "video_ids": sorted(ids),
        "dimensions_checked": sorted(checked_dimensions),
        "validation_scope": "Preparation schema only; target-specific metadata, reference alignment and official input compatibility still require review.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dimension", action="append", default=[])
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args(argv)

    def run():
        payload = load_manifest(args.manifest.expanduser().resolve(strict=True))
        return validate_manifest(payload, root=args.root, dimensions=set(args.dimension), check_files=args.check_files)
    return run_cli("Manifest validation", run, allow_nan=True)


if __name__ == "__main__":
    raise SystemExit(main())
