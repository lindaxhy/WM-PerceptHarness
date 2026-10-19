#!/usr/bin/env python3
"""Strict adapters for the retained metrics' official input files.

The adapters do not score videos, create captions, infer actions, or alter an
official rubric.  They only validate a caller-owned JSON/CSV manifest and
emit the field names used by the upstream evaluator.  A ``video_map`` sidecar
is returned for datasets whose official metadata intentionally does not carry
the local video path (for example T2V-CompBench and VBench-2.0).

Supported official formats:

* WorldModelBench ``worldmodelbench.json``
* T2V-CompBench V2 action binding, motion binding and object interactions
* VBench-2.0 ``prompts/meta_info/*.json`` records
* PhyGenBench prompt/question assets (validated without renaming fields)
* VideoPhy-2 AutoEval PC/SA CSV (``videopath`` and optional ``caption``)

No model weights or upstream code are downloaded by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from percept_harness.video_metrics._runtime import write_json_exclusive


class InputSchemaError(ValueError):
    """Raised when a caller manifest cannot be represented by the official format."""


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise InputSchemaError(f"{field} must be a non-empty string")
    return value if allow_empty else value.strip()


def _records(payload: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise InputSchemaError(f"{label} must be a non-empty JSON array")
    if not all(isinstance(row, dict) for row in payload):
        raise InputSchemaError(f"{label} must contain JSON objects")
    return payload


def _sidecar(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Return local mapping fields without leaking them into official metadata."""
    video_id = _text(row.get("video_id"), f"records[{index}].video_id")
    video = _text(row.get("video"), f"records[{index}].video")
    return {"video_id": video_id, "video": video}


def _unique_sidecars(rows: Iterable[dict[str, Any]]) -> None:
    ids: set[str] = set()
    videos: set[str] = set()
    for row in rows:
        if row["video_id"] in ids:
            raise InputSchemaError(f"duplicate video_id: {row['video_id']}")
        if row["video"] in videos:
            raise InputSchemaError(f"duplicate video path: {row['video']}")
        ids.add(row["video_id"])
        videos.add(row["video"])


def adapt_worldmodelbench(payload: Any) -> dict[str, Any]:
    """Convert local records to the exact WorldModelBench item fields.

    Official records contain ``domain``, ``subdomain``, ``text_first_frame``,
    ``text_instruction`` and ``first_frame``.  The local video path is kept in
    a separate map because the official evaluator resolves the video from the
    first-frame stem.
    """
    rows = _records(payload, "WorldModelBench records")
    official: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        local = _sidecar(row, i)
        first = _text(row.get("first_frame"), f"records[{i}].first_frame")
        item: dict[str, Any] = {
            "text_first_frame": _text(row.get("text_first_frame"), f"records[{i}].text_first_frame"),
            "text_instruction": _text(row.get("text_instruction"), f"records[{i}].text_instruction"),
            "first_frame": first,
        }
        for key in ("domain", "subdomain"):
            if key in row:
                item[key] = _text(row[key], f"records[{i}].{key}")
        official.append(item)
        mapping.append({**local, "first_frame": first})
    _unique_sidecars(mapping)
    return {"official": official, "video_map": mapping, "format": "WorldModelBench/worldmodelbench.json"}


def _phrases(row: dict[str, Any], index: int, key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise InputSchemaError(f"records[{index}].{key} must be a two-item list")
    return [_text(value[0], f"records[{index}].{key}[0]"), _text(value[1], f"records[{index}].{key}[1]")]


def adapt_action_binding(payload: Any) -> dict[str, Any]:
    """Validate/emit T2V-CompBench V2 ``meta_data/action_binding.json``."""
    rows = _records(payload, "action_binding records")
    official: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        mapping.append(_sidecar(row, i))
        official.append({"prompt": _text(row.get("prompt"), f"records[{i}].prompt"),
                         "phrase_0": _phrases(row, i, "phrase_0"),
                         "phrase_1": _phrases(row, i, "phrase_1")})
    _unique_sidecars(mapping)
    return {"official": official, "video_map": mapping, "format": "T2V-CompBench/V2/action_binding.json"}


_DIRECTIONS = {"left", "right", "up", "down"}


def adapt_motion_binding(payload: Any) -> dict[str, Any]:
    """Validate/emit T2V-CompBench V2 ``meta_data/motion_binding.json``."""
    rows = _records(payload, "motion_binding records")
    official: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        mapping.append(_sidecar(row, i))
        item = {"prompt": _text(row.get("prompt"), f"records[{i}].prompt")}
        present = []
        for obj, direction in (("object_1", "d_1"), ("object_2", "d_2")):
            obj_value = _text(row.get(obj), f"records[{i}].{obj}", allow_empty=True)
            direction_value = _text(row.get(direction), f"records[{i}].{direction}", allow_empty=True).lower()
            if bool(obj_value) != bool(direction_value):
                raise InputSchemaError(f"records[{i}] requires {obj} and {direction} together")
            if direction_value and direction_value not in _DIRECTIONS:
                raise InputSchemaError(f"records[{i}].{direction} must be one of left/right/up/down")
            item[obj] = obj_value
            item[direction] = direction_value
            present.append(bool(obj_value))
        if not present[0]:
            raise InputSchemaError(f"records[{i}].object_1 must be non-empty")
        official.append(item)
    _unique_sidecars(mapping)
    return {"official": official, "video_map": mapping, "format": "T2V-CompBench/V2/motion_binding.json"}


def adapt_object_interactions(payload: Any) -> dict[str, Any]:
    """Validate/emit T2V-CompBench V2 ``meta_data/object_interactions.json``."""
    rows = _records(payload, "object_interactions records")
    official: list[dict[str, str]] = []
    mapping: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        mapping.append(_sidecar(row, i))
        official.append({"prompt": _text(row.get("prompt"), f"records[{i}].prompt")})
    _unique_sidecars(mapping)
    return {"official": official, "video_map": mapping, "format": "T2V-CompBench/V2/object_interactions.json"}


_VB2_NAMES = {
    "Motion_Order_Understanding",
    "Motion_Rationality",
    "Mechanics",
    "Thermotics",
    "Material",
}


def adapt_vbench2(payload: Any) -> dict[str, Any]:
    """Validate/emit VBench-2.0 ``prompts/meta_info/*.json`` records.

    The emitted records intentionally use ``prompt_en``, ``dimension`` and
    ``auxiliary_info`` exactly as in ``VBench2_full_info.json``.  ``dimension``
    is a one-item list in the full-info file; the per-dimension files use the
    same value as a string, so this adapter accepts either and emits a list.
    """
    rows = _records(payload, "VBench-2.0 records")
    official: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        mapping.append(_sidecar(row, i))
        prompt = _text(row.get("prompt_en"), f"records[{i}].prompt_en")
        raw_dim = row.get("dimension")
        dims = [raw_dim] if isinstance(raw_dim, str) else raw_dim
        if not isinstance(dims, list) or len(dims) != 1 or dims[0] not in _VB2_NAMES:
            raise InputSchemaError(f"records[{i}].dimension must be one VBench-2.0 retained dimension")
        aux = row.get("auxiliary_info")
        if not isinstance(aux, list) or not aux or not all(isinstance(x, str) and x.strip() for x in aux):
            raise InputSchemaError(f"records[{i}].auxiliary_info must be a non-empty list of strings")
        if dims[0] == "Motion_Order_Understanding" and len(aux) != 2:
            raise InputSchemaError("Motion_Order_Understanding auxiliary_info must contain exactly two actions")
        # Preserve the shape of the pinned upstream asset: the per-dimension
        # files use a string, while VBench2_full_info.json uses a one-item
        # list.  Do not normalize one into the other.
        dimension_value: str | list[str] = raw_dim if isinstance(raw_dim, str) else dims
        official.append({"prompt_en": prompt, "dimension": dimension_value, "auxiliary_info": [x.strip() for x in aux]})
    _unique_sidecars(mapping)
    return {"official": official, "video_map": mapping, "format": "VBench-2.0/VBench2_full_info.json"}


def validate_phygen_eval(payload: Any, *, stage: str) -> dict[str, Any]:
    """Validate PhyGenBench assets without inventing or renaming fields.

    ``prompts`` is the official ``prompts.json`` list.  The three question
    assets are passed through as authored by PhyGenBench.  Since the upstream
    repository has changed question examples over time, this validator checks
    only the stable contract: non-empty JSON records and a stable prompt/id
    key when present; it does not impose a home-grown field name.
    """
    if stage not in {"prompts", "single", "multi", "video"}:
        raise InputSchemaError("PhyGenEval stage must be prompts, single, multi, or video")
    rows = _records(payload, f"PhyGenEval {stage}")
    for i, row in enumerate(rows):
        if stage == "prompts":
            if not any(isinstance(row.get(k), str) and row[k].strip() for k in ("prompt", "prompt_en", "caption")):
                raise InputSchemaError(f"records[{i}] must contain the official prompt text")
        # Question files are official assets; preserve all fields and reject no
        # optional official keys.  A row must still be non-empty JSON.
        if not row:
            raise InputSchemaError(f"records[{i}] must not be empty")
    return {"official": rows, "format": f"PhyGenBench/PhyGenEval/{stage}"}


def adapt_videophy_csv(rows: Any, *, task: str) -> list[dict[str, str]]:
    """Validate VideoPhy-2 AutoEval CSV rows for ``pc`` or ``sa``.

    The official example header is ``videopath`` for PC and ``caption,videopath``
    for SA.  Output is a list suitable for writing with ``csv.DictWriter``;
    no score column is accepted or fabricated.
    """
    if task not in {"pc", "sa"}:
        raise InputSchemaError("VideoPhy-2 task must be pc or sa")
    if not isinstance(rows, list) or not rows or not all(isinstance(x, dict) for x in rows):
        raise InputSchemaError("VideoPhy-2 rows must be a non-empty list of objects")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        path = _text(row.get("videopath"), f"rows[{i}].videopath")
        if path in seen:
            raise InputSchemaError(f"duplicate videopath: {path}")
        seen.add(path)
        item = {"videopath": path}
        if task == "sa":
            item = {"caption": _text(row.get("caption"), f"rows[{i}].caption"), "videopath": path}
        out.append(item)
    return out


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - CLI diagnostic
        raise InputSchemaError(f"cannot read JSON {path}: {exc}") from exc


def _write_exclusive(path: Path, payload: Any) -> None:
    if path.exists():
        raise InputSchemaError(f"refusing to overwrite existing output: {path}")
    write_json_exclusive(path, payload, parents=True, strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=("worldmodelbench", "action_binding", "motion_binding", "object_interactions", "vbench2", "phygen_prompts", "phygen_single", "phygen_multi", "phygen_video", "videophy_pc", "videophy_sa"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-map-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.adapter.startswith("videophy_"):
            payload = _load(args.input)
            result = adapt_videophy_csv(payload, task=args.adapter.rsplit("_", 1)[1])
            if args.output.exists():
                raise InputSchemaError(f"refusing to overwrite existing output: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(result[0]))
                writer.writeheader()
                writer.writerows(result)
            return 0
        payload = _load(args.input)
        if args.adapter == "worldmodelbench":
            result = adapt_worldmodelbench(payload)
        elif args.adapter == "action_binding":
            result = adapt_action_binding(payload)
        elif args.adapter == "motion_binding":
            result = adapt_motion_binding(payload)
        elif args.adapter == "object_interactions":
            result = adapt_object_interactions(payload)
        elif args.adapter == "vbench2":
            result = adapt_vbench2(payload)
        else:
            result = validate_phygen_eval(payload, stage=args.adapter.removeprefix("phygen_"))
        _write_exclusive(args.output, result["official"])
        if args.video_map_output and "video_map" in result:
            _write_exclusive(args.video_map_output, result["video_map"])
        return 0
    except (InputSchemaError, OSError) as exc:
        print(f"Input adaptation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
