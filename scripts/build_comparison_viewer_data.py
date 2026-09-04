"""Build deterministic, display-only local annotation files for the demo viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FINE_FIELDS = (
    "segment_index",
    "action_index",
    "start",
    "end",
    "description",
    "actor",
    "actor_state",
    "skill",
    "target",
    "visual_motion_state",
    "event_type",
    "confidence",
)
GROUPED_FIELDS = (
    "event_index",
    "start",
    "end",
    "description",
    "actor",
    "action",
    "target",
    "confidence",
    "source_segment_indices",
)
SCENE_FIELDS = (
    "event_index",
    "start",
    "end",
    "event_type",
    "actor",
    "target_object_id",
    "description",
    "confidence",
)
OBJECT_FIELDS = ("object_id", "name", "description")
STATE_FIELDS = ("object_id", "state", "visual_evidence", "confidence")
OUTCOME_FIELDS = ("status", "description", "confidence")


class ViewerDataError(ValueError):
    """An input cannot be safely projected into viewer data."""


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ViewerDataError(code)
    return value


def _array(value: object, code: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ViewerDataError(code)
    return value


def _finite_number(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ViewerDataError(code)
    number = float(value)
    if not math.isfinite(number):
        raise ViewerDataError(code)
    return number


def _validate_interval(item: Mapping[str, object], duration: float) -> None:
    start = _finite_number(item.get("start"), "INVALID_INTERVAL")
    end = _finite_number(item.get("end"), "INVALID_INTERVAL")
    if not 0.0 <= start < end <= duration:
        raise ViewerDataError("INVALID_INTERVAL")


def _required_fields(
    item: Mapping[str, object], fields: Sequence[str], code: str
) -> Mapping[str, object]:
    if not set(fields) <= set(item):
        raise ViewerDataError(code)
    return item


def _project_fields(value: Mapping[str, object], fields: Sequence[str]) -> dict[str, object]:
    return {field: value[field] for field in fields}


def project_local_result(
    sample_id: str,
    duration: float,
    source: Mapping[str, object],
    *,
    source_sha256: str,
) -> dict[str, object]:
    duration = _finite_number(duration, "INVALID_DURATION")
    if duration <= 0.0:
        raise ViewerDataError("INVALID_DURATION")
    fine = _array(source.get("segments"), "EXPECTED_ARRAY:segments")
    grouped = _array(
        source.get("grouped_semantic_events"),
        "EXPECTED_ARRAY:grouped_semantic_events",
    )
    scene = _array(source.get("semantic_events"), "EXPECTED_ARRAY:semantic_events")
    objects = _array(source.get("objects"), "EXPECTED_ARRAY:objects")
    initial = _array(source.get("initial_state"), "EXPECTED_ARRAY:initial_state")
    final = _array(source.get("final_state"), "EXPECTED_ARRAY:final_state")
    outcome = _mapping(source.get("outcome"), "EXPECTED_OBJECT:outcome")
    for collection in (fine, grouped, scene):
        for item in collection:
            _validate_interval(item, duration)

    return {
        "schema_version": "comparison_viewer_local_v1",
        "sample_id": sample_id,
        "duration_seconds": duration,
        "source_result_sha256": source_sha256,
        "fine_segments": [
            _project_fields(_required_fields(item, FINE_FIELDS, "MISSING_FINE_FIELD"), FINE_FIELDS)
            for item in fine
        ],
        "grouped_events": [
            _project_fields(
                _required_fields(item, GROUPED_FIELDS, "MISSING_GROUPED_FIELD"),
                GROUPED_FIELDS,
            )
            for item in grouped
        ],
        "scene_events": [
            _project_fields(_required_fields(item, SCENE_FIELDS, "MISSING_SCENE_FIELD"), SCENE_FIELDS)
            for item in scene
        ],
        "objects": [
            _project_fields(_required_fields(item, OBJECT_FIELDS, "MISSING_OBJECT_FIELD"), OBJECT_FIELDS)
            for item in objects
        ],
        "initial_state": [
            _project_fields(_required_fields(item, STATE_FIELDS, "MISSING_STATE_FIELD"), STATE_FIELDS)
            for item in initial
        ],
        "final_state": [
            _project_fields(_required_fields(item, STATE_FIELDS, "MISSING_STATE_FIELD"), STATE_FIELDS)
            for item in final
        ],
        "outcome": _project_fields(
            _required_fields(outcome, OUTCOME_FIELDS, "MISSING_OUTCOME_FIELD"),
            OUTCOME_FIELDS,
        ),
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args(argv)

    manifest = _mapping(_load_json(arguments.manifest), "INVALID_MANIFEST")
    samples = _array(manifest.get("samples"), "INVALID_MANIFEST_SAMPLES")
    sample_ids = [sample.get("sample_id") for sample in samples]
    if (
        not samples
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ViewerDataError("INVALID_MANIFEST_SAMPLES")
    expected = set(sample_ids)
    actual = {
        path.stem for path in arguments.input_dir.glob("*.json") if path.is_file()
    }
    if actual != expected:
        raise ViewerDataError("INPUT_SAMPLE_SET_INVALID")

    outputs: dict[str, bytes] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        source_path = arguments.input_dir / f"{sample_id}.json"
        raw = source_path.read_bytes()
        source = _mapping(json.loads(raw), "INVALID_LOCAL_RESULT")
        projected = project_local_result(
            sample_id,
            sample["duration_seconds"],
            source,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )
        outputs[sample_id] = _canonical_json(projected)
    for sample_id, payload in outputs.items():
        _atomic_write(arguments.output_dir / f"{sample_id}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
