"""Small JSON result reader/writer for annotations and their audit sidecars."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_record(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
        json.dumps(value, allow_nan=False)  # Reject non-finite numbers, including overflow.
    except (OSError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def audit_path(path: Path) -> Path:
    return path.with_name(path.stem + ".audit.json")


def is_annotation_record(record: dict[str, Any] | None) -> bool:
    """Distinguish public/legacy payloads from execution-only sidecars by structure."""
    return isinstance(record, dict) and bool({"duration", "events", "data"} & record.keys())


def annotation_record(path: Path) -> dict[str, Any] | None:
    """Read public+audit results, plain annotations, and legacy envelopes."""
    data = read_record(path)
    sidecar = audit_path(path)
    if sidecar.exists():
        audit = read_record(sidecar)
        if audit is None:
            raise ValueError(f"malformed annotation audit: {sidecar}")
        if is_annotation_record(audit):
            return data  # A literal *.audit.json annotation is not this file's sidecar.
        record = {**audit, "data": data}
        if audit.get("status") == "completed" and data is None:
            record.update(status="failed", error="annotation data missing")
        return record
    return data


def write_annotation(path: Path, record: dict[str, Any]) -> None:
    """Audit acts as commit marker across the two atomic file replacements."""
    sidecar = audit_path(path)
    if is_annotation_record(read_record(sidecar)):
        raise ValueError(f"Audit path is occupied by an annotation: {sidecar}")
    data = record["data"]
    audit = {key: value for key, value in record.items() if key != "data"}
    try:
        if record["status"] == "completed":
            from .pipelines.event_schemas import GroundedEvents
            GroundedEvents.model_validate(data)
            write_json(sidecar, {**audit, "status": "writing"})
            write_json(path, data)
            write_json(sidecar, audit)
        else:
            write_json(sidecar, audit)
            path.unlink(missing_ok=True)
    except BaseException:
        # A partially persisted run must never expose the preceding success.
        path.unlink(missing_ok=True)
        raise


def result_path(output: Path, video: Path) -> Path:
    """Reuse a video's filename, adding a numeric suffix for name collisions."""
    resolved = str(video.resolve())
    # Find an existing owner before filling filename gaps, including legacy names.
    for sidecar_path in sorted(output.glob("*.audit.json")):
        sidecar = read_record(sidecar_path)
        if sidecar and not is_annotation_record(sidecar) and sidecar.get("video_path") == resolved:
            return sidecar_path.with_name(sidecar_path.name[:-len(".audit.json")] + ".json")
    for candidate in sorted(output.glob("*.json")):
        record = read_record(candidate)
        if record and is_annotation_record(record) and record.get("video_path") == resolved:
            return candidate
    index = 0
    while True:
        name = video.stem if index == 0 else f"{video.stem}--{index}"
        candidate = output / f"{name}.json"
        index += 1
        if candidate.name.endswith(".audit.json"):
            continue
        if not candidate.exists() and not audit_path(candidate).exists():
            return candidate


def completed_result(path: Path, video: Path) -> dict[str, Any] | None:
    """Check completion, source path and annotation structure for result reuse."""
    from .pipelines.event_schemas import GroundedEvents
    try:
        record = annotation_record(path)
        if (not record or record.get("status") != "completed"
                or record.get("video_path") != str(video.resolve())):
            return None
        GroundedEvents.model_validate(record.get("data"))
        return record
    except (ValueError, TypeError):
        return None


def atomic_text(path: Path, text: str) -> None:
    """Replace a complete file in one step; clean up failed temporary writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".percept-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
