#!/usr/bin/env python3
"""Run the two finite, hash-bound semantic re-verification calls.

This is an experiment operator, not an application entry point.  Its public
output is deliberately limited to closed status and validation metadata; raw
prompts, payloads, and responses stay in the owner-only output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any
import zipfile

import las_repro
from las_repro.cv.summary import CvEvidenceSummary, OcclusionCandidate
from las_repro.domain import InferenceJob
from las_repro.models.ark import ArkVideoModel
from las_repro.pipelines.embodied import PromptRenderer, _validated_stage_result
from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS
from las_repro.pipelines.scene_semantics import trusted_target_skeleton
from las_repro.workers import _model_request


BUNDLE_SCHEMA = "semantic_reverification_input_v1"
MODEL_IDENTITY = "doubao-seed-2-1-pro-260628"
MODEL_ALIAS = "doubao-pro"
_STAGE_PAIRS = (
    ("full_0001", "scene_semantics", "SceneSemantics"),
    ("full_0002", "occlusion_semantics", "OcclusionDecisionSet"),
)
_SETTINGS_KEYS = {
    "timeout_seconds",
    "max_frames",
    "max_request_bytes",
    "max_output_chars",
}
_STAGE_KEYS = {
    "sample_id",
    "stage",
    "model_name",
    "payload",
    "original_template",
    "source_video_sha256",
    "original_job_payload_sha256",
}
_HASH = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MARKER = re.compile(r"\{\{(?P<name>[A-Z][A-Z0-9_]*)\}\}")
_SCENE_SUFFIX = "\n\n[CV_EVIDENCE_SUMMARY_JSON]\n"
_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_MAX_KEY_BYTES = 16 * 1024


class OperatorError(ValueError):
    """A closed failure that is safe to surface without its private cause."""

    def __init__(self, code: str, message: str = "semantic re-verification rejected") -> None:
        self.code = code
        super().__init__(f"{message} [{code}]")


def _json_bytes(value: Any, *, sort_keys: bool = True) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=sort_keys,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise OperatorError("JSON_INVALID") from None


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _checked_hash(value: Any, code: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise OperatorError(code)
    return value


def _extract_prompt_data(
    original_prompt: str,
    original_template: str,
    stage: str,
) -> tuple[dict[str, Any], str, Any | None]:
    if stage not in {pair[1] for pair in _STAGE_PAIRS}:
        raise OperatorError("STAGE_INVALID")
    if not isinstance(original_prompt, str) or not isinstance(original_template, str):
        raise OperatorError("PROMPT_INPUT_INVALID")

    core_prompt = original_prompt
    suffix_text = ""
    suffix_value: Any | None = None
    suffix_count = original_prompt.count(_SCENE_SUFFIX)
    if stage == "occlusion_semantics":
        if suffix_count:
            raise OperatorError("PROMPT_SUFFIX_INVALID")
    elif suffix_count:
        if suffix_count != 1:
            raise OperatorError("PROMPT_SUFFIX_INVALID")
        core_prompt, encoded_suffix = original_prompt.rsplit(_SCENE_SUFFIX, 1)
        try:
            suffix_value, end = json.JSONDecoder().raw_decode(encoded_suffix)
        except (ValueError, RecursionError):
            raise OperatorError("PROMPT_SUFFIX_INVALID") from None
        if end != len(encoded_suffix) or _json_bytes(suffix_value, sort_keys=False).decode() != encoded_suffix:
            raise OperatorError("PROMPT_SUFFIX_INVALID")
        suffix_text = _SCENE_SUFFIX + encoded_suffix

    matches = tuple(_MARKER.finditer(original_template))
    marker_names = tuple(match.group("name") for match in matches)
    if not marker_names or len(set(marker_names)) != len(marker_names):
        raise OperatorError("PROMPT_MARKERS_AMBIGUOUS")

    literals: list[str] = []
    prior = 0
    for match in matches:
        literals.append(original_template[prior : match.start()])
        prior = match.end()
    literals.append(original_template[prior:])

    values: dict[str, Any] = {}
    position = 0
    decoder = json.JSONDecoder()
    for index, marker_name in enumerate(marker_names):
        literal = literals[index]
        if not core_prompt.startswith(literal, position):
            raise OperatorError("PROMPT_ROUND_TRIP_FAILED")
        position += len(literal)
        try:
            value, end = decoder.raw_decode(core_prompt, position)
        except (ValueError, RecursionError):
            raise OperatorError("PROMPT_DATA_INVALID") from None
        next_literal = literals[index + 1]
        if not core_prompt.startswith(next_literal, end):
            raise OperatorError("PROMPT_DATA_AMBIGUOUS")
        values[marker_name] = value
        position = end
    position += len(literals[-1])
    if position != len(core_prompt):
        raise OperatorError("PROMPT_UNCONSUMED_TEXT")

    rebuilt_old = _render_exact_template(original_template, values)
    if rebuilt_old != core_prompt:
        raise OperatorError("PROMPT_ROUND_TRIP_FAILED")
    if values.get("VALIDATION_REPAIR_JSON", object()) is not None:
        raise OperatorError("PROMPT_INITIAL_REPAIR_INVALID")
    return values, suffix_text, suffix_value


def _render_exact_template(template: str, values: Mapping[str, Any]) -> str:
    """Render captured JSON without allowing marker-like data to be re-read."""
    rendered = template
    placeholders: dict[str, str] = {}
    encoded = {
        name: _json_bytes(value, sort_keys=False).decode("utf-8")
        for name, value in values.items()
    }
    for index, name in enumerate(sorted(values)):
        literal = "{{" + name + "}}"
        placeholder = f"\x00SEMANTIC_REVERIFY_VALUE_{index}\x00"
        if rendered.count(literal) != 1 or placeholder in rendered or any(
            placeholder in value for value in encoded.values()
        ):
            raise OperatorError("PROMPT_MARKERS_AMBIGUOUS")
        rendered = rendered.replace(literal, placeholder)
        placeholders[name] = placeholder
    if _MARKER.search(rendered) is not None:
        raise OperatorError("PROMPT_UNCONSUMED_MARKER")
    for name in sorted(values):
        rendered = rendered.replace(placeholders[name], encoded[name])
    return rendered


def rebuild_prompt(
    original_prompt: str,
    original_template: str,
    stage: str,
    repair: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Rebuild the current packaged prompt from exact old JSON data.

    The returned digest covers all immutable extracted JSON plus the optional
    scene evidence suffix.  Repair codes are intentionally excluded so the
    digest remains stable across the one permitted repair.
    """

    values, suffix_text, suffix_value = _extract_prompt_data(
        original_prompt, original_template, stage
    )
    if repair is not None:
        if set(repair) != {"issue_codes"}:
            raise OperatorError("REPAIR_INVALID")
        issue_codes = repair.get("issue_codes")
        if (
            not isinstance(issue_codes, list)
            or not issue_codes
            or len(set(issue_codes)) != len(issue_codes)
            or any(not isinstance(code, str) or not code for code in issue_codes)
        ):
            raise OperatorError("REPAIR_INVALID")
    rendered_values = dict(values)
    rendered_values["VALIDATION_REPAIR_JSON"] = (
        None if repair is None else {"issue_codes": list(repair["issue_codes"])}
    )
    try:
        prompt = PromptRenderer().render(stage, rendered_values)
    except Exception:
        raise OperatorError("CURRENT_PROMPT_RENDER_FAILED") from None
    immutable = {
        "stage": stage,
        "variables": {
            key: value
            for key, value in values.items()
            if key != "VALIDATION_REPAIR_JSON"
        },
        "scene_evidence_summary": suffix_value,
    }
    return prompt + suffix_text, _hash_json(immutable)


def _private_name(stage_record: Mapping[str, Any], suffix: str) -> str:
    return f"{stage_record['sample_id']}.{stage_record['stage']}.{suffix}"


def _open_exclusive(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags, 0o600)
    except FileExistsError:
        raise OperatorError("OUTPUT_PATH_EXISTS") from None
    except OSError:
        raise OperatorError("OUTPUT_WRITE_FAILED") from None


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = _open_exclusive(path)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise OperatorError("OUTPUT_WRITE_FAILED") from None
    _fsync_directory(path.parent)


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_exclusive(path, _json_bytes(value) + b"\n")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise OperatorError("OUTPUT_SYNC_FAILED") from None


def _ensure_private_output_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise OperatorError("OUTPUT_DIRECTORY_MISSING") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise OperatorError("OUTPUT_DIRECTORY_INVALID")
    if info.st_mode & 0o077:
        raise OperatorError("OUTPUT_DIRECTORY_NOT_PRIVATE")


def _planned_paths(stage_record: Mapping[str, Any], output_dir: Path) -> tuple[Path, ...]:
    suffixes = [
        "reservation.json",
        "call-1.prompt.private.txt",
        "call-1.payload.private.json",
        "call-1.raw.private.json",
        "call-2.prompt.private.txt",
        "call-2.payload.private.json",
        "call-2.raw.private.json",
        "validated.private.json",
        "report.json",
    ]
    return tuple(output_dir / _private_name(stage_record, suffix) for suffix in suffixes)


def _safe_usage(model: Any) -> dict[str, int]:
    getter = getattr(model, "request_metrics", None)
    if not callable(getter):
        return {}
    try:
        value = getter()
    except Exception:
        return {}
    if not isinstance(value, Mapping):
        return {}
    usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            usage[key] = item
    return usage


def _release_request(model: Any, request: Any) -> None:
    release = getattr(model, "release_request", None)
    if callable(release):
        try:
            release(request)
        except Exception:
            pass


def _result_counts(stage: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if stage == "scene_semantics":
        events = value.get("semantic_events")
        return {"semantic_event_count": len(events) if isinstance(events, list) else 0}
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        return {"decision_count": 0, "classification_counts": {}}
    classifications = Counter(
        item.get("classification")
        for item in decisions
        if isinstance(item, Mapping) and isinstance(item.get("classification"), str)
    )
    return {
        "decision_count": len(decisions),
        "classification_counts": dict(sorted(classifications.items())),
    }


def _write_report(
    stage_record: Mapping[str, Any], output_dir: Path, report: dict[str, Any]
) -> dict[str, Any]:
    _write_json_exclusive(
        output_dir / _private_name(stage_record, "report.json"), report
    )
    return report


def run_stage(
    stage_record: Mapping[str, Any],
    *,
    output_dir: Path,
    model: Any,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one already-authenticated stage with one conditional repair."""

    _ensure_private_output_directory(output_dir)
    if not isinstance(stage_record, Mapping):
        raise OperatorError("STAGE_INVALID")
    try:
        payload = stage_record["payload"]
        original_prompt = payload["prompt"]
        original_template = stage_record["original_template"]
        stage = stage_record["stage"]
        schema_name = payload["schema_name"]
        schema_context = payload["schema_context"]
    except (KeyError, TypeError):
        raise OperatorError("STAGE_INVALID") from None
    if not isinstance(payload, Mapping) or not isinstance(schema_context, Mapping):
        raise OperatorError("STAGE_INVALID")

    prompt, stable_data_digest = rebuild_prompt(
        original_prompt, original_template, stage
    )
    paths = _planned_paths(stage_record, output_dir)
    reservation_path = paths[0]
    if reservation_path.exists() or reservation_path.is_symlink():
        raise OperatorError("RESERVATION_EXISTS", "stage is already reserved")
    if any(path.exists() or path.is_symlink() for path in paths[1:]):
        raise OperatorError("OUTPUT_PATH_EXISTS")
    reservation = {
        "schema": "semantic_reverification_reservation_v1",
        "sample_id": stage_record["sample_id"],
        "stage": stage,
        "stable_data_sha256": stable_data_digest,
        "original_job_payload_sha256": stage_record.get("original_job_payload_sha256"),
    }
    _write_json_exclusive(reservation_path, reservation)

    report: dict[str, Any] = {
        "schema": "semantic_reverification_stage_report_v1",
        "sample_id": stage_record["sample_id"],
        "stage": stage,
        "status": "error",
        "call_count": 0,
        "stable_data_sha256": stable_data_digest,
        "attempts": [],
    }
    repair_codes: tuple[str, ...] | None = None
    for call_index in (1, 2):
        call_payload = dict(payload)
        call_payload["prompt"] = prompt
        job = InferenceJob(
            job_id=f"bounded-{stage}-{call_index}",
            task_id=str(stage_record["sample_id"]),
            stage=stage,
            ordinal=call_index - 1,
            payload=call_payload,
            model_name=str(stage_record["model_name"]),
        )
        try:
            request = _model_request(job)
        except Exception:
            report["error_code"] = "MODEL_REQUEST_INVALID"
            return _write_report(stage_record, output_dir, report)

        prompt_path = output_dir / _private_name(
            stage_record, f"call-{call_index}.prompt.private.txt"
        )
        payload_path = output_dir / _private_name(
            stage_record, f"call-{call_index}.payload.private.json"
        )
        _write_exclusive(prompt_path, prompt.encode("utf-8"))
        _write_json_exclusive(payload_path, call_payload)
        attempt: dict[str, Any] = {
            "ordinal": call_index - 1,
            "request_sha256": _hash_json(
                {
                    "stage": stage,
                    "model_name": stage_record["model_name"],
                    "payload": call_payload,
                }
            ),
        }
        report["call_count"] = call_index
        started = monotonic()
        try:
            raw = model.generate(request)
            elapsed = monotonic() - started
        except Exception:
            elapsed = monotonic() - started
            attempt["elapsed_seconds"] = max(0.0, float(elapsed))
            attempt["usage"] = _safe_usage(model)
            attempt["status"] = "error"
            attempt["error_code"] = "MODEL_GENERATION_FAILED"
            report["attempts"].append(attempt)
            report["error_code"] = "MODEL_GENERATION_FAILED"
            _release_request(model, request)
            return _write_report(stage_record, output_dir, report)

        attempt["elapsed_seconds"] = max(0.0, float(elapsed))
        attempt["usage"] = _safe_usage(model)
        _release_request(model, request)
        raw_path = output_dir / _private_name(
            stage_record, f"call-{call_index}.raw.private.json"
        )
        _write_json_exclusive(raw_path, raw)
        attempt["response_sha256"] = _hash_json(raw)
        if not isinstance(raw, Mapping):
            attempt["status"] = "error"
            attempt["error_code"] = "MODEL_OUTPUT_INVALID"
            report["attempts"].append(attempt)
            report["error_code"] = "MODEL_OUTPUT_INVALID"
            return _write_report(stage_record, output_dir, report)
        try:
            sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
                schema_name, raw, schema_context
            )
            validated, issue_codes, _ = _validated_stage_result(
                schema_name, sanitized, schema_context
            )
        except Exception:
            attempt["status"] = "error"
            attempt["error_code"] = "VALIDATION_CONTEXT_INVALID"
            report["attempts"].append(attempt)
            report["error_code"] = "VALIDATION_CONTEXT_INVALID"
            return _write_report(stage_record, output_dir, report)
        if issue_codes is None:
            attempt["status"] = "valid"
            attempt["issue_codes"] = []
            report["attempts"].append(attempt)
            report["status"] = "valid"
            report.update(_result_counts(stage, validated))
            _write_json_exclusive(
                output_dir / _private_name(stage_record, "validated.private.json"),
                validated,
            )
            return _write_report(stage_record, output_dir, report)

        attempt["status"] = "invalid"
        attempt["issue_codes"] = list(issue_codes)
        report["attempts"].append(attempt)
        if call_index == 2:
            report["status"] = "invalid"
            report["final_issue_codes"] = list(issue_codes)
            return _write_report(stage_record, output_dir, report)
        repair_codes = issue_codes
        report["repair_issue_codes"] = list(repair_codes)
        prompt, repaired_digest = rebuild_prompt(
            original_prompt,
            original_template,
            stage,
            repair={"issue_codes": list(repair_codes)},
        )
        if repaired_digest != stable_data_digest:
            raise OperatorError("REPAIR_DATA_CHANGED")

    raise AssertionError("bounded semantic repair loop did not terminate")


def _read_regular(path: Path, maximum: int, code: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise OperatorError(code)
            data = b""
            while len(data) <= maximum:
                part = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(data)))
                if not part:
                    break
                data += part
            if len(data) > maximum:
                raise OperatorError(code)
            return data
        finally:
            os.close(descriptor)
    except OperatorError:
        raise
    except OSError:
        raise OperatorError(code) from None


def _strict_json(data: bytes) -> Any:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise OperatorError("BUNDLE_JSON_INVALID")
            result[key] = value
        return result

    try:
        value = json.loads(data, object_pairs_hook=pairs)
    except OperatorError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise OperatorError("BUNDLE_JSON_INVALID") from None
    return value


def _verify_path_hash(path: Path, expected: str, mismatch_code: str, maximum: int) -> bytes:
    data = _read_regular(path, maximum, mismatch_code)
    if hashlib.sha256(data).hexdigest() != _checked_hash(expected, mismatch_code):
        raise OperatorError(mismatch_code)
    return data


def _verify_large_file_hash(path: Path, expected: str, mismatch_code: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OperatorError(mismatch_code)
            digest = hashlib.sha256()
            while True:
                part = os.read(descriptor, 1024 * 1024)
                if not part:
                    break
                digest.update(part)
        finally:
            os.close(descriptor)
    except OperatorError:
        raise
    except OSError:
        raise OperatorError(mismatch_code) from None
    if digest.hexdigest() != _checked_hash(expected, mismatch_code):
        raise OperatorError(mismatch_code)


def _verify_wheel(path: Path, expected_hash: str) -> None:
    _verify_path_hash(path, expected_hash, "WHEEL_HASH_MISMATCH", 512 * 1024 * 1024)
    package_root = Path(las_repro.__file__).resolve().parent
    imported = {
        "las_repro/" + source.relative_to(package_root).as_posix(): source.read_bytes()
        for source in package_root.rglob("*")
        if source.is_file()
        and "__pycache__" not in source.parts
        and source.suffix != ".pyc"
    }
    try:
        with zipfile.ZipFile(path) as archive:
            package_infos = [
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename.startswith("las_repro/")
            ]
            names = [info.filename for info in package_infos]
            if len(names) != len(set(names)) or set(names) != set(imported):
                raise OperatorError("WHEEL_PACKAGE_MISMATCH")
            for info in package_infos:
                if archive.read(info) != imported[info.filename]:
                    raise OperatorError("WHEEL_PACKAGE_MISMATCH")
    except OperatorError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        raise OperatorError("WHEEL_PACKAGE_MISMATCH") from None


def _finite_positive(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorError(code)
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise OperatorError(code)
    return result


def _validate_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SETTINGS_KEYS:
        raise OperatorError("SETTINGS_INVALID")
    timeout = _finite_positive(value["timeout_seconds"], "SETTINGS_INVALID")
    result: dict[str, Any] = {"timeout_seconds": timeout}
    for name in ("max_frames", "max_request_bytes", "max_output_chars"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise OperatorError("SETTINGS_INVALID")
        result[name] = item
    return result


def _validate_scene_alignment(
    values: Mapping[str, Any], suffix: Any | None, context: Mapping[str, Any]
) -> None:
    required = {
        "duration",
        "require_observed_content",
        "required_object_ids",
        "evidence_summary",
        "segments",
    }
    if set(context) != required:
        raise OperatorError("SCENE_CONTEXT_INVALID")
    summary_value = context["evidence_summary"]
    if summary_value is None:
        expected_suffix = None
    else:
        try:
            expected_suffix = CvEvidenceSummary.model_validate(summary_value).prompt_record()
        except Exception:
            raise OperatorError("SCENE_CONTEXT_INVALID") from None
    if suffix != expected_suffix:
        raise OperatorError("SCENE_SUFFIX_MISMATCH")
    segments = context["segments"]
    try:
        targets = trusted_target_skeleton(segments)
    except Exception:
        raise OperatorError("SCENE_CONTEXT_INVALID") from None
    if (
        values.get("SEGMENTS_JSON") != segments
        or values.get("KNOWN_TARGETS_JSON") != targets
        or values.get("VIDEO_DURATION_SECONDS_JSON") != context["duration"]
        or values.get("CV_EVIDENCE_AVAILABILITY_JSON")
        != {"available": summary_value is not None}
    ):
        raise OperatorError("SCENE_PROMPT_CONTEXT_MISMATCH")


def _validate_occlusion_alignment(
    values: Mapping[str, Any], suffix: Any | None, context: Mapping[str, Any]
) -> None:
    if suffix is not None or set(context) != {"duration", "candidates"}:
        raise OperatorError("OCCLUSION_CONTEXT_INVALID")
    raw_candidates = context["candidates"]
    if not isinstance(raw_candidates, list):
        raise OperatorError("OCCLUSION_CONTEXT_INVALID")
    try:
        candidates = [
            OcclusionCandidate.model_validate(value).prompt_record()
            for value in raw_candidates
        ]
    except Exception:
        raise OperatorError("OCCLUSION_CONTEXT_INVALID") from None
    if (
        values.get("OCCLUSION_CANDIDATES_JSON") != candidates
        or values.get("VIDEO_DURATION_SECONDS_JSON") != context["duration"]
    ):
        raise OperatorError("OCCLUSION_PROMPT_CONTEXT_MISMATCH")


def _validate_bundle(bundle: Any) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema", "model_identity", "settings", "stages"
    }:
        raise OperatorError("BUNDLE_SHAPE_INVALID")
    if bundle["schema"] != BUNDLE_SCHEMA:
        raise OperatorError("BUNDLE_SCHEMA_INVALID")
    if bundle["model_identity"] != MODEL_IDENTITY:
        raise OperatorError("MODEL_IDENTITY_INVALID")
    settings = _validate_settings(bundle["settings"])
    stages = bundle["stages"]
    if not isinstance(stages, list) or len(stages) != len(_STAGE_PAIRS):
        raise OperatorError("STAGE_PAIR_INVALID")
    validated: list[dict[str, Any]] = []
    for raw, expected in zip(stages, _STAGE_PAIRS, strict=True):
        sample_id, stage_name, schema_name = expected
        if not isinstance(raw, dict) or set(raw) != _STAGE_KEYS:
            raise OperatorError("STAGE_SHAPE_INVALID")
        if (
            raw["sample_id"] != sample_id
            or raw["stage"] != stage_name
            or raw["model_name"] != MODEL_ALIAS
        ):
            raise OperatorError("STAGE_PAIR_INVALID")
        payload = raw["payload"]
        if not isinstance(payload, dict):
            raise OperatorError("STAGE_PAYLOAD_INVALID")
        if _hash_json(payload) != _checked_hash(
            raw["original_job_payload_sha256"], "JOB_PAYLOAD_HASH_INVALID"
        ):
            raise OperatorError("JOB_PAYLOAD_HASH_MISMATCH")
        if payload.get("schema_name") != schema_name:
            raise OperatorError("STAGE_SCHEMA_INVALID")
        context = payload.get("schema_context")
        prompt = payload.get("prompt")
        template = raw["original_template"]
        if not isinstance(context, dict) or not isinstance(prompt, str) or not isinstance(template, str):
            raise OperatorError("STAGE_PAYLOAD_INVALID")
        video_value = payload.get("video_path")
        if not isinstance(video_value, str) or not Path(video_value).is_absolute():
            raise OperatorError("VIDEO_PATH_INVALID")
        _verify_large_file_hash(
            Path(video_value),
            raw["source_video_sha256"],
            "VIDEO_HASH_MISMATCH",
        )
        values, _, suffix = _extract_prompt_data(prompt, template, stage_name)
        rebuild_prompt(prompt, template, stage_name)
        if stage_name == "scene_semantics":
            _validate_scene_alignment(values, suffix, context)
        else:
            _validate_occlusion_alignment(values, suffix, context)
        validated.append(raw)
    return settings, tuple(validated)


def _prepare_output(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        try:
            os.mkdir(path, 0o700)
            os.chmod(path, 0o700)
            _fsync_directory(path.parent)
            return
        except OSError:
            raise OperatorError("OUTPUT_CREATE_FAILED") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise OperatorError("OUTPUT_DIRECTORY_INVALID")
    if any(path.glob("*.reservation.json")):
        raise OperatorError("RESERVATION_EXISTS", "stage is already reserved")
    raise OperatorError("OUTPUT_PATH_EXISTS")


def _read_api_key(path: Path) -> str:
    data = _read_regular(path, _MAX_KEY_BYTES, "CREDENTIALS_INVALID")
    try:
        value = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise OperatorError("CREDENTIALS_INVALID") from None
    if not value:
        raise OperatorError("CREDENTIALS_INVALID")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    model_factory: Callable[..., Any] = ArkVideoModel,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if not isinstance(args.source_commit, str) or _COMMIT.fullmatch(args.source_commit) is None:
            raise OperatorError("SOURCE_COMMIT_INVALID")
        bundle_data = _verify_path_hash(
            args.bundle, args.bundle_sha256, "BUNDLE_HASH_MISMATCH", _MAX_BUNDLE_BYTES
        )
        _verify_wheel(args.wheel, args.wheel_sha256)
        settings, stages = _validate_bundle(_strict_json(bundle_data))
        if not args.execute:
            print(_json_bytes({"status": "preflight_valid", "stage_count": 2}).decode())
            return 0

        _prepare_output(args.output_dir)
        api_key = _read_api_key(args.api_key_file)
        try:
            model = model_factory(
                api_key=api_key,
                model_registry={MODEL_ALIAS: MODEL_IDENTITY},
                **settings,
            )
        except Exception:
            raise OperatorError("MODEL_CONSTRUCTION_FAILED") from None
        reports: list[dict[str, Any]] = []
        try:
            for stage_record in stages:
                reports.append(
                    run_stage(stage_record, output_dir=args.output_dir, model=model)
                )
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        summary = {
            "schema": "semantic_reverification_report_v1",
            "status": "execution_complete",
            "source_commit": args.source_commit,
            "bundle_sha256": args.bundle_sha256,
            "wheel_sha256": args.wheel_sha256,
            "stages": reports,
        }
        _write_json_exclusive(args.output_dir / "report.json", summary)
        print(
            _json_bytes(
                {
                    "status": "execution_complete",
                    "stage_statuses": [report["status"] for report in reports],
                    "call_count": sum(report["call_count"] for report in reports),
                }
            ).decode()
        )
        return 0
    except OperatorError as error:
        print(f"semantic re-verification failed [{error.code}]", file=sys.stderr)
        return 2
    except Exception:
        print("semantic re-verification failed [UNEXPECTED_FAILURE]", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
