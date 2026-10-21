"""WGO-inspired segment, semantic, and joint event scoring on original video seconds."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import httpx

from percept_harness.pipelines.event_schemas import GroundedEvents
from percept_harness.models.base import parse_strict_json
from percept_harness import results
from .event_matching import check_threshold, match_weight_matrix, temporal_iou


class JudgeError(ValueError):
    """Semantic decision could not be trusted."""


def validate_decision(value):
    if not isinstance(value, dict) or set(value) != {"match", "reason"} or type(value["match"]) is not bool or not isinstance(value["reason"], str) or not value["reason"].strip():
        raise JudgeError("judge must return strict match bool and nonempty reason")
    return value


JUDGE_PROMPTS = {
    # Symmetric: both sides describe what a video shows.
    "observation": "Decide whether descriptions report the same observed action, object, destination, direction, and outcome. Ignore wording and timestamps. Return only JSON with strict boolean match and brief reason. Do not include chain of thought.",
    # Asymmetric: the reference is the instruction a generator received for an interval.
    "instruction": (
        "The reference is the instruction a video generation model received for one time interval. "
        "The prediction describes what is visible in the generated video during an overlapping interval. "
        "Decide whether the prediction shows the instructed action or change actually happening. "
        "Extra visible details in the prediction are fine and do not cause a mismatch. "
        "Require only what the instruction specifies: the subject or camera motion and its direction, "
        "the named objects, and the outcome. If the instruction is a camera motion, the prediction must show "
        "that camera motion or an equivalent viewpoint change; camera following or framing phrases alone are not required. "
        "Ignore wording, timestamps and visual quality. Return only JSON with strict boolean match and brief reason. "
        "Do not include chain of thought."),
}
# The judge owns these request fields; extra_body may add provider options only.
_RESERVED_JUDGE_KEYS = frozenset({"model", "temperature", "response_format", "messages", "stream"})


class OpenAIJudge:
    def __init__(self, base_url: str, model: str, api_key: str, *, mode: str = "observation", extra_body=None,
                 transport=None, read_timeout_seconds: float = 180.0):
        if not base_url or not model or not api_key:
            raise ValueError("judge base URL, model and API key are required")
        if mode not in JUDGE_PROMPTS:
            raise ValueError(f"unknown judge mode {mode!r}; choose from {sorted(JUDGE_PROMPTS)}")
        extra_body = {} if extra_body is None else extra_body
        if not isinstance(extra_body, dict) or _RESERVED_JUDGE_KEYS & extra_body.keys():
            raise ValueError("judge extra body must be a JSON object without " + ", ".join(sorted(_RESERVED_JUDGE_KEYS)))
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.mode, self.extra_body = mode, dict(extra_body)
        if isinstance(read_timeout_seconds, bool) or not isinstance(read_timeout_seconds, (int, float)) or not math.isfinite(read_timeout_seconds) or read_timeout_seconds <= 0:
            raise ValueError("judge read timeout must be finite and positive")
        self.transport = transport
        self.timeout = httpx.Timeout(connect=30.0, read=float(read_timeout_seconds), write=60.0, pool=30.0)

    def __call__(self, reference: str, prediction: str) -> dict:
        payload = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [
            {"role": "system", "content": JUDGE_PROMPTS[self.mode]},
            {"role": "user", "content": json.dumps({"reference": reference, "prediction": prediction}, ensure_ascii=False)},
        ], **self.extra_body}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload, headers={"Authorization": f"Bearer {self.api_key}"})
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return validate_decision(parse_strict_json(content))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise JudgeError(f"judge request or response failed: {type(error).__name__}" + (f" HTTP {error.response.status_code}" if isinstance(error, httpx.HTTPStatusError) else "")) from error


def _prf(matches, reference_count, prediction_count):
    if reference_count == prediction_count == 0:
        return {"precision": None, "recall": None, "f1": None}
    precision = matches / prediction_count if prediction_count else None
    recall = matches / reference_count if reference_count else None
    f1 = 2 * matches / (reference_count + prediction_count)
    return {"precision": precision, "recall": recall, "f1": f1}


def _check_tiou(value):
    try:
        check_threshold(value)
    except ValueError:
        raise ValueError("tIoU threshold must be finite in (0,1]") from None


def score_pair(reference: GroundedEvents, prediction: GroundedEvents, judge: Callable, tiou_threshold: float = .5) -> dict:
    _check_tiou(tiou_threshold)
    refs, preds = reference.events, prediction.events
    matrix = [[temporal_iou(a, b) for b in preds] for a in refs]
    temporal = match_weight_matrix(matrix, tiou_threshold)
    joint_matrix = [[0.0] * len(preds) for _ in refs]
    candidates = []
    for i, row in enumerate(matrix):
        for j, iou in enumerate(row):
            if iou >= tiou_threshold:
                try:
                    decision = validate_decision(judge(refs[i].description, preds[j].description))
                except Exception as error:
                    raise JudgeError(f"candidate ({i},{j}) failed: {error}") from error
                if decision["match"]:
                    joint_matrix[i][j] = iou
                candidates.append({"reference_index": i, "prediction_index": j, "iou": iou, "reference_description": refs[i].description, "prediction_description": preds[j].description, **decision})
    joint = match_weight_matrix(joint_matrix, tiou_threshold)
    temporal_passes = sum(joint_matrix[m.reference_index][m.prediction_index] > 0 for m in temporal)
    segment = _prf(len(temporal), len(refs), len(preds))
    both = _prf(len(joint), len(refs), len(preds))
    temporal_ref = {m.reference_index for m in temporal}
    temporal_pred = {m.prediction_index for m in temporal}
    joint_ref = {m.reference_index for m in joint}
    joint_pred = {m.prediction_index for m in joint}
    return {
        "reference_events": len(refs), "prediction_events": len(preds),
        "temporal_match_count": len(temporal),
        "semantic_passes_on_temporal_matches": temporal_passes,
        "joint_match_count": len(joint),
        "segment_precision": segment["precision"], "segment_recall": segment["recall"],
        "segment_f1": segment["f1"],
        "semantic_match_rate": temporal_passes / len(temporal) if temporal else None,
        "joint_precision": both["precision"], "joint_recall": both["recall"],
        "joint_f1": both["f1"],
        "candidate_judgments": candidates,
        "temporal_matches": [vars(m) for m in temporal],
        "joint_matches": [vars(m) for m in joint],
        "unmatched_temporal_reference_indices": [i for i in range(len(refs)) if i not in temporal_ref],
        "unmatched_temporal_prediction_indices": [j for j in range(len(preds)) if j not in temporal_pred],
        "unmatched_joint_reference_indices": [i for i in range(len(refs)) if i not in joint_ref],
        "unmatched_joint_prediction_indices": [j for j in range(len(preds)) if j not in joint_pred],
        "reference_events_detail": [e.model_dump(mode="json") for e in refs],
        "prediction_events_detail": [e.model_dump(mode="json") for e in preds],
    }



def aggregate(scores: list[dict]) -> dict:
    r = sum(s["reference_events"] for s in scores)
    p = sum(s["prediction_events"] for s in scores)
    t = sum(len(s["temporal_matches"]) for s in scores)
    j = sum(len(s["joint_matches"]) for s in scores)
    sem = sum(s["semantic_passes_on_temporal_matches"] for s in scores)
    def macro(keys):
        return {key: {"value": (sum(v) / len(v) if v else None), "contributing_count": len(v)} for key in keys for v in [[s[key] for s in scores if s[key] is not None]]}
    return {"totals": {"samples": len(scores), "reference_events": r, "prediction_events": p, "temporal_matches": t, "semantic_passes_on_temporal_matches": sem, "joint_matches": j}, "segment": {"micro": _prf(t, r, p), "macro": {name: macro(("segment_" + name,))["segment_" + name] for name in ("precision", "recall", "f1")}}, "semantic_match_rate": {"micro": sem / t if t else None, "macro": macro(("semantic_match_rate",))}, "joint": {"micro": _prf(j, r, p), "macro": {name: macro(("joint_" + name,))["joint_" + name] for name in ("precision", "recall", "f1")}}}


def _identity(path: Path, root: Path, record: dict) -> str:
    explicit = record["sample_id"] if "sample_id" in record else record.get("job_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("invalid explicit sample/job ID")
        return explicit
    relative = path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parent.as_posix()
    video = record.get("video_path")
    if not isinstance(video, str) or not video:
        return path.stem
    return Path(video).stem


def _ignored(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return any(part in {".percept", ".cv-archive"} for part in relative.parts) or (len(relative.parts) == 1 and bool(re.fullmatch(r"shard-\d+\.status\.json", path.name)))


def load_group(root: Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"annotation directory missing: {root}")
    items = {}
    candidates = set()
    for path in root.rglob("*.json"):
        if _ignored(root, path):
            continue
        if path.name.endswith(".audit.json") and not results.is_annotation_record(results.read_record(path)):
            path = path.with_name(path.name[:-len(".audit.json")] + ".json")
            # The primary of shard-N.status.audit.json is a skipped status file.
            if _ignored(root, path):
                continue
        candidates.add(path)
    for path in sorted(candidates):
        try:
            record = results.annotation_record(path)
            if not isinstance(record, dict):
                raise ValueError("result must be JSON object")
            sample_id = _identity(path, root, record)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"malformed primary annotation {path}: {error}") from error
        if sample_id in items:
            raise ValueError(f"duplicate or ambiguous sample ID {sample_id!r}: {items[sample_id]['path']} and {path}")
        items[sample_id] = {"path": str(path), "record": record}
    return items


def _scoring_record(record: dict) -> dict:
    if set(record) == {"duration", "events"}:
        return {"status": "completed", "data": record}
    return record


def evaluate(reference_dir: Path, systems: dict[str, Path], judge: Callable, tiou_threshold: float = .5) -> dict:
    _check_tiou(tiou_threshold)
    refs = load_group(reference_dir)
    report = {"protocol": {"name": "WGO-inspired grounded event adaptation", "version": 1, "tiou_threshold": tiou_threshold, "time_basis": "original video seconds; no duration normalization", "matching": "cardinality-first then IoU; independent temporal and joint assignments"}, "reference": str(reference_dir), "systems": {}}
    for name, directory in systems.items():
        preds = load_group(directory)
        rows, valid = {}, []
        for sample_id in sorted(refs.keys() | preds.keys()):
            if sample_id not in refs or sample_id not in preds:
                rows[sample_id] = {"status": "error", "error": "missing reference" if sample_id not in refs else "missing prediction"}
                continue
            try:
                a, b = refs[sample_id]["record"], preds[sample_id]["record"]
                a, b = [_scoring_record(record) for record in (a, b)]
                if a.get("status") != "completed" or b.get("status") != "completed":
                    raise ValueError("annotation status is not completed: " + str(a.get("error") or b.get("error") or "incomplete run"))
                if a.get("data") is None or b.get("data") is None:
                    raise ValueError("annotation data missing")
                score = score_pair(GroundedEvents.model_validate(a["data"]), GroundedEvents.model_validate(b["data"]), judge, tiou_threshold)
                rows[sample_id] = {"status": "valid", "reference_path": refs[sample_id]["path"], "prediction_path": preds[sample_id]["path"], **score}
                valid.append(score)
            except Exception as error:
                rows[sample_id] = {"status": "error", "error": f"{type(error).__name__}: {error}", "reference_path": refs[sample_id]["path"], "prediction_path": preds[sample_id]["path"]}
        invalid = len(rows) - len(valid)
        report["systems"][name] = {"samples": rows, "valid_count": len(valid), "invalid_count": invalid, "aggregate": aggregate(valid) if rows and not invalid else None}
    if not refs and not any(system["samples"] for system in report["systems"].values()):
        raise ValueError("completely empty dataset")
    return report
