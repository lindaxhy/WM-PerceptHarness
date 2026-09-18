"""Strict offline LAS alignment with cardinality-first temporal assignment.

Input JSON may retain historical whitespace. Canonical report encoding is
deterministic; duplicate keys and nonfinite numbers are never accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

FROZEN_MANIFEST_SHA256 = (
    "0ff6ee6b0df3d7aa9afe1ae74b4514be5de7fc8a378d63afb087eb0b6b5ae3bf"
)
FROZEN_QWEN = {
    "full_0001": "78d0681c013fe12b632b641d0f4e33ffd4f67e29a97e43a47b893ed9732d666b",
    "full_0002": "55422790b4b66d7409aeec052a2b5bbc71f0babd07ded2efa4b2b0641101a13f",
    "full_0004": "8f21a5e194a4781c7f55787bdf7735a091c5d9205a5aa0ded010295309bf8eb1",
    "full_0021": "d36fed02d8f904af390b95e52a2e894dc28ea63c91d7e3e875072dd8d82a7886",
    "full_0024": "18ce8fded6d1b13a06063826aa5b55e6f086056df61d276ccd01d47c3f7b4f1e",
}
OCCLUSION_TYPES = {"occlusion_enter", "occluded", "occlusion_exit"}


def canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def strict_json(raw: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON number")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    canonical_json(value)  # also rejects overflow such as 1e999
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("invalid schema keys")


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("invalid finite number")
    return value


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("invalid text")
    if re.search(r"(?i)\.(?:npz|npy|mask)\b", value) or (
        "mask" in value.casefold() and any(c in value for c in "[]{}")
    ):
        raise ValueError("prohibited raw evidence content")
    return value


def load_mapping(path) -> dict:
    value = strict_json(Path(path).read_bytes())
    _exact(
        value,
        {
            "schema_version",
            "policy",
            "action",
            "actor",
            "target",
            "state",
            "result",
            "unknown_labels",
        },
    )
    if value["schema_version"] != "las_alignment_mapping_v1":
        raise ValueError("unsupported mapping")
    for name in ("action", "actor", "target", "state", "result"):
        if type(value[name]) is not dict:
            raise ValueError("invalid mapping aliases")
        for key, label in value[name].items():
            _text(key)
            _text(label)
    if type(value["unknown_labels"]) is not list or any(
        type(x) is not str for x in value["unknown_labels"]
    ):
        raise ValueError("invalid unknown labels")
    if (
        digest(value)
        != "210a753bf46a769bb548e1763ae376f0f68c52cc646e5fe0bb69b8cca9d8fda4"
    ):
        raise ValueError("frozen mapping digest mismatch")
    return value


def _label(value, name, mapping):
    if value is None:
        return None
    normalized = " ".join(value.casefold().replace("_", " ").split())
    if normalized in mapping["unknown_labels"]:
        return None
    return mapping[name].get(normalized, normalized)


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    event_type: str
    actor: str | None = None
    target: str | None = None
    state: str | None = None
    result: str | None = None
    event_id: str = ""
    target_entity_id: str | None = None
    occluder_entity_id: str | None = None

    def __post_init__(self):
        if not 0 <= _number(self.start) < _number(self.end):
            raise ValueError("invalid positive event interval")
        _text(self.event_type)
        for label in (
            self.actor,
            self.target,
            self.state,
            self.result,
            self.target_entity_id,
            self.occluder_entity_id,
        ):
            if label is not None:
                _text(label)


@dataclass(frozen=True)
class Match:
    reference_index: int
    prediction_index: int
    iou: float


def temporal_iou(a: Event, b: Event) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    return intersection / ((a.end - a.start) + (b.end - b.start) - intersection)


def match_events(references, predictions, temporal_iou_threshold=0.3) -> list[Match]:
    """Rectangular Hungarian minimization, with one dummy per reference.

    Eligible weight is 1,000,000 + IoU. Ordered scans choose deterministic
    ties. Bound sample size so IoU sums cannot dominate cardinality.
    """
    if not 0 < _number(temporal_iou_threshold) <= 1:
        raise ValueError("invalid temporal IoU threshold")
    n, real = len(references), len(predictions)
    if max(n, real) >= 1_000_000:
        raise ValueError("too many events")
    if not n or not real:
        return []
    scores = [
        [
            temporal_iou(r, p)
            if r.event_type == p.event_type and r.event_type != "unknown"
            else -1
            for p in predictions
        ]
        for r in references
    ]
    weights = [
        [1_000_000 + iou if iou >= temporal_iou_threshold else 0 for iou in row]
        + [0] * n
        for row in scores
    ]
    m = real + n
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = -weights[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if not j0:
                break
    return sorted(
        (
            Match(p[j] - 1, j - 1, scores[p[j] - 1][j - 1])
            for j in range(1, real + 1)
            if p[j] and weights[p[j] - 1][j - 1] > 0
        ),
        key=lambda x: x.reference_index,
    )


def ratio(numerator, denominator, reason):
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "reason": None if denominator else reason,
    }


@dataclass(frozen=True)
class Annotation:
    sample_id: str
    actions: tuple[Event, ...] = ()
    occlusions: tuple[Event, ...] = ()
    intervals: tuple[Event, ...] | None = None
    component_statuses: tuple[str, ...] = ("disabled", "disabled", "disabled")
    repair_count: int | None = 0
    completed: bool = True
    provenance: dict = field(default_factory=dict)
    outcome: str | None = None


@dataclass
class SampleMetrics:
    sample_id: str
    action: dict
    occlusion: dict
    occlusion_iou: dict
    boundary_errors: dict
    fields: dict
    factual_precision: dict
    rates: dict
    provenance: dict
    occlusion_intervals: dict
    field_temporal_assignment: dict

    def to_dict(self):
        return asdict(self)


@dataclass
class AggregateMetrics:
    sample_count: int
    action: dict
    occlusion: dict
    fields: dict
    factual_precision: dict
    rates: dict
    occlusion_iou: dict
    boundary_errors: dict
    occlusion_intervals: dict

    def to_dict(self):
        return asdict(self)


def _event_metrics(refs, preds, threshold):
    matches = match_events(refs, preds, threshold)
    n = len(matches)
    return {
        "precision": ratio(n, len(preds), "no prediction events"),
        "recall": ratio(n, len(refs), "no reference events"),
        "f1": ratio(2 * n, len(refs) + len(preds), "no events"),
        "reference_count": len(refs),
        "prediction_count": len(preds),
        "matched_count": n,
        "unmatched_reference_ids": [
            e.event_id
            for i, e in enumerate(refs)
            if i not in {m.reference_index for m in matches}
        ],
        "unmatched_prediction_ids": [
            e.event_id
            for i, e in enumerate(preds)
            if i not in {m.prediction_index for m in matches}
        ],
    }


def _field_scores(pairs):
    labels = sorted({v for pair in pairs for v in pair})
    counts = {
        label: {
            "tp": sum(a == b == label for a, b in pairs),
            "fp": sum(a != label and b == label for a, b in pairs),
            "fn": sum(a == label and b != label for a, b in pairs),
        }
        for label in labels
    }
    return _field_counts(counts, len(pairs))


def _field_counts(counts, comparable):
    per_label = {
        label: ratio(
            2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"], "no comparable labels"
        )
        for label, c in counts.items()
    }
    return {
        "macro_f1": ratio(
            sum(s["value"] for s in per_label.values()),
            len(per_label),
            "no comparable labels",
        ),
        "per_label": per_label,
        "counts": {label: dict(c) for label, c in counts.items()},
        "comparable_pairs": comparable,
    }


# Occlusion boundary tolerances are deliberately asymmetric. The detector
# keeps a track alive while any sliver of the target stays visible, so a
# predicted enter is systematically later than the human-judged enter; the
# exit (full reappearance) is sharp on both sides and stays tight.
BOUNDARY_TOLERANCE_SECONDS = {"enter": 2.5, "exit": 0.5}


def _errors(values, tolerance_seconds=None):
    stats = {
        "mean": ratio(sum(values), len(values), "no matched occlusion intervals"),
        "median": {
            "value": statistics.median(values) if values else None,
            "numerator": None,
            "denominator": len(values),
            "reason": None if values else "no matched occlusion intervals",
        },
        "absolute_errors_seconds": values,
    }
    if tolerance_seconds is not None:
        stats["tolerance_seconds"] = tolerance_seconds
        stats["within_tolerance"] = ratio(
            sum(1 for value in values if value <= tolerance_seconds),
            len(values),
            "no matched occlusion intervals",
        )
    return stats


def evaluate_sample(
    reference: Annotation, prediction: Annotation, mapping, review=None
) -> SampleMetrics:
    if reference.sample_id != prediction.sample_id:
        raise ValueError("sample identity mismatch")

    def mapped(events):
        return tuple(
            Event(
                e.start,
                e.end,
                _label(e.event_type, "action", mapping) or "unknown",
                e.actor,
                e.target,
                e.state,
                e.result,
                e.event_id,
            )
            for e in events
        )

    refs, preds = mapped(reference.actions), mapped(prediction.actions)
    # Separate type-agnostic assignment measures label errors rather than
    # conditioning action labels on already having the same action type.
    matches = match_events(
        [replace(e, event_type="temporal") for e in refs],
        [replace(e, event_type="temporal") for e in preds],
        0.3,
    )
    fields = {}
    for name in ("actor", "action", "target", "state", "result"):
        attr = "event_type" if name == "action" else name
        pairs = [
            (
                _label(getattr(refs[m.reference_index], attr), name, mapping),
                _label(getattr(preds[m.prediction_index], attr), name, mapping),
            )
            for m in matches
        ]
        if name == "result" and (
            reference.outcome is not None or prediction.outcome is not None
        ):
            pairs = [
                (
                    _label(reference.outcome, name, mapping),
                    _label(prediction.outcome, name, mapping),
                )
            ]
        fields[name] = _field_scores(
            [(a, b) for a, b in pairs if a is not None and b is not None]
        )
        excluded = len(pairs) - fields[name]["comparable_pairs"]
        fields[name].update(
            matched_pairs=len(pairs),
            excluded_pairs=excluded,
            exclusion_reason="unknown or unavailable label" if excluded else None,
        )
    ri = (
        reference.intervals
        if reference.intervals is not None
        else tuple(e for e in reference.occlusions if e.event_type == "occluded")
    )
    pi = (
        prediction.intervals
        if prediction.intervals is not None
        else tuple(e for e in prediction.occlusions if e.event_type == "occluded")
    )
    interval_matches = match_events(ri, pi, 0.3)
    boundaries = {
        name: _errors(
            [
                abs(
                    getattr(ri[m.reference_index], attr)
                    - getattr(pi[m.prediction_index], attr)
                )
                for m in interval_matches
            ],
            tolerance_seconds=BOUNDARY_TOLERANCE_SECONDS[name],
        )
        for name, attr in (("enter", "start"), ("exit", "end"))
    }
    factual = ratio(0, 0, "human review not supplied")
    if review is not None:
        _exact(
            review,
            {
                "schema_version",
                "sample_id",
                "prediction_sha256",
                "reviewer",
                "reviewer_kind",
                "claims",
            },
        )
        if review["reviewer_kind"] != "human":
            raise ValueError("completed human review required")
        if (
            review["schema_version"] != "las_occlusion_review_v1"
            or review["sample_id"] != prediction.sample_id
            or review["prediction_sha256"]
            != prediction.provenance.get("prediction_sha256")
        ):
            raise ValueError("review identity mismatch")
        _text(review["reviewer"])
        claims = review["claims"]
        if type(claims) is not list:
            raise ValueError("invalid review claims")
        predicted_claims = {e.event_id: e for e in prediction.occlusions}
        if len(predicted_claims) != len(prediction.occlusions):
            raise ValueError("duplicate prediction claim identity")
        for claim in claims:
            _exact(
                claim,
                {
                    "event_id",
                    "correct",
                    "target_entity_id",
                    "occluder_entity_id",
                    "event_type",
                    "start",
                    "end",
                    "visual_reason",
                },
            )
            _text(claim["event_id"])
            if type(claim["correct"]) is not bool:
                raise ValueError("incomplete human review")
            _text(claim["visual_reason"])
            if len(claim["visual_reason"]) > 1024:
                raise ValueError("review visual reason exceeds bound")
            event = predicted_claims.get(claim["event_id"])
            if event is None:
                raise ValueError("foreign review event")
            for name in ("target_entity_id", "occluder_entity_id", "event_type"):
                if _text(claim[name]) != getattr(event, name):
                    raise ValueError("review event identity mismatch")
            if (
                _number(claim["start"]) != event.start
                or _number(claim["end"]) != event.end
            ):
                raise ValueError("review event timing mismatch")
        ids = [c["event_id"] for c in claims]
        if len(ids) != len(set(ids)) or set(ids) != {
            e.event_id for e in prediction.occlusions
        }:
            raise ValueError("incomplete human review")
        factual = ratio(
            sum(c["correct"] for c in claims),
            len(claims),
            "no positive occlusion claims",
        )
    rates = {
        "completion": ratio(int(prediction.completed), 1, "no samples"),
        "degradation": ratio(
            prediction.component_statuses.count("unavailable"),
            len(prediction.component_statuses),
            "no components",
        ),
        "repair": ratio(
            int(prediction.repair_count > 0)
            if prediction.repair_count is not None
            else 0,
            int(prediction.repair_count is not None),
            "repair history unavailable",
        ),
        "repair_count": prediction.repair_count,
    }
    return SampleMetrics(
        reference.sample_id,
        {str(t): _event_metrics(refs, preds, t) for t in (0.3, 0.5)},
        {
            str(t): _event_metrics(reference.occlusions, prediction.occlusions, t)
            for t in (0.3, 0.5)
        },
        ratio(
            sum(m.iou for m in interval_matches),
            len(interval_matches),
            "no matched occlusion intervals",
        ),
        boundaries,
        fields,
        factual,
        rates,
        {
            **reference.provenance,
            **prediction.provenance,
            "mapping_sha256": digest(mapping),
            "human_review_sha256": digest(review) if review is not None else None,
        },
        {str(t): _event_metrics(ri, pi, t) for t in (0.3, 0.5)},
        {
            "threshold": 0.3,
            "matched_count": len(matches),
            "unmatched_reference_ids": [
                e.event_id
                for i, e in enumerate(refs)
                if i not in {m.reference_index for m in matches}
            ],
            "unmatched_prediction_ids": [
                e.event_id
                for i, e in enumerate(preds)
                if i not in {m.prediction_index for m in matches}
            ],
        },
    )


def aggregate_metrics(samples: list[SampleMetrics]) -> AggregateMetrics:
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("duplicate sample")

    def summed(items, reason):
        return ratio(
            sum(x["numerator"] for x in items),
            sum(x["denominator"] for x in items),
            reason,
        )

    def events(branch):
        result = {}
        for t in ("0.3", "0.5"):
            values = [getattr(s, branch)[t] for s in samples]
            result[t] = {
                name: summed([v[name] for v in values], "no events")
                for name in ("precision", "recall", "f1")
            }
            result[t].update(
                {
                    name: sum(v[name] for v in values)
                    for name in ("reference_count", "prediction_count", "matched_count")
                }
            )
        return result

    fields = {}
    for name in ("actor", "action", "target", "state", "result"):
        counts = {}
        for sample in samples:
            for label, count in sample.fields[name]["counts"].items():
                target = counts.setdefault(label, Counter())
                target.update(count)
        fields[name] = _field_counts(
            counts, sum(s.fields[name]["comparable_pairs"] for s in samples)
        )
        excluded = sum(s.fields[name]["excluded_pairs"] for s in samples)
        fields[name].update(
            matched_pairs=sum(s.fields[name]["matched_pairs"] for s in samples),
            excluded_pairs=excluded,
            exclusion_reason="unknown or unavailable label" if excluded else None,
        )
    return AggregateMetrics(
        len(samples),
        events("action"),
        events("occlusion"),
        fields,
        summed([s.factual_precision for s in samples], "no reviewed claims"),
        {
            name: summed([s.rates[name] for s in samples], "no measured samples")
            for name in ("completion", "degradation", "repair")
        },
        summed([s.occlusion_iou for s in samples], "no matched occlusion intervals"),
        {
            name: _errors(
                [
                    x
                    for s in samples
                    for x in s.boundary_errors[name]["absolute_errors_seconds"]
                ],
                tolerance_seconds=BOUNDARY_TOLERANCE_SECONDS[name],
            )
            for name in ("enter", "exit")
        },
        events("occlusion_intervals"),
    )


def _sha(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA-256")
    return value


def read_verified(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != _sha(expected):
        raise ValueError("input SHA-256 mismatch")
    return strict_json(data)


def _strings(values):
    if type(values) is not list:
        raise ValueError("expected string list")
    for value in values:
        _text(value)


def _interval(start, end, duration):
    if not 0 <= _number(start) < _number(end) <= duration:
        raise ValueError("interval outside video")


def _confidence(value):
    if not 0 <= _number(value) <= 1:
        raise ValueError("invalid confidence")


def adapt_reference(raw, *, sample_id, duration, provenance=None) -> Annotation:
    from ..pipelines.scene_semantics import SceneEventType

    _exact(
        raw,
        {
            "summary",
            "objects",
            "initial_state",
            "final_state",
            "occlusions",
            "semantic_events",
            "outcome",
            "camera_motion",
            "quality_flags",
            "review",
        },
    )
    for name in ("summary", "initial_state", "final_state"):
        _text(raw[name])
    for name in ("objects", "semantic_events", "occlusions"):
        if type(raw[name]) is not list:
            raise ValueError("invalid reference collection")
    objects = {}
    for obj in raw["objects"]:
        _exact(obj, {"id", "name", "role", "visible_attributes"})
        if not re.fullmatch(r"obj_[0-9]{3}", _text(obj["id"])) or obj["id"] in objects:
            raise ValueError("invalid object identity")
        objects[obj["id"]] = _text(obj["name"])
        _text(obj["role"])
        _strings(obj["visible_attributes"])

    def targets(ids):
        _strings(ids)
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= objects.keys():
            raise ValueError("unclosed object references")
        return " | ".join(sorted(objects[i] for i in ids))

    actions, occlusions, intervals = [], [], []
    previous = -1
    for i, e in enumerate(raw["semantic_events"]):
        _exact(
            e,
            {
                "event_id",
                "start_s",
                "end_s",
                "type",
                "actor",
                "object_ids",
                "description",
                "visible_evidence",
                "confidence",
            },
        )
        if e["event_id"] != f"evt_{i + 1:03d}" or e["start_s"] < previous:
            raise ValueError("invalid event identity/order")
        previous = e["start_s"]
        _interval(e["start_s"], e["end_s"], duration)
        _confidence(e["confidence"])
        SceneEventType(e["type"])
        if e["actor"] not in {
            "left hand",
            "right hand",
            "both hands",
            "neither hand",
            "unknown",
            *objects,
        }:
            raise ValueError("invalid actor reference")
        _text(e["description"])
        _text(e["visible_evidence"])
        event = Event(
            e["start_s"],
            e["end_s"],
            e["type"],
            objects.get(e["actor"], e["actor"]),
            targets(e["object_ids"]),
            event_id=e["event_id"],
        )
        (occlusions if e["type"] in OCCLUSION_TYPES else actions).append(event)
    for i, e in enumerate(raw["occlusions"]):
        _exact(
            e,
            {
                "start_s",
                "end_s",
                "occluded_object_ids",
                "occluder_id",
                "visible_evidence",
            },
        )
        _interval(e["start_s"], e["end_s"], duration)
        target = targets(e["occluded_object_ids"])
        if (
            e["occluder_id"] not in objects
            or e["occluder_id"] in e["occluded_object_ids"]
        ):
            raise ValueError("invalid occluder reference")
        _text(e["visible_evidence"])
        intervals.append(
            Event(
                e["start_s"],
                e["end_s"],
                "occluded",
                target=target,
                event_id=f"interval_{i}",
            )
        )
    _exact(raw["outcome"], {"status", "reason", "visible_evidence"})
    if raw["outcome"]["status"] not in {"success", "failure", "partial", "unknown"}:
        raise ValueError("invalid outcome")
    _text(raw["outcome"]["reason"])
    _text(raw["outcome"]["visible_evidence"])
    _strings(raw["camera_motion"])
    _strings(raw["quality_flags"])
    _exact(raw["review"], {"status", "notes", "required_checks"})
    if raw["review"]["status"] != "machine_only":
        raise ValueError("invalid reference review")
    _text(raw["review"]["notes"])
    _strings(raw["review"]["required_checks"])
    return Annotation(
        sample_id,
        tuple(actions),
        tuple(occlusions),
        tuple(intervals),
        provenance=provenance or {},
        outcome=raw["outcome"]["status"],
    )


def load_references(manifest_path):
    path = Path(manifest_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen manifest SHA-256 mismatch")
    manifest = strict_json(path.read_bytes())
    read_verified_text = (
        path.parent / manifest["generator"]["prompt_file"]
    ).read_bytes()
    if (
        hashlib.sha256(read_verified_text).hexdigest()
        != manifest["generator"]["prompt_sha256"]
    ):
        raise ValueError("reference prompt SHA-256 mismatch")
    references, entries = {}, {}
    for entry in manifest["samples"]:
        sample_id = entry["sample_id"]
        raw = read_verified(
            path.parent / entry["reference_file"], entry["reference_file_sha256"]
        )
        references[sample_id] = adapt_reference(
            raw,
            sample_id=sample_id,
            duration=entry["source_video"]["duration_seconds"],
            provenance={
                "reference_sha256": entry["reference_file_sha256"],
                "reference_manifest_sha256": FROZEN_MANIFEST_SHA256,
                "source_video_sha256": entry["source_video"]["sha256"],
                "submitted_media": entry["submitted_media"],
            },
        )
        entries[sample_id] = entry
    return references, entries


def _actions(events, duration):
    result = []
    for e in events:
        _interval(e["start"], e["end"], duration)
        _confidence(e["confidence"])
        result.append(
            Event(
                e["start"],
                e["end"],
                e["action"],
                e["actor"],
                e["target"],
                event_id=f"action_{e['event_index']}",
            )
        )
    return tuple(result)


def adapt_legacy(raw, *, sample_id, duration, provenance=None):
    """Validate the historical primary projection without rewriting its bytes."""
    from ..pipelines.scene_semantics import SceneSemantics, validate_scene_semantics
    from ..pipelines.semantic_events import validate_semantic_events

    _exact(
        raw,
        {
            "task_description",
            "segments",
            "grouped_semantic_events",
            "objects",
            "initial_state",
            "final_state",
            "outcome",
            "semantic_events",
        }
        | ({"warnings"} if "warnings" in raw else set()),
    )
    _text(raw["task_description"])
    validate_semantic_events(raw["grouped_semantic_events"], raw["segments"])
    for segment in raw["segments"]:
        _interval(segment["start"], segment["end"], duration)
    if not raw["segments"]:
        raise ValueError("empty legacy result")
    scene = SceneSemantics.model_validate(
        {
            **{
                k: raw[k]
                for k in (
                    "objects",
                    "initial_state",
                    "final_state",
                    "outcome",
                    "semantic_events",
                )
            },
            "locations": [],
            "relations": [],
        }
    )
    validate_scene_semantics(scene, duration, spatial_evidence_available=False)
    if any(e["event_type"] in OCCLUSION_TYPES for e in raw["semantic_events"]):
        raise ValueError("legacy occlusion has no trusted provenance")
    return Annotation(
        sample_id,
        _actions(raw["grouped_semantic_events"], duration),
        component_statuses=(
            "disabled",
            "unavailable"
            if not raw["objects"] and raw["outcome"]["status"] == "unknown"
            else "available",
            "disabled",
        ),
        repair_count=None,
        provenance={**(provenance or {}), "adapter": "frozen_legacy_primary_v1"},
        outcome=raw["outcome"]["status"],
    )


def validate_metadata(metadata, sample_ids, role):
    _exact(
        metadata,
        {
            "schema_version",
            "model_identity",
            "provider",
            "configuration",
            "configuration_sha256",
            "runtime",
            "samples",
        },
    )
    if metadata["schema_version"] != "las_evaluation_run_v1":
        raise ValueError("invalid metadata version")
    model = _text(metadata["model_identity"])
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", model) is None:
        raise ValueError("invalid model identity")
    if metadata["provider"] not in {"qwen3_vl", "ark"}:
        raise ValueError("invalid provider")
    if (metadata["provider"] == "qwen3_vl") != model.casefold().startswith("qwen"):
        raise ValueError("model/provider mismatch")
    if metadata["provider"] == "ark" and not model.casefold().startswith("doubao"):
        raise ValueError("Doubao identity required")
    if (
        role == "qwen"
        and metadata["provider"] != "qwen3_vl"
        or role == "doubao"
        and metadata["provider"] != "ark"
    ):
        raise ValueError("control model mismatch")
    config = metadata["configuration"]
    _exact(
        config,
        {
            "fps",
            "media_resolution",
            "clip_context",
            "reasoning_effort",
            "max_fine_segment_seconds",
            "query_sha256",
            "cv",
        },
    )
    if _number(config["fps"]) <= 0 or _number(config["max_fine_segment_seconds"]) <= 0:
        raise ValueError("invalid sampling configuration")
    for name in ("media_resolution", "clip_context", "reasoning_effort"):
        if config[name] not in {"low", "medium", "high"}:
            raise ValueError("invalid semantic configuration")
    _sha(config["query_sha256"])
    if digest(config) != _sha(metadata["configuration_sha256"]):
        raise ValueError("configuration digest mismatch")
    if role != "hybrid" and config["cv"] is not None:
        raise ValueError("control must disable CV")
    if config["cv"] is not None:
        from ..cv.contracts import EvidenceThresholds, SamplingPolicy

        cv = config["cv"]
        _exact(cv, {"sampling", "thresholds", "summary_limits", "bundle_limits"})
        SamplingPolicy.model_validate(cv["sampling"])
        EvidenceThresholds.model_validate(cv["thresholds"])
        _exact(
            cv["summary_limits"],
            {
                "max_tracks",
                "max_observations_per_track",
                "max_relations",
                "max_overlays",
                "max_prompt_chars",
            },
        )
        _exact(cv["bundle_limits"], {"max_candidates", "max_prompt_chars"})
        if any(
            type(v) is not int or v <= 0
            for limits in (cv["summary_limits"], cv["bundle_limits"])
            for v in limits.values()
        ):
            raise ValueError("invalid evidence limits")
    runtime = metadata["runtime"]
    _exact(runtime, {"gpu_devices", "model_revision", "checkpoint_sha256"})
    _strings(runtime["gpu_devices"])
    if runtime["model_revision"] is not None:
        _text(runtime["model_revision"])
    if runtime["checkpoint_sha256"] is not None:
        _sha(runtime["checkpoint_sha256"])
    samples = metadata["samples"]
    if type(samples) is not list:
        raise ValueError("invalid metadata samples")
    seen = set()
    for sample in samples:
        _exact(
            sample,
            {
                "sample_id",
                "result_sha256",
                "source_video_sha256",
                "model_identity",
                "configuration_sha256",
                "status",
                "wall_seconds",
                "stage_seconds",
            },
        )
        if (
            sample["sample_id"] in seen
            or sample["model_identity"] != model
            or sample["configuration_sha256"] != metadata["configuration_sha256"]
            or sample["status"] != "COMPLETED"
        ):
            raise ValueError("mixed or incomplete run metadata")
        seen.add(sample["sample_id"])
        _sha(sample["result_sha256"])
        _sha(sample["source_video_sha256"])
        if sample["wall_seconds"] is not None and _number(sample["wall_seconds"]) < 0:
            raise ValueError("invalid timing")
        if sample["stage_seconds"] is not None and (
            type(sample["stage_seconds"]) is not dict
            or not set(sample["stage_seconds"])
            <= {
                "media_decode",
                "pass_a",
                "sam31",
                "action_enrichment",
                "occlusion",
                "scene_facts",
                "merge",
                "embodied_pass_a",
                "embodied_pass_b",
                "embodied_enrichment",
                "scene_semantics",
                "occlusion_semantics",
            }
            or any(_number(v) < 0 for v in sample["stage_seconds"].values())
        ):
            raise ValueError("invalid stage timings")
    if seen != set(sample_ids):
        raise ValueError("missing or extra metadata samples")


def acceptance_gates(
    qwen, semantic_control, hybrid, *, same_model, provenance_complete
):
    action = hybrid["action"]["0.3"]["f1"]["value"]

    def preserved(control):
        baseline = control["action"]["0.3"]["f1"]["value"] if control else None
        return baseline is not None and action is not None and action >= baseline - 0.05

    reviewed = hybrid["factual_precision"]
    return {
        "all_five_tasks_validate": all(
            x is not None
            and x["sample_count"] == 5
            and x["rates"]["completion"]["value"] == 1
            for x in (qwen, semantic_control, hybrid)
        ),
        "action_f1_preserved_vs_frozen_qwen": preserved(qwen),
        "action_f1_preserved_vs_same_model": preserved(semantic_control),
        "same_semantic_model_control": same_model,
        "positive_occlusion_f1": (hybrid["occlusion"]["0.3"]["f1"]["value"] or 0) > 0,
        "all_positive_occlusion_claims_reviewed": reviewed["denominator"]
        == hybrid["occlusion"]["0.3"]["prediction_count"],
        "reviewed_precision_at_least_0_80": reviewed["value"] is not None
        and reviewed["value"] >= 0.8,
        "provenance_hashes_present": provenance_complete,
    }


def adapt_hybrid(
    raw,
    *,
    sample_id,
    duration,
    configuration,
    artifact_root=None,
    timeline=None,
    video_sha256=None,
    provenance=None,
    require_sam31=False,
):
    """Artifact-aware adapter; timeline must come from hash-verified source media.

    The CLI supplies that source clock. This lower-level API accepts a trusted
    FrameTimeline for callers that have already performed source verification.
    It never reads mask arrays or imports model runtimes.
    """
    from ..pipelines.hybrid_result import validate_hybrid_result

    cv = raw["cv_evidence"]
    summary = bundle = None
    cv_model = None
    if cv["status"] == "available":
        if artifact_root is None or timeline is None or configuration["cv"] is None:
            raise ValueError(
                "available CV requires trusted artifact and source timeline"
            )
        from ..cv.artifacts import CvArtifactHandle, CvArtifactStore, cv_cache_key
        from ..cv.contracts import CvEvidenceRequest, EvidenceThresholds, SamplingPolicy
        from ..cv.summary import build_cv_prompt_bundle, summarize_cv_evidence

        with CvArtifactStore(Path(artifact_root)) as store:
            artifact = store.load(
                CvArtifactHandle(cv["artifact_key"], cv["manifest_sha256"])
            )
        if require_sam31 and artifact.provider != "sam31":
            raise ValueError("production evaluation requires SAM 3.1 artifacts")
        cv_model = {
            "provider": artifact.provider,
            "model_identity": artifact.model_identity,
            "checkpoint_sha256": artifact.checkpoint_sha256,
        }
        config = configuration["cv"]
        request = CvEvidenceRequest(
            schema_version="cv_request_v1",
            provider=artifact.provider,
            model_identity=artifact.model_identity,
            video_path=Path("verified-source.mp4"),
            video_sha256=video_sha256,
            duration_seconds=float(duration),
            frame_count=timeline.frames[-1].frame_index + 1,
            checkpoint_sha256=artifact.checkpoint_sha256,
            timeline=timeline,
            entities=artifact.entities,
            sampling=SamplingPolicy.model_validate(config["sampling"]),
            thresholds=EvidenceThresholds.model_validate(config["thresholds"]),
        )
        if (
            artifact.video_sha256 != video_sha256
            or cv_cache_key(request) != cv["artifact_key"]
        ):
            raise ValueError("artifact configuration or video mismatch")
        summary = summarize_cv_evidence(
            artifact,
            timeline=timeline,
            thresholds=request.thresholds,
            **config["summary_limits"],
        )
        bundle = build_cv_prompt_bundle(
            summary, request.thresholds, **config["bundle_limits"]
        )
    elif (
        cv["status"] == "disabled"
        and configuration["cv"] is not None
        or cv["status"] != "disabled"
        and configuration["cv"] is None
    ):
        raise ValueError("CV status/configuration mismatch")
    validate_hybrid_result(
        raw,
        evidence_summary=summary,
        frame_pts=[f.timestamp_seconds for f in timeline.frames]
        if summary is not None
        else None,
        occlusion_candidates=bundle.candidates if bundle is not None else None,
    )
    branches = raw["annotation_branches"]
    entities = (
        {e.entity_id: e.canonical_label for e in summary.entities} if summary else {}
    )
    occlusions = []
    for e in branches["occlusion"]["events"]:
        _interval(e["start"], e["end"], duration)
        occlusions.append(
            Event(
                e["start"],
                e["end"],
                e["event_type"],
                target=entities[e["target_entity_id"]],
                event_id=f"occlusion_{e['event_index']}",
                target_entity_id=e["target_entity_id"],
                occluder_entity_id=e["occluder_entity_id"],
            )
        )
    statuses = (
        cv["status"],
        branches["scene_facts"]["status"],
        branches["occlusion"]["status"],
    )
    return Annotation(
        sample_id,
        _actions(branches["action_events"], duration),
        tuple(occlusions),
        component_statuses=statuses,
        repair_count=raw["performance"]["repair_count"],
        provenance={
            **(provenance or {}),
            "cv_evidence": cv,
            "cv_model": cv_model,
            "adapter": "hybrid_artifact_verified_v1",
        },
        outcome=raw["outcome"]["status"],
    )


def write_report(path, report, *, replace=False):
    """Identical output is idempotent; replacing different output is explicit."""
    path = Path(path)
    data = canonical_json(report).encode()
    if path.exists():
        if path.read_bytes() == data:
            return
        if not replace:
            raise ValueError("different report exists; use --replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        import os
        import tempfile

        descriptor, temporary = tempfile.mkstemp(prefix=".las-report-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    else:
        with path.open("xb") as stream:
            stream.write(data)


def evaluate_run(
    results_dir,
    metadata,
    references,
    entries,
    mapping,
    *,
    role,
    media_dir,
    artifact_root=None,
    reviews=None,
):
    from ..cv.timeline import probe_frame_timeline

    validate_metadata(metadata, set(references), role)
    if role == "qwen" and (
        metadata["model_identity"] != "qwen3-vl-8b-instruct"
        or metadata["configuration"]
        != {
            "fps": 2,
            "media_resolution": "medium",
            "clip_context": "high",
            "reasoning_effort": "high",
            "max_fine_segment_seconds": 30.0,
            "query_sha256": "bf94e33ffb1dc1ccd40f225766281ad39b3c7f18d4b3b6d6280e2b04d37f9f2a",
            "cv": None,
        }
    ):
        raise ValueError("frozen Qwen metadata mismatch")
    root = Path(results_dir)
    if {p.stem for p in root.glob("*.json")} != set(references):
        raise ValueError("missing or extra result samples")
    samples = []
    cv_model = None
    for entry in sorted(metadata["samples"], key=lambda x: x["sample_id"]):
        sid = entry["sample_id"]
        source = entries[sid]["source_video"]
        if entry["source_video_sha256"] != source["sha256"]:
            raise ValueError("source metadata mismatch")
        if role == "qwen" and entry["result_sha256"] != FROZEN_QWEN[sid]:
            raise ValueError("frozen Qwen result mismatch")
        media = Path(media_dir) / f"{sid}.mp4"
        sha = hashlib.sha256()
        with media.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != source["sha256"]:
            raise ValueError("source video SHA-256 mismatch")
        raw = read_verified(root / f"{sid}.json", entry["result_sha256"])
        provenance = {
            "prediction_sha256": entry["result_sha256"],
            "model_identity": metadata["model_identity"],
            "provider": metadata["provider"],
            "configuration_sha256": metadata["configuration_sha256"],
            "runtime_metadata_sha256": digest(metadata),
            "timings": {
                "wall_seconds": entry["wall_seconds"],
                "stage_seconds": entry["stage_seconds"],
            },
        }
        if "annotation_branches" in raw:
            timeline = (
                probe_frame_timeline(media)
                if raw["cv_evidence"]["status"] == "available"
                else None
            )
            prediction = adapt_hybrid(
                raw,
                sample_id=sid,
                duration=source["duration_seconds"],
                configuration=metadata["configuration"],
                artifact_root=artifact_root,
                timeline=timeline,
                video_sha256=source["sha256"],
                provenance=provenance,
                require_sam31=True,
            )
        elif role == "qwen":
            prediction = adapt_legacy(
                raw,
                sample_id=sid,
                duration=source["duration_seconds"],
                provenance=provenance,
            )
        else:
            raise ValueError("new controls and hybrid require canonical branches")
        sample_cv_model = prediction.provenance.get("cv_model")
        if sample_cv_model is not None:
            if cv_model is not None and sample_cv_model != cv_model:
                raise ValueError("mixed CV model or checkpoint in evaluation run")
            cv_model = sample_cv_model
        samples.append(
            evaluate_sample(
                references[sid], prediction, mapping, (reviews or {}).get(sid)
            )
        )
    return {
        "model_identity": metadata["model_identity"],
        "cv_model": cv_model,
        "provider": metadata["provider"],
        "configuration_sha256": metadata["configuration_sha256"],
        "runtime_metadata_sha256": digest(metadata),
        "aggregate": aggregate_metrics(samples).to_dict(),
        "samples": [s.to_dict() for s in samples],
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-manifest",
        "qwen-results",
        "hybrid-results",
        "qwen-metadata",
        "hybrid-metadata",
        "mapping",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("doubao-results", "doubao-metadata", "artifact-root", "review"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument(
        "--media-dir", type=Path, default=Path("evaluation/viewer/media")
    )
    parser.add_argument("--require-factual-precision", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    try:
        if bool(args.doubao_results) != bool(args.doubao_metadata):
            raise ValueError("Doubao results and metadata must be supplied together")
        refs, entries = load_references(args.reference_manifest)
        mapping = load_mapping(args.mapping)
        metadata = {
            role: strict_json(path.read_bytes())
            for role, path in (
                ("qwen", args.qwen_metadata),
                ("hybrid", args.hybrid_metadata),
            )
        }
        if args.doubao_metadata:
            metadata["doubao"] = strict_json(args.doubao_metadata.read_bytes())
        reviews = None
        review_file_sha256 = None
        if args.review:
            review_bytes = args.review.read_bytes()
            review_file_sha256 = hashlib.sha256(review_bytes).hexdigest()
            review = strict_json(review_bytes)
            _exact(review, {"schema_version", "samples"})
            if (
                review["schema_version"] != "las_review_set_v1"
                or type(review["samples"]) is not list
            ):
                raise ValueError("invalid review set")
            reviews = {r["sample_id"]: r for r in review["samples"]}
            if len(reviews) != len(review["samples"]) or set(reviews) != set(refs):
                raise ValueError("incomplete review samples")
        if args.require_factual_precision and reviews is None:
            raise ValueError("human review required")
        runs = {
            role: evaluate_run(
                getattr(args, role + "_results"),
                meta,
                refs,
                entries,
                mapping,
                role=role,
                media_dir=args.media_dir,
                artifact_root=args.artifact_root,
                reviews=reviews if role == "hybrid" else None,
            )
            for role, meta in metadata.items()
        }
        control_name = "doubao" if metadata["hybrid"]["provider"] == "ark" else "qwen"
        control = runs.get(control_name)
        same_model = control is not None and all(
            metadata[control_name][k] == metadata["hybrid"][k]
            for k in ("provider", "model_identity")
        )
        if same_model:
            same_model = {
                k: v
                for k, v in metadata[control_name]["configuration"].items()
                if k != "cv"
            } == {
                k: v
                for k, v in metadata["hybrid"]["configuration"].items()
                if k != "cv"
            }
        gates = acceptance_gates(
            runs["qwen"]["aggregate"],
            control["aggregate"] if control else None,
            runs["hybrid"]["aggregate"],
            same_model=same_model,
            provenance_complete=True,
        )
        hybrid_f1 = runs["hybrid"]["aggregate"]["action"]["0.3"]["f1"]["value"]
        comparisons = {
            role: {
                "action_f1_0_3_delta": hybrid_f1
                - run["aggregate"]["action"]["0.3"]["f1"]["value"]
                if hybrid_f1 is not None
                and run["aggregate"]["action"]["0.3"]["f1"]["value"] is not None
                else None,
                "sam_effect_comparison": role == control_name and same_model,
            }
            for role, run in runs.items()
            if role != "hybrid"
        }
        report = {
            "schema_version": "las_alignment_report_v1",
            "reference_manifest_sha256": FROZEN_MANIFEST_SHA256,
            "mapping_sha256": digest(mapping),
            "review_file_sha256": review_file_sha256,
            "runs": runs,
            "comparisons": comparisons,
            "acceptance_gates": gates,
            "accepted": all(gates.values()),
            "limitations": [
                "References are machine-generated, not human ground truth.",
                "Label macro-F1 is conditional on comparable labels from separate type-agnostic one-to-one temporal matches at IoU 0.3, not detector-wide semantic accuracy; result compares sample outcome.status and free-text states are not inferred.",
                "Occlusion interval boundaries use matched occluded intervals, separately from semantic event scoring.",
                "Historical fine segments are not extra primary events; historical repair timings remain unavailable.",
            ],
        }
        write_report(args.output, report, replace=args.replace)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        parser.exit(
            2,
            "evaluation input validation failed; check schemas, hashes, required samples, trusted artifacts, and output replacement\n",
        )
    return 0
