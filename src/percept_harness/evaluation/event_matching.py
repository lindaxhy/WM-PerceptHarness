"""Deterministic, cardinality-first temporal matching shared by fidelity scores."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


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
