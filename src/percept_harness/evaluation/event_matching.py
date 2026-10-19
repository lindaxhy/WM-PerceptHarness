"""Deterministic cardinality-first matching for eligible event edges."""
from __future__ import annotations

import math
from dataclasses import dataclass


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("invalid finite number")
    return value


@dataclass(frozen=True)
class Match:
    reference_index: int
    prediction_index: int
    iou: float


def check_threshold(value):
    if not 0 < _number(value) <= 1:
        raise ValueError("invalid temporal IoU threshold")
    return value


def temporal_iou(a, b) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    return intersection / ((a.end - a.start) + (b.end - b.start) - intersection)


def match_weight_matrix(scores, threshold=0.75) -> list[Match]:
    """Maximize eligible pair count, then summed IoU, using one dummy per row."""
    check_threshold(threshold)
    n = len(scores)
    real = len(scores[0]) if n else 0
    if max(n, real) >= 1_000_000:
        raise ValueError("too many events")
    if any(len(row) != real for row in scores):
        raise ValueError("ragged IoU matrix")
    for row in scores:
        for value in row:
            if not 0 <= _number(value) <= 1:
                raise ValueError("invalid IoU")
    if not n or not real:
        return []
    weights = [[1_000_000 + value if value >= threshold else 0 for value in row] + [0] * n for row in scores]
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
    return sorted((Match(p[j] - 1, j - 1, scores[p[j] - 1][j - 1]) for j in range(1, real + 1) if p[j] and weights[p[j] - 1][j - 1] > 0), key=lambda x: x.reference_index)
