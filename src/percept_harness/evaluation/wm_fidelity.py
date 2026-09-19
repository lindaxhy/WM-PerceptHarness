"""World-model video fidelity: generated video vs the same-id real video.

Both videos are annotated by the same fixed pipeline; the real-video output is
the reference. Two independent columns are reported and never merged:

* temporal -- frame-level event-family mIoU. Each event type is mapped to one
  of five coarse families, the timeline is bucketed at ``FPS`` and, per family,
  intersection / union of "family active" frames is accumulated. Insensitive to
  segmentation granularity, event count, actor and object naming; sensitive to
  time coverage and family. This is frame-wise mIoU from temporal action
  segmentation (MS-TCN et al.).
* semantic -- tIoU-weighted description similarity on the tau=0.3 family-aware
  one-to-one assignment (SODA with caption score replaced by sentence-embedding
  cosine). Requires the optional ``sentence-transformers`` dependency.

Event-level soft-F1 / F1@tau and outcome agreement are kept as auxiliary
columns. The reference is model-produced, so callers should also score the
annotator against itself on repeated runs of the real videos (self-agreement)
and report generated-video scores normalised by that floor.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .las_alignment import Event, match_events

OCCLUSION_TYPES = frozenset({"occluded", "occlusion_enter", "occlusion_exit"})
FAMILY_OF = {
    "move": "displace", "transport": "displace", "lift": "displace",
    "push": "displace", "pull": "displace", "rotate": "displace",
    "approach": "approach", "reach": "approach",
    "grasp": "contact", "contact": "contact",
    "place": "place", "release": "place",
    "autonomous_motion": "passive", "state_change": "passive", "stop": "passive",
}
FAMILIES = ("displace", "approach", "contact", "place", "passive")
FPS = 30
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
MATCH_TAU = 0.3
DURATION_STRATA = ((0.0, 6.0), (6.0, 10.0), (10.0, math.inf))
EVENT_COUNT_STRATA = ((0, 3), (3, 6), (6, math.inf))


@dataclass(frozen=True)
class FidelityEvent:
    start: float
    end: float
    event_type: str
    family: str
    description: str = ""
    confidence: float | None = None

    def __post_init__(self):
        if not (0 <= self.start < self.end):
            raise ValueError("invalid positive event interval")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    duration: float
    events: tuple[FidelityEvent, ...]
    outcome: str | None = None
    task_description: str = ""


def load_sample(path: str | Path, sample_id: str | None = None) -> Sample | None:
    """Parse one pipeline output file. Returns None when the pipeline failed
    (no ``segments``), so failed samples never enter the paired set."""
    path = Path(path)
    raw = json.loads(path.read_text())
    data = raw.get("data") or {}
    if not data.get("segments"):
        return None
    events = []
    for e in data.get("semantic_events", []):
        kind = e["event_type"]
        if kind in OCCLUSION_TYPES or kind not in FAMILY_OF:
            continue
        if not (0 <= e["start"] < e["end"]):
            continue
        events.append(
            FidelityEvent(
                float(e["start"]), float(e["end"]), kind, FAMILY_OF[kind],
                e.get("description") or "", e.get("confidence"),
            )
        )
    return Sample(
        sample_id or path.stem,
        float(data.get("duration") or 0.0),
        tuple(events),
        (data.get("outcome") or {}).get("status"),
        data.get("task_description") or "",
    )


# ---------------------------------------------------------------- temporal
def _frames(event: FidelityEvent, n_frames: int) -> range:
    return range(int(event.start * FPS), min(n_frames, int(event.end * FPS) + 1))


def family_frame_sets(events, duration) -> dict[str, set[int]]:
    n = int(math.ceil(duration * FPS)) + 1
    out: dict[str, set[int]] = {f: set() for f in FAMILIES}
    for e in events:
        out[e.family].update(_frames(e, n))
    return out


def frame_overlap(reference: Sample, prediction: Sample) -> dict[str, tuple[int, int]]:
    """Per-family (intersection, union) frame counts."""
    duration = max(reference.duration, prediction.duration)
    a = family_frame_sets(reference.events, duration)
    b = family_frame_sets(prediction.events, duration)
    return {f: (len(a[f] & b[f]), len(a[f] | b[f])) for f in FAMILIES}


def frame_miou(overlap: dict[str, tuple[int, int]]) -> float:
    inter = sum(v[0] for v in overlap.values())
    union = sum(v[1] for v in overlap.values())
    return inter / union if union else 0.0


# ---------------------------------------------------------------- event level
def _as_alignment_events(events) -> list[Event]:
    return [Event(e.start, e.end, e.family) for e in events]


def event_matches(reference: Sample, prediction: Sample, tau=MATCH_TAU):
    refs, preds = _as_alignment_events(reference.events), _as_alignment_events(prediction.events)
    if not refs or not preds:
        return []
    return match_events(refs, preds, tau)


# ---------------------------------------------------------------- semantic
SimilarityFn = Callable[[str, str], float]


def sentence_embedding_similarity(model_name="sentence-transformers/all-MiniLM-L6-v2") -> SimilarityFn:
    """Cosine similarity of normalised sentence embeddings. Lazy import so the
    temporal column has no heavy dependency."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    cache: dict[str, Any] = {}

    def encode(text):
        if text not in cache:
            cache[text] = model.encode(text, normalize_embeddings=True)
        return cache[text]

    def sim(a, b):
        return float(encode(a) @ encode(b))

    return sim


# ---------------------------------------------------------------- per pair
@dataclass
class PairScore:
    sample_id: str
    overlap: dict[str, tuple[int, int]]
    n_ref: int
    n_pred: int
    matched: dict[float, int]
    iou_sum: float
    sim_weighted_sum: float | None
    aligned_pairs: list[tuple[float, float]]
    outcome: tuple[str | None, str | None]

    @property
    def miou(self) -> float:
        return frame_miou(self.overlap)

    @property
    def soft_f1(self) -> float:
        d = self.n_ref + self.n_pred
        return 2 * self.iou_sum / d if d else 0.0

    @property
    def sem_f1(self) -> float | None:
        if self.sim_weighted_sum is None:
            return None
        d = self.n_ref + self.n_pred
        return 2 * self.sim_weighted_sum / d if d else 0.0


def score_pair(reference: Sample, prediction: Sample, similarity: SimilarityFn | None = None) -> PairScore:
    if reference.sample_id != prediction.sample_id:
        raise ValueError("sample identity mismatch")
    matched = {t: len(event_matches(reference, prediction, t)) for t in TAUS}
    base = event_matches(reference, prediction, MATCH_TAU)
    iou_sum = sum(m.iou for m in base)
    sim_sum, pairs = None, []
    if similarity is not None:
        sim_sum = 0.0
        for m in base:
            s = similarity(
                reference.events[m.reference_index].description,
                prediction.events[m.prediction_index].description,
            )
            sim_sum += m.iou * s
            pairs.append((m.iou, s))
    return PairScore(
        reference.sample_id, frame_overlap(reference, prediction),
        len(reference.events), len(prediction.events), matched, iou_sum,
        sim_sum, pairs, (reference.outcome, prediction.outcome),
    )


# ---------------------------------------------------------------- aggregate
def _ratio(n, d):
    return n / d if d else None


def aggregate(scores: list[PairScore]) -> dict:
    inter = Counter(); union = Counter()
    n_ref = n_pred = 0; matched = Counter(); iou_sum = 0.0
    sem_sum = 0.0; have_sem = all(s.sim_weighted_sum is not None for s in scores) and bool(scores)
    for s in scores:
        for f, (i, u) in s.overlap.items():
            inter[f] += i; union[f] += u
        n_ref += s.n_ref; n_pred += s.n_pred
        for t, c in s.matched.items():
            matched[t] += c
        iou_sum += s.iou_sum
        if have_sem:
            sem_sum += s.sim_weighted_sum
    family_iou = {f: _ratio(inter[f], union[f]) for f in FAMILIES}
    present = [v for v in family_iou.values() if v is not None]
    outcome_pairs = [s.outcome for s in scores if s.outcome[0] and s.outcome[1]]
    f1 = {t: _ratio(2 * matched[t], n_ref + n_pred) for t in TAUS}
    out = {
        "paired_samples": len(scores),
        "reference_events": n_ref,
        "prediction_events": n_pred,
        "frame_miou_micro": _ratio(sum(inter.values()), sum(union.values())),
        "frame_miou_macro_families": statistics.mean(present) if present else None,
        "family_iou": family_iou,
        "frame_miou_per_sample": [s.miou for s in scores],
        "f1_at_tau": f1,
        "avg_f1": statistics.mean(v for v in f1.values() if v is not None) if scores else None,
        "soft_f1": _ratio(2 * iou_sum, n_ref + n_pred),
        "soft_f1_per_sample": [s.soft_f1 for s in scores],
        "mean_matched_iou": _ratio(iou_sum, matched[MATCH_TAU]),
        "outcome_agreement": {
            "agree": sum(a == b for a, b in outcome_pairs),
            "total": len(outcome_pairs),
        },
    }
    if have_sem:
        out.update(
            sem_f1=_ratio(2 * sem_sum, n_ref + n_pred),
            sem_recall=_ratio(sem_sum, n_ref),
            sem_precision=_ratio(sem_sum, n_pred),
            sem_conditional=_ratio(sem_sum, iou_sum),
            sem_f1_per_sample=[s.sem_f1 for s in scores],
        )
    return out


def bootstrap_ci(values, iterations=1000, seed=0, level=0.95):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(iterations)
    )
    lo = int((1 - level) / 2 * iterations)
    return means[lo], means[iterations - 1 - lo]


def paired_wilcoxon(a, b) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value; None when every pair ties.
    Requires scipy."""
    if len(a) != len(b) or not a:
        raise ValueError("paired series must be equal length and non-empty")
    if all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
        return None
    from scipy.stats import wilcoxon  # type: ignore

    return float(wilcoxon(a, b).pvalue)


# ---------------------------------------------------------------- breakdown
def family_recall_precision(scores: list[PairScore], refs: dict[str, Sample], preds: dict[str, Sample]) -> dict:
    """Frame-level recall = inter / reference frames, precision = inter / predicted frames."""
    inter = Counter(); ref_frames = Counter(); pred_frames = Counter()
    for s in scores:
        r, p = refs[s.sample_id], preds[s.sample_id]
        duration = max(r.duration, p.duration)
        a, b = family_frame_sets(r.events, duration), family_frame_sets(p.events, duration)
        for f in FAMILIES:
            inter[f] += len(a[f] & b[f]); ref_frames[f] += len(a[f]); pred_frames[f] += len(b[f])
    return {
        f: {"recall": _ratio(inter[f], ref_frames[f]), "precision": _ratio(inter[f], pred_frames[f])}
        for f in FAMILIES
    }


def stratify(rows: list[dict], key: str, strata) -> list[dict]:
    """rows carry ``key`` plus one score per system name; returns per-stratum means."""
    systems = [k for k in rows[0] if k not in {"sample_id", "duration", "event_count", "task"}] if rows else []
    out = []
    for lo, hi in strata:
        group = [r for r in rows if lo <= r[key] < hi]
        if not group:
            continue
        entry = {"lo": lo, "hi": hi, "n": len(group)}
        for s in systems:
            entry[s] = statistics.mean(r[s] for r in group)
        if len(systems) == 2:
            a, b = systems
            entry[f"{b}_wins"] = sum(r[b] > r[a] for r in group)
        out.append(entry)
    return out


def pearson(a, b) -> float:
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


# ---------------------------------------------------------------- directory driver
def load_group(directory: str | Path, sample_ids=None) -> dict[str, Sample]:
    """``<dir>/<sid>/<sid>.json`` layout; failed samples are dropped."""
    directory = Path(directory)
    ids = sample_ids or sorted(p.name for p in directory.iterdir() if p.is_dir())
    out = {}
    for sid in ids:
        path = directory / sid / f"{sid}.json"
        if path.exists():
            sample = load_sample(path, sid)
            if sample is not None:
                out[sid] = sample
    return out


def score_groups(refs: dict[str, Sample], preds: dict[str, Sample], similarity=None) -> list[PairScore]:
    return [score_pair(refs[sid], preds[sid], similarity) for sid in sorted(refs.keys() & preds.keys())]
