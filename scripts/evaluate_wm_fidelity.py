"""Score world-model generated videos against same-id real-video annotations.

Example (layout produced by the wm-annot runs: ``<group>-actions/<sid>/<sid>.json``):

    python scripts/evaluate_wm_fidelity.py \
        --reference data/outputs/wm-annot-20260918/gt-actions \
        --system wan=data/outputs/wm-annot-20260918/wan-actions \
        --system h3=data/outputs/wm-annot-20260918/h3-actions \
        --self-agreement run1=.../percept-doubao-sam31-20260916/stage2 \
                         run2=.../percept-doubao-sam31-rerun-20260918/stage2 \
        --semantic --out wm_fidelity_report.json

``--semantic`` needs ``sentence-transformers`` (set HF_HUB_OFFLINE=1 when the
model is cached); paired Wilcoxon needs ``scipy``. Both are optional.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from percept_harness.evaluation.wm_fidelity import (  # noqa: E402
    DURATION_STRATA,
    EVENT_COUNT_STRATA,
    FAMILIES,
    aggregate,
    bootstrap_ci,
    family_recall_precision,
    load_group,
    paired_wilcoxon,
    pearson,
    score_groups,
    sentence_embedding_similarity,
    stratify,
)


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _print_block(name, agg):
    print(f"\n===== {name}  (paired={agg['paired_samples']}, ref events={agg['reference_events']}, pred events={agg['prediction_events']}) =====")
    ci = bootstrap_ci(agg["frame_miou_per_sample"])
    print(f"  TEMPORAL  frame mIoU micro={_fmt(agg['frame_miou_micro'])}  macro-over-families={_fmt(agg['frame_miou_macro_families'])}"
          f"  per-sample mean={_fmt(statistics.mean(agg['frame_miou_per_sample']) if agg['frame_miou_per_sample'] else None)}"
          + (f" [95% CI {ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
    print("            per family: " + "  ".join(f"{f}={_fmt(agg['family_iou'][f], 2)}" for f in FAMILIES))
    if "sem_f1" in agg:
        ci = bootstrap_ci(agg["sem_f1_per_sample"])
        print(f"  SEMANTIC  sem-F1={_fmt(agg['sem_f1'])}  (R={_fmt(agg['sem_recall'])} P={_fmt(agg['sem_precision'])})"
              f"  conditional={_fmt(agg['sem_conditional'])}"
              + (f"  per-sample [95% CI {ci[0]:.3f},{ci[1]:.3f}]" if ci else ""))
    f1 = agg["f1_at_tau"]
    print(f"  AUX       soft-F1={_fmt(agg['soft_f1'])}  avg-F1={_fmt(agg['avg_f1'])}  "
          + "  ".join(f"F1@{t}={_fmt(f1[t], 2)}" for t in (0.3, 0.5, 0.7))
          + f"  outcome agreement={agg['outcome_agreement']['agree']}/{agg['outcome_agreement']['total']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="directory of annotations on the real videos")
    ap.add_argument("--system", action="append", default=[], metavar="NAME=DIR",
                    help="generated-video annotation directory; repeatable")
    ap.add_argument("--self-agreement", nargs=2, metavar=("NAME=DIR", "NAME=DIR"),
                    help="two independent annotator runs on the same real videos")
    ap.add_argument("--semantic", action="store_true", help="also compute the description-similarity column")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", type=Path, help="write the full report as JSON")
    args = ap.parse_args(argv)

    def parse(spec):
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"expected NAME=DIR, got {spec!r}")
        return name, Path(path)

    similarity = sentence_embedding_similarity(args.embedding_model) if args.semantic else None
    refs = load_group(args.reference)
    report = {"reference": str(args.reference), "systems": {}, "breakdown": {}}
    per_system_scores = {}
    for spec in args.system:
        name, path = parse(spec)
        preds = load_group(path)
        scores = score_groups(refs, preds, similarity)
        per_system_scores[name] = (scores, preds)
        agg = aggregate(scores)
        agg["family_recall_precision"] = family_recall_precision(scores, refs, preds)
        report["systems"][name] = agg
        _print_block(f"{name} vs reference", agg)
        print("  per-family R/P: " + "  ".join(
            f"{f} R={_fmt(v['recall'], 2)} P={_fmt(v['precision'], 2)}"
            for f, v in agg["family_recall_precision"].items()))

    if args.self_agreement:
        (n1, p1), (n2, p2) = map(parse, args.self_agreement)
        a, b = load_group(p1), load_group(p2)
        floor = aggregate(score_groups(a, b, similarity))
        report["self_agreement"] = floor
        _print_block(f"annotator self-agreement ({n1} vs {n2})", floor)
        if floor["frame_miou_micro"]:
            print("\n  NORMALISED BY SELF-AGREEMENT FLOOR")
            for name, agg in report["systems"].items():
                line = f"    {name}: frame mIoU {agg['frame_miou_micro']/floor['frame_miou_micro']:.2f}"
                if "sem_f1" in agg and floor.get("sem_f1"):
                    line += f"   sem-F1 {agg['sem_f1']/floor['sem_f1']:.2f}"
                print(line)

    names = list(per_system_scores)
    if len(names) == 2:
        (na, (sa, pa)), (nb, (sb, pb)) = per_system_scores.items()
        ia = {s.sample_id: s for s in sa}; ib = {s.sample_id: s for s in sb}
        common = sorted(ia.keys() & ib.keys())
        rows = [{"sample_id": sid, "duration": refs[sid].duration, "event_count": len(refs[sid].events),
                 "task": refs[sid].task_description, na: ia[sid].miou, nb: ib[sid].miou} for sid in common]
        bd = {
            "common_ids": len(common),
            "corr_between_systems": pearson([r[na] for r in rows], [r[nb] for r in rows]),
            "corr_with_duration": {n: pearson([r[n] for r in rows], [r["duration"] for r in rows]) for n in names},
            "corr_with_event_count": {n: pearson([r[n] for r in rows], [r["event_count"] for r in rows]) for n in names},
            "by_duration": stratify(rows, "duration", DURATION_STRATA),
            "by_event_count": stratify(rows, "event_count", EVENT_COUNT_STRATA),
            "per_sample": rows,
        }
        for metric, getter in (("frame_miou", lambda s: s.miou), ("soft_f1", lambda s: s.soft_f1), ("sem_f1", lambda s: s.sem_f1)):
            xa = [getter(ia[s]) for s in common]; xb = [getter(ib[s]) for s in common]
            if any(v is None for v in xa + xb):
                continue
            try:
                p = paired_wilcoxon(xa, xb)
            except ImportError:
                p = "scipy not installed"
            bd[f"paired_{metric}"] = {na: statistics.mean(xa), nb: statistics.mean(xb),
                                      f"{nb}_wins": sum(y > x for x, y in zip(xa, xb)), "wilcoxon_p": p}
            print(f"\n  PAIRED ({len(common)} ids) {metric}: {na}={statistics.mean(xa):.3f} {nb}={statistics.mean(xb):.3f}"
                  f"  {nb} wins {bd[f'paired_{metric}'][f'{nb}_wins']}/{len(common)}  Wilcoxon p={p if isinstance(p, str) else f'{p:.4f}'}")
        report["breakdown"] = bd
        print(f"\n  corr({na},{nb})={bd['corr_between_systems']:.2f}   "
              + "  ".join(f"corr({n},duration)={bd['corr_with_duration'][n]:.2f}" for n in names))
        for key in ("by_duration", "by_event_count"):
            print(f"  {key}:")
            for e in bd[key]:
                hi = "inf" if e["hi"] == float("inf") else f"{e['hi']:.0f}"
                print(f"    [{e['lo']:.0f},{hi})  n={e['n']:2d}  " + "  ".join(f"{n}={e[n]:.2f}" for n in names)
                      + f"  {nb} wins {e[f'{nb}_wins']}/{e['n']}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=lambda o: None if o != o else o) + "\n")
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
