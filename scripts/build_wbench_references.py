#!/usr/bin/env python3
"""Build per-turn WBench reference timelines for instruction-mode event scoring.

Each interaction turn of an official WBench case becomes one reference event on
the generated video's own clock. The description is the per-turn instruction
from the official prompt builder (``build_turn_prompt`` without the first-turn
scene context), loaded from a WBench git checkout pinned to WBENCH_REVISION.
References therefore state what the model was asked to do, not what a real
video shows; score them with ``percept score events --judge-mode instruction``.

Each sample directory under --videos-root is named by its WBench case ID and
holds a ``combined.json`` describing the concatenated multi-turn video:
``{"video": {"fps": 24.0, "duration": 20.6}, "turn_frames": [0, 124, ..., 496]}``
where ``turn_frames`` has one more entry than the case has turns.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from percept_harness.pipelines.event_schemas import GroundedEvents
from percept_harness.video_metrics._runtime import run_cli, verify_revision, write_json_exclusive

WBENCH_REVISION = "e9461b3555d4d18a19465cfe4cba6fab280eb7c3"
PROMPT_BUILDER = Path("src/models/text/prompt_builder.py")


def load_prompt_builder(wbench_root: Path):
    """Import only the dependency-free prompt builder from the pinned checkout."""
    verify_revision(wbench_root, WBENCH_REVISION)
    spec = importlib.util.spec_from_file_location("wbench_prompt_builder", wbench_root / PROMPT_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_turn_prompt


def reference_events(case: dict, combined: dict, build_turn_prompt) -> dict:
    turns = sorted(case["interactions"], key=lambda turn: int(turn["turn"]))
    frames, fps = combined["turn_frames"], combined["video"]["fps"]
    if len(frames) != len(turns) + 1 or frames[0] != 0 or any(a >= b for a, b in zip(frames, frames[1:])):
        raise ValueError(f"case {case['id']}: turn_frames must start at 0, increase, and bound {len(turns)} turns")
    perspective = case.get("settings", {}).get("perspective", "first_person")
    events = [{"start": frames[i] / fps, "end": frames[i + 1] / fps,
               "description": build_turn_prompt(case, turn, perspective=perspective, is_first_turn=False)}
              for i, turn in enumerate(turns)]
    duration = max(combined["video"].get("duration", 0.0), events[-1]["end"])
    return GroundedEvents.model_validate({"duration": duration, "events": events}).model_dump(mode="json")


def build(wbench_root: Path, cases_dir: Path, videos_root: Path, output: Path) -> dict:
    build_turn_prompt = load_prompt_builder(wbench_root)
    samples = sorted((p.parent for p in videos_root.glob("*/combined.json")), key=lambda p: (len(p.name), p.name))
    if not samples:
        raise ValueError(f"no <sample>/combined.json under {videos_root}")
    missing = [s.name for s in samples if not (cases_dir / f"case_{s.name}.json").is_file()]
    if missing:
        raise ValueError(f"no WBench case for samples {missing[:10]}")
    built = []  # validate every sample before writing any file
    for sample in samples:
        case_path = cases_dir / f"case_{sample.name}.json"
        raw = case_path.read_bytes()
        case = json.loads(raw)
        record = reference_events(case, json.loads((sample / "combined.json").read_text()), build_turn_prompt)
        built.append((sample.name, record, {
            "sample_id": sample.name, "case": str(case_path), "case_sha256": hashlib.sha256(raw).hexdigest(),
            "turn_types": [t["type"] for t in sorted(case["interactions"], key=lambda t: int(t["turn"]))],
            "reference": str(output / sample.name / "reference.json")}))
    for name, record, _ in built:
        write_json_exclusive(output / name / "reference.json", record, parents=True)
    provenance = [entry for _, _, entry in built]
    write_json_exclusive(output / "provenance.json", {"wbench_revision": WBENCH_REVISION, "samples": provenance})
    return {"samples": len(provenance), "output": str(output)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wbench-root", type=Path, required=True, help=f"WBench git checkout at {WBENCH_REVISION}")
    parser.add_argument("--cases-dir", type=Path, required=True, help="WBench dataset cases/ directory (case_<id>.json)")
    parser.add_argument("--videos-root", type=Path, required=True, help="directory of <case id>/combined.json")
    parser.add_argument("--output", type=Path, required=True, help="new reference directory for percept score events")
    args = parser.parse_args(argv)
    return run_cli("Reference build", lambda: build(args.wbench_root, args.cases_dir, args.videos_root, args.output))


if __name__ == "__main__":
    raise SystemExit(main())
