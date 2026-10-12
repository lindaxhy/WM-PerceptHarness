#!/usr/bin/env python3
"""Run pinned upstream evaluators for metrics whose scorer remains external.

This runner is intentionally a thin orchestration layer.  It never downloads a
repository or checkpoint, rewrites an upstream command, computes a score, or
changes an official input file.  It verifies the caller-owned source/checkpoint
and then forwards the command after ``--`` unchanged to ``subprocess.run``.

The metric specs cover the retained external evaluators: WorldModelBench,
T2V-CompBench V2, VBench-2.0, PhyGenBench, VideoPhy-2, IQA-PyTorch and the
Google Research FVD implementation.  ``--check-only`` performs all preflight
checks without starting the evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class MetricSpec:
    source_url: str
    revision: str | None
    source_label: str
    command_hint: str
    needs_source: bool = True
    needs_checkpoint: bool = False
    needs_manifest: bool = False
    needs_generated: bool = False
    needs_reference: bool = False


SPECS: dict[str, MetricSpec] = {
    "instruction_following": MetricSpec(
        "https://github.com/WorldModelBench-Team/WorldModelBench", "00b7aa17a05f9fd1ab5c8f66bcf476d04c9c33bf", "WorldModelBench",
        "python evaluate.py --model_name MODEL_NAME --video_dir VIDEO_DIR --judge JUDGE --save_name OUTPUT", needs_checkpoint=True, needs_manifest=True),
    "action_binding": MetricSpec(
        "https://github.com/KaiyueSun98/T2V-CompBench", "dd5eff7b93af0550b9efa2bdabbb21b3b017ceda", "T2V-CompBench V2",
        "python LLaVA/llava/eval/compbench_eval_action_binding.py --video-path VIDEO_DIR --output-path OUTPUT --read-prompt-file MANIFEST --t2v-model MODEL", needs_manifest=True),
    "object_interactions": MetricSpec(
        "https://github.com/KaiyueSun98/T2V-CompBench", "dd5eff7b93af0550b9efa2bdabbb21b3b017ceda", "T2V-CompBench V2",
        "python LLaVA/llava/eval/compbench_eval_interaction.py --video-path VIDEO_DIR --output-path OUTPUT --read-prompt-file MANIFEST --t2v-model MODEL", needs_manifest=True),
    "motion_binding": MetricSpec(
        "https://github.com/KaiyueSun98/T2V-CompBench", "dd5eff7b93af0550b9efa2bdabbb21b3b017ceda", "T2V-CompBench V2",
        "python Grounded-Segment-Anything/compbench_motion_binding_seg.py ...; python dot/compbench_eval_motion_binding.py ...", needs_manifest=True),
    "motion_order_understanding": MetricSpec(
        "https://github.com/Vchitect/VBench", "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490", "VBench-2.0",
        "bash evaluate.sh --max_parallel_tasks 1", needs_manifest=True),
    "motion_rationality": MetricSpec(
        "https://github.com/Vchitect/VBench", "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490", "VBench-2.0",
        "bash evaluate.sh --max_parallel_tasks 1", needs_manifest=True),
    "mechanics": MetricSpec(
        "https://github.com/Vchitect/VBench", "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490", "VBench-2.0",
        "bash evaluate.sh --max_parallel_tasks 1", needs_manifest=True),
    "thermotics": MetricSpec(
        "https://github.com/Vchitect/VBench", "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490", "VBench-2.0",
        "bash evaluate.sh --max_parallel_tasks 1", needs_manifest=True),
    "material": MetricSpec(
        "https://github.com/Vchitect/VBench", "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490", "VBench-2.0",
        "bash evaluate.sh --max_parallel_tasks 1", needs_manifest=True),
    "phygen_eval_pca": MetricSpec(
        "https://github.com/OpenGVLab/PhyGenBench", "f8642cb796f3bcb01f0b7c1b2ec53b75d357c739", "PhyGenBench",
        "python PhyGenEval/overall.py", needs_manifest=True),
    "videophy_pc": MetricSpec(
        "https://github.com/Hritikbansal/videophy", None, "VideoPhy-2",
        "python inference.py --input_csv PC.csv --checkpoint CHECKPOINT --output_csv OUTPUT --task pc", needs_checkpoint=True, needs_manifest=True),
    "videophy_sa": MetricSpec(
        "https://github.com/Hritikbansal/videophy", None, "VideoPhy-2",
        "python inference.py --input_csv SA.csv --checkpoint CHECKPOINT --output_csv OUTPUT --task sa", needs_checkpoint=True, needs_manifest=True),
    "psnr": MetricSpec(
        "https://github.com/chaofengc/IQA-PyTorch", "18dd7a19694e94aac21019170e3f5e63d6b4e19e", "IQA-PyTorch",
        "python inference_iqa.py -m PSNR -t GENERATED -r REFERENCE", needs_generated=True, needs_reference=True),
    "ssim": MetricSpec(
        "https://github.com/chaofengc/IQA-PyTorch", "18dd7a19694e94aac21019170e3f5e63d6b4e19e", "IQA-PyTorch",
        "python inference_iqa.py -m SSIM -t GENERATED -r REFERENCE", needs_generated=True, needs_reference=True),
    "lpips": MetricSpec(
        "https://github.com/chaofengc/IQA-PyTorch", "18dd7a19694e94aac21019170e3f5e63d6b4e19e", "IQA-PyTorch",
        "python inference_iqa.py -m LPIPS -t GENERATED -r REFERENCE", needs_generated=True, needs_reference=True),
    "fid": MetricSpec(
        "https://github.com/chaofengc/IQA-PyTorch", "18dd7a19694e94aac21019170e3f5e63d6b4e19e", "IQA-PyTorch",
        "pyiqa fid -t GENERATED -r REFERENCE", needs_generated=True, needs_reference=True),
    "fvd": MetricSpec(
        "https://github.com/google-research/google-research", "4700efb9afa54286b0e04473ba80a13e8461e25f", "Google Research FVD",
        "python official_fvd_runner.py --real REAL_COLLECTION --generated GENERATED_COLLECTION", needs_generated=True, needs_reference=True),
}


def _resolved(value: Path | None, *, label: str, directory: bool | None = None) -> Path | None:
    if value is None:
        return None
    path = value.expanduser().resolve(strict=True)
    if directory is True and not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    if directory is False and not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")
    return path


def verify_revision(source: Path, expected: str | None, *, runner=subprocess.run) -> dict[str, str | None]:
    if not (source / ".git").exists():
        raise ValueError(f"Source is not a git checkout: {source}")
    if expected is None:
        return {"revision": None}
    result = runner(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    actual = result.stdout.strip()
    if actual != expected:
        raise ValueError(f"{source} revision mismatch: expected {expected}, got {actual}")
    return {"revision": actual}


def preflight(*, metric: str, source_dir: Path | None, checkpoint: Path | None,
              manifest: Path | None, generated: Path | None, reference: Path | None,
              runner=subprocess.run) -> dict:
    try:
        spec = SPECS[metric]
    except KeyError as exc:
        raise ValueError(f"unknown external metric: {metric}") from exc
    if spec.needs_source and source_dir is None:
        raise ValueError(f"{metric} requires --source-dir (use the pinned official checkout)")
    source = _resolved(source_dir, label="source-dir", directory=True) if spec.needs_source else None
    if source is not None:
        revision = verify_revision(source, spec.revision, runner=runner)
    else:
        revision = {"revision": None}
    model = _resolved(checkpoint, label="checkpoint", directory=None) if spec.needs_checkpoint else checkpoint
    if spec.needs_checkpoint and model is None:
        raise ValueError(f"{metric} requires --checkpoint (no download is attempted)")
    input_manifest = _resolved(manifest, label="manifest", directory=False) if spec.needs_manifest else manifest
    if spec.needs_manifest and input_manifest is None:
        raise ValueError(f"{metric} requires --manifest (use the official adapter output)")
    generated_path = _resolved(generated, label="generated", directory=True) if spec.needs_generated else generated
    reference_path = _resolved(reference, label="reference", directory=True) if spec.needs_reference else reference
    if spec.needs_generated and generated_path is None:
        raise ValueError(f"{metric} requires --generated")
    if spec.needs_reference and reference_path is None:
        raise ValueError(f"{metric} requires --reference")
    return {"metric": metric, "source": str(source) if source else None, "source_url": spec.source_url,
            "revision": revision["revision"], "checkpoint": str(model) if model else None,
            "manifest": str(input_manifest) if input_manifest else None,
            "generated": str(generated_path) if generated_path else None,
            "reference": str(reference_path) if reference_path else None,
            "command_hint": spec.command_hint}


def run_official(*, metric: str, source_dir: Path | None, checkpoint: Path | None,
                 manifest: Path | None, generated: Path | None, reference: Path | None,
                 command: Sequence[str], output: Path | None = None,
                 check_only: bool = False, runner=subprocess.run) -> dict:
    if not command:
        raise ValueError("append the exact official command after --")
    details = preflight(metric=metric, source_dir=source_dir, checkpoint=checkpoint,
                        manifest=manifest, generated=generated, reference=reference, runner=runner)
    details["command"] = list(command)
    details["status"] = "preflight_passed"
    if check_only:
        return details
    log = None
    stream = None
    try:
        if output is not None:
            output = output.expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"output already exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.with_suffix(output.suffix + ".log")
            stream = log.open("x", encoding="utf-8")
        completed = runner(list(command), cwd=details["source"], check=False,
                           stdout=stream, stderr=subprocess.STDOUT,
                           env=os.environ.copy(), text=True)
        if completed.returncode:
            raise RuntimeError(f"official {metric} command exited with {completed.returncode}")
    finally:
        if stream is not None:
            stream.close()
    details["status"] = "complete"
    if log is not None:
        details["log"] = str(log)
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=sorted(SPECS), required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="exact upstream command; place it after --")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        result = run_official(metric=args.metric, source_dir=args.source_dir,
                              checkpoint=args.checkpoint, manifest=args.manifest,
                              generated=args.generated, reference=args.reference,
                              command=command, output=args.output,
                              check_only=args.check_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"Official external metric runner failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
