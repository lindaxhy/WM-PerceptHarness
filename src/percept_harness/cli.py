"""The `percept` command-line interface for annotation and scoring."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from . import backends
from .runner import (
    SyncRunner,
    collect_videos,
    run_batch,
)

_SCORE_MODULES = {
    "events": ".evaluation.events_cli",
    "clipiqa": ".video_metrics.clipiqa",
    "motion": ".video_metrics.motion",
    "vbench-clip": ".video_metrics.vbench_clip",
    "dover": ".video_metrics.dover",
    "external": ".video_metrics.external",
    "videophy-joint": ".video_metrics.videophy_joint",
}

_DIRECT_METRICS = {
    "imaging-quality": "MUSIQ image quality",
    "aesthetic-quality": "aesthetic quality",
    "temporal-flickering": "temporal flickering",
    "dynamic-degree": "dynamic degree",
    "subject-consistency": "subject consistency",
    "background-consistency": "background consistency",
    "overall-consistency": "video-text consistency (ViCLIP)",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _DIRECT_METRICS:
        from .video_metrics.vbench_backend import metric_main
        return metric_main(argv[0].replace("-", "_"), argv[1:])
    if argv and argv[0] == "score":
        score_parser = argparse.ArgumentParser(
            prog="percept score",
            description="Score annotations or videos in the active metric environment.",
        )
        score_parser.add_argument("metric", choices=tuple(_SCORE_MODULES))
        # Parse only the selector; each evaluator owns its remaining arguments/help.
        selected = score_parser.parse_args(argv[1:2])
        score_main = import_module(_SCORE_MODULES[selected.metric], __package__).main
        return score_main(argv[2:]) or 0
    arguments = _parser().parse_args(argv)
    if arguments.command == "annotate":
        return _eval(arguments)
    raise AssertionError("unreachable: argparse enforces the command set")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="percept",
        description="Annotate videos and compute evaluation metrics.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "score", help="score annotations or videos (use percept score --help for evaluators)",
    )
    for name, description in _DIRECT_METRICS.items():
        commands.add_parser(name, help=description)
    evaluate = commands.add_parser(
        "annotate",
        help="annotate videos to structured JSON",
    )
    inputs = evaluate.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        help="video files and/or directories (searched recursively)",
    )
    evaluate.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output directory for per-video JSON and results.jsonl",
    )
    inputs.add_argument("--manifest", type=Path, help="JSONL job_id/video_path manifest for multi-GPU annotation")
    evaluate.add_argument("--cv-devices", help="comma-separated physical GPUs for manifest mode")
    evaluate.add_argument("--workers-per-gpu", type=int, default=2)
    evaluate.add_argument("--request-concurrency", type=int, default=9)
    evaluate.add_argument("--shard-index", type=int, default=0)
    evaluate.add_argument("--shard-count", type=int, default=1)
    evaluate.add_argument("--event-workflow", choices=("reviewed", "confidence-gated"),
                          default=None, help="event workflow (default: PERCEPT_EVENT_WORKFLOW, reviewed)")
    evaluate.add_argument("--force", action="store_true", help="rerun videos without reusing VLM request cache")
    evaluate.add_argument(
        "--model",
        default=None,
        help="model alias from the backend registry (default: the registry's only entry)",
    )
    evaluate.add_argument(
        "--prompt-context",
        default=None,
        help="object naming hints passed to the annotator as data (not instructions)",
    )
    return parser


def _eval(arguments: argparse.Namespace) -> int:
    if arguments.manifest is not None:
        from .batch import run_manifest
        try:
            return run_manifest(arguments)
        except (ValueError, RuntimeError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    try:
        settings = backends.annotation_settings(arguments)
        videos = collect_videos(arguments.videos)
        if not videos:
            raise ValueError("no videos found under the given sources")
        with ExitStack() as resources:
            model, alias, closer = backends.load_model(settings, arguments.model)
            resources.callback(closer)
            cv_executor = resources.enter_context(backends.cv_executor(settings))
            print(f"model={alias} videos={len(videos)}")
            runner = SyncRunner(model, settings, model_alias=alias, cv_executor=cv_executor)
            outcomes = run_batch(
                runner, videos, arguments.output,
                prompt_context=arguments.prompt_context, force=arguments.force,
            )
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    completed = sum(outcome.status == "completed" for outcome in outcomes)
    skipped = sum(outcome.status == "skipped" for outcome in outcomes)
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    needs_review = sum(outcome.status == "needs_review" for outcome in outcomes)
    print(f"done: {completed} completed, {skipped} skipped, {needs_review} needs review, {failed} failed")
    return 0 if failed == 0 and needs_review == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
