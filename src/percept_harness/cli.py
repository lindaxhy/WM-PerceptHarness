"""The `percept` command-line interface for annotation and scoring."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .config import Settings
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
        settings = annotation_settings(arguments)
        videos = collect_videos(arguments.videos)
        if not videos:
            raise ValueError("no videos found under the given sources")
        with ExitStack() as resources:
            model, alias, closer = _load_backend(arguments, settings)
            resources.callback(closer)
            cv_executor = resources.enter_context(_cv_executor(settings))
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


def annotation_settings(arguments: argparse.Namespace) -> Settings:
    """Environment settings with command-line overrides shared by both annotation modes."""
    if arguments.event_workflow is None:
        return Settings()
    return Settings(event_workflow=arguments.event_workflow)


def _load_backend(arguments: argparse.Namespace, settings: Settings):
    """Return (model, model_alias, closer) for the selected backend."""
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        raise ValueError("PERCEPT_OPENAI_API_KEY is not configured")
    if not settings.openai_base_url:
        raise ValueError("PERCEPT_OPENAI_BASE_URL is not configured")
    registry = settings.openai_model_aliases
    if not registry:
        raise ValueError(
            "PERCEPT_OPENAI_MODEL (or PERCEPT_OPENAI_MODEL_REGISTRY) is not configured"
        )
    alias = arguments.model or _single_alias(registry, "PERCEPT_OPENAI_MODEL_REGISTRY")
    if alias not in registry:
        raise KeyError(
            f"model alias {alias!r} is absent from PERCEPT_OPENAI_MODEL_REGISTRY"
        )
    from .models.openai_compat import OpenAICompatVideoModel

    model = OpenAICompatVideoModel(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key.get_secret_value(),
        model_name=registry[alias],
        response_format=settings.openai_response_format,
        extra_body=settings.openai_extra_body,
        extra_headers=settings.openai_extra_headers,
        timeout_seconds=settings.openai_timeout_seconds,
        proxy=settings.openai_proxy.get_secret_value() if settings.openai_proxy else None,
    )
    return model, alias, model.close


def _single_alias(registry: dict, name: str) -> str:
    if len(registry) != 1:
        raise ValueError(
            f"{name} has {len(registry)} entries; pass --model to choose one"
        )
    return next(iter(registry))


def _check_sam_assets(settings: Settings) -> None:
    for name, kind in (('cv_repository_path', 'directory'), ('cv_checkpoint_path', 'file'), ('cv_bpe_path', 'file')):
        path = getattr(settings, name)
        if not (path.is_dir() if kind == 'directory' else path.is_file()):
            raise ValueError(f'{name} must be an existing local {kind}')


@contextmanager
def _cv_executor(settings: Settings):
    """Acquire SAM assets only when annotation requests CV evidence."""
    executor = None
    with ExitStack() as resources:
        def execute(payload):
            nonlocal executor
            if executor is None:
                executor = resources.enter_context(_loaded_cv_executor(settings))
            return executor(payload)

        yield execute


@contextmanager
def _loaded_cv_executor(settings: Settings):
    """Own provider and artifact lifetimes for synchronous CV execution."""
    _check_sam_assets(settings)

    from .cv.artifacts import CvArtifactStore
    from .cv.executor import SyncCvExecutor

    artifact_store = CvArtifactStore(settings.cv_cache_root)
    provider = None
    try:
        from .cv.sam31 import Sam31EvidenceProvider

        provider = Sam31EvidenceProvider.load(
            settings.cv_repository_path,
            settings.cv_checkpoint_path,
            bpe_path=settings.cv_bpe_path,
        )
        yield SyncCvExecutor(provider, artifact_store)
    finally:
        try:
            if provider is not None:
                provider.close()
        finally:
            artifact_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
