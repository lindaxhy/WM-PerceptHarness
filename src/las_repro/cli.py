"""The `percept` command-line interface: evaluate videos in one process."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import Settings
from .runner import (
    SUPPORTED_TEMPLATES,
    SyncRunner,
    collect_videos,
    default_pipeline_registry,
    run_batch,
)

_BACKENDS = ("doubao", "qwen", "fake")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "eval":
        return _eval(arguments)
    raise AssertionError("unreachable: argparse enforces the command set")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="percept",
        description="Evaluate videos with a configured VLM backend.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("eval", help="evaluate videos to structured JSON")
    evaluate.add_argument(
        "--videos",
        nargs="+",
        required=True,
        type=Path,
        help="video files and/or directories (searched recursively)",
    )
    evaluate.add_argument(
        "--template",
        required=True,
        choices=SUPPORTED_TEMPLATES,
        help="annotation template to run",
    )
    evaluate.add_argument(
        "--backend",
        required=True,
        choices=_BACKENDS,
        help="VLM backend configured in the environment",
    )
    evaluate.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output directory for per-video JSON and results.jsonl",
    )
    evaluate.add_argument(
        "--model",
        default=None,
        help="model alias from the backend registry (default: the registry's only entry)",
    )
    evaluate.add_argument(
        "--prompt-context",
        default=None,
        help="naming context for embodied_action_captioning (object hints, not an SOP)",
    )
    evaluate.add_argument(
        "--query",
        default=None,
        help="free-form query for general_video_captioning",
    )
    evaluate.add_argument(
        "--device",
        default=None,
        help="CUDA device ordinal for the qwen backend (default: first of LAS_GPU_DEVICES)",
    )
    return parser


def _eval(arguments: argparse.Namespace) -> int:
    try:
        settings = Settings.from_env()
    except Exception as error:
        print(f"error: invalid configuration: {error}", file=sys.stderr)
        return 2

    try:
        videos = collect_videos(list(arguments.videos))
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not videos:
        print("error: no videos found under the given sources", file=sys.stderr)
        return 2

    try:
        model, alias, closer = _load_backend(arguments, settings)
    except (ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"backend={arguments.backend} model={alias} videos={len(videos)}")
    try:
        runner = SyncRunner(
            model,
            settings,
            model_alias=alias,
            registry=default_pipeline_registry(),
        )
        outcomes = run_batch(
            runner,
            videos,
            arguments.template,
            arguments.output,
            prompt_context=arguments.prompt_context,
            query=arguments.query,
        )
    finally:
        closer()

    completed = sum(outcome.status == "completed" for outcome in outcomes)
    skipped = sum(outcome.status == "skipped" for outcome in outcomes)
    failed = sum(outcome.status == "failed" for outcome in outcomes)
    print(f"done: {completed} completed, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def _load_backend(arguments: argparse.Namespace, settings: Settings):
    """Return (model, model_alias, closer) for the selected backend."""
    backend = arguments.backend
    if backend == "fake":
        from .models.fake import FakeVideoModel

        alias = arguments.model or _single_alias(
            settings.model_registry, "LAS_MODEL_REGISTRY"
        )
        return FakeVideoModel(), alias, lambda: None

    if backend == "doubao":
        if settings.ark_api_key is None or not settings.ark_api_key.get_secret_value().strip():
            raise ValueError("LAS_ARK_API_KEY is not configured")
        if not settings.ark_model_registry:
            raise ValueError("LAS_ARK_MODEL_REGISTRY is empty")
        from .models.ark import ArkVideoModel

        alias = arguments.model or _single_alias(
            settings.ark_model_registry, "LAS_ARK_MODEL_REGISTRY"
        )
        if alias not in settings.ark_model_registry:
            raise KeyError(f"model alias {alias!r} is absent from LAS_ARK_MODEL_REGISTRY")
        model = ArkVideoModel(
            api_key=settings.ark_api_key.get_secret_value(),
            model_registry=settings.ark_model_registry,
            timeout_seconds=settings.ark_timeout_seconds,
            max_frames=settings.ark_max_frames,
            max_request_bytes=settings.ark_max_request_bytes,
            max_output_chars=settings.ark_max_output_chars,
            proxy=settings.ark_proxy.get_secret_value() if settings.ark_proxy else None,
        )
        return model, alias, model.close

    if backend == "qwen":
        if not settings.model_registry:
            raise ValueError("LAS_MODEL_REGISTRY is empty")
        alias = arguments.model or _single_alias(
            settings.model_registry, "LAS_MODEL_REGISTRY"
        )
        if alias not in settings.model_registry:
            raise KeyError(f"model alias {alias!r} is absent from LAS_MODEL_REGISTRY")
        device_ordinal = (
            int(arguments.device)
            if arguments.device is not None
            else settings.gpu_devices[0]
        )
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
        from .models.qwen3_vl import Qwen3VLModel

        model = Qwen3VLModel.load_alias(
            alias,
            settings.model_registry,
            f"cuda:{device_ordinal}",
            "auto",
            max_output_chars=settings.max_model_output_chars,
        )
        return model, alias, lambda: None

    raise ValueError(f"unsupported backend {backend!r}")


def _single_alias(registry: dict, name: str) -> str:
    if len(registry) != 1:
        raise ValueError(
            f"{name} has {len(registry)} entries; pass --model to choose one"
        )
    return next(iter(registry))


if __name__ == "__main__":
    raise SystemExit(main())
