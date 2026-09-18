"""The `percept` command-line interface: evaluate videos in one process."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

from .config import Settings
from .runner import (
    SUPPORTED_TEMPLATES,
    SyncRunner,
    collect_videos,
    default_pipeline_registry,
    run_batch,
)

_BACKENDS = ("openai", "fake")


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
        "--cv",
        default=None,
        choices=("disabled", "fake", "sam31"),
        help="CV evidence provider (default: PERCEPT_CV_PROVIDER from the environment)",
    )
    evaluate.add_argument(
        "--cv-device",
        default=None,
        help="CUDA device ordinal for the sam31 provider (default: PERCEPT_CV_DEVICE)",
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

    cv_provider = arguments.cv if arguments.cv is not None else settings.cv_provider
    if cv_provider != settings.cv_provider:
        # Settings validates sam31 paths only when PERCEPT_CV_PROVIDER=sam31, so
        # re-validate with the CLI override applied.
        try:
            settings = settings.model_copy(update={"cv_provider": cv_provider})
            settings = type(settings).model_validate(settings.model_dump())
        except Exception as error:
            closer()
            print(f"error: invalid CV configuration: {error}", file=sys.stderr)
            return 2

    print(f"backend={arguments.backend} model={alias} cv={cv_provider} videos={len(videos)}")
    try:
        with _cv_executor(cv_provider, arguments, settings) as cv_executor:
            runner = SyncRunner(
                model,
                settings,
                model_alias=alias,
                registry=default_pipeline_registry(),
                cv_executor=cv_executor,
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
            settings.model_registry, "PERCEPT_MODEL_REGISTRY"
        )
        return FakeVideoModel(), alias, lambda: None

    if backend == "openai":
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
            model_registry=registry,
            response_format=settings.openai_response_format,
            extra_body=settings.openai_extra_body,
            extra_headers=settings.openai_extra_headers,
            timeout_seconds=settings.openai_timeout_seconds,
            max_frames=settings.openai_max_frames,
            max_request_bytes=settings.openai_max_request_bytes,
            max_output_chars=settings.openai_max_output_chars,
            proxy=settings.openai_proxy.get_secret_value() if settings.openai_proxy else None,
        )
        return model, alias, model.close

    raise ValueError(f"unsupported backend {backend!r}")


def _single_alias(registry: dict, name: str) -> str:
    if len(registry) != 1:
        raise ValueError(
            f"{name} has {len(registry)} entries; pass --model to choose one"
        )
    return next(iter(registry))


@contextmanager
def _cv_executor(provider_name: str, arguments: argparse.Namespace, settings: Settings):
    """Yield a SyncCvExecutor for the chosen provider, or None when disabled."""
    if provider_name == "disabled":
        yield None
        return

    from .cv.artifacts import CvArtifactStore
    from .cv.executor import SyncCvExecutor

    artifact_store = CvArtifactStore(
        settings.cv_cache_root,
        max_files=settings.cv_cache_max_files,
        max_bytes=settings.cv_cache_max_bytes,
    )
    provider = None
    try:
        if provider_name == "fake":
            from .cv.base import FakeCvEvidenceProvider

            provider = FakeCvEvidenceProvider(
                execution_chunk_frames=settings.cv_execution_chunk_frames
            )
        else:  # sam31 — argparse enforces the closed set.
            cv_device = (
                int(arguments.cv_device)
                if arguments.cv_device is not None
                else settings.cv_device
            )
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cv_device))
            from .cv.sam31 import Sam31EvidenceProvider

            provider = Sam31EvidenceProvider.load(
                settings.cv_repository_path,
                settings.cv_checkpoint_path,
                settings.cv_checkpoint_sha256,
                bpe_path=settings.cv_bpe_path,
                compile_model=settings.cv_compile_model,
                max_artifact_bytes=settings.cv_cache_max_bytes,
                max_artifact_files=settings.cv_cache_max_files,
            )
            _configure_execution_chunk_frames(
                provider, settings.cv_execution_chunk_frames
            )
        yield SyncCvExecutor(provider, artifact_store)
    finally:
        try:
            if provider is not None:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
        finally:
            artifact_store.close()


def _configure_execution_chunk_frames(provider, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("requested CV execution chunk must be a positive integer")
    setter = getattr(provider, "set_execution_chunk_frames", None)
    if callable(setter):
        setter(value)
    else:
        provider.execution_chunk_frames = value
    configured = getattr(provider, "execution_chunk_frames", None)
    if configured != value:
        raise ValueError("CV provider execution chunk must equal the exact requested value")


if __name__ == "__main__":
    raise SystemExit(main())
