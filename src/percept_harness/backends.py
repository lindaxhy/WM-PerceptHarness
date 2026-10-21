"""Model, SAM and settings resources shared by single-video and manifest annotation."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager

from .config import Settings


def annotation_settings(arguments: argparse.Namespace) -> Settings:
    """Environment settings with command-line overrides shared by both annotation modes."""
    if arguments.event_workflow is None:
        return Settings()
    return Settings(event_workflow=arguments.event_workflow)


def load_model(settings: Settings, alias: str | None = None):
    """Return (model, model_alias, closer) for the configured OpenAI-compatible API."""
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        raise ValueError("PERCEPT_OPENAI_API_KEY is not configured")
    if not settings.openai_base_url:
        raise ValueError("PERCEPT_OPENAI_BASE_URL is not configured")
    registry = settings.openai_model_aliases
    if not registry:
        raise ValueError(
            "PERCEPT_OPENAI_MODEL (or PERCEPT_OPENAI_MODEL_REGISTRY) is not configured"
        )
    if alias is None:
        if len(registry) != 1:
            raise ValueError(
                f"PERCEPT_OPENAI_MODEL_REGISTRY has {len(registry)} entries; pass --model to choose one"
            )
        alias = next(iter(registry))
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


def _check_sam_assets(settings: Settings) -> None:
    for name, kind in (('cv_repository_path', 'directory'), ('cv_checkpoint_path', 'file'), ('cv_bpe_path', 'file')):
        path = getattr(settings, name)
        if not (path.is_dir() if kind == 'directory' else path.is_file()):
            raise ValueError(f'{name} must be an existing local {kind}')


@contextmanager
def cv_executor(settings: Settings):
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
