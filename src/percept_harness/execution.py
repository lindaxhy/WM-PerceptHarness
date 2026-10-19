"""Direct model/CV stage execution with an optional local request cache."""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import results
from .media import TimeSpan
from .models.base import ModelOutputError, ModelRequest, VideoModel, VisualEvidence


logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Base class for stable execution-boundary failures."""


class SemanticResultCache:
    """Local cache of raw per-request model results.

    A repair-loop retry or a whole-task rerun re-issues many requests whose
    inputs are byte-identical to ones that already succeeded. Keyed on the
    video path/size/mtime, span, prompt, schema context, and the backend's own
    ``semantic_cache_identity``, a hit returns the stored result instead of
    paying for another inference.

    Entries hold the *raw* model output for the request, so a hit replays it
    through the same ``sanitize`` step a live call would run. Caching the
    sanitized shape instead is wrong: re-sanitizing an envelope treats it as raw
    model output, and the pipeline's own envelope unpacking is only defined for
    some schemas, so a replayed normalized envelope would fail there. Failed
    attempts are never cached: a retry must reach the model again. Entries are
    JSON files written atomically; a corrupt or unreadable entry is treated as a
    miss and rewritten.
    """

    # Bumped when the stored payload's meaning changes, so entries written by
    # an incompatible version are ignored rather than misread. Version 1 stored
    # the sanitized shape; version 2 stores the raw model output.
    _FORMAT_VERSION = 2

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
    def key(
        self,
        request: ModelRequest,
        backend_identity: Mapping[str, Any],
        schema_context: Mapping[str, Any] | None,
    ) -> str:
        video = request.video_path.resolve(strict=True)
        status = video.stat()
        identity = {
            "cache_version": 3,
            "video": [str(video), status.st_size, status.st_mtime_ns],
            "span": [request.span.start, request.span.end],
            "fps": request.fps,
            "stage": request.stage,
            "prompt": request.prompt,
            "schema_name": request.schema_name,
            "schema_context": dict(schema_context) if schema_context else None,
            "backend": dict(backend_identity),
            "visual_evidence": [e.identity() for e in request.visual_evidence],
        }
        # Opaque cache filename only; no content integrity verification.
        return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _entry_path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._entry_path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(entry, dict)
            or entry.get("format_version") != self._FORMAT_VERSION
            or not isinstance(entry.get("result"), dict)
        ):
            return None
        return entry["result"]

    def publish(self, key: str, result: Mapping[str, Any]) -> bool:
        try:
            results.write_json(self._entry_path(key), {
                "format_version": self._FORMAT_VERSION,
                "result": dict(result),
            })
        except OSError:
            logger.debug("semantic cache publish failed for %s", key, exc_info=True)
            return False
        return True


class StageExecutor:
    """Execute a model or CV stage directly, with optional request caching."""

    def __init__(
        self,
        model: VideoModel,
        *,
        default_model_alias: str,
        cv_executor=None,
        semantic_cache: SemanticResultCache | None = None,
    ) -> None:
        self.model = model
        self.default_model_alias = default_model_alias
        self._cv_executor = cv_executor
        self._semantic_cache = (
            semantic_cache
            if getattr(model, "supports_semantic_result_cache", False)
            else None
        )

    def execute(self, stage: str, payload: Mapping[str, Any],
                model_name: str | None = None) -> dict[str, Any]:
        if stage == "cv_evidence":
            if self._cv_executor is None:
                raise ExecutionError("no CV executor is configured")
            result, _ = self._cv_executor(payload)
        else:
            result = self._execute_model(stage, payload, model_name or self.default_model_alias)
        return result

    def _execute_model(self, stage: str, payload: Mapping[str, Any],
                       model_name: str) -> dict[str, Any]:
        from .pipelines import output_validation as schemas

        request = _model_request(stage, payload, model_name)
        context = _schema_validation_context(payload)
        request = replace(request, response_contract=schemas.response_contract(request.schema_name))
        cache_key: str | None = None
        if self._semantic_cache is not None:
            identity = getattr(self.model, "semantic_cache_identity", None)
            if callable(identity):
                try:
                    cache_key = self._semantic_cache.key(request, identity(request), context)
                except (OSError, TypeError, ValueError):
                    logger.debug("semantic cache key derivation failed", exc_info=True)
            if cache_key is not None:
                cached = self._semantic_cache.load(cache_key)
                if cached is not None:
                    # Replay the raw output through the same sanitize step a
                    # live call runs, so the pipeline sees an identical result.
                    try:
                        result = schemas.sanitize(request.schema_name, cached, context)
                    except (ValueError, TypeError):
                        result = None
                    if result is not None and schemas.failure_codes(request.schema_name, result) is None:
                        return result
        cacheable = True
        try:
            generated = self.model.generate(request)
            if not isinstance(generated, Mapping):
                raise ModelOutputError("model output must be a structured object")
            raw = dict(generated)
            result = schemas.sanitize(request.schema_name, raw, context)
            if schemas.failure_codes(request.schema_name, result) is not None:
                # An invalid-output envelope is a terminal verdict for this
                # attempt, not a result: caching it would replay the rejection
                # on every retry instead of reaching the model again.
                cacheable = False
        except ModelOutputError:
            result = schemas.model_output_failure(request.schema_name)
            if result is None:
                raise
            cacheable = False  # never replay a malformed-output placeholder
        if cache_key is not None and cacheable:
            self._semantic_cache.publish(cache_key, raw)
        return result


def _model_request(stage: str, payload: Mapping[str, Any], model_name: str) -> ModelRequest:
    video_value = payload.get("video_path")
    if not isinstance(video_value, str) or not video_value:
        raise ValueError("model stage video_path must be a non-blank local path")
    video_path = Path(video_value)
    if not video_path.is_absolute():
        raise ValueError("model stage video_path must be absolute")
    try:
        video_path = video_path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("model stage video_path is unavailable") from None
    if not video_path.is_file():
        raise ValueError("model stage video_path must be a regular file")

    span_value = payload.get("span")
    if isinstance(span_value, Mapping):
        start = span_value.get("start")
        end = span_value.get("end")
    else:
        start = payload.get("start")
        end = payload.get("end")
    return ModelRequest(
        stage=stage,
        video_path=video_path,
        span=TimeSpan(start, end),
        fps=payload.get("fps", 2.0),
        prompt=payload.get("prompt", ""),
        schema_name=payload.get("schema_name", stage),
        model_name=model_name,
        max_frames=payload.get("max_frames", 128),
        max_output_tokens=payload.get("max_output_tokens"),
        visual_evidence=tuple(VisualEvidence.from_payload(e) for e in payload.get("visual_evidence", ())),
    )


def _schema_validation_context(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    context = payload.get("schema_context")
    if context is None:
        return None
    if not isinstance(context, Mapping):
        raise ValueError("model stage schema context must be an object")
    return context
