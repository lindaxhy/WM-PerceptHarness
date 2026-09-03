"""Shared local video-model protocol and strict structured-output parsing."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from ..media import FrameRef, TimeSpan, VideoMetadata
from ..model_alias import DEFAULT_MODEL_ALIAS, validate_model_alias


DEFAULT_MAX_MODEL_OUTPUT_CHARS = 1_000_000
TuningLevel = Literal["low", "medium", "high"]
_COMPLETE_JSON_FENCE = re.compile(
    r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?```",
    flags=re.IGNORECASE | re.DOTALL,
)


class ModelOutputError(ValueError):
    """A model response is not one safe, complete structured JSON object."""


class ModelRequestError(ValueError):
    """A worker attempted to construct an invalid local model request."""


@dataclass
class VideoSession:
    """Worker-local video state reused by all stages for one task."""

    metadata: VideoMetadata
    sampled_frames: tuple[FrameRef, ...] = ()
    backend_cache: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    """Immutable visual-only input for one local model inference stage."""

    stage: str
    video_path: Path
    span: TimeSpan
    fps: float
    prompt: str
    schema_name: str
    video_session_id: str | None
    model_name: str = DEFAULT_MODEL_ALIAS
    media_resolution: TuningLevel | None = None
    reasoning_effort: TuningLevel | None = None
    clip_context: TuningLevel | None = None
    video_session: VideoSession | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ModelRequestError("stage must be a non-blank string")
        if not isinstance(self.video_path, Path):
            raise ModelRequestError("video_path must be a pathlib.Path")
        if not isinstance(self.span, TimeSpan):
            raise ModelRequestError("span must be a TimeSpan")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ModelRequestError("fps must be a finite positive number")
        if not isinstance(self.prompt, str):
            raise ModelRequestError("prompt must be a string")
        if not isinstance(self.schema_name, str) or not self.schema_name.strip():
            raise ModelRequestError("schema_name must be a non-blank string")
        if self.video_session_id is not None and (
            not isinstance(self.video_session_id, str) or not self.video_session_id.strip()
        ):
            raise ModelRequestError("video_session_id must be a non-blank string or None")
        try:
            validate_model_alias(self.model_name)
        except ValueError:
            raise ModelRequestError("model_name must be a local model alias") from None
        if self.video_session is not None and not isinstance(
            self.video_session,
            VideoSession,
        ):
            raise ModelRequestError("video_session must be a VideoSession or None")
        for field_name in ("media_resolution", "reasoning_effort", "clip_context"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or value not in ("low", "medium", "high")
            ):
                raise ModelRequestError(
                    f"{field_name} must be low, medium, high, or None"
                )


@runtime_checkable
class VideoModel(Protocol):
    """A self-hosted backend that returns one validated structured result."""

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        """Run one visual-only inference stage."""


def parse_strict_json(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_MODEL_OUTPUT_CHARS,
) -> dict[str, Any]:
    """Parse exactly one JSON object, optionally wrapped in one JSON fence.

    Model output is deliberately not recovered from surrounding prose.  A
    repair prompt needs to see malformed output as malformed rather than have
    a parser guess which embedded object was intended.
    """
    if not isinstance(text, str):
        raise ModelOutputError("model output must be text")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if len(text) > max_chars:
        raise ModelOutputError("model output exceeds the configured maximum size")

    payload = text.strip()
    if not payload:
        raise ModelOutputError("model output is empty")

    if payload.startswith("```") or payload.endswith("```"):
        fenced = _COMPLETE_JSON_FENCE.fullmatch(payload)
        if fenced is None:
            raise ModelOutputError("model output must use one complete JSON Markdown fence")
        payload = fenced.group("body")

    payload = payload.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(payload)
    except (ValueError, RecursionError):
        raise ModelOutputError("model output is not valid JSON") from None
    if payload[end:].strip():
        raise ModelOutputError("model output contains trailing prose or multiple JSON values")
    if not isinstance(value, dict):
        raise ModelOutputError("model output must be a JSON object")
    try:
        _require_finite_numbers(value)
    except RecursionError:
        raise ModelOutputError("model output is not valid JSON") from None
    return value


def _require_finite_numbers(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelOutputError("model output numeric fields must be finite")
        return
    if isinstance(value, dict):
        for child in value.values():
            _require_finite_numbers(child)
        return
    if isinstance(value, list):
        for child in value:
            _require_finite_numbers(child)
