"""Shared local video-model protocol and strict structured-output parsing."""

from __future__ import annotations

import json
import hashlib
import base64
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from .response_contract import ModelResponseContract

from ..media import FrameRef, TimeSpan, VideoMetadata


def validate_model_alias(value: Any) -> str:
    """Validate the local name used to select a configured provider model."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ValueError("model alias must be an opaque local name")
    return value


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
class VisualEvidence:
    """Immutable PNG bytes resolved through an authenticated CV artifact store."""
    png_bytes: bytes
    timestamp: float
    entity_id: str
    track_id: str
    source: str

    def __post_init__(self):
        if type(self.png_bytes) is not bytes or not self.png_bytes.startswith(b'\x89PNG\r\n\x1a\n') or len(self.png_bytes) > 8 * 1024 * 1024:
            raise ModelRequestError('visual evidence must be a bounded PNG')
        if type(self.timestamp) not in (int, float) or not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise ModelRequestError('visual evidence time must be finite original seconds')
        for value in (self.entity_id, self.track_id, self.source):
            if type(value) is not str or not value.strip() or len(value) > 256:
                raise ModelRequestError('invalid visual evidence identity')

    def identity(self):
        return {'sha256': hashlib.sha256(self.png_bytes).hexdigest(), 'timestamp': self.timestamp,
                'entity_id': self.entity_id, 'track_id': self.track_id, 'source': self.source}

    def payload(self):
        return {**self.identity(), 'png_base64': base64.b64encode(self.png_bytes).decode('ascii')}

    @classmethod
    def from_payload(cls, value):
        if type(value) is not dict or set(value) != {'sha256', 'timestamp', 'entity_id', 'track_id', 'source', 'png_base64'}:
            raise ModelRequestError('invalid visual evidence payload')
        encoded = value['png_base64']
        if type(encoded) is not str or len(encoded) > 12 * 1024 * 1024:
            raise ModelRequestError('invalid visual evidence byte budget')
        result = cls(png_bytes=base64.b64decode(encoded, validate=True), timestamp=value['timestamp'],
                     entity_id=value['entity_id'], track_id=value['track_id'], source=value['source'])
        if result.identity()['sha256'] != value['sha256']:
            raise ModelRequestError('visual evidence digest mismatch')
        return result


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
    model_name: str
    media_resolution: TuningLevel | None = None
    reasoning_effort: TuningLevel | None = None
    clip_context: TuningLevel | None = None
    video_session: VideoSession | None = None
    response_contract: ModelResponseContract | None = None
    visual_evidence: tuple[VisualEvidence, ...] = ()

    def __post_init__(self) -> None:
        if type(self.visual_evidence) is not tuple or len(self.visual_evidence) > 96 or any(type(e) is not VisualEvidence for e in self.visual_evidence):
            raise ModelRequestError('invalid visual evidence collection')
        if sum(len(e.png_bytes) for e in self.visual_evidence) > 64 * 1024 * 1024:
            raise ModelRequestError('visual evidence exceeds byte budget')
        if any(not self.span.start <= e.timestamp < self.span.end for e in self.visual_evidence):
            raise ModelRequestError('visual evidence outside original video span')
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
    forbid_duplicate_keys: bool = False,
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
    decoder = (json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
               if forbid_duplicate_keys else json.JSONDecoder())
    try:
        value, end = decoder.raw_decode(payload)
    except ModelOutputError:
        raise
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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelOutputError("model output contains duplicate keys")
        result[key] = value
    return result


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
