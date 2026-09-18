"""Bounded visual-only adapter for any OpenAI-compatible chat completions API.

One backend covers every provider that speaks the OpenAI chat completions
surface: Doubao ARK, DashScope, OpenAI, Gemini's compatibility layer, and any
local model served by vLLM or SGLang. Provider quirks are absorbed by two
pass-through settings (``extra_body``, ``extra_headers``) instead of vendor
adapters.
"""

from __future__ import annotations

import base64
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import httpx

from ..media import FrameRef, TimeSpan, extract_frames
from ..model_alias import validate_model_alias
from .base import ModelOutputError, ModelRequest, parse_strict_json
from .response_contract import canonical

STAGE_MAX_OUTPUT_TOKENS = {
    "active_objects": 1_024,
    "general_segment": 4_096,
    "general_summary": 2_048,
    "embodied_pass_a": 4_096,
    "embodied_pass_b": 8_192,
    "embodied_enrichment": 4_096,
    "scene_semantics": 8_192,
    "occlusion_semantics": 4_096,
}
RESPONSE_FORMAT_MODES = ("json_schema", "json_object", "none")
# Overriding these through extra_body would silently invalidate the semantic
# cache identity or replace the visual payload; providers never need them.
_RESERVED_BODY_KEYS = frozenset({"model", "messages", "stream", "max_tokens", "response_format"})
FrameExtractor = Callable[[Path, TimeSpan, float, Path], list[FrameRef]]


class OpenAICompatBackendError(RuntimeError):
    """A sanitized transport, service, or configuration failure."""


class OpenAICompatVideoModel:
    """Send explicitly sampled JPEG frames to one OpenAI-compatible endpoint."""

    supports_semantic_result_cache = True
    # Bump these contracts when payload defaults or frame encoding/sampling change.
    adapter_contract_version = "openai-compat-chat-completions-v1"
    frame_extraction_contract_version = "extract-frames-jpeg-timestamps-v1"

    def __init__(self, *, base_url: str, api_key: str,
                 model_registry: Mapping[str, str],
                 response_format: str = "json_object",
                 extra_body: Mapping[str, Any] | None = None,
                 extra_headers: Mapping[str, str] | None = None,
                 timeout_seconds: float = 180.0, max_frames: int = 128,
                 max_request_bytes: int = 32 * 1024 * 1024,
                 max_output_chars: int = 1_000_000, proxy: str | None = None,
                 transport: httpx.BaseTransport | None = None,
                 client: httpx.Client | None = None,
                 frame_extractor: FrameExtractor = extract_frames) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise OpenAICompatBackendError("API credentials are not configured")
        if (not isinstance(base_url, str)
                or not base_url.startswith(("https://", "http://"))):
            raise OpenAICompatBackendError("base_url must be an http(s) URL")
        if response_format not in RESPONSE_FORMAT_MODES:
            raise OpenAICompatBackendError("response_format mode is not supported")
        extra_body = dict(extra_body or {})
        reserved = set(extra_body) & _RESERVED_BODY_KEYS
        if reserved:
            raise OpenAICompatBackendError(
                f"extra_body must not override reserved keys: {sorted(reserved)}"
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for name, value in (("max_frames", max_frames), ("max_request_bytes", max_request_bytes),
                            ("max_output_chars", max_output_chars)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._key = api_key
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._registry = dict(model_registry)
        self.response_format = response_format
        self._extra_body = extra_body
        self._extra_headers = dict(extra_headers or {})
        self.timeout_seconds, self.max_frames = timeout_seconds, max_frames
        self.max_request_bytes, self.max_output_chars = max_request_bytes, max_output_chars
        self._extract = frame_extractor
        if client is not None and transport is not None:
            raise ValueError("client and transport are mutually exclusive")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds, proxy=proxy,
                                               transport=transport, trust_env=False,
                                               follow_redirects=False)
        self._metrics: dict[str, int] = {}

    def semantic_cache_identity(self, request: ModelRequest) -> dict[str, Any]:
        """Effective request settings plus explicitly recorded, ignored hints."""
        return {
            "adapter_contract_version": self.adapter_contract_version,
            "frame_extraction_contract_version": self.frame_extraction_contract_version,
            "endpoint": self._endpoint,
            "model_alias": request.model_name,
            "resolved_model_id": self._registry[request.model_name],
            "max_frames": self.max_frames,
            "max_request_bytes": self.max_request_bytes,
            "max_output_chars": self.max_output_chars,
            "max_output_tokens": STAGE_MAX_OUTPUT_TOKENS[request.stage],
            "response_format": {
                "mode": self.response_format,
                "contract": (request.response_contract.cache_identity()
                             if request.response_contract is not None else None),
            },
            "extra_body": canonical(self._extra_body),
            "stream": False,
            "recorded_hints": {
                "media_resolution": request.media_resolution,
                "reasoning_effort": request.reasoning_effort,
                "clip_context": request.clip_context,
            },
        }

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        self._metrics = {}
        if request.stage not in STAGE_MAX_OUTPUT_TOKENS:
            raise OpenAICompatBackendError("stage has no configured output budget")
        try:
            alias = validate_model_alias(request.model_name)
            model_id = self._registry[alias]
        except (ValueError, KeyError, TypeError):
            raise OpenAICompatBackendError("model alias is not allowlisted") from None
        temporary = Path(tempfile.mkdtemp(prefix="percept-frames-"))
        try:
            try:
                frames = self._extract(request.video_path, request.span, request.fps, temporary)
                content = self._content(request, frames)
            except OpenAICompatBackendError:
                raise
            except Exception:
                raise OpenAICompatBackendError("visual preparation failed") from None
            payload: dict[str, Any] = {
                "model": model_id, "stream": False,
                "max_tokens": STAGE_MAX_OUTPUT_TOKENS[request.stage],
                "messages": [{"role": "user", "content": content}],
            }
            if self.response_format == "json_schema" and request.response_contract is not None:
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": request.response_contract.name, "strict": True,
                    "schema": json.loads(request.response_contract.schema_json)}}
            elif self.response_format in ("json_schema", "json_object"):
                payload["response_format"] = {"type": "json_object"}
            payload.update(self._extra_body)
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if len(encoded) > self.max_request_bytes:
                raise OpenAICompatBackendError("request exceeds configured size limit")
            try:
                with self._client.stream("POST", self._endpoint,
                    headers={"Authorization": f"Bearer {self._key}",
                             "Content-Type": "application/json", **self._extra_headers},
                    content=encoded, timeout=self.timeout_seconds,
                    follow_redirects=False) as response:
                    if response.status_code != 200:
                        raise OpenAICompatBackendError(
                            f"service request failed with status {response.status_code}"
                        )
                    raw = self._bounded_body(response)
            except (OpenAICompatBackendError, ModelOutputError):
                raise
            except (httpx.HTTPError, OSError):
                raise OpenAICompatBackendError("transport failed") from None
            return self._parse(raw)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _content(self, request: ModelRequest, frames: list[FrameRef]) -> list[dict[str, Any]]:
        if not frames or len(frames) > self.max_frames:
            raise OpenAICompatBackendError("frame count is outside configured bounds")
        previous = -math.inf
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        for frame in frames:
            if (not math.isfinite(frame.timestamp) or frame.timestamp < request.span.start
                    or frame.timestamp >= request.span.end or frame.timestamp <= previous):
                raise OpenAICompatBackendError("frame timestamps are invalid")
            previous = frame.timestamp
            try:
                jpeg = frame.path.read_bytes()
            except OSError:
                raise OpenAICompatBackendError("extracted frame is unavailable") from None
            content.extend((
                {"type": "text", "text": f"Frame timestamp: {frame.timestamp:.6f} seconds"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}},
            ))
        return content

    def _bounded_body(self, response: httpx.Response) -> bytes:
        parts, size = [], 0
        for part in response.iter_bytes():
            size += len(part)
            if size > self.max_output_chars * 4 + 1_000_000:
                raise ModelOutputError("response exceeds configured size limit")
            parts.append(part)
        return b"".join(parts)

    def _parse(self, raw: bytes) -> dict[str, Any]:
        try:
            envelope = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelOutputError("response is not valid JSON") from None
        if not isinstance(envelope, dict):
            raise ModelOutputError("response envelope must be an object")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ModelOutputError("response must contain exactly one choice")
        choice = choices[0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise ModelOutputError(
                f"response finished abnormally: {choice.get('finish_reason')!r}"
            )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ModelOutputError("response must contain one assistant message")
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise ModelOutputError("response message content must be non-empty text")
        metrics: dict[str, int] = {}
        usage = envelope.get("usage")
        if isinstance(usage, dict):
            for source, target in (("prompt_tokens", "input_tokens"),
                                   ("completion_tokens", "output_tokens")):
                value = usage.get(source)
                if value is None:
                    continue  # some compatible servers report partial usage
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ModelOutputError("response usage is invalid")
                metrics[target] = value
        result = parse_strict_json(text, max_chars=self.max_output_chars,
                                   forbid_duplicate_keys=True)
        self._metrics = metrics
        return result

    def request_metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def release_request(self, request: ModelRequest) -> None:
        self._metrics = {}

    def release_video_session(self, session_id: str) -> None:
        return None

    def close(self) -> None:
        self._metrics = {}
        if self._owns_client:
            self._client.close()
