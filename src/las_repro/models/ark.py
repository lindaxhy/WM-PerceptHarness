"""Bounded visual-only adapter for the Volcengine ARK Responses API."""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import httpx

from ..media import FrameRef, TimeSpan, extract_frames
from ..model_alias import validate_model_alias
from .base import ModelOutputError, ModelRequest

ARK_RESPONSES_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/responses"
ARK_STAGE_MAX_OUTPUT_TOKENS = {
    "active_objects": 1_024,
    "general_segment": 4_096,
    "general_summary": 2_048,
    "embodied_pass_a": 4_096,
    "embodied_pass_b": 8_192,
    "embodied_enrichment": 4_096,
    "scene_semantics": 8_192,
    "occlusion_semantics": 4_096,
}
_MEDIA_MAX_PIXELS = {"low": 65_536, "medium": 131_072, "high": 262_144}
FrameExtractor = Callable[[Path, TimeSpan, float, Path], list[FrameRef]]


class ArkBackendError(RuntimeError):
    """A sanitized ARK transport, service, or configuration failure."""


class ArkVideoModel:
    """Send explicitly sampled JPEG frames to one allowlisted ARK model."""

    supports_semantic_result_cache = True
    # Bump these contracts when payload defaults or frame encoding/sampling change.
    adapter_contract_version = "ark-responses-scene-choices-v3"
    frame_extraction_contract_version = "extract-frames-jpeg-timestamps-v1"

    def semantic_cache_identity(self, request: ModelRequest) -> dict[str, Any]:
        """Effective request settings plus explicitly recorded, ignored hints."""
        return {
            "adapter_contract_version": self.adapter_contract_version,
            "response_format": (request.response_contract.cache_identity()
                                if request.response_contract is not None else None),
            "frame_extraction_contract_version": self.frame_extraction_contract_version,
            "model_alias": request.model_name,
            "resolved_model_id": self._registry[request.model_name],
            "max_frames": self.max_frames,
            "ark_max_request_bytes": self.max_request_bytes,
            "ark_max_output_chars": self.max_output_chars,
            "max_output_tokens": ARK_STAGE_MAX_OUTPUT_TOKENS[request.stage],
            "thinking": {"type": "disabled"},
            "store": False,
            "stream": False,
            "image_pixel_limit": None if request.media_resolution is None else {
                "min_pixels": 4096,
                "max_pixels": _MEDIA_MAX_PIXELS[request.media_resolution],
            },
            "media_resolution": request.media_resolution,
            "recorded_hints": {
                "reasoning_effort": request.reasoning_effort,
                "clip_context": request.clip_context,
            },
        }

    def __init__(self, *, api_key: str, model_registry: Mapping[str, str],
                 timeout_seconds: float = 180.0, max_frames: int = 128,
                 max_request_bytes: int = 32 * 1024 * 1024,
                 max_output_chars: int = 1_000_000, proxy: str | None = None,
                 transport: httpx.BaseTransport | None = None,
                 client: httpx.Client | None = None,
                 frame_extractor: FrameExtractor = extract_frames) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ArkBackendError("ARK credentials are not configured")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        for name, value in (("max_frames", max_frames), ("max_request_bytes", max_request_bytes),
                            ("max_output_chars", max_output_chars)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._key = api_key
        self._registry = dict(model_registry)
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

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        self._metrics = {}
        if request.stage not in ARK_STAGE_MAX_OUTPUT_TOKENS:
            raise ArkBackendError("ARK stage has no configured output budget")
        max_output_tokens = ARK_STAGE_MAX_OUTPUT_TOKENS[request.stage]
        try:
            alias = validate_model_alias(request.model_name)
            model_id = self._registry[alias]
        except (ValueError, KeyError, TypeError):
            raise ArkBackendError("ARK model alias is not allowlisted") from None
        temporary = Path(tempfile.mkdtemp(prefix="las-ark-frames-"))
        try:
            try:
                frames = self._extract(request.video_path, request.span, request.fps, temporary)
                content = self._content(request, frames)
            except ArkBackendError:
                raise
            except Exception:
                raise ArkBackendError("ARK visual preparation failed") from None
            payload = {"model": model_id, "store": False, "stream": False,
                       "thinking": {"type": "disabled"},
                       "max_output_tokens": max_output_tokens,
                       "input": [{"role": "user", "content": content}]}
            if request.response_contract is not None:
                payload["text"] = {"format": request.response_contract.format()}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if len(encoded) > self.max_request_bytes:
                raise ArkBackendError("ARK request exceeds configured size limit")
            try:
                with self._client.stream("POST", ARK_RESPONSES_ENDPOINT,
                    headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                    content=encoded, timeout=self.timeout_seconds,
                    follow_redirects=False) as response:
                    if response.status_code != 200:
                        raise ArkBackendError("ARK service request failed")
                    raw = self._bounded_body(response)
            except ArkBackendError:
                raise
            except (httpx.HTTPError, OSError):
                raise ArkBackendError("ARK transport failed") from None
            return self._parse(raw, model_id)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _content(self, request: ModelRequest, frames: list[FrameRef]) -> list[dict[str, Any]]:
        if not frames or len(frames) > self.max_frames:
            raise ArkBackendError("ARK frame count is outside configured bounds")
        previous = -math.inf
        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.prompt}]
        for frame in frames:
            if (not math.isfinite(frame.timestamp) or frame.timestamp < request.span.start
                    or frame.timestamp >= request.span.end or frame.timestamp <= previous):
                raise ArkBackendError("ARK frame timestamps are invalid")
            previous = frame.timestamp
            try: jpeg = frame.path.read_bytes()
            except OSError: raise ArkBackendError("ARK extracted frame is unavailable") from None
            visual: dict[str, Any] = {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            }
            if request.media_resolution is not None:
                visual["image_pixel_limit"] = {
                    "min_pixels": 4_096,
                    "max_pixels": _MEDIA_MAX_PIXELS[request.media_resolution],
                }
            content.extend((
                {"type": "input_text", "text": f"Frame timestamp: {frame.timestamp:.6f} seconds"},
                visual,
            ))
        return content

    def _bounded_body(self, response: httpx.Response) -> bytes:
        parts, size = [], 0
        for part in response.iter_bytes():
            size += len(part)
            if size > self.max_output_chars * 4 + 1_000_000:
                raise ModelOutputError("ARK response exceeds configured size limit")
            parts.append(part)
        return b"".join(parts)

    def _parse(self, raw: bytes, model_id: str) -> dict[str, Any]:
        try: envelope = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ModelOutputError("ARK response is not valid JSON") from None
        if not isinstance(envelope, dict) or envelope.get("status") != "completed" or envelope.get("model") != model_id:
            raise ModelOutputError("ARK response identity or status is invalid")
        output = envelope.get("output")
        if not isinstance(output, list):
            raise ModelOutputError("ARK response output must be a list")
        messages = [item for item in output if isinstance(item, dict) and item.get("type") == "message"]
        if len(messages) != 1 or messages[0].get("role") != "assistant" or messages[0].get("status") not in (None, "completed"):
            raise ModelOutputError("ARK response must contain one completed assistant message")
        content = messages[0].get("content")
        if not isinstance(content, list):
            raise ModelOutputError("ARK response message content must be a list")
        outputs = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "output_text"]
        if len(outputs) != 1:
            raise ModelOutputError("ARK response must contain one output text")
        usage = envelope.get("usage")
        if not isinstance(usage, dict): raise ModelOutputError("ARK usage is invalid")
        metrics = {}
        for key in ("input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelOutputError("ARK usage is invalid")
            metrics[key] = value
        result = _strict_no_duplicates(outputs[0], self.max_output_chars)
        self._metrics = metrics
        return result

    def request_metrics(self) -> dict[str, int]: return dict(self._metrics)
    def release_request(self, request: ModelRequest) -> None: self._metrics = {}
    def release_video_session(self, session_id: str) -> None: return None
    def close(self) -> None:
        self._metrics = {}
        if self._owns_client: self._client.close()


def _strict_no_duplicates(text: Any, max_chars: int) -> dict[str, Any]:
    if not isinstance(text, str) or len(text) > max_chars:
        raise ModelOutputError("model output exceeds the configured maximum size")
    payload = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?```", payload,
                          flags=re.IGNORECASE | re.DOTALL)
    if payload.startswith("```") or payload.endswith("```"):
        if fenced is None:
            raise ModelOutputError("model output must use one complete JSON Markdown fence")
        payload = fenced.group("body")
    payload = payload.lstrip()
    if not payload:
        raise ModelOutputError("model output is empty")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result: raise ModelOutputError("model output contains duplicate keys")
            result[key] = value
        return result
    try:
        value, end = json.JSONDecoder(object_pairs_hook=pairs).raw_decode(payload)
    except ModelOutputError:
        raise
    except (ValueError, RecursionError):
        raise ModelOutputError("model output is not valid JSON") from None
    if payload[end:].strip():
        raise ModelOutputError("model output contains trailing prose or multiple JSON values")
    if not isinstance(value, dict):
        raise ModelOutputError("model output must be a JSON object")
    try:
        _require_finite(value)
    except RecursionError:
        raise ModelOutputError("model output is not valid JSON") from None
    return value


def _require_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ModelOutputError("model output numeric fields must be finite")
    if isinstance(value, dict):
        for child in value.values(): _require_finite(child)
    elif isinstance(value, list):
        for child in value: _require_finite(child)
