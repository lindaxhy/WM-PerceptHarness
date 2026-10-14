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
import logging
import math
import random
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

from ..media import FrameRef, TimeSpan, extract_frames
from .base import ModelOutputError, ModelRequest, parse_strict_json
from .response_contract import canonical

# Provider statuses worth one more attempt: throttling, and transient
# service-side failures. 4xx besides 429 are deterministic and never retried.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Payload thinning floor: below this frame count a request that still exceeds
# the byte budget is a genuine configuration problem, not a dense clip.
_MIN_FRAMES = 8
# Overriding these through extra_body would silently invalidate the semantic
# cache identity or replace the visual payload; providers never need them.
_RESERVED_BODY_KEYS = frozenset({"model", "messages", "stream", "max_tokens", "response_format"})
FrameExtractor = Callable[[Path, TimeSpan, float, Path], list[FrameRef]]


class OpenAICompatBackendError(RuntimeError):
    """A sanitized transport, service, or configuration failure."""


class OpenAICompatVideoModel:
    """Send sampled frames using connection values validated by Settings/CLI.

    Direct Python callers must provide valid connection settings. The adapter
    validates provider-specific body overrides and all external responses.
    """

    TRANSPORT_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 5.0
    MAX_REQUEST_BYTES = 32 * 1024 * 1024
    MAX_OUTPUT_CHARS = 1_000_000
    OUTPUT_TOKENS = {
        "confidence_draft": 16384, "grounded_draft": 16384, "evidence_review": 8192,
        "sam_prompt_review": 4096, "grounded_revision": 16384,
    }

    supports_semantic_result_cache = True
    # Bump these contracts when payload defaults or frame encoding/sampling change.
    adapter_contract_version = "openai-compat-chat-completions-v1"
    frame_extraction_contract_version = "extract-frames-source-pts-v2"

    def __init__(self, *, base_url: str, api_key: str,
                 model_name: str,
                 response_format: str = "json_object",
                 extra_body: Mapping[str, Any] | None = None,
                 extra_headers: Mapping[str, str] | None = None,
                 timeout_seconds: float = 180.0,
                 proxy: str | None = None,
                 transport: httpx.BaseTransport | None = None,
                 frame_extractor: FrameExtractor = extract_frames) -> None:
        extra_body = dict(extra_body or {})
        reserved = set(extra_body) & _RESERVED_BODY_KEYS
        if reserved:
            raise OpenAICompatBackendError(
                f"extra_body must not override reserved keys: {sorted(reserved)}"
            )
        self._key = api_key
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model_name = model_name
        self.response_format = response_format
        self._extra_body = extra_body
        self._extra_headers = dict(extra_headers or {})
        self.timeout_seconds = timeout_seconds
        self._extract = frame_extractor
        self._client = httpx.Client(timeout=timeout_seconds, proxy=proxy,
                                    transport=transport, trust_env=False,
                                    follow_redirects=False)
        self._metrics: dict[str, int] = {}


    def semantic_cache_identity(self, request: ModelRequest) -> dict[str, Any]:
        """Effective settings that determine the model request."""
        return {
            "adapter_contract_version": self.adapter_contract_version,
            "frame_extraction_contract_version": self.frame_extraction_contract_version,
            "endpoint": self._endpoint,
            "model_alias": request.model_name,
            "resolved_model_id": self.model_name,
            "max_frames": request.max_frames,
            "max_request_bytes": self.MAX_REQUEST_BYTES,
            "max_output_chars": self.MAX_OUTPUT_CHARS,
            "max_output_tokens": (request.max_output_tokens or self.OUTPUT_TOKENS[request.stage]),
            "response_format": {
                "mode": self.response_format,
                "contract": (request.response_contract.cache_identity()
                             if request.response_contract is not None else None),
            },
            "extra_body": canonical(self._extra_body),
            "stream": False,
        }

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        self._metrics = {}
        if request.max_output_tokens is None and request.stage not in self.OUTPUT_TOKENS:
            raise OpenAICompatBackendError("stage has no configured output budget")
        temporary = Path(tempfile.mkdtemp(prefix="percept-frames-"))
        try:
            span_seconds = max(0.0, request.span.end - request.span.start)
            evidence_count = len(request.visual_evidence)
            original_budget = request.max_frames - evidence_count
            if original_budget < 1:
                raise OpenAICompatBackendError('mandatory evidence exceeds frame budget')
            fps = request.fps
            if span_seconds > 0 and fps * span_seconds > original_budget:
                # A long clip at the requested cadence would exceed max_frames
                # and previously failed outright. Spread the frame budget
                # evenly across the span instead (0.5 guards the < end loop
                # boundary against rounding up to max_frames + 1).
                fps = (original_budget - 0.5) / span_seconds
            try:
                frames = self._extract(request.video_path, request.span, fps, temporary)
                content = self._content(request, frames)
            except OpenAICompatBackendError:
                raise
            except Exception:
                raise OpenAICompatBackendError("visual preparation failed") from None
            payload: dict[str, Any] = {
                "model": self.model_name, "stream": False,
                "max_tokens": (request.max_output_tokens or self.OUTPUT_TOKENS[request.stage]),
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
            while len(encoded) > self.MAX_REQUEST_BYTES and len(frames) > _MIN_FRAMES:
                # High-bitrate sources can exceed the request budget at any
                # frame count. Thin the sampled frames evenly (keeping the
                # first frame) instead of failing the whole task.
                keep = max(_MIN_FRAMES, int(len(frames) * self.MAX_REQUEST_BYTES
                                            / len(encoded) * 0.9))
                if keep >= len(frames):
                    keep = len(frames) - 1
                step = (len(frames) - 1) / (keep - 1) if keep > 1 else 1
                frames = [frames[round(i * step)] for i in range(keep)]
                payload["messages"] = [
                    {"role": "user", "content": self._content(request, frames)}
                ]
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if len(encoded) > self.MAX_REQUEST_BYTES:
                raise OpenAICompatBackendError("request exceeds configured size limit")
            raw = self._post_with_retries(encoded)
            return self._parse(raw)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _post_with_retries(self, encoded: bytes) -> bytes:
        """POST once, then retry transient transport/service failures with backoff.

        Deterministic failures (non-retryable statuses, oversized or malformed
        responses) raise immediately. The final attempt's failure is raised
        with its underlying cause preserved for diagnosis.
        """
        attempts = self.TRANSPORT_RETRIES + 1
        for attempt in range(attempts):
            retryable: str | None = None
            cause: BaseException | None = None
            try:
                with self._client.stream("POST", self._endpoint,
                    headers={"Authorization": f"Bearer {self._key}",
                             "Content-Type": "application/json", **self._extra_headers},
                    content=encoded, timeout=self.timeout_seconds,
                    follow_redirects=False) as response:
                    if response.status_code == 200:
                        return self._bounded_body(response)
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        retryable = f"service status {response.status_code}"
                    else:
                        raise OpenAICompatBackendError(
                            f"service request failed with status {response.status_code}"
                        )
            except (OpenAICompatBackendError, ModelOutputError):
                raise
            except (httpx.TimeoutException, httpx.TransportError) as error:
                retryable = f"transport {type(error).__name__}"
                cause = error
            except (httpx.HTTPError, OSError) as error:
                raise OpenAICompatBackendError("transport failed") from error
            if attempt + 1 == attempts:
                message = f"service request failed after {attempts} attempts: {retryable}"
                raise OpenAICompatBackendError(message) from cause
            delay = self.RETRY_BACKOFF_SECONDS * (2 ** attempt) * (1 + random.random())
            logger.warning(
                "retrying request (%s; attempt %d/%d) in %.1fs",
                retryable, attempt + 1, attempts, delay,
            )
            time.sleep(delay)
        raise OpenAICompatBackendError("transport failed")  # pragma: no cover

    def _content(self, request: ModelRequest, frames: list[FrameRef]) -> list[dict[str, Any]]:
        if not frames or len(frames) + len(request.visual_evidence) > request.max_frames:
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
        for evidence in request.visual_evidence:
            content.extend((
                {'type': 'text', 'text': 'SAM overlay metadata: ' + json.dumps(evidence.identity(), sort_keys=True)},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(evidence.png_bytes).decode('ascii')}}))
        return content

    def _bounded_body(self, response: httpx.Response) -> bytes:
        parts, size = [], 0
        for part in response.iter_bytes():
            size += len(part)
            if size > self.MAX_OUTPUT_CHARS * 4 + 1_000_000:
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
        result = parse_strict_json(text, max_chars=self.MAX_OUTPUT_CHARS,
                                   forbid_duplicate_keys=True)
        self._metrics = metrics
        return result

    def request_metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def close(self) -> None:
        self._metrics = {}
        self._client.close()
