"""Canonical validation for inference-job completion metrics."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

_MAX_JOB_METRICS_BYTES = 4096
_JOB_METRIC_KEYS = frozenset(
    {
        "inference_seconds",
        "input_tokens",
        "output_tokens",
        "peak_allocated_bytes",
        "processed_frames",
        "execution_chunk_frames",
        "entity_prompts",
        "track_count",
        "cache_hit",
        "oom_retry",
        "semantic_cache_hit",
        "semantic_cache_published",
        "semantic_cache_key",
        "semantic_cache_result_sha256",
    }
)
_JOB_BOOLEAN_METRIC_KEYS = frozenset(
    {"cache_hit", "oom_retry", "semantic_cache_hit", "semantic_cache_published"}
)
_JOB_DIGEST_METRIC_KEYS = frozenset(
    {"semantic_cache_key", "semantic_cache_result_sha256"}
)
_JOB_INTEGER_METRIC_KEYS = (
    _JOB_METRIC_KEYS - _JOB_BOOLEAN_METRIC_KEYS - _JOB_DIGEST_METRIC_KEYS
    - {"inference_seconds"}
)


def validate_inference_job_metrics(metrics: Mapping[str, Any] | None) -> None:
    """Validate completion metrics with the store-era canonical rules."""
    if metrics is None:
        return
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping or None")
    values = dict(metrics)
    unknown = values.keys() - _JOB_METRIC_KEYS
    if unknown:
        raise ValueError("metrics contain unknown keys")
    if "inference_seconds" in values:
        duration = values["inference_seconds"]
        if type(duration) is not float or not math.isfinite(duration) or duration < 0:
            raise ValueError("metrics inference_seconds must be a finite non-negative float")
    for key in _JOB_INTEGER_METRIC_KEYS & values.keys():
        value = values[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"metrics {key} must be a non-negative integer")
    for key in _JOB_BOOLEAN_METRIC_KEYS & values.keys():
        if type(values[key]) is not bool:
            raise ValueError(f"metrics {key} must be a boolean")
    for key in _JOB_DIGEST_METRIC_KEYS & values.keys():
        if (not isinstance(values[key], str)
                or re.fullmatch(r"[0-9a-f]{64}", values[key]) is None):
            raise ValueError(f"metrics {key} must be a lowercase SHA256 digest")
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded.encode("utf-8")) > _MAX_JOB_METRICS_BYTES:
        raise ValueError("metrics canonical JSON exceeds 4096 bytes")
