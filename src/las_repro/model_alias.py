"""Stable local model-alias rules shared by API, storage, and workers."""

from __future__ import annotations

import re
from typing import Any


DEFAULT_MODEL_ALIAS = "qwen3-vl-8b-instruct"
_MODEL_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_model_alias(value: Any) -> str:
    """Return one opaque local alias, rejecting paths and remote model IDs."""
    if not isinstance(value, str) or _MODEL_ALIAS.fullmatch(value) is None:
        raise ValueError("model alias must be an opaque local name")
    return value
