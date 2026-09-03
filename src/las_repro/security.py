"""Authentication and non-mutating sensitive-value redaction helpers."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException, Request, status

from .config import Settings


_SENSITIVE_KEY_PARTS = ("key", "token", "authorization", "secret", "password")


def verify_bearer(request: Request, settings: Settings) -> None:
    """Require a valid bearer token without retaining or comparing its plaintext."""
    authorization = request.headers.get("Authorization")
    if authorization is None:
        _raise_unauthorized()

    scheme, separator, token = authorization.partition(" ")
    if (
        scheme.lower() != "bearer"
        or not separator
        or not token
        or token != token.strip()
        or any(character.isspace() for character in token)
    ):
        _raise_unauthorized()

    supplied_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expected_hash = settings.api_key_sha256.lower()
    if not hmac.compare_digest(supplied_hash, expected_hash):
        _raise_unauthorized()


def redact(value: Any) -> Any:
    """Return a recursively redacted copy of mappings and lists.

    Sensitive field detection is deliberately based on names, never values, so
    callers may safely retain ordinary user-facing strings and structured data.
    """
    if isinstance(value, Mapping):
        return {
            key: "***" if _is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower()
    return any(marker in normalized for marker in _SENSITIVE_KEY_PARTS)


def _raise_unauthorized() -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )
