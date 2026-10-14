"""Local video-model adapters and their shared protocol."""

from .base import (
    ModelOutputError,
    ModelRequest,
    ModelRequestError,
    VideoModel,
    parse_strict_json,
)

__all__ = [
    "ModelOutputError",
    "ModelRequest",
    "ModelRequestError",
    "VideoModel",
    "parse_strict_json",
]
