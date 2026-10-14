"""Local video-model adapters and their shared protocol."""

from .base import (
    ModelOutputError,
    ModelRequest,
    ModelRequestError,
    VideoModel,
    parse_strict_json,
)
from .fake import FakeVideoModel, UnknownModelStageError

__all__ = [
    "FakeVideoModel",
    "ModelOutputError",
    "ModelRequest",
    "ModelRequestError",
    "UnknownModelStageError",
    "VideoModel",
    "parse_strict_json",
]
