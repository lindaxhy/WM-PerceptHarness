"""Execution context shared by event annotation stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Settings
from ..runner import StageExecutor


class SafePipelineError(RuntimeError):
    """A stable, pre-sanitized pipeline failure safe for task persistence."""


@dataclass(frozen=True)
class PipelineContext:
    """Task-local dependencies supplied to one pipeline invocation."""

    executor: StageExecutor
    settings: Settings
    media_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
