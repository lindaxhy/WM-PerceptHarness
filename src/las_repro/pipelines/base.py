"""Shared pipeline protocol, execution context, and template registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config import Settings
from ..domain import TaskRecord
from ..media import MediaResolver
from ..store import SQLiteTaskStore


class SafePipelineError(RuntimeError):
    """A stable, pre-sanitized pipeline failure safe for task persistence."""


@dataclass(frozen=True)
class PipelineContext:
    """Task-local dependencies supplied to one pipeline invocation."""

    store: SQLiteTaskStore
    media_resolver: MediaResolver
    settings: Settings
    task_dir: Path
    media_path: Path | None = None


@runtime_checkable
class Pipeline(Protocol):
    """A template-specific orchestration pipeline."""

    def run(self, task: TaskRecord, context: PipelineContext) -> dict[str, Any]:
        """Run one claimed top-level task to a JSON-compatible result."""


PipelineFactory = Callable[[], Pipeline]


class PipelineRegistry:
    """Map persisted task templates to lazily-created local pipelines."""

    def __init__(self) -> None:
        self._factories: dict[str, PipelineFactory] = {}

    def register(self, template: str, factory: PipelineFactory) -> None:
        if not isinstance(template, str) or not template.strip():
            raise ValueError("pipeline template must be a non-blank string")
        if not callable(factory):
            raise TypeError("pipeline factory must be callable")
        if template in self._factories:
            raise ValueError(f"pipeline template {template!r} is already registered")
        self._factories[template] = factory

    def create(self, template: str) -> Pipeline:
        try:
            factory = self._factories[template]
        except KeyError:
            raise KeyError(f"pipeline template {template!r} is not registered") from None
        pipeline = factory()
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline factory must return an object with run(task, context)")
        return pipeline
