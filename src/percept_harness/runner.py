"""Synchronous video evaluation runner — no service, no task queue, no workers."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import result_store
from .config import Settings
from .domain import TaskRecord
from .execution import SyncJobStore
from .models.base import VideoModel
from .pipelines.base import PipelineContext, SafePipelineError
from .pipelines.embodied import EmbodiedActionPipeline

_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


@dataclass(frozen=True)
class EvalOutcome:
    """The terminal record of one evaluated video."""

    video_path: Path
    status: str  # "completed" | "failed" | "skipped"
    data: dict[str, Any] | None = None
    error: str | None = None
    elapsed_seconds: float | None = None
    result_file: str | None = None
    metadata: dict[str, Any] | None = None


class SyncRunner:
    """Evaluate videos one at a time in the current process."""

    def __init__(
        self,
        model: VideoModel,
        settings: Settings,
        *,
        model_alias: str,
        cv_executor=None,
        work_root: Path | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.model_alias = model_alias
        self.cv_executor = cv_executor
        self.work_root = work_root if work_root is not None else settings.work_root
        self.semantic_cache = None
        if settings.semantic_result_cache and getattr(
            model, "supports_semantic_result_cache", False
        ):
            from .execution import SemanticResultCache

            self.semantic_cache = SemanticResultCache(
                Path(self.work_root) / "semantic-cache"
            )

    def evaluate(
        self,
        video_path: Path,
        *,
        prompt_context: str | None = None,
    ) -> EvalOutcome:
        """Evaluate one video end-to-end and return its terminal outcome."""
        started = time.monotonic()
        video_path = Path(video_path)
        context = None
        try:
            from .models.fake import FakeVideoModel
            if self.settings.cv_provider == 'disabled':
                raise SafePipelineError('Annotation requires SAM; disabled CV is not supported')
            if self.settings.cv_provider == 'fake' and not isinstance(self.model, FakeVideoModel):
                raise SafePipelineError('Fake CV is allowed only with an explicit fake model')
            resolved = video_path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError("video path must be a regular file")
            task = _task_record(resolved, self.model_alias,
                                prompt_context=prompt_context)
            pipeline = EmbodiedActionPipeline()
            store = SyncJobStore(
                self.model,
                default_model_alias=self.model_alias,
                cv_executor=self.cv_executor,
                semantic_cache=self.semantic_cache,
            )
            try:
                with _task_dir(self.work_root, task.task_id) as task_dir:
                    context = PipelineContext(
                        store=store,
                        media_resolver=None,
                        settings=self.settings,
                        task_dir=task_dir,
                        media_path=resolved,
                    )
                    result = pipeline.run(task, context)
            finally:
                store.close()
            if not isinstance(result, Mapping):
                raise TypeError("pipeline result must be a mapping")
            return EvalOutcome(
                video_path=resolved,
                status="completed",
                data=dict(result),
                metadata=dict(context.metadata),
                elapsed_seconds=time.monotonic() - started,
            )
        except SafePipelineError as error:
            return EvalOutcome(
                video_path=video_path,
                status="failed",
                metadata=dict(context.metadata) if context is not None else None,
                error=str(error),
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception as error:
            return EvalOutcome(
                video_path=video_path,
                status="failed",
                metadata=dict(context.metadata) if context is not None else None,
                error=f"{type(error).__name__}: {error}",
                elapsed_seconds=time.monotonic() - started,
            )


def collect_videos(sources: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated video list."""
    videos: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        source = Path(source)
        if source.is_dir():
            candidates = sorted(
                child for child in source.rglob("*")
                if child.is_file() and child.suffix.lower() in _VIDEO_SUFFIXES
            )
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"video source does not exist: {source}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                videos.append(resolved)
    return videos


def run_batch(
    runner: SyncRunner,
    videos: list[Path],
    output_dir: Path,
    *,
    prompt_context: str | None = None,
    force: bool = False,
    log=print,
) -> list[EvalOutcome]:
    """Skip completed valid results by default; force explicitly reruns them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        runner.semantic_cache = None
    outcomes = []
    for index, source in enumerate(videos, start=1):
        video = Path(source).resolve()
        path = result_store.result_path(output_dir, video)
        previous = result_store.completed_result(path, video) if not force and video.is_file() else None
        if previous is not None:
            outcome = EvalOutcome(video_path=video, status="skipped", data=previous["data"],
                                  result_file=path.name, metadata=previous.get("metadata"))
            log(f"[{index}/{len(videos)}] skip {video.name} (completed result)")
        else:
            log(f"[{index}/{len(videos)}] eval {video.name} ...")
            outcome = replace(runner.evaluate(video, prompt_context=prompt_context), result_file=path.name)
            _write_result(path, outcome)
            log(f"[{index}/{len(videos)}] {outcome.status} {video.name} -> {path.name}")
        outcomes.append(outcome)
    _write_aggregate(output_dir / "results.jsonl", outcomes)
    return outcomes


def _task_record(
    video_path: Path,
    model_alias: str,
    *,
    prompt_context: str | None,
) -> TaskRecord:
    payload: dict[str, Any] = {
        "video_url": str(video_path),
        "model_name": model_alias,
    }
    if prompt_context is not None:
        payload["task_context"] = {"prompt_context": prompt_context}
    return TaskRecord(
        task_id=str(uuid.uuid4()),
        operator_id="percept_eval",
        operator_version="v1",
        payload=payload,
    )


@contextmanager
def _task_dir(work_root: Path, task_id: str) -> Iterator[Path]:
    work_root.mkdir(parents=True, exist_ok=True)
    root = work_root.resolve(strict=True)
    destination = root / task_id
    destination.mkdir(exist_ok=True)
    try:
        yield destination
    finally:
        shutil.rmtree(destination, ignore_errors=True)


def _write_result(result_path: Path, outcome: EvalOutcome) -> None:
    record = {
        "video_path": str(outcome.video_path),
        "status": outcome.status,
        "data": outcome.data,
        "metadata": outcome.metadata,
        "error": outcome.error,
        "elapsed_seconds": outcome.elapsed_seconds,
        "result_file": outcome.result_file,
    }
    result_store.write_annotation(result_path, record)


def _write_aggregate(jsonl_path: Path, outcomes: list[EvalOutcome]) -> None:
    lines = [json.dumps({
        "video_path": str(outcome.video_path),
        "status": outcome.status,
        "data": outcome.data,
        "metadata": outcome.metadata,
        "error": outcome.error,
        "result_file": outcome.result_file,
    }, ensure_ascii=False, allow_nan=False) + "\n" for outcome in outcomes]
    result_store.atomic_text(jsonl_path, "".join(lines))
