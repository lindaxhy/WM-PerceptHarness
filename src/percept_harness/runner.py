"""Synchronous video evaluation runner — no service, no task queue, no workers."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import result_store
from .config import Settings
from .domain import TaskRecord
from .execution import SyncJobStore
from .models.base import VideoModel
from .pipelines.base import PipelineContext, PipelineRegistry, SafePipelineError

SUPPORTED_TEMPLATES = (
    "general_video_captioning",
    "embodied_active_object_detection",
    "embodied_action_captioning",
)

_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


@dataclass(frozen=True)
class EvalOutcome:
    """The terminal record of one evaluated video."""

    video_path: Path
    template: str
    status: str  # "completed" | "failed" | "skipped"
    data: dict[str, Any] | None = None
    error: str | None = None
    elapsed_seconds: float | None = None
    provenance: dict[str, Any] | None = None
    result_file: str | None = None


def default_pipeline_registry() -> PipelineRegistry:
    from .pipelines.embodied import EmbodiedActionPipeline, EmbodiedActiveObjectsPipeline
    from .pipelines.general import GeneralCaptionPipeline

    registry = PipelineRegistry()
    registry.register("general_video_captioning", GeneralCaptionPipeline)
    registry.register("embodied_active_object_detection", EmbodiedActiveObjectsPipeline)
    registry.register("embodied_action_captioning", EmbodiedActionPipeline)
    return registry


class SyncRunner:
    """Evaluate videos one at a time in the current process."""

    def __init__(
        self,
        model: VideoModel,
        settings: Settings,
        *,
        model_alias: str,
        registry: PipelineRegistry | None = None,
        cv_executor=None,
        work_root: Path | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.model_alias = model_alias
        self.custom_registry = registry is not None
        self.registry = registry if registry is not None else default_pipeline_registry()
        self.cv_executor = cv_executor
        self.work_root = work_root if work_root is not None else settings.work_root

    def evaluate(
        self,
        video_path: Path,
        template: str,
        *,
        prompt_context: str | None = None,
        query: str | None = None,
    ) -> EvalOutcome:
        """Evaluate one video end-to-end and return its terminal outcome."""
        started = time.monotonic()
        video_path = Path(video_path)
        try:
            resolved = video_path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError("video path must be a regular file")
            task = _task_record(resolved, template, self.model_alias,
                                prompt_context=prompt_context, query=query)
            pipeline = self.registry.create(template)
            store = SyncJobStore(
                self.model,
                default_model_alias=self.model_alias,
                cv_executor=self.cv_executor,
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
                template=template,
                status="completed",
                data=dict(result),
                elapsed_seconds=time.monotonic() - started,
            )
        except SafePipelineError as error:
            return EvalOutcome(
                video_path=video_path,
                template=template,
                status="failed",
                error=str(error),
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception as error:
            return EvalOutcome(
                video_path=video_path,
                template=template,
                status="failed",
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
    template: str,
    output_dir: Path,
    *,
    prompt_context: str | None = None,
    query: str | None = None,
    force: bool = False,
    log=print,
) -> list[EvalOutcome]:
    """Serialize writers and reuse only results from the same verified experiment."""
    with result_store.output_lock(Path(output_dir)):
        return _run_batch(runner, videos, template, output_dir,
                          prompt_context=prompt_context, query=query, force=force, log=log)


def _run_batch(
    runner: SyncRunner,
    videos: list[Path],
    template: str,
    output_dir: Path,
    *,
    prompt_context: str | None = None,
    query: str | None = None,
    force: bool = False,
    log=print,
) -> list[EvalOutcome]:
    """Reuse only intact results with matching input and verified execution identity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run = result_store.run_identity(runner, template, prompt_context, query)
    manifest = output_dir / ".percept" / "runs" / f"{uuid.uuid4().hex}.json"
    run_record = {"run": run, "run_fingerprint": result_store.json_digest(run),
                  "force": force, "status": "running", "results": [],
                  "started_at": datetime.now(timezone.utc).isoformat()}
    result_store.write_json(manifest, run_record)
    outcomes: list[EvalOutcome] = []
    if not run["reusable"]:
        log("resume disabled: custom model, pipeline, or CV execution identity is not verified")
    try:
        for index, source in enumerate(videos, start=1):
            video = Path(source).resolve()
            path = result_store.result_path(output_dir, video)
            try:
                identity = result_store.provenance(video, run)
            except OSError:
                identity = None  # The evaluator reports the normal missing/unreadable-input failure.
            previous = (result_store.completed_result(path, identity)
                        if identity is not None and not force else None)
            if previous is not None:
                try:
                    if result_store.file_digest(video) != identity["input"]["sha256"]:
                        previous = None
                        identity = result_store.provenance(video, run)
                except OSError:
                    previous = None
                    identity = None
            if previous is not None:
                outcome = EvalOutcome(
                    video_path=video, template=template, status="skipped",
                    data=previous["data"], provenance=identity, result_file=path.name,
                )
                log(f"[{index}/{len(videos)}] skip {video.name} (matching input and run)")
            else:
                log(f"[{index}/{len(videos)}] eval {video.name} ...")
                outcome = runner.evaluate(
                    video, template, prompt_context=prompt_context, query=query
                )
                if outcome.status == "completed":
                    try:
                        unchanged = identity is not None and result_store.file_digest(video) == identity["input"]["sha256"]
                    except OSError:
                        unchanged = False
                    unchanged = unchanged and result_store.run_identity(
                        runner, template, prompt_context, query
                    ) == run
                    if not unchanged:
                        outcome = EvalOutcome(video_path=video, template=template, status="failed",
                                              error="Input or run configuration changed during annotation; result discarded")
                outcome = replace(outcome, provenance=identity, result_file=path.name)
                result_store.archive_result(path, output_dir)
                _write_result(path, outcome)
                log(f"[{index}/{len(videos)}] {outcome.status} {video.name} -> {path.name}")
            outcomes.append(outcome)
            run_record["results"].append({"video_path": str(video), "result_file": path.name,
                                          "status": outcome.status, "input": identity["input"] if identity else None,
                                          "record_sha256": result_store.file_digest(path)})
            result_store.write_json(manifest, run_record)
        _write_aggregate(output_dir / "results.jsonl", outcomes)
        run_record["status"] = "completed" if all(o.status != "failed" for o in outcomes) else "failed"
    except BaseException:
        run_record["status"] = "interrupted"
        raise
    finally:
        run_record["finished_at"] = datetime.now(timezone.utc).isoformat()
        result_store.write_json(manifest, run_record)
    return outcomes


def _task_record(
    video_path: Path,
    template: str,
    model_alias: str,
    *,
    prompt_context: str | None,
    query: str | None,
) -> TaskRecord:
    if template not in SUPPORTED_TEMPLATES:
        raise ValueError(
            f"unsupported template {template!r}; expected one of {SUPPORTED_TEMPLATES}"
        )
    payload: dict[str, Any] = {
        "video_url": str(video_path),
        "task_template": template,
        "model_name": model_alias,
    }
    if query is not None:
        payload["query"] = query
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
        "template": outcome.template,
        "status": outcome.status,
        "data": outcome.data,
        "error": outcome.error,
        "elapsed_seconds": outcome.elapsed_seconds,
        "provenance": outcome.provenance,
        "data_sha256": result_store.json_digest(outcome.data),
        "result_file": outcome.result_file,
    }
    result_store.write_json(result_path, record)


def _write_aggregate(jsonl_path: Path, outcomes: list[EvalOutcome]) -> None:
    lines = [json.dumps({
        "video_path": str(outcome.video_path),
        "template": outcome.template,
        "status": outcome.status,
        "data": outcome.data,
        "error": outcome.error,
        "provenance": outcome.provenance,
        "result_file": outcome.result_file,
    }, ensure_ascii=False, allow_nan=False) + "\n" for outcome in outcomes]
    result_store.atomic_text(jsonl_path, "".join(lines))
