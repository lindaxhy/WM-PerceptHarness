"""Video annotation orchestration: one runner, video discovery and batch results."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import results
from .config import Settings
from .execution import SemanticResultCache, StageExecutor
from .models.base import VideoModel

_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


@dataclass(frozen=True)
class EvalOutcome:
    """The terminal record of one evaluated video."""

    video_path: Path
    status: str  # "completed" | "needs_review" | "failed" | "skipped"
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
    ) -> None:
        self.model = model
        self.settings = settings
        self.model_alias = model_alias
        self.cv_executor = cv_executor
        self.semantic_cache = None
        if settings.semantic_result_cache and getattr(model, "supports_semantic_result_cache", False):
            self.semantic_cache = SemanticResultCache(Path(settings.work_root) / "semantic-cache")

    def evaluate(
        self,
        video_path: Path,
        *,
        prompt_context: str | None = None,
        review_from: Path | None = None,
    ) -> EvalOutcome:
        """Evaluate one video end-to-end and return its terminal outcome."""
        from .pipelines.base import PipelineContext, SafePipelineError
        from .pipelines.embodied import EmbodiedActionPipeline

        started = time.monotonic()
        video_path = Path(video_path)
        context = None
        try:
            resolved = video_path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError("video path must be a regular file")
            executor = StageExecutor(
                self.model, default_model_alias=self.model_alias,
                cv_executor=self.cv_executor, semantic_cache=self.semantic_cache,
            )
            context = PipelineContext(
                executor=executor, settings=self.settings, media_path=resolved,
            )
            prior_review = None
            if review_from is not None:
                prior_review = results.read_record(Path(review_from))
                if (not prior_review or prior_review.get('status') != 'needs_review' or
                        prior_review.get('video_path') != str(resolved)):
                    raise SafePipelineError('review_from must be a needs_review audit for this source video')
                context.metadata['prior_review_source'] = str(Path(review_from).resolve())
            result = EmbodiedActionPipeline().run(context, prompt_context=prompt_context, prior_review=prior_review)
            return EvalOutcome(
                video_path=resolved,
                status=("needs_review" if context.metadata.get("revision_validation", {}).get("status") == "needs_review" else "completed"),
                data=dict(result),
                metadata=dict(context.metadata),
                elapsed_seconds=time.monotonic() - started,
            )
        except Exception as error:
            return EvalOutcome(
                video_path=video_path,
                status="failed",
                metadata=dict(context.metadata) if context is not None else None,
                error=str(error) if isinstance(error, SafePipelineError) else f"{type(error).__name__}: {error}",
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
    review_from: Path | None = None,
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
        path = results.result_path(output_dir, video)
        previous = results.completed_result(path, video) if not force and review_from is None and video.is_file() else None
        if previous is not None:
            outcome = EvalOutcome(video_path=video, status="skipped", data=previous["data"],
                                  result_file=path.name, metadata=previous.get("metadata"))
            log(f"[{index}/{len(videos)}] skip {video.name} (completed result)")
        else:
            log(f"[{index}/{len(videos)}] eval {video.name} ...")
            outcome = replace(runner.evaluate(video, prompt_context=prompt_context, review_from=review_from),
                              result_file=path.name)
            write_outcome(path, outcome)
            log(f"[{index}/{len(videos)}] {outcome.status} {video.name} -> {path.name}")
        outcomes.append(outcome)
    _write_aggregate(output_dir / "results.jsonl", outcomes)
    return outcomes


def _record(outcome: EvalOutcome) -> dict[str, Any]:
    return {"video_path": str(outcome.video_path), "status": outcome.status, "data": outcome.data,
            "metadata": outcome.metadata, "error": outcome.error,
            "elapsed_seconds": outcome.elapsed_seconds, "result_file": outcome.result_file}


def write_outcome(result_path: Path, outcome: EvalOutcome) -> None:
    """Persist one outcome as its public annotation and audit sidecar."""
    results.write_annotation(result_path, _record(outcome))


def _write_aggregate(jsonl_path: Path, outcomes: list[EvalOutcome]) -> None:
    lines = [json.dumps({k: v for k, v in _record(outcome).items() if k != "elapsed_seconds"},
                        ensure_ascii=False, allow_nan=False) + "\n" for outcome in outcomes]
    results.atomic_text(jsonl_path, "".join(lines))
