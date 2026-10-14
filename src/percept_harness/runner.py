"""Video annotation: orchestration, stage execution and request caching.

The public entry points are SyncRunner, collect_videos and run_batch. Stage
records and execution helpers below them support the annotation pipelines.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from . import results
from .config import Settings
from .media import TimeSpan, VideoMetadata
from .models.base import ModelOutputError, ModelRequest, VideoModel, VideoSession, VisualEvidence

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
        from .pipelines.base import PipelineContext, SafePipelineError
        from .pipelines.embodied import EmbodiedActionPipeline

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
        path = results.result_path(output_dir, video)
        previous = results.completed_result(path, video) if not force and video.is_file() else None
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
    results.write_annotation(result_path, record)


def _write_aggregate(jsonl_path: Path, outcomes: list[EvalOutcome]) -> None:
    lines = [json.dumps({
        "video_path": str(outcome.video_path),
        "status": outcome.status,
        "data": outcome.data,
        "metadata": outcome.metadata,
        "error": outcome.error,
        "result_file": outcome.result_file,
    }, ensure_ascii=False, allow_nan=False) + "\n" for outcome in outcomes]
    results.atomic_text(jsonl_path, "".join(lines))


class InferenceStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class InferenceJobSpec:
    """A deterministic definition of one inference stage job."""

    stage: str
    ordinal: int
    payload: Mapping[str, Any]
    model_name: str | None = None


@dataclass(frozen=True)
class TaskRecord:
    """Input to one annotation pipeline; execution state lives in its outcome."""

    task_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class InferenceJob:
    job_id: str
    task_id: str
    stage: str
    ordinal: int
    payload: dict[str, Any]
    model_name: str
    status: InferenceStatus = InferenceStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None


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
    }
)
_JOB_BOOLEAN_METRIC_KEYS = frozenset(
    {"cache_hit", "oom_retry", "semantic_cache_hit", "semantic_cache_published"}
)
_JOB_CACHE_KEY_METRICS = frozenset(
    {"semantic_cache_key"}
)
_JOB_INTEGER_METRIC_KEYS = (
    _JOB_METRIC_KEYS - _JOB_BOOLEAN_METRIC_KEYS - _JOB_CACHE_KEY_METRICS
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
    for key in _JOB_CACHE_KEY_METRICS & values.keys():
        if (not isinstance(values[key], str)
                or re.fullmatch(r"[0-9a-f]{64}", values[key]) is None):
            raise ValueError(f"metrics {key} must be a lowercase hexadecimal cache key")
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded.encode("utf-8")) > _MAX_JOB_METRICS_BYTES:
        raise ValueError("metrics canonical JSON exceeds 4096 bytes")


logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Base class for stable execution-boundary failures."""


class JobWaitTimeout(ExecutionError):
    """Requested inference jobs did not become terminal before the deadline."""


class InferenceJobFailed(ExecutionError):
    """One requested inference job reached the failed terminal state."""

    def __init__(self, job: InferenceJob) -> None:
        self.job_id = job.job_id
        self.stage = job.stage
        self.ordinal = job.ordinal
        self.job_error = job.error
        detail = f": {job.error}" if job.error else ""
        super().__init__(f"inference job failed at ordinal {job.ordinal}{detail}")


class JobStore(Protocol):
    """The minimal store surface pipelines depend on."""

    def create_inference_jobs(
        self, task_id: str, specs: Iterable[InferenceJobSpec]
    ) -> list[InferenceJob]: ...

    def list_inference_jobs(self, task_id: str) -> list[InferenceJob]: ...


class CvJobExecutor(Protocol):
    """Execute one CV evidence job payload to its published-result mapping."""

    def __call__(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: ...


class SemanticResultCache:
    """Content-addressed store of sanitized per-request model results.

    A repair-loop retry or a whole-task rerun re-issues many requests whose
    inputs are byte-identical to ones that already succeeded. Keyed on the
    video bytes, span, prompt, schema context, and the backend's own
    ``semantic_cache_identity``, a hit returns the stored result instead of
    paying for another inference.

    Entries hold the *raw* model output for the request, so a hit replays it
    through the same ``sanitize`` step a live call would run. Caching the
    sanitized shape instead is wrong: re-sanitizing an envelope treats it as raw
    model output, and the pipeline's own envelope unpacking is only defined for
    some schemas, so a replayed normalized envelope would fail there. Failed
    attempts are never cached: a retry must reach the model again. Entries are
    JSON files written atomically; a corrupt or unreadable entry is treated as a
    miss and rewritten.
    """

    # Bumped when the stored payload's meaning changes, so entries written by
    # an incompatible version are ignored rather than misread. Version 1 stored
    # the sanitized shape; version 2 stores the raw model output.
    _FORMAT_VERSION = 2

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
    def key(
        self,
        request: ModelRequest,
        backend_identity: Mapping[str, Any],
        schema_context: Mapping[str, Any] | None,
    ) -> str:
        video = request.video_path.resolve(strict=True)
        status = video.stat()
        identity = {
            "cache_version": 3,
            "video": [str(video), status.st_size, status.st_mtime_ns],
            "span": [request.span.start, request.span.end],
            "fps": request.fps,
            "stage": request.stage,
            "prompt": request.prompt,
            "schema_name": request.schema_name,
            "schema_context": dict(schema_context) if schema_context else None,
            "backend": dict(backend_identity),
            "visual_evidence": [e.identity() for e in request.visual_evidence],
        }
        # Opaque cache filename only; no content integrity verification.
        return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _entry_path(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._entry_path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(entry, dict)
            or entry.get("format_version") != self._FORMAT_VERSION
            or not isinstance(entry.get("result"), dict)
        ):
            return None
        return entry["result"]

    def publish(self, key: str, result: Mapping[str, Any]) -> bool:
        path = self._entry_path(key)
        encoded = json.dumps(
            {
                "key": key,
                "format_version": self._FORMAT_VERSION,
                "result": dict(result),
            },
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, staging = tempfile.mkstemp(
                dir=path.parent, prefix=".staging-", suffix=".json"
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(encoded)
                os.replace(staging, path)
            except BaseException:
                try:
                    os.unlink(staging)
                except OSError:
                    pass
                raise
        except OSError:
            logger.debug("semantic cache publish failed for %s", key, exc_info=True)
            return False
        return True


class SyncJobStore:
    """Execute inference jobs eagerly and remember their terminal records."""

    def __init__(
        self,
        model: VideoModel,
        *,
        default_model_alias: str,
        output_schemas: Any | None = None,
        cv_executor: CvJobExecutor | None = None,
        semantic_cache: SemanticResultCache | None = None,
    ) -> None:
        if output_schemas is None:
            # Imported lazily: output_validation transitively imports modules
            # that need this module's metrics-free surface at import time.
            from .pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

            output_schemas = DEFAULT_OUTPUT_SCHEMAS
        self.model = model
        self.default_model_alias = default_model_alias
        self._output_schemas = output_schemas
        self._cv_executor = cv_executor
        self._semantic_cache = (
            semantic_cache
            if getattr(model, "supports_semantic_result_cache", False)
            else None
        )
        self._jobs: dict[str, InferenceJob] = {}
        self._sessions: dict[str, VideoSession] = {}

    # -- store surface used by pipelines ------------------------------------

    def create_inference_jobs(
        self,
        task_id: str,
        specs: Iterable[InferenceJobSpec],
    ) -> list[InferenceJob]:
        jobs: list[InferenceJob] = []
        for spec in specs:
            job_id = f"{task_id}:{spec.stage}:{spec.ordinal}"
            existing = self._jobs.get(job_id)
            if existing is not None:
                jobs.append(existing)
                continue
            model_name = spec.model_name or self.default_model_alias
            base = InferenceJob(
                job_id=job_id,
                task_id=task_id,
                stage=spec.stage,
                ordinal=spec.ordinal,
                payload=dict(spec.payload),
                model_name=model_name,
            )
            try:
                if spec.stage == "cv_evidence":
                    result, metrics = self._execute_cv(base)
                else:
                    result, metrics = self._execute_model(base)
                validate_inference_job_metrics(metrics)
                job = replace(
                    base,
                    status=InferenceStatus.COMPLETED,
                    result=result,
                    metrics=metrics,
                )
            except Exception as error:
                job = replace(
                    base,
                    status=InferenceStatus.FAILED,
                    error=str(error) or "inference failed",
                )
            self._jobs[job_id] = job
            jobs.append(job)
        return jobs

    def list_inference_jobs(self, task_id: str) -> list[InferenceJob]:
        return [job for job in self._jobs.values() if job.task_id == task_id]

    def close(self) -> None:
        """Release any cached per-video session state."""
        for session_id, session in tuple(self._sessions.items()):
            release = getattr(self.model, "release_video_session", None)
            try:
                if callable(release):
                    release(session_id)
            except Exception:
                pass
            session.sampled_frames = ()
            session.backend_cache.clear()
        self._sessions.clear()

    # -- execution -----------------------------------------------------------

    def _execute_cv(self, job: InferenceJob) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._cv_executor is None:
            raise ExecutionError("no CV executor is configured")
        return self._cv_executor(job.payload)

    def _execute_model(self, job: InferenceJob) -> tuple[dict[str, Any], dict[str, Any]]:
        request = self._attach_session(_model_request(job), job)
        context = _schema_validation_context(job.payload)
        request = replace(
            request,
            response_contract=self._output_schemas.model_response_contract(
                request.schema_name, context
            ),
        )
        cache_key: str | None = None
        if self._semantic_cache is not None:
            identity = getattr(self.model, "semantic_cache_identity", None)
            if callable(identity):
                try:
                    cache_key = self._semantic_cache.key(request, identity(request), context)
                except (OSError, TypeError, ValueError):
                    logger.debug("semantic cache key derivation failed", exc_info=True)
            if cache_key is not None:
                cached = self._semantic_cache.load(cache_key)
                if cached is not None:
                    # Replay the raw output through the same sanitize step a
                    # live call runs, so the pipeline sees an identical result.
                    try:
                        result = self._output_schemas.sanitize(
                            request.schema_name, cached, context
                        )
                    except (ValueError, TypeError):
                        result = None
                    if result is not None and self._output_schemas.failure_codes(
                        request.schema_name, result
                    ) is None:
                        return result, {
                            "inference_seconds": 0.0,
                            "semantic_cache_hit": True,
                            "semantic_cache_key": cache_key,
                        }
        started = time.monotonic()
        cacheable = True
        try:
            try:
                generated = self.model.generate(request)
            finally:
                inference_seconds = time.monotonic() - started
                self._remember_session(request)
                release = getattr(self.model, "release_request", None)
                if callable(release):
                    release(request)
            if not isinstance(generated, Mapping):
                raise ModelOutputError("model output must be a structured object")
            raw = dict(generated)
            result = self._output_schemas.sanitize(request.schema_name, raw, context)
            if self._output_schemas.failure_codes(request.schema_name, result) is not None:
                # An invalid-output envelope is a terminal verdict for this
                # attempt, not a result: caching it would replay the rejection
                # on every retry instead of reaching the model again.
                cacheable = False
        except ModelOutputError:
            result = self._output_schemas.model_output_failure(request.schema_name)
            if result is None:
                raise
            cacheable = False  # never replay a malformed-output placeholder
        metrics = _model_request_metrics(self.model, inference_seconds=inference_seconds)
        if cache_key is not None and cacheable:
            published = self._semantic_cache.publish(cache_key, raw)
            metrics["semantic_cache_key"] = cache_key
            metrics["semantic_cache_published"] = published
        return result, metrics

    def _attach_session(self, request: ModelRequest, job: InferenceJob) -> ModelRequest:
        if request.video_session_id is None:
            return request
        metadata = _video_session_metadata(job.payload)
        if metadata is None:
            return request
        session = self._sessions.get(request.video_session_id)
        if session is None or session.metadata != metadata:
            session = VideoSession(metadata=metadata)
        return replace(request, video_session=session)

    def _remember_session(self, request: ModelRequest) -> None:
        if request.video_session_id is not None and request.video_session is not None:
            self._sessions[request.video_session_id] = request.video_session


def wait_for_jobs(
    store: JobStore,
    task_id: str,
    job_ids: Sequence[str],
    timeout: float,
    **_: Any,
) -> list[dict[str, Any]]:
    """Return results for already-terminal jobs in requested ID order.

    Synchronous execution completes every job inside ``create_inference_jobs``,
    so this never sleeps; the signature matches the queue-era helper that
    pipelines inject.
    """
    if not (isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
            and math.isfinite(float(timeout)) and float(timeout) >= 0):
        raise ValueError("timeout must be a finite non-negative number")
    requested = tuple(job_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("job_ids must not contain duplicates")
    by_id = {job.job_id: job for job in store.list_inference_jobs(task_id)}
    if any(job_id not in by_id for job_id in requested):
        raise KeyError("one or more inference jobs were not found for the task")
    jobs = [by_id[job_id] for job_id in requested]
    failed = next((job for job in jobs if job.status is InferenceStatus.FAILED), None)
    if failed is not None:
        raise InferenceJobFailed(failed)
    pending = [job for job in jobs if job.status is not InferenceStatus.COMPLETED]
    if pending:
        raise JobWaitTimeout("inference jobs did not become terminal")
    return [dict(job.result or {}) for job in jobs]


def _model_request(job: InferenceJob) -> ModelRequest:
    payload = job.payload
    video_value = payload.get("video_path")
    if not isinstance(video_value, str) or not video_value:
        raise ValueError("model job video_path must be a non-blank local path")
    video_path = Path(video_value)
    if not video_path.is_absolute():
        raise ValueError("model job video_path must be absolute")
    try:
        video_path = video_path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("model job video_path is unavailable") from None
    if not video_path.is_file():
        raise ValueError("model job video_path must be a regular file")

    span_value = payload.get("span")
    if isinstance(span_value, Mapping):
        start = span_value.get("start")
        end = span_value.get("end")
    else:
        start = payload.get("start")
        end = payload.get("end")
    return ModelRequest(
        stage=job.stage,
        video_path=video_path,
        span=TimeSpan(
            _finite_number(start, "model job start"),
            _finite_number(end, "model job end"),
        ),
        fps=_finite_number(payload.get("fps", 2.0), "model job fps"),
        prompt=payload.get("prompt", ""),
        schema_name=payload.get("schema_name", job.stage),
        video_session_id=payload.get("video_session_id", job.task_id),
        model_name=job.model_name,
        media_resolution=payload.get("media_resolution"),
        reasoning_effort=payload.get("reasoning_effort"),
        clip_context=payload.get("clip_context"),
        visual_evidence=tuple(VisualEvidence.from_payload(e) for e in payload.get("visual_evidence", ())),
    )


def _model_request_metrics(model: VideoModel, *, inference_seconds: float) -> dict[str, Any]:
    request_metrics = getattr(model, "request_metrics", None)
    reported = request_metrics() if callable(request_metrics) else None
    if reported is None:
        metrics: dict[str, Any] = {}
    elif isinstance(reported, Mapping):
        metrics = dict(reported)
    else:
        raise TypeError("model request_metrics must return a mapping or None")
    metrics["inference_seconds"] = inference_seconds
    return metrics


def _schema_validation_context(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    context = payload.get("schema_context")
    if context is None:
        return None
    if not isinstance(context, Mapping):
        raise ValueError("model job schema context must be an object")
    return context


def _video_session_metadata(payload: Mapping[str, Any]) -> VideoMetadata | None:
    value = payload.get("video_metadata")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"duration", "width", "height", "fps"}:
        raise ValueError("model job video metadata must be an exact object")
    width = value["width"]
    height = value["height"]
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("model job video dimensions must be positive integers")
    duration = _finite_number(value["duration"], "model job video duration")
    fps = _finite_number(value["fps"], "model job video fps")
    if duration <= 0 or fps <= 0:
        raise ValueError("model job video metadata must be positive")
    return VideoMetadata(duration=duration, width=width, height=height, fps=fps)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real number")
    return numeric
