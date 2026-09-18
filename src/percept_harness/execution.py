"""Synchronous in-memory job execution replacing the queue/worker service layer.

Pipelines were written against a store that queues inference jobs for worker
processes. Here `create_inference_jobs` executes each job immediately in the
current process, so `wait_for_jobs` only collects already-terminal results.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from .domain import InferenceJob, InferenceJobSpec, InferenceStatus
from .media import TimeSpan, VideoMetadata
from .models.base import ModelOutputError, ModelRequest, VideoModel, VideoSession
from .metrics import validate_inference_job_metrics


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
        super().__init__(f"inference job failed at ordinal {job.ordinal}")


class JobStore(Protocol):
    """The minimal store surface pipelines depend on."""

    def create_inference_jobs(
        self, task_id: str, specs: Iterable[InferenceJobSpec], *, now: float | None = None
    ) -> list[InferenceJob]: ...

    def get_inference_job(self, job_id: str) -> InferenceJob | None: ...

    def list_inference_jobs(self, task_id: str) -> list[InferenceJob]: ...


class CvJobExecutor(Protocol):
    """Execute one CV evidence job payload to its published-result mapping."""

    def __call__(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]: ...


class SyncJobStore:
    """Execute inference jobs eagerly and remember their terminal records."""

    def __init__(
        self,
        model: VideoModel,
        *,
        default_model_alias: str,
        output_schemas: Any | None = None,
        cv_executor: CvJobExecutor | None = None,
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
        self._jobs: dict[str, InferenceJob] = {}
        self._sessions: dict[str, VideoSession] = {}

    # -- store surface used by pipelines ------------------------------------

    def create_inference_jobs(
        self,
        task_id: str,
        specs: Iterable[InferenceJobSpec],
        *,
        now: float | None = None,
    ) -> list[InferenceJob]:
        created_at = time.time() if now is None else now
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
                created_at=created_at,
                updated_at=created_at,
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
                    completed_by="sync",
                    started_at=created_at,
                    finished_at=time.time(),
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

    def get_inference_job(self, job_id: str) -> InferenceJob | None:
        return self._jobs.get(job_id)

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
        context = _schema_validation_context(job.payload, request)
        request = replace(
            request,
            response_contract=self._output_schemas.model_response_contract(
                request.schema_name, context
            ),
        )
        started = time.monotonic()
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
            result = self._output_schemas.sanitize(request.schema_name, generated, context)
        except ModelOutputError:
            result = self._output_schemas.model_output_failure(request.schema_name)
            if result is None:
                raise
        metrics = _model_request_metrics(self.model, inference_seconds=inference_seconds)
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
    request: ModelRequest,
) -> Mapping[str, Any] | None:
    context = payload.get("schema_context")
    if context is None:
        if request.schema_name == "general_segment":
            return {"span": {"start": request.span.start, "end": request.span.end}}
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
