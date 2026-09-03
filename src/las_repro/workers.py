"""Leased coordinator and local GPU inference worker primitives."""

from __future__ import annotations

import math
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .domain import InferenceJob, InferenceStatus, TaskRecord, TaskStatus
from .media import MediaResolver, TimeSpan, VideoMetadata
from .models.base import ModelOutputError, ModelRequest, VideoModel, VideoSession
from .model_alias import DEFAULT_MODEL_ALIAS, validate_model_alias
from .pipelines.base import PipelineContext, PipelineRegistry, SafePipelineError
from .pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS, OutputSchemaRegistry
from .store import InvalidTransition, SQLiteTaskStore, WorkerMismatch


class WorkerError(RuntimeError):
    """Base class for stable worker-boundary failures."""


class JobWaitTimeout(WorkerError):
    """Requested inference jobs did not become terminal before the deadline."""


class InferenceJobFailed(WorkerError):
    """One requested inference job reached the failed terminal state."""

    def __init__(self, job: InferenceJob) -> None:
        self.job_id = job.job_id
        self.stage = job.stage
        self.ordinal = job.ordinal
        super().__init__(f"inference job failed at ordinal {job.ordinal}")


class _LeaseLost(WorkerError):
    """The current iteration can no longer safely publish its result."""


class StopSignal(Protocol):
    """Minimal interruptible wait surface used by long-running worker loops."""

    def is_set(self) -> bool:
        """Return whether shutdown was requested."""

    def wait(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds and return whether shutdown occurred."""


@dataclass(frozen=True)
class _SessionUse:
    task_id: str
    last_used: float
    session: VideoSession | None


class _LeaseKeeper:
    """Renew one owned lease while a synchronous expensive call is active."""

    def __init__(
        self,
        renew: Callable[[float | None], object],
        *,
        interval: float,
        fixed_now: float | None,
    ) -> None:
        self._renew = renew
        self._interval = interval
        self._fixed_now = fixed_now
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def __enter__(self) -> _LeaseKeeper:
        self._renew(self._fixed_now)
        if self._fixed_now is None:
            self._thread = threading.Thread(
                target=self._run,
                name="las-lease-heartbeat",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(0.1, self._interval * 2))
        if exc_type is None and self._error is not None:
            raise _LeaseLost("worker lease could not be renewed") from None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._renew(None)
            except Exception as error:
                self._error = error
                self._stop.set()
                return


class GPUWorker:
    """Claim and execute at most one inference job on an assigned device."""

    def __init__(
        self,
        store: SQLiteTaskStore,
        model: VideoModel,
        worker_id: str,
        device: str,
        *,
        model_name: str = DEFAULT_MODEL_ALIAS,
        lease_seconds: float = 300.0,
        heartbeat_interval: float | None = None,
        session_idle_seconds: float = 900.0,
        monotonic: Callable[[], float] = time.monotonic,
        output_schemas: OutputSchemaRegistry = DEFAULT_OUTPUT_SCHEMAS,
    ) -> None:
        self.store = store
        self.model = model
        self.worker_id = _nonblank(worker_id, "worker_id")
        self.device = _nonblank(device, "device")
        self.model_name = validate_model_alias(model_name)
        self.lease_seconds = _positive_finite(lease_seconds, "lease_seconds")
        default_interval = self.lease_seconds / 3.0
        self.heartbeat_interval = _positive_finite(
            default_interval if heartbeat_interval is None else heartbeat_interval,
            "heartbeat_interval",
        )
        if self.heartbeat_interval >= self.lease_seconds:
            raise ValueError("heartbeat_interval must be shorter than lease_seconds")
        self.session_idle_seconds = _positive_finite(
            session_idle_seconds,
            "session_idle_seconds",
        )
        self._monotonic = monotonic
        self._output_schemas = output_schemas
        self._sessions: dict[str, _SessionUse] = {}

    def run_once(self, *, now: float | None = None) -> bool:
        """Process at most one eligible job, returning whether one was claimed."""
        self._release_expired_sessions()
        job: InferenceJob | None = None
        claim_returned = False
        request: ModelRequest | None = None

        def register_claim(claimed: InferenceJob) -> None:
            nonlocal job
            job = claimed

        try:
            claimed = self.store.claim_inference_job(
                self.worker_id,
                model_name=self.model_name,
                lease_seconds=self.lease_seconds,
                now=now,
                on_claim=register_claim,
            )
            claim_returned = True
            if claimed is None:
                return False
            job = claimed
            with _LeaseKeeper(
                lambda heartbeat_now: self.store.heartbeat_inference_job(
                    job.job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                    attempt=job.attempt,
                    now=heartbeat_now,
                ),
                interval=self.heartbeat_interval,
                fixed_now=now,
            ):
                request = self._attach_video_session(_model_request(job), job)
                try:
                    try:
                        generated = self.model.generate(request)
                        if not isinstance(generated, Mapping):
                            raise ModelOutputError(
                                "model output must be a structured object"
                            )
                        result = self._output_schemas.sanitize(
                            request.schema_name,
                            generated,
                            _schema_validation_context(job.payload, request),
                        )
                    except ModelOutputError:
                        result = self._output_schemas.model_output_failure(
                            request.schema_name
                        )
                        if result is None:
                            raise
                finally:
                    try:
                        self._release_request(request)
                    finally:
                        self._remember_session(request, job.task_id)
            self.store.complete_inference_job(
                job.job_id,
                result,
                worker_id=self.worker_id,
                attempt=job.attempt,
                now=now,
            )
        except (InvalidTransition, WorkerMismatch):
            if not claim_returned:
                raise
            # A recovered owner won the lease race.  The stale result is dropped.
            pass
        except Exception:
            if not claim_returned:
                if job is not None:
                    try:
                        self.store.expire_inference_job_lease(
                            job.job_id,
                            self.worker_id,
                            attempt=job.attempt,
                            now=now,
                        )
                    except (InvalidTransition, WorkerMismatch):
                        pass
                raise
            try:
                self.store.fail_inference_job(
                    job.job_id,
                    "model inference failed",
                    worker_id=self.worker_id,
                    attempt=job.attempt,
                    now=now,
                )
            except (InvalidTransition, WorkerMismatch):
                # Recovery may have completed while the failed call was unwinding.
                pass
        except BaseException:
            if job is not None:
                try:
                    self.store.expire_inference_job_lease(
                        job.job_id,
                        self.worker_id,
                        attempt=job.attempt,
                        now=now,
                    )
                except (InvalidTransition, WorkerMismatch):
                    pass
            raise
        finally:
            request = None
        return True

    def run_forever(
        self,
        stop: StopSignal,
        *,
        no_work_backoff: float = 0.05,
        max_no_work_backoff: float = 1.0,
        error_backoff: float = 1.0,
    ) -> None:
        """Run iterations until stopped, backing off only when idle or errored."""
        _run_forever(
            self.run_once,
            stop,
            no_work_backoff=no_work_backoff,
            max_no_work_backoff=max_no_work_backoff,
            error_backoff=error_backoff,
        )

    def close(self) -> None:
        """Release every worker-owned visual session before process teardown."""
        for session_id in tuple(self._sessions):
            self._release_session(session_id)

    def _remember_session(self, request: ModelRequest, task_id: str) -> None:
        if request.video_session_id is not None:
            previous = self._sessions.get(request.video_session_id)
            self._sessions[request.video_session_id] = _SessionUse(
                task_id=task_id,
                last_used=_finite_clock(self._monotonic()),
                session=(
                    request.video_session
                    if request.video_session is not None
                    else previous.session
                    if previous is not None
                    else None
                ),
            )

    def _attach_video_session(
        self,
        request: ModelRequest,
        job: InferenceJob,
    ) -> ModelRequest:
        if request.video_session_id is None:
            return request
        metadata = _video_session_metadata(job.payload)
        if metadata is None:
            return request
        previous = self._sessions.get(request.video_session_id)
        if (
            previous is not None
            and previous.task_id == job.task_id
            and previous.session is not None
            and previous.session.metadata == metadata
        ):
            session = previous.session
        else:
            session = VideoSession(metadata=metadata)
        return replace(request, video_session=session)

    def _release_request(self, request: ModelRequest) -> None:
        release = getattr(self.model, "release_request", None)
        if callable(release):
            release(request)

    def _release_expired_sessions(self) -> None:
        if not self._sessions:
            return
        current = _finite_clock(self._monotonic())
        for session_id, use in tuple(self._sessions.items()):
            task = self.store.get_task(use.task_id)
            terminal = task is None or task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
            idle = current - use.last_used >= self.session_idle_seconds
            if not terminal and not idle:
                continue
            self._release_session(session_id)

    def _release_session(self, session_id: str) -> None:
        use = self._sessions.get(session_id)
        if use is None:
            return
        release = getattr(self.model, "release_video_session", None)
        try:
            if callable(release):
                release(session_id)
        except Exception:
            pass
        finally:
            if use.session is not None:
                use.session.sampled_frames = ()
                use.session.backend_cache.clear()
            self._sessions.pop(session_id, None)


class Coordinator:
    """Claim and dispatch at most one top-level task to a local pipeline."""

    def __init__(
        self,
        store: SQLiteTaskStore,
        media_resolver: MediaResolver,
        settings: Settings,
        pipelines: PipelineRegistry,
        *,
        worker_id: str = "coordinator",
        lease_seconds: float | None = None,
        heartbeat_interval: float | None = None,
        cleanup_on_terminal: bool = True,
    ) -> None:
        self.store = store
        self.media_resolver = media_resolver
        self.settings = settings
        self.pipelines = pipelines
        self.worker_id = _nonblank(worker_id, "worker_id")
        configured_lease = settings.lease_seconds if lease_seconds is None else lease_seconds
        self.lease_seconds = _positive_finite(configured_lease, "lease_seconds")
        default_interval = self.lease_seconds / 3.0
        self.heartbeat_interval = _positive_finite(
            default_interval if heartbeat_interval is None else heartbeat_interval,
            "heartbeat_interval",
        )
        if self.heartbeat_interval >= self.lease_seconds:
            raise ValueError("heartbeat_interval must be shorter than lease_seconds")
        self.cleanup_on_terminal = cleanup_on_terminal

    def run_once(self, *, now: float | None = None) -> bool:
        """Resolve and run at most one eligible top-level task."""
        task: TaskRecord | None = None
        claim_returned = False
        task_dir: Path | None = None
        terminal_by_self = False

        def register_claim(claimed: TaskRecord) -> None:
            nonlocal task
            task = claimed

        try:
            claimed = self.store.claim_task(
                self.worker_id,
                lease_seconds=self.lease_seconds,
                now=now,
                on_claim=register_claim,
            )
            claim_returned = True
            if claimed is None:
                return False
            task = claimed
            with _LeaseKeeper(
                lambda heartbeat_now: self.store.heartbeat_task(
                    task.task_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                    attempt=task.attempt,
                    now=heartbeat_now,
                ),
                interval=self.heartbeat_interval,
                fixed_now=now,
            ):
                template = task.payload.get("task_template")
                if not isinstance(template, str):
                    raise ValueError("task template is invalid")
                pipeline = self.pipelines.create(template)
                task_dir = _prepare_task_dir(self.settings.work_root, task.task_id)
                media_path = self.media_resolver.resolve(
                    task.payload.get("video_url"),
                    task_dir,
                )
                context = PipelineContext(
                    self.store,
                    self.media_resolver,
                    self.settings,
                    task_dir,
                    media_path,
                )
                result = pipeline.run(task, context)
                if not isinstance(result, Mapping):
                    raise TypeError("pipeline result must be a mapping")
            self.store.complete_task(
                task.task_id,
                dict(result),
                worker_id=self.worker_id,
                attempt=task.attempt,
                now=now,
            )
            terminal_by_self = True
        except (InvalidTransition, WorkerMismatch):
            if not claim_returned:
                raise
            # A stale coordinator must not publish or remove the new owner's files.
            pass
        except SafePipelineError as error:
            if not claim_returned:
                raise
            try:
                self.store.fail_task(
                    task.task_id,
                    str(error),
                    worker_id=self.worker_id,
                    attempt=task.attempt,
                    now=now,
                )
                terminal_by_self = True
            except (InvalidTransition, WorkerMismatch):
                pass
        except Exception:
            if not claim_returned:
                if task is not None:
                    try:
                        self.store.expire_task_lease(
                            task.task_id,
                            self.worker_id,
                            attempt=task.attempt,
                            now=now,
                        )
                    except (InvalidTransition, WorkerMismatch):
                        pass
                raise
            try:
                self.store.fail_task(
                    task.task_id,
                    "task execution failed",
                    worker_id=self.worker_id,
                    attempt=task.attempt,
                    now=now,
                )
                terminal_by_self = True
            except (InvalidTransition, WorkerMismatch):
                pass
        except BaseException:
            if task is not None:
                try:
                    self.store.expire_task_lease(
                        task.task_id,
                        self.worker_id,
                        attempt=task.attempt,
                        now=now,
                    )
                except (InvalidTransition, WorkerMismatch):
                    pass
            raise
        finally:
            if terminal_by_self and task_dir is not None and self.cleanup_on_terminal:
                _cleanup_task_dir(task_dir, self.settings.work_root)
        return True

    def run_forever(
        self,
        stop: StopSignal,
        *,
        no_work_backoff: float = 0.05,
        max_no_work_backoff: float = 1.0,
        error_backoff: float = 1.0,
    ) -> None:
        """Run iterations until stopped, backing off only when idle or errored."""
        _run_forever(
            self.run_once,
            stop,
            no_work_backoff=no_work_backoff,
            max_no_work_backoff=max_no_work_backoff,
            error_backoff=error_backoff,
        )


def wait_for_jobs(
    store: SQLiteTaskStore,
    task_id: str,
    job_ids: Sequence[str],
    timeout: float,
    *,
    poll_interval: float = 0.05,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Wait a bounded time for requested jobs and return results in ID order."""
    timeout_value = _nonnegative_finite(timeout, "timeout")
    interval = _positive_finite(poll_interval, "poll_interval")
    requested = tuple(job_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("job_ids must not contain duplicates")

    started = _finite_clock(monotonic())
    deadline = started + timeout_value
    if not math.isfinite(deadline):
        raise ValueError("timeout must produce a finite deadline")

    while True:
        by_id = {job.job_id: job for job in store.list_inference_jobs(task_id)}
        if any(job_id not in by_id for job_id in requested):
            raise KeyError("one or more inference jobs were not found for the task")
        jobs = [by_id[job_id] for job_id in requested]
        failed = next((job for job in jobs if job.status is InferenceStatus.FAILED), None)
        if failed is not None:
            raise InferenceJobFailed(failed)
        if all(job.status is InferenceStatus.COMPLETED for job in jobs):
            return [dict(job.result or {}) for job in jobs]

        current = _finite_clock(monotonic())
        remaining = deadline - current
        if remaining <= 0:
            raise JobWaitTimeout("inference jobs did not finish before the deadline")
        sleep(min(interval, remaining))


def _run_forever(
    run_once: Callable[[], bool],
    stop: StopSignal,
    *,
    no_work_backoff: float,
    max_no_work_backoff: float,
    error_backoff: float,
) -> None:
    initial_delay = _positive_finite(no_work_backoff, "no_work_backoff")
    maximum_delay = _positive_finite(
        max_no_work_backoff,
        "max_no_work_backoff",
    )
    error_delay = _positive_finite(error_backoff, "error_backoff")
    if maximum_delay < initial_delay:
        raise ValueError("max_no_work_backoff must be at least no_work_backoff")

    idle_delay = initial_delay
    while not stop.is_set():
        try:
            worked = run_once()
        except Exception:
            idle_delay = initial_delay
            if stop.wait(error_delay):
                return
            continue
        if worked:
            idle_delay = initial_delay
            continue
        if stop.wait(idle_delay):
            return
        idle_delay = min(maximum_delay, idle_delay * 2.0)


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
    prompt = payload.get("prompt", "")
    schema_name = payload.get("schema_name", job.stage)
    session_id = payload.get("video_session_id", job.task_id)
    return ModelRequest(
        stage=job.stage,
        video_path=video_path,
        span=TimeSpan(
            _finite_number(start, "model job start"),
            _finite_number(end, "model job end"),
        ),
        fps=_finite_number(payload.get("fps", 2.0), "model job fps"),
        prompt=prompt,
        schema_name=schema_name,
        video_session_id=session_id,
        model_name=job.model_name,
        media_resolution=payload.get("media_resolution"),
        reasoning_effort=payload.get("reasoning_effort"),
        clip_context=payload.get("clip_context"),
    )


def _schema_validation_context(
    payload: Mapping[str, Any],
    request: ModelRequest,
) -> Mapping[str, Any] | None:
    context = payload.get("schema_context")
    if context is None:
        if request.schema_name == "general_segment":
            return {
                "span": {
                    "start": request.span.start,
                    "end": request.span.end,
                }
            }
        return None
    if not isinstance(context, Mapping):
        raise ValueError("model job schema context must be an object")
    return context


def _video_session_metadata(payload: Mapping[str, Any]) -> VideoMetadata | None:
    value = payload.get("video_metadata")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "duration",
        "width",
        "height",
        "fps",
    }:
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


def _prepare_task_dir(work_root: Path, task_id: str) -> Path:
    work_root.mkdir(parents=True, exist_ok=True)
    root = work_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("work root must be a directory")
    destination = root / task_id
    if destination.is_symlink():
        raise ValueError("task directory must not be a symbolic link")
    destination.mkdir(exist_ok=True)
    resolved = destination.resolve(strict=True)
    if not resolved.is_dir() or resolved.parent != root:
        raise ValueError("task directory escaped the work root")
    return resolved


def _cleanup_task_dir(task_dir: Path, work_root: Path) -> None:
    try:
        root = work_root.resolve(strict=True)
        resolved = task_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        return
    if resolved.parent != root or resolved == root or not resolved.is_dir():
        return
    try:
        shutil.rmtree(resolved)
    except OSError:
        pass


def _nonblank(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    return value


def _positive_finite(value: Real, name: str) -> float:
    numeric = _finite_number(value, name)
    if numeric <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _nonnegative_finite(value: Real, name: str) -> float:
    numeric = _finite_number(value, name)
    if numeric < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return numeric


def _finite_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(numeric):
        if name == "timeout":
            raise ValueError("timeout must be a finite non-negative number")
        raise ValueError(f"{name} must be a finite real number")
    return numeric


def _finite_clock(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError("monotonic clock must return a finite real number")
    return float(value)
