"""Authenticated LAS-compatible submit and poll control-plane routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings
from .contracts import PollRequest, PollResponse, ResponseMetadata, SubmitRequest, SubmitResponse
from .domain import InferenceStatus, TaskRecord, TaskStatus
from .security import redact, verify_bearer
from .store import SQLiteTaskStore


_TASK_NOT_FOUND = "TASK_NOT_FOUND"
_OPERATOR_MISMATCH = "OPERATOR_MISMATCH"
_OPERATOR_VERSION_MISMATCH = "OPERATOR_VERSION_MISMATCH"
_TASK_FAILED = "TASK_FAILED"


def create_app(settings: Settings, store: SQLiteTaskStore) -> FastAPI:
    """Build the small API process around an initialized durable task store."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed"},
        )

    def require_bearer(request: Request) -> None:
        verify_bearer(request, settings)

    @app.post(
        "/api/v1/submit",
        response_model=SubmitResponse,
        response_model_exclude_unset=True,
        dependencies=[Depends(require_bearer)],
    )
    def submit(submission: SubmitRequest) -> dict[str, Any] | JSONResponse:
        if submission.data.model_name not in settings.model_registry:
            return JSONResponse(
                status_code=422,
                content={"detail": "Request validation failed"},
            )
        task = store.create_task(
            redact(submission.sanitized_data()),
            operator_id=submission.operator_id,
            operator_version=submission.operator_version,
        )
        metadata: dict[str, Any] = {
            "task_id": task.task_id,
            "task_status": task.status,
            "business_code": "0",
            "error_msg": "",
        }
        warnings = submission.compatibility_warnings()
        if warnings:
            metadata["warnings"] = warnings
        return {"metadata": metadata}

    @app.post(
        "/api/v1/poll",
        response_model=PollResponse,
        dependencies=[Depends(require_bearer)],
    )
    def poll(request: PollRequest) -> PollResponse | JSONResponse:
        task = store.get_task(request.task_id)
        if task is None:
            return _business_error(
                request.task_id, 404, _TASK_NOT_FOUND, "Task was not found"
            )
        if task.operator_id != request.operator_id:
            return _business_error(
                task.task_id, 409, _OPERATOR_MISMATCH, "Task operator does not match request"
            )
        if task.operator_version != request.operator_version:
            return _business_error(
                task.task_id,
                409,
                _OPERATOR_VERSION_MISMATCH,
                "Task operator version does not match request",
            )

        metadata = ResponseMetadata(
            task_id=task.task_id,
            task_status=task.status,
            progress=_progress(task, store),
        )
        if task.status is TaskStatus.COMPLETED:
            return PollResponse(metadata=metadata, data=redact(task.result or {}))
        if task.status is TaskStatus.FAILED:
            metadata.business_code = _TASK_FAILED
            metadata.error_msg = _failure_summary(task.error)
        return PollResponse(metadata=metadata)

    return app


def _business_error(task_id: str, status_code: int, business_code: str, error_msg: str) -> JSONResponse:
    """Return stable poll-domain errors without using exception detail wrappers."""
    response = PollResponse(
        metadata=ResponseMetadata(
            task_id=task_id,
            task_status=TaskStatus.FAILED,
            business_code=business_code,
            error_msg=error_msg,
        )
    )
    return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))


def _progress(task: TaskRecord, store: SQLiteTaskStore) -> dict[str, Any] | None:
    """Expose persisted progress when present, otherwise derive it from jobs."""
    if isinstance(task.result, Mapping):
        persisted = task.result.get("progress")
        if isinstance(persisted, Mapping):
            return redact(dict(persisted))

    jobs = store.list_inference_jobs(task.task_id)
    if not jobs:
        return None
    active = next(
        (job for job in jobs if job.status is InferenceStatus.RUNNING),
        next((job for job in jobs if job.status is InferenceStatus.PENDING), None),
    )
    return {
        "stage": active.stage if active is not None else jobs[-1].stage,
        "completed": sum(job.status is InferenceStatus.COMPLETED for job in jobs),
        "total": len(jobs),
    }


def _failure_summary(error: str | None) -> str:
    """Return a stable public failure summary without exposing worker diagnostics."""
    return "Task execution failed"
