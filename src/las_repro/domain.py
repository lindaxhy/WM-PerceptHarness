"""Immutable-value domain records shared by the API, store, and workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .model_alias import DEFAULT_MODEL_ALIAS


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class InferenceStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class InferenceJobSpec:
    """A deterministic definition of one inference stage job."""

    stage: str
    ordinal: int
    payload: Mapping[str, Any]
    affinity_worker_id: str | None = None
    affinity_fallback_at: float | None = None
    affinity_fallback_seconds: float | None = None


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    operator_id: str
    operator_version: str
    payload: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0
    lease_until: float | None = None
    worker_id: str | None = None
    attempt: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class InferenceJob:
    job_id: str
    task_id: str
    stage: str
    ordinal: int
    payload: dict[str, Any]
    model_name: str = DEFAULT_MODEL_ALIAS
    status: InferenceStatus = InferenceStatus.PENDING
    created_at: float = 0.0
    updated_at: float = 0.0
    lease_until: float | None = None
    worker_id: str | None = None
    attempt: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    affinity_worker_id: str | None = None
    affinity_fallback_at: float | None = None
    completed_by: str | None = None
