"""Input and inference-result records for local evaluation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

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
