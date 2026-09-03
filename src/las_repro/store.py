"""Durable SQLite storage for leased coordinator tasks and inference jobs."""

from __future__ import annotations

import json
import math
import os
import signal
import sqlite3
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Iterator

from .domain import InferenceJob, InferenceJobSpec, InferenceStatus, TaskRecord, TaskStatus
from .model_alias import DEFAULT_MODEL_ALIAS, validate_model_alias


class StoreError(RuntimeError):
    """Base error for store operations."""


class InvalidTransition(StoreError):
    """Raised when a state-machine transition is not legal."""


class WorkerMismatch(StoreError):
    """Raised when a worker attempts to mutate another worker's lease."""


class DuplicateInferenceJob(StoreError):
    """Raised when an existing stage/ordinal has a conflicting definition."""


_DATABASE_DIRECTORY_ERROR = (
    "database directory must be an owned non-writable-by-others directory"
)
_DATABASE_ANCESTRY_ERROR = (
    "database directory must have trusted POSIX ancestry"
)
_DATABASE_FILE_ERROR = "database file must be an owner-only regular file"
_DATABASE_SIDECAR_ERROR = "database sidecar must be an owner-only regular file"
_LEGACY_MODEL_MIGRATION_BATCH_SIZE = 16
_MAX_LEGACY_TASK_PAYLOAD_BYTES = 1_048_576


class SQLiteTaskStore:
    """A short-transaction SQLite queue with recoverable worker leases."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self._initialized = False

    @contextmanager
    def _connect(
        self,
        database_path: Path | None = None,
    ) -> Iterator[sqlite3.Connection]:
        target = database_path if database_path is not None else self.database_path
        directory_descriptor, directory_identity = _open_trusted_database_directory(
            target.parent
        )
        try:
            _validate_database_sidecars(target)
            descriptor, target_identity = _open_database_no_follow(target)
            try:
                connection = sqlite3.connect(
                    f"{target.absolute().as_uri()}?mode=rw",
                    timeout=5.0,
                    isolation_level=None,
                    uri=True,
                )
                try:
                    if not _trusted_directory_has_identity(
                        target.parent, directory_identity
                    ):
                        raise StoreError(_DATABASE_DIRECTORY_ERROR)
                    if not _owned_private_file_has_identity(target, target_identity):
                        raise StoreError("database path changed while it was opened")
                    _validate_database_sidecars(target)
                    connection.row_factory = sqlite3.Row
                    connection.execute("PRAGMA journal_mode=WAL")
                    _validate_database_sidecars(target)
                    connection.execute("PRAGMA foreign_keys=ON")
                    connection.execute("PRAGMA busy_timeout=5000")
                    yield connection
                finally:
                    connection.close()
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_descriptor)

    def initialize(self) -> None:
        self._initialized = False
        _validate_trusted_database_directory(self.database_path.parent)
        with _defer_termination_signals():
            _validate_database_sidecars(self.database_path)
            if _path_entry_exists(self.database_path):
                self._initialize_schema()
            else:
                self._initialize_fresh_database()
            self._initialized = True

    def _initialize_fresh_database(self) -> None:
        _reject_orphan_sidecars(self.database_path)
        private_path, private_identity = _create_private_database(self.database_path)
        may_own_sidecars = False
        try:
            _reject_private_sidecars(private_path)
            may_own_sidecars = True
            self._initialize_schema(private_path)
            _reject_orphan_sidecars(self.database_path)
            try:
                os.link(private_path, self.database_path, follow_symlinks=False)
            except FileExistsError:
                self._initialize_schema()
        finally:
            sidecar_identities = (
                _regular_file_identities(_database_sidecars(private_path))
                if may_own_sidecars
                else {}
            )
            _cleanup_owned_database(
                private_path,
                private_identity,
                sidecar_identities,
            )

    def _initialize_schema(self, database_path: Path | None = None) -> None:
        with self._connect(database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    operator_id TEXT NOT NULL,
                    operator_version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_until REAL,
                    worker_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    result TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS inference_jobs (
                    job_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    stage TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    model_name TEXT NOT NULL DEFAULT 'qwen3-vl-8b-instruct',
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_until REAL,
                    worker_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    affinity_worker_id TEXT,
                    affinity_fallback_at REAL,
                    completed_by TEXT,
                    UNIQUE(task_id, stage, ordinal)
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_claim
                    ON tasks(status, lease_until, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_claim
                    ON inference_jobs(status, lease_until, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_task
                    ON inference_jobs(task_id, stage, ordinal);
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                columns = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA table_info(inference_jobs)"
                    ).fetchall()
                }
                if "model_name" not in columns:
                    connection.execute(
                        "ALTER TABLE inference_jobs ADD COLUMN model_name TEXT "
                        f"NOT NULL DEFAULT '{DEFAULT_MODEL_ALIAS}'"
                    )
                    _backfill_legacy_job_model_aliases(connection)
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_model_claim
                    ON inference_jobs(model_name, status, lease_until, created_at)
                    """
                )
                connection.execute(
                    """
                    UPDATE inference_jobs
                    SET status = ?,
                        updated_at = MAX(
                            inference_jobs.updated_at,
                            (
                                SELECT parent.updated_at
                                FROM tasks AS parent
                                WHERE parent.task_id = inference_jobs.task_id
                            )
                        ),
                        lease_until = NULL,
                        worker_id = NULL,
                        attempt = attempt + 1,
                        result = NULL,
                        error = ?,
                        completed_by = NULL
                    WHERE status IN (?, ?)
                      AND EXISTS (
                          SELECT 1
                          FROM tasks AS parent
                          WHERE parent.task_id = inference_jobs.task_id
                            AND parent.status IN (?, ?)
                      )
                    """,
                    (
                        InferenceStatus.FAILED.value,
                        _json_dump("parent task is terminal"),
                        InferenceStatus.PENDING.value,
                        InferenceStatus.RUNNING.value,
                        TaskStatus.COMPLETED.value,
                        TaskStatus.FAILED.value,
                    ),
                )
                # Foreign keys prevent this state in databases created by this
                # store. Older databases may have been written with FK checks
                # disabled, so fence such jobs rather than leave them pending.
                connection.execute(
                    """
                    UPDATE inference_jobs
                    SET status = ?, lease_until = NULL, worker_id = NULL,
                        attempt = attempt + 1, result = NULL, error = ?,
                        completed_by = NULL
                    WHERE status IN (?, ?)
                      AND NOT EXISTS (
                          SELECT 1
                          FROM tasks AS parent
                          WHERE parent.task_id = inference_jobs.task_id
                      )
                    """,
                    (
                        InferenceStatus.FAILED.value,
                        _json_dump("parent task is unavailable"),
                        InferenceStatus.PENDING.value,
                        InferenceStatus.RUNNING.value,
                    ),
                )
                connection.commit()
                if database_path is not None:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            except Exception:
                connection.rollback()
                raise

    def close(self) -> None:
        """Checkpoint pending WAL pages and release the short-lived connection.

        Store operations already close their own connections.  This explicit,
        idempotent lifecycle boundary lets process entry points checkpoint the
        WAL after all worker threads have stopped, which also makes subsequent
        backup and restart behavior deterministic.
        """
        if not self._initialized or not self.database_path.is_file():
            return
        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()

    def create_task(
        self,
        payload: Mapping[str, Any],
        *,
        operator_id: str = "las_long_video_understand",
        operator_version: str = "v1",
        now: float | None = None,
    ) -> TaskRecord:
        created_at = self._now(now)
        task_id = str(uuid.uuid4())
        payload_json = _json_dump(payload)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, operator_id, operator_version, payload, status,
                    created_at, updated_at, lease_until, worker_id, attempt, result, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, NULL, NULL)
                """,
                (task_id, operator_id, operator_version, payload_json, TaskStatus.PENDING.value, created_at, created_at),
            )
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return _task_from_row(row)

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return _task_from_row(row) if row is not None else None

    def get_inference_job(self, job_id: str) -> InferenceJob | None:
        """Return one detached inference-job record, if it exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inference_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def claim_task(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        now: float | None = None,
        on_claim: Callable[[TaskRecord], None] | None = None,
    ) -> TaskRecord | None:
        registration = (
            (lambda row: on_claim(_task_from_row(row)))
            if on_claim is not None
            else None
        )
        claimed = self._claim(
            table="tasks",
            identifier="task_id",
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            now=self._now(now),
            on_claim=registration,
        )
        return _task_from_row(claimed) if claimed is not None else None

    def heartbeat_task(
        self,
        task_id: str,
        worker_id: str,
        *,
        lease_seconds: float,
        attempt: int,
        now: float | None = None,
    ) -> TaskRecord:
        row = self._heartbeat(
            table="tasks",
            identifier="task_id",
            value=task_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            attempt=attempt,
            now=self._now(now),
        )
        return _task_from_row(row)

    def expire_task_lease(
        self,
        task_id: str,
        worker_id: str,
        *,
        attempt: int,
        now: float | None = None,
    ) -> TaskRecord:
        """Make one owned task generation immediately recoverable."""
        row = self._expire_lease(
            table="tasks",
            identifier="task_id",
            value=task_id,
            worker_id=worker_id,
            attempt=attempt,
            now=self._now(now),
        )
        return _task_from_row(row)

    def complete_task(
        self,
        task_id: str,
        result: Mapping[str, Any],
        *,
        worker_id: str,
        attempt: int,
        now: float | None = None,
    ) -> TaskRecord:
        row = self._finish(
            table="tasks",
            identifier="task_id",
            value=task_id,
            worker_id=worker_id,
            attempt=attempt,
            status=TaskStatus.COMPLETED.value,
            result=result,
            error=None,
            now=self._now(now),
        )
        return _task_from_row(row)

    def fail_task(
        self,
        task_id: str,
        error: str,
        *,
        worker_id: str,
        attempt: int,
        now: float | None = None,
    ) -> TaskRecord:
        row = self._finish(
            table="tasks",
            identifier="task_id",
            value=task_id,
            worker_id=worker_id,
            attempt=attempt,
            status=TaskStatus.FAILED.value,
            result=None,
            error=error,
            now=self._now(now),
        )
        return _task_from_row(row)

    def create_inference_jobs(
        self, task_id: str, specs: Iterable[InferenceJobSpec], *, now: float | None = None
    ) -> list[InferenceJob]:
        created_at = self._now(now)
        definitions = [_job_definition(spec) for spec in specs]
        jobs: list[InferenceJob] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                parent = _required_row(connection, "tasks", "task_id", task_id)
                model_name = _task_model_alias(parent["payload"])
                if parent["status"] in {
                    TaskStatus.COMPLETED.value,
                    TaskStatus.FAILED.value,
                }:
                    raise InvalidTransition("cannot create inference jobs for a terminal task")
                for spec, absolute_fallback_at, fallback_seconds in definitions:
                    payload_json = _json_dump(spec.payload)
                    existing = connection.execute(
                        "SELECT * FROM inference_jobs WHERE task_id = ? AND stage = ? AND ordinal = ?",
                        (task_id, spec.stage, spec.ordinal),
                    ).fetchone()
                    if existing is None:
                        affinity_fallback_at = _computed_affinity_fallback(
                            created_at,
                            absolute_fallback_at,
                            fallback_seconds,
                        )
                        job_id = str(uuid.uuid4())
                        connection.execute(
                            """
                            INSERT INTO inference_jobs (
                                job_id, task_id, stage, ordinal, model_name, payload, status,
                                created_at, updated_at, lease_until, worker_id, attempt,
                                result, error, affinity_worker_id, affinity_fallback_at, completed_by
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, NULL, NULL, ?, ?, NULL)
                            """,
                            (
                                job_id,
                                task_id,
                                spec.stage,
                                spec.ordinal,
                                model_name,
                                payload_json,
                                InferenceStatus.PENDING.value,
                                created_at,
                                created_at,
                                spec.affinity_worker_id,
                                affinity_fallback_at,
                            ),
                        )
                        existing = connection.execute(
                            "SELECT * FROM inference_jobs WHERE job_id = ?", (job_id,)
                        ).fetchone()
                    elif not _same_job_definition(
                        existing,
                        model_name,
                        payload_json,
                        spec,
                        absolute_fallback_at,
                        fallback_seconds,
                    ):
                        raise DuplicateInferenceJob(
                            f"job already exists for task={task_id!r}, stage={spec.stage!r}, ordinal={spec.ordinal}"
                        )
                    jobs.append(_job_from_row(existing))
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()
        return jobs

    def claim_inference_job(
        self,
        worker_id: str,
        *,
        model_name: str = DEFAULT_MODEL_ALIAS,
        lease_seconds: float,
        now: float | None = None,
        on_claim: Callable[[InferenceJob], None] | None = None,
    ) -> InferenceJob | None:
        alias = validate_model_alias(model_name)
        registration = (
            (lambda row: on_claim(_job_from_row(row)))
            if on_claim is not None
            else None
        )
        claimed = self._claim(
            table="inference_jobs",
            identifier="job_id",
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            now=self._now(now),
            affinity=True,
            model_name=alias,
            on_claim=registration,
        )
        return _job_from_row(claimed) if claimed is not None else None

    def complete_inference_job(
        self,
        job_id: str,
        result: Mapping[str, Any],
        *,
        worker_id: str,
        attempt: int,
        now: float | None = None,
    ) -> InferenceJob:
        row = self._finish(
            table="inference_jobs",
            identifier="job_id",
            value=job_id,
            worker_id=worker_id,
            attempt=attempt,
            status=InferenceStatus.COMPLETED.value,
            result=result,
            error=None,
            now=self._now(now),
            completed_by=worker_id,
        )
        return _job_from_row(row)

    def heartbeat_inference_job(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: float,
        attempt: int,
        now: float | None = None,
    ) -> InferenceJob:
        """Renew a running inference job only for its current owner."""
        row = self._heartbeat(
            table="inference_jobs",
            identifier="job_id",
            value=job_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            attempt=attempt,
            now=self._now(now),
        )
        return _job_from_row(row)

    def expire_inference_job_lease(
        self,
        job_id: str,
        worker_id: str,
        *,
        attempt: int,
        now: float | None = None,
    ) -> InferenceJob:
        """Make one owned inference generation immediately recoverable."""
        row = self._expire_lease(
            table="inference_jobs",
            identifier="job_id",
            value=job_id,
            worker_id=worker_id,
            attempt=attempt,
            now=self._now(now),
        )
        return _job_from_row(row)

    def fail_inference_job(
        self,
        job_id: str,
        error: str,
        *,
        worker_id: str,
        attempt: int,
        now: float | None = None,
    ) -> InferenceJob:
        row = self._finish(
            table="inference_jobs",
            identifier="job_id",
            value=job_id,
            worker_id=worker_id,
            attempt=attempt,
            status=InferenceStatus.FAILED.value,
            result=None,
            error=error,
            now=self._now(now),
            completed_by=None,
        )
        return _job_from_row(row)

    def list_inference_jobs(self, task_id: str) -> list[InferenceJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM inference_jobs WHERE task_id = ? ORDER BY stage, ordinal", (task_id,)
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def _claim(
        self,
        *,
        table: str,
        identifier: str,
        worker_id: str,
        lease_seconds: float,
        now: float,
        affinity: bool = False,
        model_name: str | None = None,
        on_claim: Callable[[sqlite3.Row], None] | None = None,
    ) -> sqlite3.Row | None:
        lease_seconds = _finite_time(lease_seconds, "lease_seconds")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        lease_until = _finite_time(now + lease_seconds, "lease deadline")
        affinity_sql = ""
        affinity_params: tuple[Any, ...] = ()
        if affinity:
            affinity_sql = """
                AND (
                    affinity_worker_id IS NULL
                    OR affinity_worker_id = ?
                    OR (affinity_fallback_at IS NOT NULL AND affinity_fallback_at <= ?)
                )
            """
            affinity_params = (worker_id, now)
        model_sql = ""
        model_params: tuple[Any, ...] = ()
        if model_name is not None:
            if table != "inference_jobs":
                raise ValueError("model_name filter is only valid for inference jobs")
            model_sql = "AND model_name = ?"
            model_params = (model_name,)
        parent_sql = ""
        parent_params: tuple[Any, ...] = ()
        if table == "inference_jobs":
            parent_sql = """
                AND EXISTS (
                    SELECT 1
                    FROM tasks AS parent
                    WHERE parent.task_id = inference_jobs.task_id
                      AND parent.status IN (?, ?)
                )
            """
            parent_params = (TaskStatus.PENDING.value, TaskStatus.RUNNING.value)
        order_by = (
            "created_at, task_id, stage, ordinal, job_id"
            if affinity
            else f"created_at, {identifier}"
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"""
                    SELECT * FROM {table}
                    WHERE (status = ? OR (status = ? AND lease_until <= ?))
                    {model_sql}
                    {affinity_sql}
                    {parent_sql}
                    ORDER BY {order_by}
                    LIMIT 1
                    """,
                    (
                        "PENDING",
                        "RUNNING",
                        now,
                        *model_params,
                        *affinity_params,
                        *parent_params,
                    ),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                changed = connection.execute(
                    f"""
                    UPDATE {table}
                    SET status = ?, worker_id = ?, lease_until = ?, attempt = attempt + 1, updated_at = ?
                    WHERE {identifier} = ?
                      AND (status = ? OR (status = ? AND lease_until <= ?))
                      {model_sql}
                      {affinity_sql}
                      {parent_sql}
                    """,
                    (
                        "RUNNING",
                        worker_id,
                        lease_until,
                        now,
                        row[identifier],
                        "PENDING",
                        "RUNNING",
                        now,
                        *model_params,
                        *affinity_params,
                        *parent_params,
                    ),
                )
                if changed.rowcount != 1:
                    connection.commit()
                    return None
                claimed = connection.execute(
                    f"SELECT * FROM {table} WHERE {identifier} = ?", (row[identifier],)
                ).fetchone()
                if on_claim is not None:
                    on_claim(claimed)
                connection.commit()
                return claimed
            except BaseException:
                connection.rollback()
                raise

    def _heartbeat(
        self,
        *,
        table: str,
        identifier: str,
        value: str,
        worker_id: str,
        lease_seconds: float,
        attempt: int,
        now: float,
    ) -> sqlite3.Row:
        lease_seconds = _finite_time(lease_seconds, "lease_seconds")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        lease_until = _finite_time(now + lease_seconds, "lease deadline")
        attempt = _lease_attempt(attempt)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = _required_row(connection, table, identifier, value)
                _require_running_owner(current, worker_id, attempt)
                changed = connection.execute(
                    f"""
                    UPDATE {table} SET lease_until = ?, updated_at = ?
                    WHERE {identifier} = ? AND status = 'RUNNING'
                      AND worker_id = ? AND attempt = ?
                    """,
                    (lease_until, now, value, worker_id, attempt),
                )
                if changed.rowcount != 1:
                    raise WorkerMismatch("lease owner or generation changed")
                updated = _required_row(connection, table, identifier, value)
                connection.commit()
                return updated
            except Exception:
                connection.rollback()
                raise

    def _finish(
        self,
        *,
        table: str,
        identifier: str,
        value: str,
        worker_id: str,
        attempt: int,
        status: str,
        result: Mapping[str, Any] | None,
        error: str | None,
        now: float,
        completed_by: str | None = None,
    ) -> sqlite3.Row:
        result_json = _json_dump(result) if result is not None else None
        error_json = _json_dump(error) if error is not None else None
        attempt = _lease_attempt(attempt)
        completed_by_sql = ", completed_by = ?" if table == "inference_jobs" else ""
        params: tuple[Any, ...] = (status, now, result_json, error_json, *(() if table == "tasks" else (completed_by,)), value)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = _required_row(connection, table, identifier, value)
                _require_running_owner(current, worker_id, attempt)
                changed = connection.execute(
                    f"""
                    UPDATE {table}
                    SET status = ?, updated_at = ?, lease_until = NULL, result = ?, error = ?{completed_by_sql}
                    WHERE {identifier} = ? AND status = 'RUNNING'
                      AND worker_id = ? AND attempt = ?
                    """,
                    (*params, worker_id, attempt),
                )
                if changed.rowcount != 1:
                    raise WorkerMismatch("lease owner or generation changed")
                if table == "tasks":
                    connection.execute(
                        """
                        UPDATE inference_jobs
                        SET status = ?, updated_at = ?, lease_until = NULL,
                            worker_id = NULL, attempt = attempt + 1,
                            result = NULL, error = ?, completed_by = NULL
                        WHERE task_id = ? AND status IN (?, ?)
                        """,
                        (
                            InferenceStatus.FAILED.value,
                            now,
                            _json_dump("parent task is terminal"),
                            value,
                            InferenceStatus.PENDING.value,
                            InferenceStatus.RUNNING.value,
                        ),
                    )
                updated = _required_row(connection, table, identifier, value)
                connection.commit()
                return updated
            except Exception:
                connection.rollback()
                raise

    def _expire_lease(
        self,
        *,
        table: str,
        identifier: str,
        value: str,
        worker_id: str,
        attempt: int,
        now: float,
    ) -> sqlite3.Row:
        attempt = _lease_attempt(attempt)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = _required_row(connection, table, identifier, value)
                _require_running_owner(current, worker_id, attempt)
                # Revoking the owner is the fence.  A heartbeat that resumes
                # after this transaction cannot extend the interrupted claim.
                changed = connection.execute(
                    f"""
                    UPDATE {table}
                    SET lease_until = ?, updated_at = ?, worker_id = NULL
                    WHERE {identifier} = ? AND status = 'RUNNING'
                      AND worker_id = ? AND attempt = ?
                    """,
                    (now, now, value, worker_id, attempt),
                )
                if changed.rowcount != 1:
                    raise WorkerMismatch("lease owner or generation changed")
                updated = _required_row(connection, table, identifier, value)
                connection.commit()
                return updated
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _now(value: float | None) -> float:
        return _finite_time(time.time() if value is None else value, "now")


def _required_row(connection: sqlite3.Connection, table: str, identifier: str, value: str) -> sqlite3.Row:
    row = connection.execute(f"SELECT * FROM {table} WHERE {identifier} = ?", (value,)).fetchone()
    if row is None:
        raise KeyError(value)
    return row


def _require_running_owner(row: sqlite3.Row, worker_id: str, attempt: int) -> None:
    if row["status"] != "RUNNING":
        raise InvalidTransition(f"cannot transition {row['status']}")
    if row["worker_id"] != worker_id:
        raise WorkerMismatch(f"lease belongs to {row['worker_id']!r}")
    if row["attempt"] != attempt:
        raise WorkerMismatch("lease generation no longer belongs to this claim")


def _lease_attempt(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("attempt must be a non-negative integer")
    return value


def _same_job_definition(
    row: sqlite3.Row,
    model_name: str,
    payload_json: str,
    spec: InferenceJobSpec,
    absolute_fallback_at: float | None,
    fallback_seconds: float | None,
) -> bool:
    expected_fallback_at = _computed_affinity_fallback(
        row["created_at"],
        absolute_fallback_at,
        fallback_seconds,
    )
    return (
        row["model_name"] == model_name
        and row["payload"] == payload_json
        and row["affinity_worker_id"] == spec.affinity_worker_id
        and row["affinity_fallback_at"] == expected_fallback_at
    )


def _job_definition(
    spec: InferenceJobSpec,
) -> tuple[InferenceJobSpec, float | None, float | None]:
    absolute = _optional_finite_time(
        spec.affinity_fallback_at,
        "affinity_fallback_at",
    )
    relative = _optional_finite_time(
        spec.affinity_fallback_seconds,
        "affinity_fallback_seconds",
    )
    if absolute is not None and relative is not None:
        raise ValueError(
            "affinity_fallback_at and affinity_fallback_seconds are mutually exclusive"
        )
    if relative is not None and relative <= 0:
        raise ValueError("affinity_fallback_seconds must be positive")
    return spec, absolute, relative


def _computed_affinity_fallback(
    created_at: float,
    absolute_fallback_at: float | None,
    fallback_seconds: float | None,
) -> float | None:
    if fallback_seconds is None:
        return absolute_fallback_at
    return _finite_time(
        created_at + fallback_seconds,
        "affinity fallback deadline",
    )


def _create_private_database(path: Path) -> tuple[Path, tuple[int, int]]:
    descriptor, private_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".init",
        dir=path.parent,
    )
    private_path = Path(private_name)
    identity: tuple[int, int] | None = None
    try:
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        os.fchmod(descriptor, 0o600)
        return private_path, identity
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        if identity is not None:
            _cleanup_owned_database(private_path, identity, {})
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _reject_orphan_sidecars(path: Path) -> None:
    if _path_entry_exists(path):
        return
    if any(_path_entry_exists(sidecar) for sidecar in _database_sidecars(path)):
        raise StoreError("database sidecar exists without its database")


def _reject_private_sidecars(path: Path) -> None:
    if any(_path_entry_exists(sidecar) for sidecar in _database_sidecars(path)):
        raise StoreError("private database sidecar existed before initialization")


def _cleanup_owned_database(
    path: Path,
    identity: tuple[int, int],
    sidecar_identities: Mapping[Path, tuple[int, int]],
) -> None:
    if not _regular_file_has_identity(path, identity):
        return
    for sidecar in _database_sidecars(path):
        if not _regular_file_has_identity(path, identity):
            return
        sidecar_identity = sidecar_identities.get(sidecar)
        if sidecar_identity is not None:
            _quarantine_and_unlink_owned_file(sidecar, sidecar_identity)
    if _regular_file_has_identity(path, identity):
        _quarantine_and_unlink_owned_file(path, identity, identity_already_checked=True)


def _quarantine_and_unlink_owned_file(
    path: Path,
    identity: tuple[int, int],
    *,
    identity_already_checked: bool = False,
) -> None:
    """Remove an owned inode without deleting a late pathname replacement.

    Moving the entry onto a reserved random name creates an atomic boundary. If
    a replacement won after the identity check, its inode is detected at the
    quarantine name and linked back instead of being unlinked. A competing
    entry at the original name leaves the replacement preserved in quarantine.
    Replacement of the unpredictable quarantine name by another same-privilege
    directory mutator is outside this local-work-directory trust boundary.
    """
    if not identity_already_checked and not _regular_file_has_identity(path, identity):
        return
    descriptor, quarantine_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".cleanup",
        dir=path.parent,
    )
    quarantine = Path(quarantine_name)
    placeholder = os.fstat(descriptor)
    placeholder_identity = (placeholder.st_dev, placeholder.st_ino)
    os.close(descriptor)
    preserve_quarantine = False
    try:
        try:
            os.replace(path, quarantine)
        except FileNotFoundError:
            return
        if _regular_file_has_identity(quarantine, identity):
            _unlink_regular_file(quarantine, identity)
            return

        moved_identity = _regular_file_identity(quarantine)
        if moved_identity is None:
            preserve_quarantine = True
            return
        try:
            os.link(quarantine, path, follow_symlinks=False)
        except FileExistsError:
            preserve_quarantine = True
            return
        _unlink_regular_file(quarantine, moved_identity)
    finally:
        if not preserve_quarantine:
            _unlink_regular_file(quarantine, placeholder_identity)


def _database_sidecars(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(f"{path.name}-wal"),
        path.with_name(f"{path.name}-shm"),
    )


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _validate_trusted_database_directory(path: Path) -> tuple[int, int]:
    try:
        status = path.lstat()
    except (FileNotFoundError, OSError):
        raise StoreError(_DATABASE_DIRECTORY_ERROR) from None
    if not stat.S_ISDIR(status.st_mode):
        raise StoreError(_DATABASE_DIRECTORY_ERROR)
    if os.name == "posix":
        effective_uid = os.geteuid()
        if status.st_uid != effective_uid or stat.S_IMODE(status.st_mode) & 0o022:
            raise StoreError(_DATABASE_DIRECTORY_ERROR)
        _validate_trusted_database_ancestry(path, effective_uid)
    return status.st_dev, status.st_ino


def _validate_trusted_database_ancestry(path: Path, effective_uid: int) -> None:
    absolute = path.absolute()
    for component in reversed((absolute, *absolute.parents)):
        try:
            status = component.lstat()
        except OSError:
            raise StoreError(_DATABASE_ANCESTRY_ERROR) from None
        permissions = stat.S_IMODE(status.st_mode)
        writable_by_others = bool(permissions & 0o022)
        sticky = bool(permissions & stat.S_ISVTX)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid not in {0, effective_uid}
            or (writable_by_others and not sticky)
        ):
            raise StoreError(_DATABASE_ANCESTRY_ERROR)


def _open_trusted_database_directory(path: Path) -> tuple[int, tuple[int, int]]:
    expected_identity = _validate_trusted_database_directory(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise StoreError(_DATABASE_DIRECTORY_ERROR) from None
    try:
        opened = os.fstat(descriptor)
        opened_identity = opened.st_dev, opened.st_ino
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened_identity != expected_identity
            or not _trusted_directory_has_identity(path, opened_identity)
        ):
            raise StoreError(_DATABASE_DIRECTORY_ERROR)
        return descriptor, opened_identity
    except BaseException:
        os.close(descriptor)
        raise


def _trusted_directory_has_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        return _validate_trusted_database_directory(path) == identity
    except StoreError:
        return False


def _open_database_no_follow(path: Path) -> tuple[int, tuple[int, int]]:
    expected_identity = _owned_private_file_identity(path, _DATABASE_FILE_ERROR)
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise StoreError(_DATABASE_FILE_ERROR) from None
    try:
        opened = os.fstat(descriptor)
        opened_identity = opened.st_dev, opened.st_ino
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened_identity != expected_identity
            or not _owned_private_file_has_identity(path, opened_identity)
        ):
            raise StoreError("database path changed while it was opened")
        return descriptor, opened_identity
    except BaseException:
        os.close(descriptor)
        raise


def _validate_database_sidecars(path: Path) -> None:
    for sidecar in _database_sidecars(path):
        try:
            status = sidecar.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise StoreError(_DATABASE_SIDECAR_ERROR) from None
        _owned_private_file_identity_from_status(status, _DATABASE_SIDECAR_ERROR)


def _owned_private_file_identity(path: Path, error_message: str) -> tuple[int, int]:
    try:
        status = path.lstat()
    except (FileNotFoundError, OSError):
        raise StoreError(error_message) from None
    return _owned_private_file_identity_from_status(status, error_message)


def _owned_private_file_identity_from_status(
    status: os.stat_result,
    error_message: str,
) -> tuple[int, int]:
    if not stat.S_ISREG(status.st_mode):
        raise StoreError(error_message)
    if os.name == "posix":
        if os.geteuid() != status.st_uid or stat.S_IMODE(status.st_mode) != 0o600:
            raise StoreError(error_message)
    return status.st_dev, status.st_ino


def _owned_private_file_has_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        return _owned_private_file_identity(path, _DATABASE_FILE_ERROR) == identity
    except StoreError:
        return False


def _regular_file_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(status.st_mode):
        return None
    return status.st_dev, status.st_ino


def _regular_file_identities(
    paths: Iterable[Path],
) -> dict[Path, tuple[int, int]]:
    return {
        path: identity
        for path in paths
        if (identity := _regular_file_identity(path)) is not None
    }


def _regular_file_has_identity(path: Path, identity: tuple[int, int]) -> bool:
    return _regular_file_identity(path) == identity


def _unlink_regular_file(path: Path, identity: tuple[int, int]) -> None:
    if not _regular_file_has_identity(path, identity):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def _defer_termination_signals() -> Iterator[None]:
    """Deliver TERM/INT only after one store initialization is complete."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handled = tuple(
        item
        for item in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None))
        if item is not None
    )
    mask_signals = getattr(signal, "pthread_sigmask", None)
    entry_mask: set[signal.Signals] | None = None
    if callable(mask_signals):
        entry_mask = mask_signals(signal.SIG_BLOCK, handled)
    previous = {item: signal.getsignal(item) for item in handled}
    deferred: list[tuple[int, Any]] = []

    def remember(signum: int, frame: Any) -> None:
        deferred.append((signum, frame))

    operation_error: BaseException | None = None
    operation_traceback: Any = None
    cleanup_error: BaseException | None = None
    cleanup_traceback: Any = None
    try:
        try:
            for item in handled:
                signal.signal(item, remember)
            if entry_mask is not None:
                mask_signals(signal.SIG_SETMASK, entry_mask)
            yield
        except BaseException as error:
            # Keep the initialization failure alive across handler restoration.
            # Unmasking can synchronously raise the outer CLI's shutdown
            # exception; that signal must not turn a corrupt/failed bootstrap
            # into a clean exit.
            operation_error = error
            operation_traceback = error.__traceback__
    finally:
        try:
            try:
                if callable(mask_signals):
                    mask_signals(signal.SIG_BLOCK, handled)
                    _remember_pending_signals(handled, entry_mask, deferred)
            finally:
                try:
                    for item, handler in previous.items():
                        signal.signal(item, handler)
                finally:
                    if callable(mask_signals) and entry_mask is not None:
                        mask_signals(signal.SIG_SETMASK, entry_mask)
        except BaseException as error:
            cleanup_error = error
            cleanup_traceback = error.__traceback__

    if operation_error is not None:
        raise operation_error.with_traceback(operation_traceback)
    if cleanup_error is not None:
        raise cleanup_error.with_traceback(cleanup_traceback)

    for signum, frame in deferred:
        handler = previous[signum]
        if handler is signal.SIG_IGN:
            continue
        if handler is signal.SIG_DFL:
            signal.raise_signal(signum)
        else:
            handler(signum, frame)


def _remember_pending_signals(
    handled: tuple[signal.Signals, ...],
    entry_mask: set[signal.Signals] | None,
    deferred: list[tuple[int, Any]],
) -> None:
    pending_signals = getattr(signal, "sigpending", None)
    wait_signal = getattr(signal, "sigwait", None)
    if not callable(pending_signals) or not callable(wait_signal):
        return
    pending = pending_signals()
    originally_blocked = entry_mask or set()
    for signum in handled:
        if signum in pending and signum not in originally_blocked:
            deferred.append((wait_signal({signum}), None))


def _finite_time(value: Real, name: str) -> float:
    """Return a SQLite-safe timestamp or duration without silently coercing input."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real number")
    return numeric


def _optional_finite_time(value: float | None, name: str) -> float | None:
    return None if value is None else _finite_time(value, name)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_load(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _task_model_alias(payload_json: str) -> str:
    try:
        payload = _json_load(payload_json)
        if not isinstance(payload, Mapping):
            raise ValueError
        return validate_model_alias(payload.get("model_name", DEFAULT_MODEL_ALIAS))
    except (TypeError, ValueError, json.JSONDecodeError):
        raise StoreError("task model alias is invalid") from None


def _backfill_legacy_job_model_aliases(connection: sqlite3.Connection) -> None:
    """Bind pre-column jobs to their parent's validated model alias.

    This runs only in the transaction that adds ``model_name``.  Keyset batches
    bound Python memory use, while the byte check happens inside SQLite before
    any legacy payload is materialized in Python.
    """
    missing_parent = connection.execute(
        """
        SELECT 1
        FROM inference_jobs AS job
        LEFT JOIN tasks AS parent ON parent.task_id = job.task_id
        WHERE parent.task_id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if missing_parent is not None:
        raise StoreError("inference job parent task is unavailable")

    oversized_payload = connection.execute(
        """
        SELECT 1
        FROM tasks AS parent
        WHERE length(CAST(parent.payload AS BLOB)) > ?
          AND EXISTS (
              SELECT 1
              FROM inference_jobs AS job
              WHERE job.task_id = parent.task_id
          )
        LIMIT 1
        """,
        (_MAX_LEGACY_TASK_PAYLOAD_BYTES,),
    ).fetchone()
    if oversized_payload is not None:
        raise StoreError("task payload exceeds migration limit")

    last_task_id: str | None = None
    while True:
        parents = connection.execute(
            """
            SELECT parent.task_id, parent.payload
            FROM tasks AS parent
            WHERE (? IS NULL OR parent.task_id > ?)
              AND EXISTS (
                  SELECT 1
                  FROM inference_jobs AS job
                  WHERE job.task_id = parent.task_id
              )
            ORDER BY parent.task_id
            LIMIT ?
            """,
            (
                last_task_id,
                last_task_id,
                _LEGACY_MODEL_MIGRATION_BATCH_SIZE,
            ),
        ).fetchall()
        if not parents:
            return
        for parent in parents:
            model_name = _task_model_alias(parent["payload"])
            connection.execute(
                "UPDATE inference_jobs SET model_name = ? WHERE task_id = ?",
                (model_name, parent["task_id"]),
            )
        last_task_id = parents[-1]["task_id"]


def _task_from_row(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"],
        operator_id=row["operator_id"],
        operator_version=row["operator_version"],
        payload=_json_load(row["payload"]),
        status=TaskStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        lease_until=row["lease_until"],
        worker_id=row["worker_id"],
        attempt=row["attempt"],
        result=_json_load(row["result"]),
        error=_json_load(row["error"]),
    )


def _job_from_row(row: sqlite3.Row) -> InferenceJob:
    return InferenceJob(
        job_id=row["job_id"],
        task_id=row["task_id"],
        stage=row["stage"],
        ordinal=row["ordinal"],
        model_name=row["model_name"],
        payload=_json_load(row["payload"]),
        status=InferenceStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        lease_until=row["lease_until"],
        worker_id=row["worker_id"],
        attempt=row["attempt"],
        result=_json_load(row["result"]),
        error=_json_load(row["error"]),
        affinity_worker_id=row["affinity_worker_id"],
        affinity_fallback_at=row["affinity_fallback_at"],
        completed_by=row["completed_by"],
    )
