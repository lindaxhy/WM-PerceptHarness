from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import gc
import json
import os
import sqlite3
import stat
import sys
import threading
from pathlib import Path

import pytest

from las_repro.domain import InferenceJobSpec, InferenceStatus, TaskStatus
from las_repro.store import (
    DuplicateInferenceJob,
    InvalidTransition,
    SQLiteTaskStore,
    StoreError,
    WorkerMismatch,
)


@pytest.fixture
def store(tmp_path):
    value = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    value.initialize()
    return value


def test_claim_is_atomic_and_expired_lease_is_recoverable(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})

    first = store.claim_task("coordinator-a", lease_seconds=1, now=100.0)
    assert first is not None
    assert first.task_id == task.task_id
    assert store.claim_task("coordinator-b", lease_seconds=1, now=100.5) is None

    recovered = store.claim_task("coordinator-b", lease_seconds=1, now=101.1)
    assert recovered is not None
    assert recovered.task_id == task.task_id
    assert recovered.attempt == 2


def test_claim_registration_interrupt_rolls_back_before_commit(store):
    task = store.create_task({"video_url": "/v.mp4"})
    registered = []

    def interrupt(claimed):
        registered.append(claimed)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        store.claim_task(
            "coordinator",
            lease_seconds=30.0,
            now=100.0,
            on_claim=interrupt,
        )

    assert [item.task_id for item in registered] == [task.task_id]
    current = store.get_task(task.task_id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.worker_id is None
    assert current.attempt == 0


def test_terminal_task_cannot_transition(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    claimed = store.claim_task("w", lease_seconds=10, now=1)
    assert claimed is not None
    store.complete_task(
        task.task_id, {"summary": "done"}, worker_id="w", attempt=claimed.attempt
    )

    with pytest.raises(InvalidTransition):
        store.fail_task(
            task.task_id, "late failure", worker_id="w", attempt=claimed.attempt
        )


def test_concurrent_task_claims_have_exactly_one_winner(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})

    with ThreadPoolExecutor(max_workers=20) as executor:
        claimed = list(
            executor.map(
                lambda index: store.claim_task(f"worker-{index}", lease_seconds=10, now=100.0),
                range(20),
            )
        )

    winners = [item for item in claimed if item is not None]
    assert [item.task_id for item in winners] == [task.task_id]
    assert winners[0].attempt == 1


def test_task_json_values_are_detached_from_callers_and_reads(store):
    payload = {"video_url": "/v.mp4", "task_template": "general_video_captioning", "tuning": {"fps": 2}}
    task = store.create_task(payload)
    payload["tuning"]["fps"] = 99

    first = store.get_task(task.task_id)
    assert first is not None
    assert first.payload["tuning"] == {"fps": 2}
    first.payload["tuning"]["fps"] = 3

    second = store.get_task(task.task_id)
    assert second is not None
    assert second.payload["tuning"] == {"fps": 2}


def test_task_owner_must_match_for_heartbeat_and_completion(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    claimed = store.claim_task("owner", lease_seconds=10, now=100.0)
    assert claimed is not None

    with pytest.raises(WorkerMismatch):
        store.heartbeat_task(
            task.task_id,
            "other",
            lease_seconds=10,
            attempt=claimed.attempt,
            now=101.0,
        )
    with pytest.raises(WorkerMismatch):
        store.complete_task(
            task.task_id,
            {"summary": "wrong worker"},
            worker_id="other",
            attempt=claimed.attempt,
        )


def test_reused_task_worker_id_cannot_renew_or_finish_an_expired_generation(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    expired = store.claim_task("coordinator", lease_seconds=1, now=100.0)
    replacement = store.claim_task("coordinator", lease_seconds=10, now=101.1)
    assert expired is not None
    assert replacement is not None
    assert replacement.attempt == expired.attempt + 1

    with pytest.raises(WorkerMismatch):
        store.heartbeat_task(
            task.task_id,
            "coordinator",
            lease_seconds=10,
            attempt=expired.attempt,
            now=101.2,
        )
    with pytest.raises(WorkerMismatch):
        store.complete_task(
            task.task_id,
            {"summary": "stale"},
            worker_id="coordinator",
            attempt=expired.attempt,
            now=101.2,
        )

    current = store.get_task(task.task_id)
    assert current is not None
    assert current.status is TaskStatus.RUNNING
    assert current.attempt == replacement.attempt
    assert current.result is None


def test_task_state_machine_rejects_pending_completion_and_keeps_terminal_data(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    with pytest.raises(InvalidTransition):
        store.complete_task(
            task.task_id, {"summary": "too early"}, worker_id="w", attempt=0
        )

    claimed = store.claim_task("w", lease_seconds=10, now=1.0)
    assert claimed is not None
    completed = store.complete_task(
        task.task_id,
        {"summary": "done"},
        worker_id="w",
        attempt=claimed.attempt,
        now=2.0,
    )

    assert completed.status is TaskStatus.COMPLETED
    assert completed.worker_id == "w"
    assert completed.lease_until is None
    assert completed.result == {"summary": "done"}
    assert store.get_task(task.task_id) == completed


def test_inference_jobs_are_idempotent_and_reject_conflicting_duplicate(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    specs = [
        InferenceJobSpec("coarse", 0, {"start": 0}, affinity_worker_id="gpu-a", affinity_fallback_at=50.0),
        InferenceJobSpec("coarse", 1, {"start": 30}),
    ]

    created = store.create_inference_jobs(task.task_id, specs, now=10.0)
    repeated = store.create_inference_jobs(task.task_id, specs, now=20.0)

    assert [job.job_id for job in repeated] == [job.job_id for job in created]
    assert [job.created_at for job in repeated] == [10.0, 10.0]
    with pytest.raises(DuplicateInferenceJob):
        store.create_inference_jobs(task.task_id, [InferenceJobSpec("coarse", 0, {"start": 1})])


def test_relative_affinity_deadline_uses_durable_job_creation_time_and_is_idempotent(store):
    """A follow-up must receive its own full grace window after long prior stages."""
    task = store.create_task(
        {"video_url": "/v.mp4", "task_template": "embodied_action_captioning"}
    )
    spec = InferenceJobSpec(
        "embodied_enrichment",
        0,
        {"stage": "enrichment"},
        affinity_worker_id="gpu-a",
        affinity_fallback_seconds=2.5,
    )

    [created] = store.create_inference_jobs(task.task_id, [spec], now=250.0)
    [repeated] = store.create_inference_jobs(task.task_id, [spec], now=900.0)

    assert created.created_at == 250.0
    assert created.affinity_fallback_at == 252.5
    assert repeated.job_id == created.job_id
    assert repeated.affinity_fallback_at == created.affinity_fallback_at
    assert store.claim_inference_job("gpu-b", lease_seconds=1.0, now=252.49) is None
    assert store.claim_inference_job("gpu-a", lease_seconds=1.0, now=252.49) is not None


def test_affinity_definition_rejects_absolute_and_relative_deadlines_together(store):
    task = store.create_task(
        {"video_url": "/v.mp4", "task_template": "embodied_action_captioning"}
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        store.create_inference_jobs(
            task.task_id,
            [
                InferenceJobSpec(
                    "coarse",
                    0,
                    {},
                    affinity_worker_id="gpu-a",
                    affinity_fallback_at=5.0,
                    affinity_fallback_seconds=1.0,
                )
            ],
        )


def test_store_explicitly_closes_every_short_lived_sqlite_connection(
    tmp_path, monkeypatch
):
    """Queue polling must not rely on cyclic GC to release database descriptors."""
    opened = 0
    closed = 0
    real_connect = sqlite3.connect

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            nonlocal closed
            closed += 1
            return super().close()

    def tracking_connect(*args, **kwargs):
        nonlocal opened
        opened += 1
        return real_connect(*args, **kwargs, factory=TrackingConnection)

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    gc.disable()
    try:
        tracked = SQLiteTaskStore(tmp_path / "tracked.sqlite3")
        tracked.initialize()
        task = tracked.create_task(
            {"video_url": "/v.mp4", "task_template": "general_video_captioning"}
        )
        for _ in range(300):
            assert tracked.get_task(task.task_id) is not None
            assert tracked.list_inference_jobs(task.task_id) == []
    finally:
        gc.enable()

    assert opened == closed


def test_fresh_initialize_failure_removes_only_its_database_and_sidecars(
    tmp_path,
    monkeypatch,
):
    """Leaving the first-created file would publish an unusable empty schema."""
    database_path = tmp_path / "failed.sqlite3"
    real_connect = sqlite3.connect

    def deny_schema_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)

        def authorizer(action, *_):
            if action == sqlite3.SQLITE_CREATE_TABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        return connection

    monkeypatch.setattr(sqlite3, "connect", deny_schema_connect)
    store = SQLiteTaskStore(database_path)

    with pytest.raises(sqlite3.DatabaseError):
        store.initialize()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_initialize_rejects_a_missing_database_directory_without_creating_it(tmp_path):
    database_path = tmp_path / "missing" / "tasks.sqlite3"

    with pytest.raises(
        StoreError,
        match="database directory must be an owned non-writable-by-others directory",
    ):
        SQLiteTaskStore(database_path).initialize()

    assert not database_path.parent.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
@pytest.mark.parametrize("mode", [0o720, 0o707], ids=("group-writable", "world-writable"))
def test_initialize_rejects_an_unsafe_database_directory_before_mutation(
    tmp_path,
    mode,
):
    database_directory = tmp_path / "unsafe"
    database_directory.mkdir(mode=0o700)
    database_directory.chmod(mode)

    with pytest.raises(
        StoreError,
        match="database directory must be an owned non-writable-by-others directory",
    ):
        SQLiteTaskStore(database_directory / "tasks.sqlite3").initialize()

    assert list(database_directory.iterdir()) == []
    assert stat.S_IMODE(database_directory.stat().st_mode) == mode


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_initialize_rejects_a_database_directory_not_owned_by_effective_uid(
    tmp_path,
    monkeypatch,
):
    database_directory = tmp_path / "foreign-owner"
    database_directory.mkdir(mode=0o700)
    actual_uid = os.geteuid()
    monkeypatch.setattr(os, "geteuid", lambda: actual_uid + 1)

    with pytest.raises(
        StoreError,
        match="database directory must be an owned non-writable-by-others directory",
    ):
        SQLiteTaskStore(database_directory / "tasks.sqlite3").initialize()

    assert list(database_directory.iterdir()) == []


def test_initialize_rejects_a_symlink_database_directory_before_mutation(tmp_path):
    real_directory = tmp_path / "real-database-directory"
    real_directory.mkdir(mode=0o700)
    configured_directory = tmp_path / "configured-database-directory"
    configured_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(
        StoreError,
        match="database directory must be an owned non-writable-by-others directory",
    ):
        SQLiteTaskStore(configured_directory / "tasks.sqlite3").initialize()

    assert list(real_directory.iterdir()) == []
    assert configured_directory.is_symlink()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_initialize_rejects_a_private_database_directory_below_unsafe_ancestry(
    tmp_path,
):
    unsafe_ancestor = tmp_path / "unsafe-ancestor"
    unsafe_ancestor.mkdir(mode=0o700)
    unsafe_ancestor.chmod(0o777)
    database_directory = unsafe_ancestor / "private-database"
    database_directory.mkdir(mode=0o700)
    database_path = database_directory / "tasks.sqlite3"

    with pytest.raises(
        StoreError,
        match="database directory must have trusted POSIX ancestry",
    ):
        SQLiteTaskStore(database_path).initialize()

    assert not database_path.exists()
    assert list(database_directory.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_initialize_accepts_linux_style_sticky_temporary_ancestry(tmp_path):
    sticky_ancestor = tmp_path / "linux-tmp"
    sticky_ancestor.mkdir(mode=0o700)
    sticky_ancestor.chmod(0o1777)
    database_directory = sticky_ancestor / "service-owned"
    database_directory.mkdir(mode=0o700)
    database_path = database_directory / "tasks.sqlite3"

    SQLiteTaskStore(database_path).initialize()

    assert database_path.is_file()
    assert stat.S_IMODE(database_path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_initialize_rejects_an_intermediate_symlink_in_database_ancestry(tmp_path):
    real_ancestor = tmp_path / "real-ancestor"
    database_directory = real_ancestor / "private-database"
    database_directory.mkdir(mode=0o700, parents=True)
    configured_ancestor = tmp_path / "configured-ancestor"
    configured_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    database_path = configured_ancestor / "private-database" / "tasks.sqlite3"

    with pytest.raises(
        StoreError,
        match="database directory must have trusted POSIX ancestry",
    ):
        SQLiteTaskStore(database_path).initialize()

    assert not (database_directory / "tasks.sqlite3").exists()
    assert configured_ancestor.is_symlink()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_initialize_accepts_owned_non_writable_by_others_database_directory(tmp_path):
    database_directory = tmp_path / "trusted"
    database_directory.mkdir(mode=0o755)
    database_path = database_directory / "tasks.sqlite3"

    SQLiteTaskStore(database_path).initialize()

    assert database_path.is_file()
    assert stat.S_IMODE(database_path.stat().st_mode) == 0o600


def test_fresh_reservation_permission_failure_removes_its_empty_database(
    tmp_path,
    monkeypatch,
):
    """Failure after private-file creation must leave no bootstrap artifact."""
    database_path = tmp_path / "permission-failed.sqlite3"

    def fail_permissions(*_):
        raise OSError("injected permission failure")

    monkeypatch.setattr(os, "fchmod", fail_permissions)
    store = SQLiteTaskStore(database_path)

    with pytest.raises(OSError, match="permission failure"):
        store.initialize()

    assert list(tmp_path.iterdir()) == []


def test_fresh_initialize_failure_never_deletes_a_concurrently_replaced_database(
    tmp_path,
    monkeypatch,
):
    """Cleanup must compare the owned inode before removing a fresh path."""
    database_path = tmp_path / "raced.sqlite3"
    replacement_path = tmp_path / "replacement.sqlite3"
    real_connect = sqlite3.connect
    with real_connect(replacement_path) as replacement:
        replacement.execute("CREATE TABLE preserved (value TEXT NOT NULL)")
        replacement.execute("INSERT INTO preserved VALUES ('replacement-won')")
    replaced = False

    def replace_after_connect(*args, **kwargs):
        nonlocal replaced
        connection = real_connect(*args, **kwargs)
        if not replaced:
            replaced = True
            connection.close()
            os.replace(replacement_path, database_path)
            raise RuntimeError("injected connect publication race")
        return connection

    monkeypatch.setattr(sqlite3, "connect", replace_after_connect)
    store = SQLiteTaskStore(database_path)

    with pytest.raises(RuntimeError, match="publication race"):
        store.initialize()

    assert replaced is True
    assert database_path.is_file()
    with real_connect(database_path) as preserved:
        assert preserved.execute("SELECT value FROM preserved").fetchone() == (
            "replacement-won",
        )


def test_failed_fresh_initializer_never_deletes_a_concurrent_success(
    tmp_path,
    monkeypatch,
):
    """A private failed bootstrap must not own another initializer's database."""
    database_path = tmp_path / "overlap.sqlite3"
    first_inside_schema = threading.Event()
    release_first = threading.Event()
    first_errors = []
    original_initialize_schema = SQLiteTaskStore._initialize_schema

    def coordinate_schema(store, *args):
        if threading.current_thread().name == "failing-initializer":
            first_inside_schema.set()
            assert release_first.wait(timeout=3.0)
            raise RuntimeError("first initializer failed")
        return original_initialize_schema(store, *args)

    monkeypatch.setattr(SQLiteTaskStore, "_initialize_schema", coordinate_schema)

    def fail_first():
        try:
            SQLiteTaskStore(database_path).initialize()
        except BaseException as error:
            first_errors.append(error)

    first = threading.Thread(target=fail_first, name="failing-initializer")
    first.start()
    assert first_inside_schema.wait(timeout=3.0)
    try:
        successful = SQLiteTaskStore(database_path)
        successful.initialize()
        accepted = successful.create_task({"video_url": "/accepted.mp4"})
    finally:
        release_first.set()
        first.join(timeout=3.0)

    assert not first.is_alive()
    assert len(first_errors) == 1
    assert isinstance(first_errors[0], RuntimeError)
    assert database_path.is_file()
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"tasks", "inference_jobs"} <= tables
        assert connection.execute(
            "SELECT task_id FROM tasks"
        ).fetchone() == (accepted.task_id,)


def test_fresh_initialize_rejects_and_preserves_orphan_sidecars_byte_for_byte(
    tmp_path,
):
    """A fresh database must never let SQLite consume preexisting sidecars."""
    database_path = tmp_path / "orphaned.sqlite3"
    wal_path = database_path.with_name(f"{database_path.name}-wal")
    shm_path = database_path.with_name(f"{database_path.name}-shm")
    wal_path.write_bytes(b"preexisting-wal-marker")
    shm_path.write_bytes(b"preexisting-shm-marker")

    with pytest.raises(StoreError, match="sidecar"):
        SQLiteTaskStore(database_path).initialize()

    assert not database_path.exists()
    assert wal_path.read_bytes() == b"preexisting-wal-marker"
    assert shm_path.read_bytes() == b"preexisting-shm-marker"


def test_initialize_rejects_a_dangling_database_symlink_without_touching_target(
    tmp_path,
):
    database_path = tmp_path / "configured.sqlite3"
    redirected_path = tmp_path / "redirected.sqlite3"
    database_path.symlink_to(redirected_path)

    with pytest.raises(StoreError, match="regular file"):
        SQLiteTaskStore(database_path).initialize()

    assert database_path.is_symlink()
    assert not redirected_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_initialize_rejects_an_existing_database_with_non_private_mode(tmp_path):
    database_path = tmp_path / "configured.sqlite3"
    SQLiteTaskStore(database_path).initialize()
    before = database_path.read_bytes()
    database_path.chmod(0o640)

    with pytest.raises(
        StoreError,
        match="database file must be an owner-only regular file",
    ):
        SQLiteTaskStore(database_path).initialize()

    assert database_path.read_bytes() == before
    assert stat.S_IMODE(database_path.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_initialize_rejects_a_non_private_database_sidecar_without_mutating_it(
    tmp_path,
    suffix,
):
    database_path = tmp_path / "configured.sqlite3"
    SQLiteTaskStore(database_path).initialize()
    sidecar = database_path.with_name(database_path.name + suffix)
    marker = b"preexisting-sidecar-must-remain"
    sidecar.write_bytes(marker)
    sidecar.chmod(0o644)

    with pytest.raises(
        StoreError,
        match="database sidecar must be an owner-only regular file",
    ):
        SQLiteTaskStore(database_path).initialize()

    assert sidecar.read_bytes() == marker
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_initialize_rejects_a_symlink_database_sidecar_without_touching_target(
    tmp_path,
    suffix,
):
    database_path = tmp_path / "configured.sqlite3"
    SQLiteTaskStore(database_path).initialize()
    victim = tmp_path / "sidecar-victim"
    marker = b"sidecar target must remain untouched"
    victim.write_bytes(marker)
    victim.chmod(0o600)
    sidecar = database_path.with_name(database_path.name + suffix)
    sidecar.symlink_to(victim)

    with pytest.raises(
        StoreError,
        match="database sidecar must be an owner-only regular file",
    ):
        SQLiteTaskStore(database_path).initialize()

    assert sidecar.is_symlink()
    assert victim.read_bytes() == marker


def test_initialize_tolerates_a_sidecar_removed_by_another_trusted_worker(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "configured.sqlite3"
    SQLiteTaskStore(database_path).initialize()
    sidecar = database_path.with_name(database_path.name + "-wal")
    sidecar.write_bytes(b"transient trusted worker sidecar")
    sidecar.chmod(0o600)
    real_lstat = Path.lstat
    removed = False

    def remove_after_first_sidecar_lstat(path):
        nonlocal removed
        status = real_lstat(path)
        if path == sidecar and not removed:
            removed = True
            sidecar.unlink()
        return status

    monkeypatch.setattr(Path, "lstat", remove_after_first_sidecar_lstat)

    SQLiteTaskStore(database_path).initialize()

    assert removed is True
    assert not sidecar.exists()


def test_link_loser_rejects_a_symlink_without_touching_its_target(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "configured.sqlite3"
    redirected_path = tmp_path / "redirected.sqlite3"

    def publish_symlink_instead(*_args, **_kwargs):
        database_path.symlink_to(redirected_path)
        raise FileExistsError

    monkeypatch.setattr(os, "link", publish_symlink_instead)

    with pytest.raises(StoreError, match="regular file"):
        SQLiteTaskStore(database_path).initialize()

    assert database_path.is_symlink()
    assert not redirected_path.exists()
    assert not list(tmp_path.glob(".*.init"))


def test_connect_time_symlink_swap_cannot_create_or_initialize_redirected_target(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "configured.sqlite3"
    preserved_path = tmp_path / "preserved.sqlite3"
    redirected_path = tmp_path / "redirected.sqlite3"
    SQLiteTaskStore(database_path).initialize()
    real_connect = sqlite3.connect
    swapped = False

    def swap_inside_connect(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.replace(database_path, preserved_path)
            database_path.symlink_to(redirected_path)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", swap_inside_connect)

    with pytest.raises((sqlite3.OperationalError, StoreError)):
        SQLiteTaskStore(database_path).initialize()

    assert swapped is True
    assert database_path.is_symlink()
    assert not redirected_path.exists()
    with real_connect(preserved_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        } >= {"tasks", "inference_jobs"}


def test_connect_time_symlink_swap_cannot_modify_existing_redirected_database(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "configured.sqlite3"
    preserved_path = tmp_path / "preserved.sqlite3"
    redirected_path = tmp_path / "redirected.sqlite3"
    SQLiteTaskStore(database_path).initialize()
    with sqlite3.connect(redirected_path) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        connection.execute("INSERT INTO sentinel VALUES ('must-remain-byte-identical')")
    redirected_before = redirected_path.read_bytes()
    real_connect = sqlite3.connect
    swapped = False

    def swap_inside_connect(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            os.replace(database_path, preserved_path)
            database_path.symlink_to(redirected_path)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", swap_inside_connect)

    with pytest.raises(StoreError, match="changed while it was opened"):
        SQLiteTaskStore(database_path).initialize()

    assert swapped is True
    assert redirected_path.read_bytes() == redirected_before
    with real_connect(redirected_path) as connection:
        assert connection.execute("SELECT value FROM sentinel").fetchone() == (
            "must-remain-byte-identical",
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('tasks', 'inference_jobs')"
        ).fetchone() == (0,)


def test_fresh_publication_is_private_and_leaves_no_bootstrap_artifacts(tmp_path):
    database_path = tmp_path / "configured.sqlite3"

    SQLiteTaskStore(database_path).initialize()

    assert database_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".*.init"))
    assert not list(tmp_path.glob(".*.cleanup"))


def test_two_successful_initializers_share_one_complete_publication(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "concurrent.sqlite3"
    real_link = os.link
    publication_barrier = threading.Barrier(2)

    def publish_together(*args, **kwargs):
        publication_barrier.wait(timeout=3.0)
        return real_link(*args, **kwargs)

    monkeypatch.setattr(os, "link", publish_together)
    stores = [SQLiteTaskStore(database_path), SQLiteTaskStore(database_path)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda item: item.initialize(), stores))

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"tasks", "inference_jobs"} <= tables
    assert database_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".*.init"))


def test_cleanup_does_not_unlink_a_replacement_at_the_private_name(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "configured.sqlite3"
    replacement_path = tmp_path / "replacement.bin"
    replacement_payload = b"concurrent replacement must survive"
    replacement_path.write_bytes(replacement_payload)
    owned_stash = tmp_path / "owned-stash.sqlite3"
    private_path = None
    identity_checks = 0

    import las_repro.store as store_module

    real_identity_check = store_module._regular_file_has_identity

    def replace_after_last_identity_check(path, identity):
        nonlocal private_path, identity_checks
        result = real_identity_check(path, identity)
        if path.name.endswith(".init") and result:
            identity_checks += 1
            if identity_checks == 4:
                private_path = path
                os.replace(path, owned_stash)
                os.replace(replacement_path, path)
        return result

    monkeypatch.setattr(
        store_module,
        "_regular_file_has_identity",
        replace_after_last_identity_check,
    )

    SQLiteTaskStore(database_path).initialize()

    assert private_path is not None
    assert private_path.read_bytes() == replacement_payload
    assert SQLiteTaskStore(database_path).initialize() is None


def test_inference_affinity_then_fallback_and_expired_lease_reclaim(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    [job] = store.create_inference_jobs(
        task.task_id,
        [InferenceJobSpec("coarse", 0, {"start": 0}, affinity_worker_id="gpu-a", affinity_fallback_at=50.0)],
    )

    assert store.claim_inference_job("gpu-b", lease_seconds=10, now=40.0) is None
    first = store.claim_inference_job("gpu-a", lease_seconds=1, now=40.0)
    assert first is not None
    assert first.job_id == job.job_id
    assert store.claim_inference_job("gpu-b", lease_seconds=1, now=40.5) is None

    recovered = store.claim_inference_job("gpu-b", lease_seconds=1, now=50.0)
    assert recovered is not None
    assert recovered.job_id == job.job_id
    assert recovered.attempt == 2


def test_inference_jobs_persist_task_model_alias_and_claim_only_exact_matches(store):
    """Two model workers must not consume one another's durable queues."""
    first_task = store.create_task(
        {
            "video_url": "/v.mp4",
            "task_template": "general_video_captioning",
            "model_name": "model-a",
        },
        now=1.0,
    )
    second_task = store.create_task(
        {
            "video_url": "/v.mp4",
            "task_template": "general_video_captioning",
            "model_name": "model-b",
        },
        now=2.0,
    )
    [first_job] = store.create_inference_jobs(
        first_task.task_id,
        [InferenceJobSpec("general_segment", 0, {"span": "first"})],
        now=3.0,
    )
    [second_job] = store.create_inference_jobs(
        second_task.task_id,
        [InferenceJobSpec("general_segment", 0, {"span": "second"})],
        now=4.0,
    )

    assert first_job.model_name == "model-a"
    assert second_job.model_name == "model-b"
    assert (
        store.claim_inference_job(
            "gpu-missing",
            model_name="model-c",
            lease_seconds=10.0,
            now=5.0,
        )
        is None
    )
    claimed_b = store.claim_inference_job(
        "gpu-b", model_name="model-b", lease_seconds=10.0, now=5.0
    )
    claimed_a = store.claim_inference_job(
        "gpu-a", model_name="model-a", lease_seconds=10.0, now=5.0
    )
    assert claimed_b is not None and claimed_b.job_id == second_job.job_id
    assert claimed_a is not None and claimed_a.job_id == first_job.job_id


def test_model_filtered_claim_survives_restart_and_expired_lease_recovery(tmp_path):
    """Restart and lease recovery must retain the job's original model alias."""
    database_path = tmp_path / "tasks.sqlite3"
    store = SQLiteTaskStore(database_path)
    store.initialize()
    task = store.create_task(
        {
            "video_url": "/v.mp4",
            "task_template": "general_video_captioning",
            "model_name": "model-a",
        }
    )
    [job] = store.create_inference_jobs(
        task.task_id,
        [InferenceJobSpec("general_segment", 0, {})],
    )
    first = store.claim_inference_job(
        "gpu-a-old", model_name="model-a", lease_seconds=1.0, now=10.0
    )
    assert first is not None
    store.close()

    reopened = SQLiteTaskStore(database_path)
    reopened.initialize()
    assert reopened.get_inference_job(job.job_id).model_name == "model-a"
    assert (
        reopened.claim_inference_job(
            "gpu-b", model_name="model-b", lease_seconds=10.0, now=12.0
        )
        is None
    )
    recovered = reopened.claim_inference_job(
        "gpu-a-new", model_name="model-a", lease_seconds=10.0, now=12.0
    )
    assert recovered is not None
    assert recovered.job_id == job.job_id
    assert recovered.attempt == first.attempt + 1
    reopened.close()


def test_legacy_model_alias_migration_backfills_all_jobs_and_filters_claims(tmp_path):
    """A legacy non-default task must not silently enter the default queue."""
    database_path = tmp_path / "legacy-models.sqlite3"
    _create_legacy_model_alias_store(
        database_path,
        task_payloads={
            "task-b": json.dumps(
                {
                    "video_url": "/b.mp4",
                    "task_template": "general_video_captioning",
                    "model_name": "model-b",
                }
            ),
            "task-default": json.dumps(
                {
                    "video_url": "/default.mp4",
                    "task_template": "general_video_captioning",
                }
            ),
        },
        jobs=(
            ("job-b-pending", "task-b", "PENDING", None, None, 0, "gpu-b", 100.0),
            ("job-b-running", "task-b", "RUNNING", 1.0, "gpu-old", 2, None, None),
            ("job-default", "task-default", "PENDING", None, None, 0, None, None),
        ),
    )

    store = SQLiteTaskStore(database_path)
    store.initialize()

    migrated = {
        job.job_id: job for job in store.list_inference_jobs("task-b")
    }
    assert migrated["job-b-pending"].model_name == "model-b"
    assert migrated["job-b-running"].model_name == "model-b"
    [default_job] = store.list_inference_jobs("task-default")
    assert default_job.model_name == "qwen3-vl-8b-instruct"
    assert (
        store.claim_inference_job(
            "gpu-default-wrong",
            model_name="qwen3-vl-8b-instruct",
            lease_seconds=10.0,
            now=10.0,
        ).job_id
        == "job-default"
    )
    assert (
        store.claim_inference_job(
            "gpu-other",
            model_name="model-b",
            lease_seconds=10.0,
            now=10.0,
        ).job_id
        == "job-b-running"
    )
    assert (
        store.claim_inference_job(
            "gpu-other",
            model_name="model-b",
            lease_seconds=10.0,
            now=10.0,
        )
        is None
    )
    affinity_claim = store.claim_inference_job(
        "gpu-b", model_name="model-b", lease_seconds=1.0, now=10.0
    )
    assert affinity_claim is not None
    assert affinity_claim.job_id == "job-b-pending"
    store.close()

    reopened = SQLiteTaskStore(database_path)
    reopened.initialize()
    assert (
        reopened.claim_inference_job(
            "gpu-default",
            model_name="qwen3-vl-8b-instruct",
            lease_seconds=10.0,
            now=12.0,
        )
        is None
    )
    recovered = reopened.claim_inference_job(
        "gpu-b", model_name="model-b", lease_seconds=10.0, now=12.0
    )
    assert recovered is not None
    assert recovered.job_id == "job-b-pending"
    assert recovered.attempt == affinity_claim.attempt + 1
    reopened.close()


@pytest.mark.parametrize(
    "parent_payload",
    [
        '{"model_name":"/etc/passwd"}',
        '{"model_name":7}',
        '{"model_name":',
    ],
)
def test_legacy_model_alias_migration_rolls_back_on_invalid_parent_payload(
    tmp_path,
    parent_payload,
):
    """One untrusted legacy alias must roll back the entire schema migration."""
    database_path = tmp_path / "legacy-invalid.sqlite3"
    _create_legacy_model_alias_store(
        database_path,
        task_payloads={
            "a-valid": '{"model_name":"model-b"}',
            "z-invalid": parent_payload,
        },
        jobs=(
            ("valid-job", "a-valid", "PENDING", None, None, 0, None, None),
            ("invalid-job", "z-invalid", "PENDING", None, None, 0, None, None),
        ),
    )

    with pytest.raises(StoreError, match="task model alias is invalid"):
        SQLiteTaskStore(database_path).initialize()

    with closing(sqlite3.connect(database_path)) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(inference_jobs)")
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        jobs = connection.execute(
            "SELECT job_id, status FROM inference_jobs ORDER BY job_id"
        ).fetchall()
    assert "model_name" not in columns
    assert "idx_jobs_model_claim" not in indexes
    assert jobs == [("invalid-job", "PENDING"), ("valid-job", "PENDING")]


def test_legacy_model_alias_migration_rejects_oversized_parent_payload(tmp_path):
    """Migration must not parse an attacker-controlled unbounded legacy payload."""
    database_path = tmp_path / "legacy-oversized.sqlite3"
    payload = json.dumps({"model_name": "model-b", "padding": "x" * 2_000_000})
    _create_legacy_model_alias_store(
        database_path,
        task_payloads={"task-b": payload},
        jobs=(("job-b", "task-b", "PENDING", None, None, 0, None, None),),
    )

    with pytest.raises(StoreError, match="task payload exceeds migration limit"):
        SQLiteTaskStore(database_path).initialize()

    with closing(sqlite3.connect(database_path)) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(inference_jobs)")
        }
    assert "model_name" not in columns


def test_initialize_never_rewrites_model_aliases_after_column_exists(tmp_path):
    """Normal restarts must not rederive or overwrite durable job routing."""
    database_path = tmp_path / "current-schema.sqlite3"
    store = SQLiteTaskStore(database_path)
    store.initialize()
    task = store.create_task(
        {
            "video_url": "/v.mp4",
            "task_template": "general_video_captioning",
            "model_name": "model-a",
        }
    )
    [job] = store.create_inference_jobs(
        task.task_id,
        [InferenceJobSpec("general_segment", 0, {})],
    )
    store.close()
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            "UPDATE inference_jobs SET model_name = 'model-b' WHERE job_id = ?",
            (job.job_id,),
        )
        connection.execute(
            "UPDATE tasks SET payload = ? WHERE task_id = ?",
            ('{"model_name":"model-c"}', task.task_id),
        )

    reopened = SQLiteTaskStore(database_path)
    reopened.initialize()
    assert reopened.get_inference_job(job.job_id).model_name == "model-b"
    reopened.initialize()
    assert reopened.get_inference_job(job.job_id).model_name == "model-b"
    reopened.close()


def test_completed_job_records_immutable_completed_by(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    [job] = store.create_inference_jobs(task.task_id, [InferenceJobSpec("coarse", 0, {})])
    claimed = store.claim_inference_job("gpu-a", lease_seconds=10, now=1.0)
    assert claimed is not None

    completed = store.complete_inference_job(
        job.job_id,
        {"captions": []},
        worker_id="gpu-a",
        attempt=claimed.attempt,
        now=2.0,
    )
    assert completed.status is InferenceStatus.COMPLETED
    assert completed.completed_by == "gpu-a"
    with pytest.raises(InvalidTransition):
        store.fail_inference_job(
            job.job_id, "late", worker_id="gpu-a", attempt=claimed.attempt
        )
    assert store.list_inference_jobs(task.task_id)[0].completed_by == "gpu-a"


def test_job_owner_mismatch_and_pending_transition_are_rejected(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    [job] = store.create_inference_jobs(task.task_id, [InferenceJobSpec("coarse", 0, {})])

    with pytest.raises(InvalidTransition):
        store.fail_inference_job(job.job_id, "too early", worker_id="gpu-a", attempt=0)
    claimed = store.claim_inference_job("gpu-a", lease_seconds=10, now=1.0)
    assert claimed is not None
    with pytest.raises(WorkerMismatch):
        store.complete_inference_job(
            job.job_id, {}, worker_id="gpu-b", attempt=claimed.attempt
        )


def test_reused_gpu_worker_id_cannot_renew_or_finish_an_expired_generation(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    [job] = store.create_inference_jobs(task.task_id, [InferenceJobSpec("coarse", 0, {})])
    expired = store.claim_inference_job("gpu-0", lease_seconds=1, now=100.0)
    replacement = store.claim_inference_job("gpu-0", lease_seconds=10, now=101.1)
    assert expired is not None
    assert replacement is not None
    assert replacement.attempt == expired.attempt + 1

    with pytest.raises(WorkerMismatch):
        store.heartbeat_inference_job(
            job.job_id,
            "gpu-0",
            lease_seconds=10,
            attempt=expired.attempt,
            now=101.2,
        )
    with pytest.raises(WorkerMismatch):
        store.fail_inference_job(
            job.job_id,
            "stale failure",
            worker_id="gpu-0",
            attempt=expired.attempt,
            now=101.2,
        )

    current = store.get_inference_job(job.job_id)
    assert current is not None
    assert current.status is InferenceStatus.RUNNING
    assert current.attempt == replacement.attempt
    assert current.error is None


def test_explicit_expiry_fences_a_stalled_inference_heartbeat(store):
    task = store.create_task({"video_url": "/v.mp4"})
    [job] = store.create_inference_jobs(
        task.task_id,
        [InferenceJobSpec("coarse", 0, {})],
    )
    claimed = store.claim_inference_job("gpu-old", lease_seconds=30.0, now=100.0)
    assert claimed is not None
    heartbeat_started = threading.Event()
    release_heartbeat = threading.Event()

    def delayed_heartbeat():
        heartbeat_started.set()
        assert release_heartbeat.wait(timeout=2.0)
        return store.heartbeat_inference_job(
            job.job_id,
            "gpu-old",
            lease_seconds=30.0,
            attempt=claimed.attempt,
            now=101.0,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        stale = executor.submit(delayed_heartbeat)
        assert heartbeat_started.wait(timeout=1.0)
        expired = store.expire_inference_job_lease(
            job.job_id,
            worker_id="gpu-old",
            attempt=claimed.attempt,
            now=100.5,
        )
        release_heartbeat.set()
        with pytest.raises(WorkerMismatch):
            stale.result(timeout=1.0)

    assert expired.worker_id is None
    assert expired.lease_until == 100.5
    recovered = store.claim_inference_job("gpu-new", lease_seconds=30.0, now=100.5)
    assert recovered is not None
    assert recovered.job_id == job.job_id
    assert recovered.attempt == claimed.attempt + 1


@pytest.mark.parametrize("invalid_time", [float("nan"), float("inf"), float("-inf")])
def test_task_time_inputs_reject_non_finite_values_without_stuck_rows(store, invalid_time):
    payload = {"video_url": "/v.mp4", "task_template": "general_video_captioning"}

    with pytest.raises(ValueError, match="finite"):
        store.create_task(payload, now=invalid_time)
    with closing(sqlite3.connect(store.database_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

    task = store.create_task(payload, now=1.0)
    with pytest.raises(ValueError, match="finite"):
        store.claim_task("worker", lease_seconds=1.0, now=invalid_time)
    with pytest.raises(ValueError, match="finite"):
        store.claim_task("worker", lease_seconds=invalid_time, now=1.0)
    assert store.get_task(task.task_id).status is TaskStatus.PENDING

    claimed = store.claim_task("worker", lease_seconds=1.0, now=1.0)
    assert claimed is not None
    with pytest.raises(ValueError, match="finite"):
        store.heartbeat_task(
            task.task_id,
            "worker",
            lease_seconds=1.0,
            attempt=claimed.attempt,
            now=invalid_time,
        )
    with pytest.raises(ValueError, match="finite"):
        store.heartbeat_task(
            task.task_id,
            "worker",
            lease_seconds=invalid_time,
            attempt=claimed.attempt,
            now=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        store.complete_task(
            task.task_id,
            {},
            worker_id="worker",
            attempt=claimed.attempt,
            now=invalid_time,
        )

    recovered = store.claim_task("other", lease_seconds=1.0, now=3.0)
    assert recovered is not None
    assert recovered.task_id == task.task_id


@pytest.mark.parametrize("invalid_time", [float("nan"), float("inf"), float("-inf")])
def test_job_time_inputs_and_affinity_reject_non_finite_values(store, invalid_time):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})

    with pytest.raises(ValueError, match="finite"):
        store.create_inference_jobs(task.task_id, [InferenceJobSpec("bad-now", 0, {})], now=invalid_time)
    with pytest.raises(ValueError, match="finite"):
        store.create_inference_jobs(
            task.task_id, [InferenceJobSpec("bad-affinity", 0, {}, affinity_fallback_at=invalid_time)]
        )
    assert store.list_inference_jobs(task.task_id) == []

    [job] = store.create_inference_jobs(task.task_id, [InferenceJobSpec("coarse", 0, {})], now=1.0)
    with pytest.raises(ValueError, match="finite"):
        store.claim_inference_job("worker", lease_seconds=1.0, now=invalid_time)
    with pytest.raises(ValueError, match="finite"):
        store.claim_inference_job("worker", lease_seconds=invalid_time, now=1.0)
    assert store.list_inference_jobs(task.task_id)[0].status is InferenceStatus.PENDING

    claimed = store.claim_inference_job("worker", lease_seconds=1.0, now=1.0)
    assert claimed is not None
    with pytest.raises(ValueError, match="finite"):
        store.complete_inference_job(
            job.job_id,
            {},
            worker_id="worker",
            attempt=claimed.attempt,
            now=invalid_time,
        )
    with pytest.raises(ValueError, match="finite"):
        store.fail_inference_job(
            job.job_id,
            "failure",
            worker_id="worker",
            attempt=claimed.attempt,
            now=invalid_time,
        )

    recovered = store.claim_inference_job("other", lease_seconds=1.0, now=3.0)
    assert recovered is not None
    assert recovered.job_id == job.job_id


def test_finite_inputs_cannot_overflow_computed_lease_deadlines(store):
    task = store.create_task({"video_url": "/v.mp4", "task_template": "general_video_captioning"})
    [job] = store.create_inference_jobs(task.task_id, [InferenceJobSpec("coarse", 0, {})])

    with pytest.raises(ValueError, match="finite"):
        store.claim_task("worker", lease_seconds=sys.float_info.max, now=sys.float_info.max)
    assert store.get_task(task.task_id).status is TaskStatus.PENDING
    with pytest.raises(ValueError, match="finite"):
        store.claim_inference_job("worker", lease_seconds=sys.float_info.max, now=sys.float_info.max)
    assert store.list_inference_jobs(task.task_id)[0].status is InferenceStatus.PENDING

    claimed = store.claim_task("worker", lease_seconds=1.0, now=1.0)
    assert claimed is not None
    with pytest.raises(ValueError, match="finite"):
        store.heartbeat_task(
            task.task_id,
            "worker",
            lease_seconds=sys.float_info.max,
            attempt=claimed.attempt,
            now=sys.float_info.max,
        )
    assert store.get_task(task.task_id).lease_until == 2.0


@pytest.mark.parametrize("terminal_status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_parent_terminalization_atomically_fails_all_nonterminal_children(
    store,
    terminal_status,
):
    """Media cleanup must never leave pending or running child jobs claimable."""
    task = store.create_task(
        {"video_url": "/v.mp4", "task_template": "general_video_captioning"},
        now=1.0,
    )
    parent = store.claim_task("coordinator", lease_seconds=10.0, now=2.0)
    assert parent is not None
    jobs = store.create_inference_jobs(
        task.task_id,
        [
            InferenceJobSpec("general_segment", 0, {"span": 0}),
            InferenceJobSpec("general_segment", 1, {"span": 1}),
            InferenceJobSpec("general_segment", 2, {"span": 2}),
        ],
        now=3.0,
    )
    completed_claim = store.claim_inference_job("gpu-complete", lease_seconds=10, now=4.0)
    assert completed_claim is not None
    store.complete_inference_job(
        completed_claim.job_id,
        {"segments": []},
        worker_id="gpu-complete",
        attempt=completed_claim.attempt,
        now=4.1,
    )
    running_claim = store.claim_inference_job("gpu-stale", lease_seconds=10, now=4.2)
    assert running_claim is not None

    if terminal_status is TaskStatus.COMPLETED:
        store.complete_task(
            task.task_id,
            {"summary": "fallback"},
            worker_id="coordinator",
            attempt=parent.attempt,
            now=5.0,
        )
    else:
        store.fail_task(
            task.task_id,
            "general segment inference timed out",
            worker_id="coordinator",
            attempt=parent.attempt,
            now=5.0,
        )

    by_id = {job.job_id: job for job in store.list_inference_jobs(task.task_id)}
    assert by_id[jobs[0].job_id].status is InferenceStatus.COMPLETED
    for original in jobs[1:]:
        cancelled = by_id[original.job_id]
        assert cancelled.status is InferenceStatus.FAILED
        assert cancelled.error == "parent task is terminal"
        assert cancelled.worker_id is None
        assert cancelled.lease_until is None
        prior_attempt = (
            running_claim.attempt
            if original.job_id == running_claim.job_id
            else original.attempt
        )
        assert cancelled.attempt == prior_attempt + 1
    assert store.claim_inference_job("gpu-late", lease_seconds=10, now=20.0) is None


def test_terminal_parent_rejects_new_inference_jobs(store):
    """A stale coordinator must not recreate work after the parent is terminal."""
    task = store.create_task(
        {"video_url": "/v.mp4", "task_template": "general_video_captioning"}
    )
    parent = store.claim_task("coordinator", lease_seconds=10)
    assert parent is not None
    store.fail_task(
        task.task_id,
        "task execution failed",
        worker_id="coordinator",
        attempt=parent.attempt,
    )

    with pytest.raises(InvalidTransition, match="terminal task"):
        store.create_inference_jobs(
            task.task_id,
            [InferenceJobSpec("general_segment", 0, {})],
        )


@pytest.mark.parametrize("terminal_status", [TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_initialize_reconciles_legacy_terminal_parent_children_idempotently(
    tmp_path,
    terminal_status,
):
    """Opening a pre-fix database fences abandoned work without losing evidence."""
    database_path = tmp_path / "legacy.sqlite3"
    _create_legacy_store_with_inconsistent_children(database_path, terminal_status)

    store = SQLiteTaskStore(database_path)
    store.initialize()
    store.initialize()

    jobs = {job.job_id: job for job in store.list_inference_jobs("terminal-parent")}
    assert jobs["legacy-pending"].status is InferenceStatus.FAILED
    assert jobs["legacy-pending"].attempt == 1
    assert jobs["legacy-running"].status is InferenceStatus.FAILED
    assert jobs["legacy-running"].attempt == 4
    for job_id in ("legacy-pending", "legacy-running"):
        fenced = jobs[job_id]
        assert fenced.error == "parent task is terminal"
        assert fenced.worker_id is None
        assert fenced.lease_until is None
        assert fenced.result is None
        assert fenced.completed_by is None

    completed = jobs["legacy-completed"]
    assert completed.status is InferenceStatus.COMPLETED
    assert completed.attempt == 2
    assert completed.result == {"evidence": "kept"}
    assert completed.completed_by == "gpu-complete"
    assert store.claim_inference_job("gpu-late", lease_seconds=10, now=100.0) is None
    with pytest.raises(InvalidTransition):
        store.complete_inference_job(
            "legacy-running",
            {"evidence": "stale"},
            worker_id="gpu-stale",
            attempt=3,
            now=100.0,
        )


def test_inference_claim_requires_existing_nonterminal_parent_even_before_reconciliation(store):
    """Claim eligibility itself fences terminal-parent and impossible orphan rows."""
    task = store.create_task(
        {"video_url": "/v.mp4", "task_template": "general_video_captioning"},
        now=1.0,
    )
    parent = store.claim_task("coordinator", lease_seconds=10, now=2.0)
    assert parent is not None
    store.fail_task(
        task.task_id,
        "task execution failed",
        worker_id="coordinator",
        attempt=parent.attempt,
        now=3.0,
    )
    with closing(sqlite3.connect(store.database_path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        _insert_raw_inference_job(
            connection,
            job_id="bypassed-terminal",
            task_id=task.task_id,
            status="PENDING",
            updated_at=4.0,
        )
        _insert_raw_inference_job(
            connection,
            job_id="impossible-orphan",
            task_id="missing-parent",
            status="RUNNING",
            updated_at=4.0,
            lease_until=5.0,
            worker_id="gpu-stale",
            attempt=2,
        )

    assert store.claim_inference_job("gpu-late", lease_seconds=10, now=100.0) is None

    store.initialize()
    terminal_child = store.get_inference_job("bypassed-terminal")
    orphan = store.get_inference_job("impossible-orphan")
    assert terminal_child is not None
    assert terminal_child.status is InferenceStatus.FAILED
    assert terminal_child.error == "parent task is terminal"
    assert orphan is not None
    assert orphan.status is InferenceStatus.FAILED
    assert orphan.error == "parent task is unavailable"
    assert orphan.attempt == 3


def _create_legacy_store_with_inconsistent_children(database_path, terminal_status) -> None:
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE tasks (
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
            CREATE TABLE inference_jobs (
                job_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                stage TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
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
            """
        )
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, operator_id, operator_version, payload, status,
                created_at, updated_at, attempt, result, error
            ) VALUES (
                'terminal-parent', 'las_long_video_understand', 'v1', '{}',
                ?, 1.0, 10.0, 1, ?, ?
            )
            """,
            (
                terminal_status.value,
                '{"summary":"done"}' if terminal_status is TaskStatus.COMPLETED else None,
                '"task execution failed"' if terminal_status is TaskStatus.FAILED else None,
            ),
        )
        _insert_raw_inference_job(
            connection,
            job_id="legacy-pending",
            task_id="terminal-parent",
            status="PENDING",
            updated_at=2.0,
        )
        _insert_raw_inference_job(
            connection,
            job_id="legacy-running",
            task_id="terminal-parent",
            status="RUNNING",
            updated_at=3.0,
            lease_until=4.0,
            worker_id="gpu-stale",
            attempt=3,
            result='{"partial":"discard"}',
            completed_by="gpu-stale",
        )
        _insert_raw_inference_job(
            connection,
            job_id="legacy-completed",
            task_id="terminal-parent",
            status="COMPLETED",
            updated_at=5.0,
            worker_id="gpu-complete",
            attempt=2,
            result='{"evidence":"kept"}',
            completed_by="gpu-complete",
        )
    database_path.chmod(0o600)


def _create_legacy_model_alias_store(
    database_path,
    *,
    task_payloads,
    jobs,
) -> None:
    """Create the last schema that predated durable model-alias routing."""
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE tasks (
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
            CREATE TABLE inference_jobs (
                job_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                stage TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
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
            CREATE INDEX idx_tasks_claim
                ON tasks(status, lease_until, created_at);
            CREATE INDEX idx_jobs_claim
                ON inference_jobs(status, lease_until, created_at);
            CREATE INDEX idx_jobs_task
                ON inference_jobs(task_id, stage, ordinal);
            """
        )
        for ordinal, (task_id, payload) in enumerate(task_payloads.items()):
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, operator_id, operator_version, payload, status,
                    created_at, updated_at, attempt
                ) VALUES (?, 'las_long_video_understand', 'v1', ?, 'RUNNING', ?, ?, 1)
                """,
                (task_id, payload, float(ordinal + 1), float(ordinal + 1)),
            )
        for ordinal, (
            job_id,
            task_id,
            status,
            lease_until,
            worker_id,
            attempt,
            affinity_worker_id,
            affinity_fallback_at,
        ) in enumerate(jobs):
            connection.execute(
                """
                INSERT INTO inference_jobs (
                    job_id, task_id, stage, ordinal, payload, status,
                    created_at, updated_at, lease_until, worker_id, attempt,
                    result, error, affinity_worker_id, affinity_fallback_at,
                    completed_by
                ) VALUES (?, ?, 'general_segment', ?, '{}', ?, ?, ?, ?, ?, ?,
                          NULL, NULL, ?, ?, NULL)
                """,
                (
                    job_id,
                    task_id,
                    ordinal,
                    status,
                    float(ordinal + 1),
                    float(ordinal + 1),
                    lease_until,
                    worker_id,
                    attempt,
                    affinity_worker_id,
                    affinity_fallback_at,
                ),
            )
    database_path.chmod(0o600)


def _insert_raw_inference_job(
    connection,
    *,
    job_id,
    task_id,
    status,
    updated_at,
    lease_until=None,
    worker_id=None,
    attempt=0,
    result=None,
    completed_by=None,
) -> None:
    connection.execute(
        """
        INSERT INTO inference_jobs (
            job_id, task_id, stage, ordinal, payload, status,
            created_at, updated_at, lease_until, worker_id, attempt,
            result, error, affinity_worker_id, affinity_fallback_at, completed_by
        ) VALUES (?, ?, 'general_segment', ?, '{}', ?, 1.0, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
        """,
        (
            job_id,
            task_id,
            {"legacy-pending": 0, "legacy-running": 1, "legacy-completed": 2}.get(job_id, 0),
            status,
            updated_at,
            lease_until,
            worker_id,
            attempt,
            result,
            completed_by,
        ),
    )
