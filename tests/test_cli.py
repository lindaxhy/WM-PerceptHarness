from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import select
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import las_repro.cli as cli
from las_repro.api import create_app
from las_repro.config import Settings
from las_repro.domain import TaskStatus
from las_repro.store import SQLiteTaskStore


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SECRET_SENTINEL = "test-secret-must-not-be-printed"


def _cli_environment(tmp_path: Path, media_root: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LAS_")
    }
    environment.update(
        {
            "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
            "LAS_API_KEY_SHA256": hashlib.sha256(b"local-test-key").hexdigest(),
            "LAS_DATABASE_PATH": str(tmp_path / "tasks.sqlite3"),
            "LAS_WORK_ROOT": str(tmp_path / "work"),
            "LAS_ALLOWED_MEDIA_ROOTS": str(media_root),
            "LAS_MODEL_REGISTRY": json.dumps(
                {"qwen3-vl-8b-instruct": str(tmp_path / "model")}
            ),
            "LAS_BACKEND": "qwen3_vl",
            "LAS_GPU_DEVICES": "0,1,2,3",
            "LAS_API_HOST": "127.0.0.1",
            "LAS_API_PORT": "8000",
            "LAS_TOS_ACCESS_KEY": SECRET_SENTINEL,
            "LAS_TOS_SECRET_KEY": SECRET_SENTINEL,
        }
    )
    return environment


def _run_cli(
    *arguments: str,
    environment: dict[str, str],
    timeout: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "las_repro.cli", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _assert_secret_absent(completed: subprocess.CompletedProcess[str]) -> None:
    assert SECRET_SENTINEL not in completed.stdout
    assert SECRET_SENTINEL not in completed.stderr


def test_help_exposes_exact_process_roles_without_printing_secrets(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    completed = _run_cli(
        "--help",
        environment=_cli_environment(tmp_path, media_root),
    )

    assert completed.returncode == 0, completed.stderr
    for command in ("init-db", "api", "coordinator", "gpu-worker", "run-fake"):
        assert command in completed.stdout
    _assert_secret_absent(completed)


def test_distribution_registers_the_las_repro_console_script() -> None:
    entry_points = {
        entry.name: entry.value
        for entry in importlib.metadata.distribution("las-repro").entry_points
        if entry.group == "console_scripts"
    }

    assert entry_points["las-repro"] == "las_repro.cli:main"


def test_init_db_and_empty_run_fake_once_are_successful_and_quiet_about_secrets(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)

    initialized = _run_cli("init-db", environment=environment)
    assert initialized.returncode == 0, initialized.stderr
    assert (tmp_path / "tasks.sqlite3").is_file()
    _assert_secret_absent(initialized)

    fake = _run_cli("run-fake", "--once", environment=environment)
    assert fake.returncode == 0, fake.stderr
    _assert_secret_absent(fake)


def test_run_fake_once_drains_each_pipeline_without_gpu_or_network(
    tmp_path: Path,
    short_video: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    video = media_root / "silent.mp4"
    shutil.copyfile(short_video, video)
    environment = _cli_environment(tmp_path, media_root)
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store.initialize()
    tasks = [
        store.create_task(
            {
                "video_url": str(video),
                "task_template": template,
                "model_name": "qwen3-vl-8b-instruct",
            },
            operator_id=(
                "las_long_video_understand"
                if index % 2 == 0
                else "las_video_understanding"
            ),
        )
        for index, template in enumerate(
            (
                "general_video_captioning",
                "embodied_active_object_detection",
                "embodied_action_captioning",
            )
        )
    ]

    completed = _run_cli("run-fake", "--once", environment=environment)

    assert completed.returncode == 0, completed.stderr
    for submitted in tasks:
        task = store.get_task(submitted.task_id)
        assert task is not None
        assert task.status is TaskStatus.COMPLETED
        assert task.result
    _assert_secret_absent(completed)


def test_run_fake_services_each_configured_model_alias_without_cross_claiming(
    tmp_path: Path,
    short_video: Path,
) -> None:
    """Fake mode must apply the same registry and per-alias claim contract."""
    media_root = tmp_path / "media"
    media_root.mkdir()
    video = media_root / "silent.mp4"
    shutil.copyfile(short_video, video)
    environment = _cli_environment(tmp_path, media_root)
    environment["LAS_MODEL_REGISTRY"] = json.dumps(
        {"model-a": str(tmp_path / "model-a"), "model-b": str(tmp_path / "model-b")}
    )
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store.initialize()
    tasks = [
        store.create_task(
            {
                "video_url": str(video),
                "task_template": "general_video_captioning",
                "model_name": model_name,
            }
        )
        for model_name in ("model-a", "model-b")
    ]

    completed = _run_cli("run-fake", "--once", environment=environment)

    assert completed.returncode == 0, completed.stderr
    for task, expected_alias in zip(tasks, ("model-a", "model-b"), strict=True):
        persisted = store.get_task(task.task_id)
        assert persisted is not None and persisted.status is TaskStatus.COMPLETED
        jobs = store.list_inference_jobs(task.task_id)
        assert jobs and {job.model_name for job in jobs} == {expected_alias}
        assert all(job.completed_by == f"fake-gpu-{expected_alias}" for job in jobs)
    _assert_secret_absent(completed)


def test_http_submit_and_poll_complete_embodied_actions_beyond_two_seconds(
    tmp_path: Path,
    longer_silent_video: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    video = media_root / "silent-2.2s.mp4"
    shutil.copyfile(longer_silent_video, video)
    environment = _cli_environment(tmp_path, media_root)
    settings = Settings(
        api_key_sha256=environment["LAS_API_KEY_SHA256"],
        database_path=tmp_path / "tasks.sqlite3",
        work_root=tmp_path / "work",
        allowed_media_roots=(media_root,),
    )
    store = SQLiteTaskStore(settings.database_path)
    store.initialize()
    headers = {"Authorization": "Bearer local-test-key"}

    with TestClient(create_app(settings, store)) as client:
        submitted = client.post(
            "/api/v1/submit",
            headers=headers,
            json={
                "operator_id": "las_video_understanding",
                "operator_version": "v1",
                "data": {
                    "video_url": str(video),
                    "task_template": "embodied_action_captioning",
                },
            },
        )
        assert submitted.status_code == 200
        task_id = submitted.json()["metadata"]["task_id"]

        completed = _run_cli("run-fake", "--once", environment=environment)
        assert completed.returncode == 0, completed.stderr

        polled = client.post(
            "/api/v1/poll",
            headers=headers,
            json={
                "operator_id": "las_video_understanding",
                "operator_version": "v1",
                "task_id": task_id,
            },
        )

    assert polled.status_code == 200
    body = polled.json()
    assert body["metadata"]["task_status"] == "COMPLETED"
    intervals = [
        (segment["start"], segment["end"])
        for segment in body["data"]["segments"]
    ]
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == 2.2
    assert all(0.0 < end - start <= 1.0 for start, end in intervals)
    _assert_secret_absent(completed)


def test_coordinator_role_never_imports_gpu_backend_or_optional_dependencies(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)
    script = """
import importlib.abc
import sys

class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'las_repro.models.qwen3_vl' or fullname.split('.')[0] in {
            'torch', 'transformers', 'qwen_vl_utils', 'accelerate'
        }:
            raise AssertionError(f'GPU dependency imported by coordinator: {fullname}')
        return None

sys.meta_path.insert(0, BlockGPU())
from las_repro.cli import main
raise SystemExit(main(['coordinator', '--once']))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    _assert_secret_absent(completed)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
def test_idle_coordinator_sigterm_is_clean_across_twenty_ready_handshakes(
    tmp_path: Path,
) -> None:
    """Removing the pre-settings signal boundary makes these exits negative."""
    for ordinal in range(20):
        run_root = tmp_path / f"run-{ordinal}"
        media_root = run_root / "media"
        media_root.mkdir(parents=True)
        environment = _cli_environment(run_root, media_root)
        read_fd, write_fd = os.pipe()
        environment["_LAS_REPRO_SIGNAL_READY_FD"] = str(write_fd)
        process = subprocess.Popen(
            [sys.executable, "-m", "las_repro.cli", "coordinator"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            pass_fds=(write_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        os.close(write_fd)
        stdout = ""
        stderr = ""
        try:
            readable, _, _ = select.select([read_fd], [], [], 5.0)
            assert readable, f"run {ordinal} did not publish signal readiness"
            assert os.read(read_fd, 1) == b"R"
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5.0)
        finally:
            os.close(read_fd)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)

        assert process.returncode == 0, f"run {ordinal}: {stderr}"
        assert SECRET_SENTINEL not in stdout + stderr
        database = run_root / "tasks.sqlite3"
        if database.exists():
            with sqlite3.connect(database) as connection:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
                ).fetchone()
                if table is not None:
                    assert connection.execute(
                        "SELECT COUNT(*) FROM tasks WHERE worker_id IS NOT NULL"
                    ).fetchone() == (0,)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
def test_main_handles_sigterm_before_settings_and_restores_the_previous_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving handler installation below Settings.from_env makes this escape."""

    class UnprotectedStartupSignal(BaseException):
        pass

    previous = signal.getsignal(signal.SIGTERM)

    def unprotected_handler(_: int, __: Any) -> None:
        raise UnprotectedStartupSignal

    def interrupt_settings() -> Settings:
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail("SIGTERM did not interrupt settings initialization")

    signal.signal(signal.SIGTERM, unprotected_handler)
    monkeypatch.setattr(Settings, "from_env", interrupt_settings)
    try:
        assert cli.main(["coordinator"]) == 0
        assert signal.getsignal(signal.SIGTERM) is unprotected_handler
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask") or not hasattr(signal, "SIGUSR1"),
    reason="POSIX signal masks are unavailable",
)
def test_signal_boundary_restores_the_mask_captured_at_entry() -> None:
    """Restoring the current exit mask leaks mask changes made by a command."""
    target = signal.SIGUSR1
    initial = signal.pthread_sigmask(signal.SIG_BLOCK, ())
    try:
        with cli._stop_on_signals(threading.Event()):
            operation = signal.SIG_UNBLOCK if target in initial else signal.SIG_BLOCK
            signal.pthread_sigmask(operation, (target,))
            assert (target in signal.pthread_sigmask(signal.SIG_BLOCK, ())) is (
                target not in initial
            )

        assert signal.pthread_sigmask(signal.SIG_BLOCK, ()) == initial
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, initial)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="POSIX signal masks are unavailable",
)
def test_signal_boundary_restores_handlers_when_teardown_blocking_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed teardown block must not bypass handler/mask restoration."""
    real_mask = signal.pthread_sigmask
    initial_mask = real_mask(signal.SIG_BLOCK, ())
    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    block_calls = 0

    def fail_teardown_block(how: int, mask: Any) -> set[signal.Signals]:
        nonlocal block_calls
        if how == signal.SIG_BLOCK and set(mask) == {signal.SIGTERM, signal.SIGINT}:
            block_calls += 1
            if block_calls == 2:
                raise OSError("injected teardown mask failure")
        return real_mask(how, mask)

    monkeypatch.setattr(signal, "pthread_sigmask", fail_teardown_block)
    try:
        with pytest.raises(OSError, match="injected teardown mask failure"):
            with cli._stop_on_signals(threading.Event()):
                assert signal.getsignal(signal.SIGTERM) is not previous[signal.SIGTERM]

        assert {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGTERM, signal.SIGINT)
        } == previous
        assert real_mask(signal.SIG_BLOCK, ()) == initial_mask
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        real_mask(signal.SIG_SETMASK, initial_mask)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
@pytest.mark.parametrize(
    "arguments",
    [
        ["init-db"],
        ["api"],
        ["coordinator"],
        ["gpu-worker", "--device", "0"],
        ["run-fake", "--once"],
    ],
    ids=("init-db", "api", "coordinator", "gpu-worker", "run-fake"),
)
def test_each_role_closes_an_initialized_store_when_startup_sigterm_arrives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    """A TERM immediately after SQLite creation must not escape store cleanup."""
    settings = Settings(
        database_path=tmp_path / "tasks.sqlite3",
        work_root=tmp_path / "work",
        backend="qwen3_vl",
        gpu_devices=(0,),
    )
    initialized = 0
    closed = 0
    original_initialize = SQLiteTaskStore.initialize
    original_close = SQLiteTaskStore.close

    def interrupt_after_initialize(store: SQLiteTaskStore) -> None:
        nonlocal initialized
        original_initialize(store)
        initialized += 1
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail("SIGTERM did not interrupt post-initialize startup")

    def record_close(store: SQLiteTaskStore) -> None:
        nonlocal closed
        closed += 1
        original_close(store)

    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr(SQLiteTaskStore, "initialize", interrupt_after_initialize)
    monkeypatch.setattr(SQLiteTaskStore, "close", record_close)

    assert cli.main(arguments) == 0
    assert initialized == 1
    assert closed == 1
    with sqlite3.connect(settings.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE worker_id IS NOT NULL"
        ).fetchone() == (0,)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
@pytest.mark.parametrize(
    "arguments",
    [
        ["init-db"],
        ["api"],
        ["coordinator"],
        ["gpu-worker", "--device", "0"],
        ["run-fake", "--once"],
    ],
    ids=("init-db", "api", "coordinator", "gpu-worker", "run-fake"),
)
def test_each_role_defers_sigterm_from_the_first_sqlite_connect_until_schema_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    """A connect-time TERM is delivered only after the full schema is durable."""
    settings = Settings(
        database_path=tmp_path / "tasks.sqlite3",
        work_root=tmp_path / "work",
        backend="qwen3_vl",
        gpu_devices=(0,),
    )
    real_connect = sqlite3.connect
    connect_calls = 0

    def interrupt_first_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal connect_calls
        connection = real_connect(*args, **kwargs)
        connect_calls += 1
        if connect_calls == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return connection

    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr(sqlite3, "connect", interrupt_first_connect)

    assert cli.main(arguments) == 0
    assert connect_calls >= 1
    with real_connect(settings.database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"tasks", "inference_jobs"} <= tables
        assert connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE worker_id IS NOT NULL"
        ).fetchone() == (0,)


def test_store_context_does_not_reopen_a_database_when_initialize_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling close on an uninitialized store used to open the empty file."""
    settings = Settings(database_path=tmp_path / "failed.sqlite3")
    real_connect = sqlite3.connect
    reopened = 0

    def fail_initialize(store: SQLiteTaskStore) -> None:
        store.database_path.touch()
        raise RuntimeError("injected initialization failure")

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal reopened
        reopened += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(SQLiteTaskStore, "initialize", fail_initialize)
    monkeypatch.setattr(sqlite3, "connect", tracking_connect)

    with pytest.raises(RuntimeError, match="injected initialization failure"):
        with cli._store(settings):
            pytest.fail("an uninitialized store must never be yielded")

    assert reopened == 0


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
def test_failed_fresh_initialize_is_an_error_even_with_deferred_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A deferred TERM must not hide a simultaneous schema failure as success."""
    settings = Settings(database_path=tmp_path / "failed.sqlite3")
    real_connect = sqlite3.connect
    connected = 0

    def interrupt_and_deny_schema(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal connected
        connection = real_connect(*args, **kwargs)
        connected += 1

        def authorizer(action: int, *_: Any) -> int:
            if action == sqlite3.SQLITE_CREATE_TABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        if connected == 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return connection

    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr(sqlite3, "connect", interrupt_and_deny_schema)

    assert cli.main(["init-db"]) == 1
    assert "DatabaseError" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="POSIX signal masks are unavailable",
)
def test_pending_teardown_sigterm_does_not_hide_an_initialization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pending TERM must stay deferred until a failing initialization reports 1."""
    settings = Settings(database_path=tmp_path / "failed.sqlite3")
    real_connect = sqlite3.connect
    real_mask = signal.pthread_sigmask
    initial_mask = real_mask(signal.SIG_BLOCK, ())
    block_calls = 0

    def deny_schema_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)

        def authorizer(action: int, *_: Any) -> int:
            if action == sqlite3.SQLITE_CREATE_TABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        return connection

    def make_term_pending_during_store_teardown(
        how: int,
        mask: Any,
    ) -> set[signal.Signals]:
        nonlocal block_calls
        result = real_mask(how, mask)
        if how == signal.SIG_BLOCK and set(mask) == {signal.SIGTERM, signal.SIGINT}:
            block_calls += 1
            if block_calls == 3:
                os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr(sqlite3, "connect", deny_schema_connect)
    monkeypatch.setattr(signal, "pthread_sigmask", make_term_pending_during_store_teardown)
    try:
        assert cli.main(["init-db"]) == 1
        assert "DatabaseError" in capsys.readouterr().err
        assert list(tmp_path.iterdir()) == []
    finally:
        real_mask(signal.SIG_SETMASK, initial_mask)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="POSIX signal masks are unavailable",
)
def test_sigterm_during_deferred_handler_restore_does_not_hide_init_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An init error wins even when TERM arrives after the pending-set drain."""
    settings = Settings(database_path=tmp_path / "failed.sqlite3")
    real_connect = sqlite3.connect
    real_signal = signal.signal
    signal_calls = 0

    def deny_schema_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)

        def authorizer(action: int, *_: Any) -> int:
            if action == sqlite3.SQLITE_CREATE_TABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        return connection

    def interrupt_first_store_handler_restore(signum: int, handler: Any) -> Any:
        nonlocal signal_calls
        signal_calls += 1
        if signal_calls == 5:
            os.kill(os.getpid(), signal.SIGTERM)
        return real_signal(signum, handler)

    monkeypatch.setattr(Settings, "from_env", lambda: settings)
    monkeypatch.setattr(sqlite3, "connect", deny_schema_connect)
    monkeypatch.setattr(signal, "signal", interrupt_first_store_handler_restore)

    assert cli.main(["init-db"]) == 1
    assert "DatabaseError" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
@pytest.mark.parametrize("once", [False, True], ids=("forever", "once"))
def test_active_coordinator_sigterm_expires_its_current_lease(
    tmp_path: Path,
    short_video: Path,
    once: bool,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    video = media_root / "silent.mp4"
    shutil.copyfile(short_video, video)
    environment = _cli_environment(tmp_path, media_root)
    environment["LAS_LEASE_SECONDS"] = "30"
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store.initialize()
    task = store.create_task(
        {
            "video_url": str(video),
            "task_template": "general_video_captioning",
            "model_name": "qwen3-vl-8b-instruct",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "las_repro.cli",
            "coordinator",
            *(["--once"] if once else []),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    try:
        deadline = time.monotonic() + 5.0
        while True:
            claimed = store.get_task(task.task_id)
            assert claimed is not None
            if claimed.status is TaskStatus.RUNNING and store.list_inference_jobs(task.task_id):
                break
            assert process.poll() is None
            if time.monotonic() >= deadline:
                pytest.fail("coordinator did not enter its active wait")
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            pytest.fail("active coordinator did not exit after SIGTERM")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)

    assert process.returncode == 0, stderr
    interrupted = store.get_task(task.task_id)
    assert interrupted is not None
    assert interrupted.status is TaskStatus.RUNNING
    assert interrupted.lease_until is not None
    assert interrupted.lease_until <= time.time()
    recovered = store.claim_task("recovery", lease_seconds=30.0)
    assert recovered is not None
    assert recovered.task_id == task.task_id
    assert recovered.attempt == 2
    assert SECRET_SENTINEL not in stdout + stderr


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
def test_run_fake_sigterm_gracefully_stops_api_and_worker_roles(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment["LAS_API_PORT"] = str(port)
    process = subprocess.Popen(
        [sys.executable, "-m", "las_repro.cli", "run-fake"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while True:
            assert process.poll() is None
            with socket.socket() as client:
                client.settimeout(0.05)
                if client.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if time.monotonic() >= deadline:
                pytest.fail("run-fake did not open its configured listener")
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)

    assert process.returncode == 0, stderr
    assert SECRET_SENTINEL not in stdout
    assert SECRET_SENTINEL not in stderr


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="SIGTERM is unavailable")
def test_api_sigterm_runs_the_store_close_boundary_and_exits_cleanly(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment["LAS_API_PORT"] = str(port)
    process = subprocess.Popen(
        [sys.executable, "-m", "las_repro.cli", "api"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while True:
            assert process.poll() is None
            with socket.socket() as client:
                client.settimeout(0.05)
                if client.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if time.monotonic() >= deadline:
                pytest.fail("api did not open its configured listener")
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5.0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)

    assert process.returncode == 0, stderr
    assert (tmp_path / "tasks.sqlite3").is_file()
    assert SECRET_SENTINEL not in stdout + stderr


def test_cli_settings_validate_listen_address_fields() -> None:
    settings = Settings(api_host="127.0.0.1", api_port=8123)

    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8123


def test_store_close_is_idempotent_and_leaves_persisted_state_readable(
    tmp_path: Path,
) -> None:
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store.initialize()
    task = store.create_task({"video_url": "/allowed/video.mp4"})

    store.close()
    store.close()

    reopened = SQLiteTaskStore(store.database_path)
    reopened.initialize()
    assert reopened.get_task(task.task_id) == task
    reopened.close()


def test_gpu_worker_loads_one_model_on_exactly_the_configured_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)
    for key, value in environment.items():
        if key.startswith("LAS_"):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv(
        "LAS_MODEL_REGISTRY",
        json.dumps({"alternate-model": str(tmp_path / "alternate-model")}),
    )
    loaded: list[tuple[Any, ...]] = []
    constructed: list[tuple[Any, ...]] = []
    signal_boundaries: list[threading.Event] = []
    worker_closes = 0
    model = object()

    from las_repro.models.qwen3_vl import Qwen3VLModel
    from las_repro import workers

    def load_alias(*args: Any, **kwargs: Any) -> object:
        loaded.append((*args, kwargs))
        return model

    class RecordingWorker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append((*args, kwargs))

        def run_once(self) -> bool:
            return False

        def close(self) -> None:
            nonlocal worker_closes
            worker_closes += 1

    monkeypatch.setattr(Qwen3VLModel, "load_alias", load_alias)
    monkeypatch.setattr(workers, "GPUWorker", RecordingWorker)

    @contextmanager
    def record_signal_boundary(stop: threading.Event):
        signal_boundaries.append(stop)
        yield

    monkeypatch.setattr(cli, "_stop_on_signals", record_signal_boundary)

    assert (
        cli.main(
            [
                "gpu-worker",
                "--device",
                "3",
                "--model-name",
                "alternate-model",
                "--once",
            ]
        )
        == 0
    )
    assert len(loaded) == 1
    assert loaded[0][0] == "alternate-model"
    assert loaded[0][2] == "cuda:3"
    assert len(constructed) == 1
    assert constructed[0][1] is model
    assert constructed[0][2:4] == ("gpu-3", "cuda:3")
    assert constructed[0][4]["model_name"] == "alternate-model"
    assert len(signal_boundaries) == 1
    assert worker_closes == 1


@pytest.mark.parametrize(
    ("run_error", "expected_status"),
    [(RuntimeError("run failed"), 1), (cli._SignalShutdown(), 0)],
    ids=("run-error", "signal"),
)
def test_gpu_worker_role_closes_worker_and_store_when_run_unwinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_error: BaseException,
    expected_status: int,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)
    for key, value in environment.items():
        if key.startswith("LAS_"):
            monkeypatch.setenv(key, value)
    worker_closes = 0
    store_closes = 0

    from las_repro.models.qwen3_vl import Qwen3VLModel
    from las_repro import workers

    class FailingWorker:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def run_once(self) -> bool:
            raise run_error

        def close(self) -> None:
            nonlocal worker_closes
            worker_closes += 1

    original_close = SQLiteTaskStore.close

    def record_store_close(store: SQLiteTaskStore) -> None:
        nonlocal store_closes
        store_closes += 1
        original_close(store)

    monkeypatch.setattr(Qwen3VLModel, "load_alias", lambda *args, **kwargs: object())
    monkeypatch.setattr(workers, "GPUWorker", FailingWorker)
    monkeypatch.setattr(SQLiteTaskStore, "close", record_store_close)

    assert cli.main(["gpu-worker", "--device", "3", "--once"]) == expected_status
    assert worker_closes == 1
    assert store_closes == 1


def test_gpu_worker_role_closes_store_when_model_loading_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)
    for key, value in environment.items():
        if key.startswith("LAS_"):
            monkeypatch.setenv(key, value)
    store_closes = 0

    from las_repro.models.qwen3_vl import Qwen3VLModel

    original_close = SQLiteTaskStore.close

    def record_store_close(store: SQLiteTaskStore) -> None:
        nonlocal store_closes
        store_closes += 1
        original_close(store)

    def fail_load(*_: Any, **__: Any) -> object:
        raise RuntimeError("model load failed")

    monkeypatch.setattr(Qwen3VLModel, "load_alias", fail_load)
    monkeypatch.setattr(SQLiteTaskStore, "close", record_store_close)

    assert cli.main(["gpu-worker", "--device", "3", "--once"]) == 1
    assert store_closes == 1


def test_gpu_worker_rejects_a_device_outside_config_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    environment = _cli_environment(tmp_path, media_root)
    environment["LAS_GPU_DEVICES"] = "0,1"
    for key, value in environment.items():
        if key.startswith("LAS_"):
            monkeypatch.setenv(key, value)

    from las_repro.models.qwen3_vl import Qwen3VLModel

    attempted_loads = 0

    def must_not_load(*_: Any, **__: Any) -> object:
        nonlocal attempted_loads
        attempted_loads += 1
        return object()

    monkeypatch.setattr(Qwen3VLModel, "load_alias", must_not_load)

    assert cli.main(["gpu-worker", "--device", "3", "--once"]) == 1
    assert attempted_loads == 0


def test_run_fake_shutdown_keeps_inference_alive_until_current_coordinator_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from las_repro import media, workers
    from las_repro.models import fake

    started = {"coordinator": threading.Event(), "gpu": threading.Event()}
    stop_signals: dict[str, threading.Event] = {}
    gpu_closed = threading.Event()

    class RecordingCoordinator:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def run_forever(self, stop: threading.Event) -> None:
            stop_signals["coordinator"] = stop
            started["coordinator"].set()
            assert stop.wait(timeout=2.0)

    class RecordingGPUWorker:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def run_forever(self, stop: threading.Event) -> None:
            stop_signals["gpu"] = stop
            started["gpu"].set()
            assert stop.wait(timeout=2.0)

        def close(self) -> None:
            gpu_closed.set()

    monkeypatch.setattr(workers, "Coordinator", RecordingCoordinator)
    monkeypatch.setattr(workers, "GPUWorker", RecordingGPUWorker)
    monkeypatch.setattr(fake, "FakeVideoModel", lambda: object())
    monkeypatch.setattr(media, "MediaResolver", lambda *args, **kwargs: object())
    monkeypatch.setattr(media, "TosAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_pipeline_registry", lambda: object())
    monkeypatch.setattr(cli, "create_app", lambda *args, **kwargs: object())

    def stop_api(_: Any, __: str, ___: int, *, stop: threading.Event) -> None:
        assert started["coordinator"].wait(timeout=2.0)
        assert started["gpu"].wait(timeout=2.0)
        stop.set()

    monkeypatch.setattr(cli, "_serve", stop_api)
    settings = Settings(
        database_path=tmp_path / "tasks.sqlite3",
        work_root=tmp_path / "work",
    )

    assert (
        cli._run_fake(
            Namespace(once=False, host=None, port=None),
            settings,
            threading.Event(),
        )
        == 0
    )
    assert stop_signals["coordinator"] is not stop_signals["gpu"]
    assert gpu_closed.is_set()


def test_run_fake_once_uses_main_signal_boundary_around_coordinator_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from las_repro import media, workers
    from las_repro.models import fake

    boundary_active = False
    coordinator_calls = 0
    gpu_closes = 0

    class RecordingCoordinator:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def run_once(self) -> bool:
            nonlocal coordinator_calls
            assert boundary_active
            coordinator_calls += 1
            return False

    class RecordingGPUWorker:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def run_forever(self, stop: threading.Event) -> None:
            assert stop.wait(timeout=2.0)

        def close(self) -> None:
            nonlocal gpu_closes
            gpu_closes += 1

    @contextmanager
    def record_signal_boundary(_: threading.Event):
        nonlocal boundary_active
        boundary_active = True
        try:
            yield
        finally:
            boundary_active = False

    monkeypatch.setattr(workers, "Coordinator", RecordingCoordinator)
    monkeypatch.setattr(workers, "GPUWorker", RecordingGPUWorker)
    monkeypatch.setattr(fake, "FakeVideoModel", lambda: object())
    monkeypatch.setattr(media, "MediaResolver", lambda *args, **kwargs: object())
    monkeypatch.setattr(media, "TosAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_pipeline_registry", lambda: object())
    monkeypatch.setattr(cli, "_stop_on_signals", record_signal_boundary)
    settings = Settings(
        database_path=tmp_path / "tasks.sqlite3",
        work_root=tmp_path / "work",
    )
    monkeypatch.setattr(Settings, "from_env", lambda: settings)

    assert cli.main(["run-fake", "--once"]) == 0
    assert coordinator_calls == 1
    assert boundary_active is False
    assert gpu_closes == 1
