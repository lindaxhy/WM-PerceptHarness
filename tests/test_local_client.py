from __future__ import annotations

import importlib
import json
import os
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest


SECRET_SENTINEL = "local-client-secret-must-not-survive"
SERVICE_IDENTITY = "http://127.0.0.1:8123/"


def _api() -> Any:
    return importlib.import_module("las_repro.local_client")


def _submission(*, query: str = "describe visible actions") -> dict[str, Any]:
    return {
        "operator_id": "las_video_understanding",
        "operator_version": "v1",
        "data": {
            "video_url": "/allowed/silent.mp4",
            "query": query,
            "ark_api_key": SECRET_SENTINEL,
        },
    }


def test_stateful_submit_resumes_existing_task_without_duplicate_request(
    tmp_path: Path,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    posted: list[dict[str, Any]] = []

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        posted.append(payload)
        return {
            "metadata": {
                "task_id": "task-123",
                "task_status": "PENDING",
                "business_code": "0",
                "error_msg": "",
            }
        }

    assert client.stateful_submit(
        state_path, _submission(), post, service_identity=SERVICE_IDENTITY
    ) == "task-123"
    assert client.stateful_submit(
        state_path, _submission(), post, service_identity=SERVICE_IDENTITY
    ) == "task-123"
    assert len(posted) == 1
    assert posted[0]["data"]["task_template"] == "general_video_captioning"
    assert SECRET_SENTINEL not in json.dumps(posted)
    assert SECRET_SENTINEL not in state_path.read_text(encoding="utf-8")
    assert json.loads(state_path.read_text(encoding="utf-8"))["service_identity"] == (
        "http://127.0.0.1:8123"
    )


def test_unknown_submit_outcome_is_fenced_before_any_retry(tmp_path: Path) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    calls = 0

    def interrupted(_: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise ConnectionError("connection lost after request body")

    with pytest.raises(ConnectionError):
        client.stateful_submit(
            state_path,
            _submission(),
            interrupted,
            service_identity=SERVICE_IDENTITY,
        )

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "SUBMIT_UNKNOWN"
    assert "task_id" not in persisted

    def must_not_post(_: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("unknown submissions must never be retried")

    with pytest.raises(client.UnknownSubmitOutcome, match="must not be repeated"):
        client.stateful_submit(
            state_path,
            _submission(),
            must_not_post,
            service_identity=SERVICE_IDENTITY,
        )
    assert calls == 1


def test_state_write_does_not_follow_the_old_predictable_temporary_name(
    tmp_path: Path,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    predictable = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
    predictable.symlink_to(victim)

    task_id = client.stateful_submit(
        state_path,
        _submission(),
        lambda _: {"metadata": {"task_id": "safe-task", "business_code": "0"}},
        service_identity=SERVICE_IDENTITY,
    )

    assert task_id == "safe-task"
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert state_path.is_file()
    assert not state_path.is_symlink()
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600


def test_stateful_submit_rejects_a_symlink_destination_before_post(
    tmp_path: Path,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    state_path.symlink_to(tmp_path / "missing-target.json")

    with pytest.raises(client.SubmissionStateError, match="symlink"):
        client.stateful_submit(
            state_path,
            _submission(),
            lambda _: pytest.fail("a symlink state path must fail before Submit"),
            service_identity=SERVICE_IDENTITY,
        )

    assert state_path.is_symlink()


def test_stateful_submit_rejects_a_symlink_lock_file(
    tmp_path: Path,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    victim = tmp_path / "lock-victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    lock_path.symlink_to(victim)

    with pytest.raises(client.SubmissionStateError, match="lock"):
        client.stateful_submit(
            state_path,
            _submission(),
            lambda _: pytest.fail("an unsafe lock must fail before Submit"),
            service_identity=SERVICE_IDENTITY,
        )

    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert not state_path.exists()


def test_atomic_state_temp_is_private_unpredictable_and_cleaned_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    observed: dict[str, Any] = {}

    def interrupt_replace(source: str | os.PathLike[str], destination: Any) -> None:
        temporary = Path(source)
        observed["temporary"] = temporary
        observed["destination"] = Path(destination)
        observed["mode"] = stat.S_IMODE(temporary.stat().st_mode)
        raise KeyboardInterrupt

    monkeypatch.setattr(client.os, "replace", interrupt_replace)

    with pytest.raises(KeyboardInterrupt):
        client.stateful_submit(
            state_path,
            _submission(),
            lambda _: pytest.fail("an interrupted state publish must not Submit"),
            service_identity=SERVICE_IDENTITY,
        )

    temporary = observed["temporary"]
    assert observed["destination"] == state_path
    assert temporary.parent == state_path.parent
    assert temporary.name != f".{state_path.name}.{os.getpid()}.tmp"
    assert observed["mode"] == 0o600
    assert not temporary.exists()
    assert not state_path.exists()


def test_concurrent_same_state_callers_perform_exactly_one_submit(
    tmp_path: Path,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    calls = 0
    calls_lock = threading.Lock()

    def post(_: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.02)
        return {"metadata": {"task_id": "one-task", "business_code": "0"}}

    def submit(_: int) -> str:
        return client.stateful_submit(
            state_path,
            _submission(),
            post,
            service_identity=SERVICE_IDENTITY,
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        task_ids = list(executor.map(submit, range(10)))

    assert task_ids == ["one-task"] * 10
    assert calls == 1


def test_state_file_identity_mismatch_refuses_to_reuse_output(tmp_path: Path) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    client.stateful_submit(
        state_path,
        _submission(query="first request"),
        lambda _: {"metadata": {"task_id": "task-first", "business_code": "0"}},
        service_identity=SERVICE_IDENTITY,
    )

    with pytest.raises(client.StateIdentityMismatch, match="different local service or request"):
        client.stateful_submit(
            state_path,
            _submission(query="second request"),
            lambda _: {"metadata": {"task_id": "must-not-be-created"}},
            service_identity=SERVICE_IDENTITY,
        )


def test_state_file_identity_is_scoped_to_normalized_service_url(
    tmp_path: Path,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"
    client.stateful_submit(
        state_path,
        _submission(),
        lambda _: {
            "metadata": {"task_id": "first-deployment", "business_code": "0"}
        },
        service_identity="HTTP://LOCALHOST:8123/",
    )
    assert client.stateful_submit(
        state_path,
        _submission(),
        lambda _: pytest.fail("normalized deployment identity must resume"),
        service_identity="http://localhost:8123",
    ) == "first-deployment"

    with pytest.raises(
        client.StateIdentityMismatch,
        match="different local service or request",
    ):
        client.stateful_submit(
            state_path,
            _submission(),
            lambda _: pytest.fail("a different deployment must not reuse or submit"),
            service_identity="http://localhost:9123",
        )


@pytest.mark.parametrize(
    "service_identity",
    (
        "",
        "localhost:8123",
        "ftp://localhost",
        "http://user:password@localhost",
        "http://localhost?token=secret",
        "http://localhost#fragment",
        "http://localhost:invalid",
    ),
)
def test_service_identity_requires_a_noncredentialed_http_base_url(
    tmp_path: Path,
    service_identity: str,
) -> None:
    client = _api()

    with pytest.raises(ValueError, match="service_identity"):
        client.stateful_submit(
            tmp_path / "state.json",
            _submission(),
            lambda _: pytest.fail("invalid identity must fail before Submit"),
            service_identity=service_identity,
        )


@pytest.mark.parametrize(
    "response",
    [
        {"metadata": {"business_code": "0"}},
        {"metadata": {"business_code": "TASK_REJECTED", "task_id": "ignored"}},
        ["not", "an", "object"],
    ],
)
def test_missing_task_identity_or_rejection_never_blindly_resubmits(
    tmp_path: Path,
    response: Any,
) -> None:
    client = _api()
    state_path = tmp_path / "state.json"

    with pytest.raises(client.SubmissionStateError):
        client.stateful_submit(
            state_path,
            _submission(),
            lambda _: response,
            service_identity=SERVICE_IDENTITY,
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if isinstance(response, dict) and response.get("metadata", {}).get("business_code") == "TASK_REJECTED":
        assert state["status"] == "SUBMIT_REJECTED"
    else:
        assert state["status"] == "SUBMIT_UNKNOWN"
    assert SECRET_SENTINEL not in json.dumps(state)
