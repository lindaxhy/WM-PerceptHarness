"""State-file guard for interruption-safe local Submit clients.

The helper owns only submission identity and persistence.  Callers supply the
actual localhost HTTP function, so this module neither embeds an address nor
accepts or forwards Ark credentials.
"""

from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from .contracts import SubmitRequest


class SubmissionStateError(RuntimeError):
    """A durable local submission cannot safely proceed."""


class StateIdentityMismatch(SubmissionStateError):
    """A state file belongs to a different local request."""


class UnknownSubmitOutcome(SubmissionStateError):
    """A prior Submit may have succeeded but returned no durable task ID."""


class SubmissionRejected(SubmissionStateError):
    """The local API returned a known business rejection."""


PostJSON = Callable[[dict[str, Any]], Mapping[str, Any]]


def stateful_submit(
    state_path: str | os.PathLike[str],
    submission: SubmitRequest | Mapping[str, Any],
    post: PostJSON,
    *,
    service_identity: str,
) -> str:
    """Submit once or return the persisted task ID for the same request.

    ``SUBMITTING`` is written before calling ``post``.  Any interruption,
    transport error, malformed success, or missing task ID is persisted as
    ``SUBMIT_UNKNOWN``.  A later invocation then refuses to call ``post``
    again, preventing duplicate local tasks after an ambiguous outcome.
    """
    if not callable(post):
        raise TypeError("post must be callable")
    normalized_service = _normalize_service_identity(service_identity)
    request = (
        submission
        if isinstance(submission, SubmitRequest)
        else SubmitRequest.model_validate(submission)
    )
    local_payload = {
        "operator_id": request.operator_id,
        "operator_version": request.operator_version,
        "data": request.sanitized_data(),
    }
    identity = _identity(
        {"service_identity": normalized_service, "request": local_payload}
    )
    path = Path(state_path)

    with _state_lock(path):
        state = _read_state(path)
        if state is not None:
            if state.get("identity") != identity:
                raise StateIdentityMismatch(
                    "state file belongs to a different local service or request"
                )
            task_id = state.get("task_id")
            if isinstance(task_id, str) and task_id.strip():
                return task_id.strip()
            if state.get("status") in {"SUBMITTING", "SUBMIT_UNKNOWN"}:
                raise UnknownSubmitOutcome(
                    "the previous Submit outcome is unknown and must not be repeated"
                )

        state = {
            "identity": identity,
            "service_identity": normalized_service,
            "status": "SUBMITTING",
            "updated_at": time.time(),
        }
        _write_state(path, state)
        try:
            response = post(copy.deepcopy(local_payload))
        except BaseException:
            state.update(status="SUBMIT_UNKNOWN", updated_at=time.time())
            _write_state(path, state)
            raise

        if not isinstance(response, Mapping):
            state.update(status="SUBMIT_UNKNOWN", updated_at=time.time())
            _write_state(path, state)
            raise SubmissionStateError(
                "local Submit returned no reliable response object"
            )
        metadata = response.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = response
        business_code = str(metadata.get("business_code", ""))
        if business_code not in {"", "0"}:
            state.update(
                status="SUBMIT_REJECTED",
                business_code=business_code,
                updated_at=time.time(),
            )
            _write_state(path, state)
            raise SubmissionRejected("local Submit was rejected")
        task_id = metadata.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            state.update(status="SUBMIT_UNKNOWN", updated_at=time.time())
            _write_state(path, state)
            raise SubmissionStateError(
                "local Submit returned no task ID; submission must not be repeated"
            )

        durable_task_id = task_id.strip()
        status = metadata.get("task_status")
        state.update(
            status=(status if isinstance(status, str) and status else "PENDING"),
            task_id=durable_task_id,
            updated_at=time.time(),
        )
        _write_state(path, state)
        return durable_task_id


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_service_identity(value: str) -> str:
    """Return a nonsecret, stable HTTP deployment identity."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("service_identity must be a non-empty HTTP(S) base URL")
    parsed = urlsplit(value.strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("service_identity must be an HTTP(S) base URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("service_identity must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("service_identity must not contain a query or fragment")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("service_identity contains an invalid port") from error
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, host, path, "", ""))


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        path_status = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(path_status.st_mode):
        raise SubmissionStateError("state path must not be a symlink")
    if not stat.S_ISREG(path_status.st_mode):
        raise SubmissionStateError("state path must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP or path.is_symlink():
            raise SubmissionStateError("state path must not be a symlink") from None
        raise SubmissionStateError("state file is unreadable") from None
    try:
        opened_status = os.fstat(descriptor)
        current_status = path.lstat()
        if (
            not stat.S_ISREG(opened_status.st_mode)
            or (opened_status.st_dev, opened_status.st_ino)
            != (current_status.st_dev, current_status.st_ino)
        ):
            raise SubmissionStateError("state file changed while it was opened")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SubmissionStateError("state file is unreadable") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise SubmissionStateError("state file must contain one JSON object")
    return value


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _reject_unsafe_destination(path)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        if error.errno == errno.ELOOP or lock_path.is_symlink():
            raise SubmissionStateError("state lock must not be a symlink") from None
        raise SubmissionStateError("state lock is unavailable") from None
    try:
        lock_status = os.fstat(descriptor)
        current_status = lock_path.lstat()
        if (
            not stat.S_ISREG(lock_status.st_mode)
            or lock_status.st_nlink != 1
            or (lock_status.st_dev, lock_status.st_ino)
            != (current_status.st_dev, current_status.st_ino)
        ):
            raise SubmissionStateError("state lock is not a safe regular file")
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "r+b")
        descriptor = -1
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    with stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _reject_unsafe_destination(path: Path) -> None:
    try:
        path_status = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(path_status.st_mode):
        raise SubmissionStateError("state path must not be a symlink")
    if not stat.S_ISREG(path_status.st_mode):
        raise SubmissionStateError("state path must be a regular file")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
