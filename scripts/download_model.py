#!/usr/bin/env python3
"""Download one allowlisted Qwen snapshot on a connected machine."""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import io
import json
import multiprocessing
import os
import queue as queue_module
import re
import signal
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ALLOWED_MODEL_IDS = frozenset({DEFAULT_MODEL_ID})
MANIFEST_NAME = "sha256-manifest.json"
STATE_NAME = ".las-download-state.json"
STATE_FORMAT = "las-repro-download-state-v1"
PUBLISH_STATE_FORMAT = "las-repro-empty-destination-state-v2"
INCOMPLETE_SUFFIX = ".las-incomplete"
LOCK_SUFFIX = ".las-download.lock"
PUBLISH_STATE_SUFFIX = ".las-publish-state.json"
EMPTY_DESTINATION_SUFFIX = ".las-empty-destination"
RETIRED_EMPTY_DESTINATION_SUFFIX = ".las-retired-empty-destination"
COMMITTED_STATE_SUFFIX = ".las-retired-empty-state.json"
_OUTPUT_CAPTURE_LIMIT = 64 * 1024
_PROCESS_REAP_GRACE_SECONDS = 0.5
_COMMIT_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")
_LIFETIME_OUTPUT_RESOURCES: list[Any] = []


class DownloadSafetyError(ValueError):
    """A requested snapshot target is outside the exporter's narrow policy."""


class _ParentTermination(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__("download parent termination requested")


class _DirectoryIdentity(NamedTuple):
    device: int
    inode: int
    mode: int


class _PathIdentity(NamedTuple):
    device: int
    inode: int
    mode: int


class _FileIdentity(NamedTuple):
    device: int
    inode: int
    mode: int


class _PublicationRecord(NamedTuple):
    destination: _DirectoryIdentity
    marker: _FileIdentity


def download_model(
    destination: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = "main",
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Resume an approved snapshot and atomically publish it after hashing."""
    _validate_model_id(model_id)
    _validate_revision(revision)
    final_dir = _resolve_destination(destination)
    with _destination_lock(final_dir):
        return _download_model_locked(
            final_dir,
            model_id=model_id,
            revision=revision,
            snapshot_download=snapshot_download,
        )


def _download_model_locked(
    destination: Path,
    *,
    model_id: str,
    revision: str,
    snapshot_download: Callable[..., str] | None,
) -> Path:
    final_dir, local_dir, empty_destination_identity = _prepare_destination(
        destination, model_id=model_id, revision=revision
    )
    if local_dir is None:
        return final_dir / MANIFEST_NAME
    with _suppress_dependency_output():
        downloader = snapshot_download or _import_snapshot_download()
        downloaded = downloader(
            repo_id=model_id,
            repo_type="model",
            revision=revision,
            local_dir=str(local_dir),
        )
    try:
        downloaded_path = Path(downloaded).resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        raise DownloadSafetyError("snapshot downloader returned an invalid location") from None
    if downloaded_path != local_dir:
        raise DownloadSafetyError("snapshot downloader escaped the destination")
    _reject_symlinks(local_dir)

    manifest = {
        "files": _manifest_files(local_dir),
        "format": "las-repro-model-sha256-v1",
        "model_id": model_id,
        "revision": revision,
    }
    manifest_path = local_dir / MANIFEST_NAME
    _atomic_json_write(manifest_path, manifest)
    _publish_snapshot(
        local_dir,
        final_dir,
        model_id=model_id,
        revision=revision,
        empty_destination_identity=empty_destination_identity,
    )
    return final_dir / MANIFEST_NAME


def _validate_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or model_id not in ALLOWED_MODEL_IDS:
        raise DownloadSafetyError("model ID is not allowlisted")


def _validate_revision(revision: str) -> None:
    if not isinstance(revision, str) or (
        revision != "main" and _COMMIT_REVISION.fullmatch(revision) is None
    ):
        raise DownloadSafetyError(
            "revision must be main or an immutable 40-hex commit"
        )


def _prepare_destination(
    destination: str | Path,
    *,
    model_id: str,
    revision: str,
) -> tuple[Path, Path | None, _DirectoryIdentity | None]:
    resolved = _resolve_destination(destination)
    parent = resolved.parent
    incomplete = parent / f".{resolved.name}{INCOMPLETE_SUFFIX}"
    publish_state = parent / f".{resolved.name}{PUBLISH_STATE_SUFFIX}"
    empty_destination = parent / f".{resolved.name}{EMPTY_DESTINATION_SUFFIX}"
    state = _download_state(model_id=model_id, revision=revision)
    empty_destination_identity = _recover_publication(
        resolved,
        incomplete=incomplete,
        publish_state=publish_state,
        empty_destination=empty_destination,
        state=state,
        model_id=model_id,
        revision=revision,
    )
    if resolved.exists() and empty_destination_identity is None:
        return resolved, None, None
    _prepare_staging(incomplete, state)
    return resolved, incomplete.resolve(strict=True), empty_destination_identity


def _resolve_destination(destination: str | Path) -> Path:
    if not isinstance(destination, (str, Path)) or not str(destination):
        raise DownloadSafetyError("destination must be an explicit directory")
    candidate = Path(destination).expanduser()
    if candidate.is_symlink():
        raise DownloadSafetyError("destination must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=False)
        home = Path.home().resolve(strict=True)
        current = Path.cwd().resolve(strict=True)
    except (OSError, RuntimeError):
        raise DownloadSafetyError("destination cannot be resolved safely") from None
    if resolved == Path(resolved.anchor) or resolved in {home, current}:
        raise DownloadSafetyError("destination is too broad")
    try:
        parent = resolved.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DownloadSafetyError("destination parent is unavailable") from None
    if not parent.is_dir() or resolved.parent != parent:
        raise DownloadSafetyError("destination parent is unavailable")
    return resolved


def _download_state(*, model_id: str, revision: str) -> dict[str, str]:
    return {
        "format": STATE_FORMAT,
        "model_id": model_id,
        "revision": revision,
    }


@contextlib.contextmanager
def _destination_lock(destination: Path) -> Any:
    lock_path = destination.parent / f".{destination.name}{LOCK_SUFFIX}"
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError(errno.EPERM, "unsafe lock inode")
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise DownloadSafetyError("destination lock is unavailable") from None
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _recover_publication(
    destination: Path,
    *,
    incomplete: Path,
    publish_state: Path,
    empty_destination: Path,
    state: dict[str, str],
    model_id: str,
    revision: str,
) -> _DirectoryIdentity | None:
    retired_empty = destination.parent / (
        f".{destination.name}{RETIRED_EMPTY_DESTINATION_SUFFIX}"
    )
    committed_state = destination.parent / (
        f".{destination.name}{COMMITTED_STATE_SUFFIX}"
    )
    if (
        publish_state.is_symlink()
        or committed_state.is_symlink()
    ):
        raise DownloadSafetyError("publication state is invalid")
    transaction_exists = publish_state.exists()
    committed_exists = committed_state.exists()
    if transaction_exists and committed_exists:
        raise DownloadSafetyError("publication state is invalid")
    if not transaction_exists:
        if _path_present(empty_destination):
            raise DownloadSafetyError("publication state is invalid")
        if committed_exists:
            committed = _read_publication_state(
                committed_state, model_id=model_id, revision=revision
            )
            if not _path_present(retired_empty):
                if _published_snapshot_matches(
                    destination, model_id=model_id, revision=revision
                ):
                    return None
                raise DownloadSafetyError("publication state is invalid")
            if not _path_matches_identity(
                retired_empty, committed.destination
            ):
                raise DownloadSafetyError("publication state is invalid")
            published = _published_snapshot_matches(
                destination, model_id=model_id, revision=revision
            )
            if published and _is_empty_directory(retired_empty):
                return None
            if (published or not _path_present(destination)) and (
                _rollback_changed_publication(
                    destination,
                    incomplete=incomplete,
                    empty_destination=empty_destination,
                    retired_empty=retired_empty,
                    publish_state=committed_state,
                    marker_identity=committed.marker,
                    expected_placeholder_identity=committed.destination,
                    model_id=model_id,
                    revision=revision,
                )
            ):
                raise DownloadSafetyError(
                    "empty destination changed during publication"
                )
            raise DownloadSafetyError("publication state is invalid")
        if _path_present(retired_empty):
            raise DownloadSafetyError("publication state is invalid")

    identity: _DirectoryIdentity | None = None
    if transaction_exists:
        record = _read_publication_state(
            publish_state, model_id=model_id, revision=revision
        )
        identity = record.destination
        marker_identity = record.marker
        if _path_present(empty_destination) and _path_present(retired_empty):
            raise DownloadSafetyError("publication state is invalid")
        placeholder = (
            empty_destination
            if _path_present(empty_destination)
            else retired_empty if _path_present(retired_empty) else None
        )
        placeholder_identity = (
            _capture_path_identity(placeholder)
            if placeholder is not None
            else None
        )
        if placeholder_identity is not None and placeholder_identity != (
            _directory_as_path_identity(identity)
        ):
            if (
                _published_snapshot_matches(
                    destination, model_id=model_id, revision=revision
                )
                or not _path_present(destination)
            ) and _rollback_changed_handoff(
                destination,
                incomplete=incomplete,
                moved_path=placeholder,
                moved_identity=placeholder_identity,
                publish_state=publish_state,
                marker_identity=marker_identity,
                model_id=model_id,
                revision=revision,
            ):
                raise DownloadSafetyError(
                    "empty destination changed during publication"
                )
            raise DownloadSafetyError("publication state is invalid")

        if _published_snapshot_matches(
            destination, model_id=model_id, revision=revision
        ):
            if placeholder is not None:
                if _path_matches_identity(placeholder, identity) and (
                    _is_empty_directory(placeholder)
                ):
                    try:
                        _remove_published_placeholder(
                            empty_destination,
                            retired_empty=retired_empty,
                            identity=identity,
                        )
                    except _DestinationChanged:
                        pass
                    else:
                        _commit_publication_marker(
                            publish_state,
                            committed_state,
                            expected_identity=marker_identity,
                        )
                        try:
                            _require_unchanged_empty_directory(
                                retired_empty, identity
                            )
                        except _DestinationChanged:
                            if _rollback_changed_publication(
                                destination,
                                incomplete=incomplete,
                                empty_destination=empty_destination,
                                retired_empty=retired_empty,
                                publish_state=committed_state,
                                marker_identity=marker_identity,
                                expected_placeholder_identity=identity,
                                model_id=model_id,
                                revision=revision,
                            ):
                                raise DownloadSafetyError(
                                    "empty destination changed during publication"
                                ) from None
                            raise DownloadSafetyError(
                                "publication state is invalid"
                            ) from None
                        return None
                if _rollback_changed_publication(
                    destination,
                    incomplete=incomplete,
                    empty_destination=empty_destination,
                    retired_empty=retired_empty,
                    publish_state=publish_state,
                    marker_identity=marker_identity,
                    model_id=model_id,
                    revision=revision,
                ):
                    raise DownloadSafetyError(
                        "empty destination changed during publication"
                    )
                raise DownloadSafetyError("publication state is invalid")
            _discard_publication_marker(
                publish_state, expected_identity=marker_identity
            )
            return None

        if not _path_present(destination):
            if placeholder is not None:
                if _path_matches_identity(placeholder, identity) and (
                    _is_empty_directory(placeholder)
                ):
                    return identity
                if _restore_placeholder_name(
                    destination,
                    placeholder=placeholder,
                    publish_state=publish_state,
                    marker_identity=marker_identity,
                ):
                    raise DownloadSafetyError(
                        "empty destination changed during publication"
                    )
                raise DownloadSafetyError("publication state is invalid")
            # A rollback or original-directory retirement completed before its
            # marker handoff. The deterministic staging directory remains the
            # only resumable snapshot location.
            _discard_publication_marker(
                publish_state, expected_identity=marker_identity
            )
            identity = None
            transaction_exists = False
        elif placeholder is None:
            if _path_matches_identity(destination, identity):
                _discard_publication_marker(
                    publish_state, expected_identity=marker_identity
                )
                raise DownloadSafetyError(
                    "empty destination changed during publication"
                )
            # Rollback restored or a non-cooperating writer installed another
            # destination. Retire only the exact owned marker and then apply
            # the ordinary destination policy without touching that content.
            _discard_publication_marker(
                publish_state, expected_identity=marker_identity
            )
            identity = None
            transaction_exists = False
        else:
            raise DownloadSafetyError("empty destination changed during publication")

    if destination.exists():
        if not destination.is_dir():
            raise DownloadSafetyError("destination has the wrong type")
        _reject_symlinks(destination)
        if _is_empty_directory(destination):
            if empty_destination.exists() or retired_empty.exists():
                raise DownloadSafetyError("publication state is invalid")
            observed = _capture_empty_directory_identity(destination)
            if identity is not None and observed != identity:
                raise DownloadSafetyError("empty destination changed during publication")
            return observed
        if _read_state(destination / STATE_NAME) != state:
            raise DownloadSafetyError("published download state does not match request")
        _verify_manifest(destination, model_id=model_id, revision=revision)
        return None
    return None


def _capture_empty_directory_identity(path: Path) -> _DirectoryIdentity:
    try:
        before = path.lstat()
    except OSError:
        raise DownloadSafetyError("empty destination changed during publication") from None
    identity = _DirectoryIdentity(before.st_dev, before.st_ino, before.st_mode)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise DownloadSafetyError("destination has the wrong type")
    if not _is_empty_directory(path):
        raise DownloadSafetyError("empty destination changed during publication")
    if not _path_matches_identity(path, identity):
        raise DownloadSafetyError("empty destination changed during publication")
    return identity


def _path_matches_identity(path: Path, identity: _DirectoryIdentity) -> bool:
    try:
        observed = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_dev == identity.device
        and observed.st_ino == identity.inode
        and observed.st_mode == identity.mode
    )


def _prepare_staging(incomplete: Path, state: dict[str, str]) -> None:

    if incomplete.is_symlink():
        raise DownloadSafetyError("incomplete download must not be a symbolic link")
    if incomplete.exists():
        if not incomplete.is_dir():
            raise DownloadSafetyError("incomplete download has the wrong type")
        _reject_symlinks(incomplete)
        marker = _read_state(incomplete / STATE_NAME)
        if marker != state:
            raise DownloadSafetyError("incomplete download state does not match request")
        return

    claim = Path(
        tempfile.mkdtemp(
            prefix=f".{incomplete.name.removeprefix('.')}.las-claim-",
            dir=incomplete.parent,
        )
    )
    claim_identity = claim.stat(follow_symlinks=False)
    try:
        _atomic_json_write(claim / STATE_NAME, state)
        _rename_exclusive(claim, incomplete)
        _fsync_directory(incomplete.parent)
    except FileExistsError:
        _validate_existing_staging(incomplete, state)
    except OSError:
        raise DownloadSafetyError("incomplete download could not be created") from None
    finally:
        _remove_empty_claim(
            claim,
            device=claim_identity.st_dev,
            inode=claim_identity.st_ino,
        )


def _validate_existing_staging(incomplete: Path, state: dict[str, str]) -> None:
    if incomplete.is_symlink() or not incomplete.is_dir():
        raise DownloadSafetyError("incomplete download has the wrong type")
    _reject_symlinks(incomplete)
    if _read_state(incomplete / STATE_NAME) != state:
        raise DownloadSafetyError("incomplete download state does not match request")


def _remove_empty_claim(claim: Path, *, device: int, inode: int) -> None:
    # A pathname unlink/rmdir cannot be conditioned on the inode on every
    # supported platform. Unique claims are therefore retained if they were
    # not atomically moved into staging; this never blocks deterministic retry.
    del claim, device, inode


def _read_state(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DownloadSafetyError("incomplete download state is invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise DownloadSafetyError("incomplete download state is invalid") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "model_id", "revision"}
        or any(not isinstance(item, str) for item in value.values())
    ):
        raise DownloadSafetyError("incomplete download state is invalid")
    return value


def _publication_state(
    *,
    model_id: str,
    revision: str,
    identity: _DirectoryIdentity,
) -> dict[str, Any]:
    return {
        "destination": {
            "device": identity.device,
            "inode": identity.inode,
            "mode": identity.mode,
        },
        "format": PUBLISH_STATE_FORMAT,
        "model_id": model_id,
        "revision": revision,
    }


def _read_publication_state(
    path: Path, *, model_id: str, revision: str
) -> _PublicationRecord:
    marker_identity = _capture_regular_file_identity(path)
    if marker_identity is None:
        raise DownloadSafetyError("publication state is invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise DownloadSafetyError("publication state is invalid") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"destination", "format", "model_id", "revision"}
        or value.get("format") != PUBLISH_STATE_FORMAT
        or value.get("model_id") != model_id
        or value.get("revision") != revision
    ):
        raise DownloadSafetyError("publication state does not match request")
    raw_identity = value.get("destination")
    if not isinstance(raw_identity, dict) or set(raw_identity) != {
        "device",
        "inode",
        "mode",
    }:
        raise DownloadSafetyError("publication state is invalid")
    fields = tuple(raw_identity[name] for name in ("device", "inode", "mode"))
    if any(isinstance(field, bool) or not isinstance(field, int) for field in fields):
        raise DownloadSafetyError("publication state is invalid")
    if not _path_matches_file_identity(path, marker_identity):
        raise DownloadSafetyError("publication state changed during inspection")
    return _PublicationRecord(_DirectoryIdentity(*fields), marker_identity)


def _capture_regular_file_identity(path: Path) -> _FileIdentity | None:
    try:
        observed = path.lstat()
    except OSError:
        return None
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_nlink != 1
    ):
        return None
    return _FileIdentity(observed.st_dev, observed.st_ino, observed.st_mode)


def _path_matches_file_identity(path: Path, identity: _FileIdentity) -> bool:
    observed = _capture_regular_file_identity(path)
    return observed == identity


def _reject_symlinks(root: Path) -> None:
    try:
        candidates = tuple(root.rglob("*"))
    except (OSError, RuntimeError):
        raise DownloadSafetyError("downloaded snapshot cannot be inspected") from None
    if any(path.is_symlink() for path in candidates):
        raise DownloadSafetyError("downloaded snapshot contains a symbolic link")


def _is_empty_directory(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    try:
        with os.scandir(path) as entries:
            return next(entries, None) is None
    except OSError:
        raise DownloadSafetyError("directory contents cannot be inspected") from None


def _publish_snapshot(
    incomplete: Path,
    destination: Path,
    *,
    model_id: str,
    revision: str,
    empty_destination_identity: _DirectoryIdentity | None,
) -> None:
    if empty_destination_identity is None:
        if destination.exists() or destination.is_symlink():
            raise DownloadSafetyError("destination appeared before publication")
        try:
            _rename_exclusive(incomplete, destination)
            _fsync_directory(destination.parent)
        except OSError:
            raise DownloadSafetyError("snapshot could not be published") from None
        return

    identity = empty_destination_identity
    publish_state = destination.parent / f".{destination.name}{PUBLISH_STATE_SUFFIX}"
    empty_destination = (
        destination.parent / f".{destination.name}{EMPTY_DESTINATION_SUFFIX}"
    )
    retired_empty = destination.parent / (
        f".{destination.name}{RETIRED_EMPTY_DESTINATION_SUFFIX}"
    )
    committed_state = destination.parent / (
        f".{destination.name}{COMMITTED_STATE_SUFFIX}"
    )
    if committed_state.exists() or committed_state.is_symlink():
        raise DownloadSafetyError("publication state is invalid")
    if publish_state.exists():
        record = _read_publication_state(
            publish_state, model_id=model_id, revision=revision
        )
        if record.destination != identity:
            raise DownloadSafetyError("publication state does not match request")
        marker_identity = record.marker
    else:
        marker_identity = _atomic_json_write_exclusive(
            publish_state,
            _publication_state(
                model_id=model_id,
                revision=revision,
                identity=identity,
            ),
        )

    try:
        if destination.exists():
            if empty_destination.exists() or retired_empty.exists():
                raise DownloadSafetyError("publication state is invalid")
            _require_unchanged_empty_directory(destination, identity)
            _rename_exclusive(destination, empty_destination)
            moved_identity = _capture_path_identity(empty_destination)
            if moved_identity is None:
                raise _DestinationChanged
            if moved_identity != _directory_as_path_identity(identity):
                raise _ChangedHandoff(empty_destination, moved_identity)
            _fsync_directory(destination.parent)
        placeholder = (
            empty_destination if empty_destination.exists() else retired_empty
        )
        _require_unchanged_empty_directory(placeholder, identity)

        _require_unchanged_empty_directory(placeholder, identity)
        _rename_exclusive(incomplete, destination)
        _fsync_directory(destination.parent)
        _require_unchanged_empty_directory(placeholder, identity)
        _remove_published_placeholder(
            empty_destination,
            retired_empty=retired_empty,
            identity=identity,
        )
        _commit_publication_marker(
            publish_state,
            committed_state,
            expected_identity=marker_identity,
        )
        _require_unchanged_empty_directory(retired_empty, identity)
        return
    except _ChangedHandoff as handoff:
        _rollback_changed_handoff(
            destination,
            incomplete=incomplete,
            moved_path=handoff.path,
            moved_identity=handoff.identity,
            publish_state=publish_state,
            marker_identity=marker_identity,
            model_id=model_id,
            revision=revision,
        )
        raise DownloadSafetyError(
            "empty destination changed during publication"
        ) from None
    except _DestinationChanged:
        marker_path, rollback_identity = _marker_for_rollback(
            publish_state,
            committed_state=committed_state,
            expected_identity=marker_identity,
        )
        _rollback_changed_publication(
            destination,
            incomplete=incomplete,
            empty_destination=empty_destination,
            retired_empty=retired_empty,
            publish_state=marker_path,
            marker_identity=rollback_identity,
            expected_placeholder_identity=(
                identity if retired_empty.exists() else None
            ),
            model_id=model_id,
            revision=revision,
        )
        raise DownloadSafetyError(
            "empty destination changed during publication"
        ) from None
    except BaseException as error:
        if _published_snapshot_matches(destination, model_id=model_id, revision=revision):
            remaining = (
                empty_destination
                if empty_destination.exists()
                else retired_empty if retired_empty.exists() else None
            )
            if remaining is None or (
                _path_matches_identity(remaining, identity)
                and _is_empty_directory(remaining)
            ):
                if isinstance(error, OSError):
                    raise DownloadSafetyError("snapshot could not be published") from None
                raise
        marker_path, rollback_identity = _marker_for_rollback(
            publish_state,
            committed_state=committed_state,
            expected_identity=marker_identity,
        )
        _rollback_changed_publication(
            destination,
            incomplete=incomplete,
            empty_destination=empty_destination,
            retired_empty=retired_empty,
            publish_state=marker_path,
            marker_identity=rollback_identity,
            expected_placeholder_identity=(
                identity if retired_empty.exists() else None
            ),
            model_id=model_id,
            revision=revision,
        )
        if isinstance(error, OSError):
            raise DownloadSafetyError("snapshot could not be published") from None
        raise


class _DestinationChanged(RuntimeError):
    pass


class _ChangedHandoff(RuntimeError):
    def __init__(self, path: Path, identity: _PathIdentity) -> None:
        self.path = path
        self.identity = identity
        super().__init__("changed destination handoff")


def _require_unchanged_empty_directory(
    path: Path, identity: _DirectoryIdentity
) -> None:
    if not _path_matches_identity(path, identity) or not _is_empty_directory(path):
        raise _DestinationChanged


def _published_snapshot_matches(
    destination: Path, *, model_id: str, revision: str
) -> bool:
    try:
        if destination.is_symlink() or not destination.is_dir():
            return False
        if _read_state(destination / STATE_NAME) != _download_state(
            model_id=model_id, revision=revision
        ):
            return False
        _verify_manifest(destination, model_id=model_id, revision=revision)
    except DownloadSafetyError:
        return False
    return True


def _rollback_changed_publication(
    destination: Path,
    *,
    incomplete: Path,
    empty_destination: Path,
    retired_empty: Path,
    publish_state: Path,
    marker_identity: _FileIdentity | None,
    expected_placeholder_identity: _DirectoryIdentity | None = None,
    model_id: str,
    revision: str,
) -> bool:
    try:
        placeholder = (
            empty_destination
            if empty_destination.exists()
            else retired_empty if retired_empty.exists() else None
        )
        if expected_placeholder_identity is not None and (
            placeholder is None
            or not _path_matches_identity(
                placeholder, expected_placeholder_identity
            )
        ):
            return False
        if destination.exists():
            if _published_snapshot_matches(
                destination, model_id=model_id, revision=revision
            ):
                if incomplete.exists() or incomplete.is_symlink():
                    return False
                expected_snapshot = _capture_directory_identity(destination)
                if expected_snapshot is None:
                    return False
                try:
                    _rename_exclusive(destination, incomplete)
                except BaseException:
                    if incomplete.exists() and not (
                        _path_matches_identity(incomplete, expected_snapshot)
                        and _published_snapshot_matches(
                            incomplete, model_id=model_id, revision=revision
                        )
                    ):
                        _restore_moved_directory(
                            incomplete,
                            destination,
                            expected_identity=_capture_directory_identity(incomplete),
                        )
                    raise
                if not _path_matches_identity(incomplete, expected_snapshot) or not (
                    _published_snapshot_matches(
                        incomplete, model_id=model_id, revision=revision
                    )
                ):
                    _restore_moved_directory(
                        incomplete,
                        destination,
                        expected_identity=_capture_directory_identity(incomplete),
                    )
                    return False
            elif placeholder is None:
                if not _retire_expected_marker(publish_state, marker_identity):
                    return False
                return True
            else:
                return False
        placeholder = (
            empty_destination
            if empty_destination.exists()
            else retired_empty if retired_empty.exists() else None
        )
        if placeholder is not None:
            if expected_placeholder_identity is not None and not (
                _path_matches_identity(
                    placeholder, expected_placeholder_identity
                )
            ):
                _restore_published_snapshot(
                    incomplete,
                    destination,
                    model_id=model_id,
                    revision=revision,
                )
                return False
            moved_identity = (
                expected_placeholder_identity
                or _capture_directory_identity(placeholder)
            )
            if moved_identity is None:
                return False
            _rename_exclusive(placeholder, destination)
            if not _path_matches_identity(destination, moved_identity):
                raced_identity = _capture_directory_identity(destination)
                if raced_identity is not None and not placeholder.exists():
                    _restore_moved_directory(
                        destination,
                        placeholder,
                        expected_identity=raced_identity,
                    )
                _restore_published_snapshot(
                    incomplete,
                    destination,
                    model_id=model_id,
                    revision=revision,
                )
                return False
        if not destination.exists():
            return False
        if not _retire_expected_marker(publish_state, marker_identity):
            return False
        _fsync_directory(destination.parent)
        return True
    except (DownloadSafetyError, OSError):
        return False


def _restore_published_snapshot(
    incomplete: Path,
    destination: Path,
    *,
    model_id: str,
    revision: str,
) -> bool:
    if destination.exists() or not _published_snapshot_matches(
        incomplete, model_id=model_id, revision=revision
    ):
        return False
    identity = _capture_directory_identity(incomplete)
    if identity is None:
        return False
    return _restore_moved_directory(
        incomplete, destination, expected_identity=identity
    ) and _published_snapshot_matches(
        destination, model_id=model_id, revision=revision
    )


def _restore_placeholder_name(
    destination: Path,
    *,
    placeholder: Path,
    publish_state: Path,
    marker_identity: _FileIdentity,
) -> bool:
    if destination.exists() or not placeholder.exists():
        return False
    try:
        moved_identity = _capture_directory_identity(placeholder)
        if moved_identity is None:
            return False
        _rename_exclusive(placeholder, destination)
        if not _path_matches_identity(destination, moved_identity):
            return False
        _discard_publication_marker(
            publish_state, expected_identity=marker_identity
        )
        _fsync_directory(destination.parent)
    except (DownloadSafetyError, OSError):
        return False
    return True


def _retire_expected_marker(
    publish_state: Path, marker_identity: _FileIdentity | None
) -> bool:
    if marker_identity is None:
        return not publish_state.exists() and not publish_state.is_symlink()
    _discard_publication_marker(
        publish_state, expected_identity=marker_identity
    )
    return True


def _remove_published_placeholder(
    empty_destination: Path,
    *,
    retired_empty: Path,
    identity: _DirectoryIdentity,
) -> None:
    if empty_destination.exists():
        if retired_empty.exists():
            raise _DestinationChanged
        _require_unchanged_empty_directory(empty_destination, identity)
        _rename_exclusive(empty_destination, retired_empty)
        moved_identity = _capture_path_identity(retired_empty)
        if moved_identity is None:
            raise _DestinationChanged
        if moved_identity != _directory_as_path_identity(identity):
            raise _ChangedHandoff(retired_empty, moved_identity)
        _fsync_directory(empty_destination.parent)
    _require_unchanged_empty_directory(retired_empty, identity)


def _capture_directory_identity(path: Path) -> _DirectoryIdentity | None:
    try:
        observed = path.lstat()
    except OSError:
        return None
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        return None
    return _DirectoryIdentity(observed.st_dev, observed.st_ino, observed.st_mode)


def _capture_path_identity(path: Path) -> _PathIdentity | None:
    try:
        observed = path.lstat()
    except OSError:
        return None
    return _PathIdentity(observed.st_dev, observed.st_ino, observed.st_mode)


def _path_matches_path_identity(path: Path, identity: _PathIdentity) -> bool:
    return _capture_path_identity(path) == identity


def _path_present(path: Path) -> bool:
    return _capture_path_identity(path) is not None


def _directory_as_path_identity(identity: _DirectoryIdentity) -> _PathIdentity:
    return _PathIdentity(identity.device, identity.inode, identity.mode)


def _restore_moved_directory(
    source: Path,
    destination: Path,
    *,
    expected_identity: _DirectoryIdentity | None,
) -> bool:
    if expected_identity is None or destination.exists() or not source.exists():
        return False
    try:
        if not _path_matches_identity(source, expected_identity):
            return False
        _rename_exclusive(source, destination)
        return _path_matches_identity(destination, expected_identity)
    except OSError:
        return False


def _restore_moved_entry(
    source: Path,
    destination: Path,
    *,
    expected_identity: _PathIdentity,
) -> bool:
    if _path_present(destination) or not _path_matches_path_identity(
        source, expected_identity
    ):
        return False
    try:
        _rename_exclusive(source, destination)
    except OSError:
        return False
    if _path_matches_path_identity(destination, expected_identity):
        return True
    # A post-rename replacement at the destination is unowned. Leave it
    # visible; moving it backward would hide user content and could cause the
    # caller to restore the model snapshot over the requested name.
    return False


def _rollback_changed_handoff(
    destination: Path,
    *,
    incomplete: Path,
    moved_path: Path,
    moved_identity: _PathIdentity,
    publish_state: Path,
    marker_identity: _FileIdentity,
    model_id: str,
    revision: str,
) -> bool:
    if not _path_matches_path_identity(moved_path, moved_identity):
        return False
    snapshot_moved = False
    try:
        if _path_present(destination):
            if not _published_snapshot_matches(
                destination, model_id=model_id, revision=revision
            ) or _path_present(incomplete):
                return False
            snapshot_identity = _capture_directory_identity(destination)
            if snapshot_identity is None:
                return False
            try:
                _rename_exclusive(destination, incomplete)
            except BaseException:
                moved_candidate = _capture_path_identity(incomplete)
                if moved_candidate is not None and not (
                    _path_matches_identity(incomplete, snapshot_identity)
                    and _published_snapshot_matches(
                        incomplete, model_id=model_id, revision=revision
                    )
                ):
                    _restore_moved_entry(
                        incomplete,
                        destination,
                        expected_identity=moved_candidate,
                    )
                raise
            if not _path_matches_identity(incomplete, snapshot_identity) or not (
                _published_snapshot_matches(
                    incomplete, model_id=model_id, revision=revision
                )
            ):
                moved_candidate = _capture_path_identity(incomplete)
                if moved_candidate is not None:
                    _restore_moved_entry(
                        incomplete,
                        destination,
                        expected_identity=moved_candidate,
                    )
                return False
            snapshot_moved = True
        if not _restore_moved_entry(
            moved_path, destination, expected_identity=moved_identity
        ):
            if not _path_present(moved_path):
                # The carried entry was restored (or moved on by its owner),
                # and its owner may also have moved it away again. Never
                # republish over the requested name; retire our exact marker
                # and leave the verified snapshot staged for a clean retry.
                _discard_publication_marker(
                    publish_state, expected_identity=marker_identity
                )
                _fsync_directory(destination.parent)
                return True
            if snapshot_moved:
                _restore_published_snapshot(
                    incomplete,
                    destination,
                    model_id=model_id,
                    revision=revision,
                )
            return False
        _discard_publication_marker(
            publish_state, expected_identity=marker_identity
        )
        _fsync_directory(destination.parent)
        return True
    except (DownloadSafetyError, OSError):
        return False


def _marker_for_rollback(
    publish_state: Path,
    *,
    committed_state: Path,
    expected_identity: _FileIdentity,
) -> tuple[Path, _FileIdentity | None]:
    if _path_matches_file_identity(publish_state, expected_identity):
        return publish_state, expected_identity
    if _path_matches_file_identity(committed_state, expected_identity):
        return committed_state, expected_identity
    return publish_state, None


def _commit_publication_marker(
    publish_state: Path,
    committed_state: Path,
    *,
    expected_identity: _FileIdentity,
) -> None:
    """Durably retain provenance for the original empty directory inode."""
    if committed_state.exists() or committed_state.is_symlink():
        raise DownloadSafetyError("committed publication state already exists")
    try:
        if not _path_matches_file_identity(publish_state, expected_identity):
            raise _DestinationChanged
        _rename_exclusive(publish_state, committed_state)
        if not _path_matches_file_identity(committed_state, expected_identity):
            raise _DestinationChanged
        _fsync_directory(publish_state.parent)
    except BaseException as error:
        if isinstance(error, _DestinationChanged):
            raise DownloadSafetyError(
                "publication state changed during commit"
            ) from None
        if isinstance(error, OSError):
            raise DownloadSafetyError(
                "publication state could not be committed"
            ) from None
        raise


def _discard_publication_marker(
    publish_state: Path, *, expected_identity: _FileIdentity
) -> None:
    """Retire the exact marker inode without unlinking a raced replacement."""
    if not _path_matches_file_identity(publish_state, expected_identity):
        raise DownloadSafetyError("publication state changed during cleanup")
    cleanup_root = Path(
        tempfile.mkdtemp(
            prefix=f".{publish_state.name.removeprefix('.')}.las-retired-",
            dir=publish_state.parent,
        )
    )
    retired = cleanup_root / "owned-publication-marker"
    try:
        if not _path_matches_file_identity(publish_state, expected_identity):
            raise _DestinationChanged
        _rename_exclusive(publish_state, retired)
        if not _path_matches_file_identity(retired, expected_identity):
            if not publish_state.exists() and retired.exists():
                try:
                    _rename_exclusive(retired, publish_state)
                except OSError:
                    pass
            raise _DestinationChanged
        _fsync_directory(publish_state.parent)
    except BaseException as error:
        if retired.exists() and not _path_matches_file_identity(
            retired, expected_identity
        ):
            moved_identity = _capture_regular_file_identity(retired)
            if (
                moved_identity is not None
                and not publish_state.exists()
                and _path_matches_file_identity(retired, moved_identity)
            ):
                try:
                    _rename_exclusive(retired, publish_state)
                except OSError:
                    pass
        if isinstance(error, _DestinationChanged):
            raise DownloadSafetyError(
                "publication state changed during cleanup"
            ) from None
        if isinstance(error, OSError):
            raise DownloadSafetyError(
                "publication state could not be cleaned"
            ) from None
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically rename a path without replacing an existing target."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise OSError(errno.ENOTSUP, "exclusive rename is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


class _FdText(io.TextIOBase):
    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8", errors="replace")
        offset = 0
        while offset < len(encoded):
            offset += os.write(self._descriptor, encoded[offset:])
        return len(value)

    def flush(self) -> None:
        return None


@contextlib.contextmanager
def _suppress_dependency_output() -> Any:
    """Capture bounded Python/native dependency streams, then discard them."""
    saved: list[tuple[int, int]] = []
    temporary_files = [
        tempfile.TemporaryFile(prefix="las-download-output-") for _ in range(2)
    ]
    pipes = [os.pipe() for _ in range(2)]
    threads = [
        threading.Thread(
            target=_drain_bounded,
            args=(read_fd, temporary),
            daemon=True,
        )
        for (read_fd, _), temporary in zip(pipes, temporary_files, strict=True)
    ]
    try:
        for thread in threads:
            thread.start()
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for target, (_, write_fd) in zip((1, 2), pipes, strict=True):
            saved.append((target, os.dup(target)))
            os.dup2(write_fd, target)
            os.close(write_fd)
        with contextlib.redirect_stdout(_FdText(1)), contextlib.redirect_stderr(
            _FdText(2)
        ):
            yield
    finally:
        for target, descriptor in reversed(saved):
            try:
                os.dup2(descriptor, target)
            finally:
                os.close(descriptor)
        for thread in threads:
            thread.join(timeout=1.0)
        for temporary in temporary_files:
            temporary.close()


def _drain_bounded(read_fd: int, destination: Any) -> None:
    remaining = _OUTPUT_CAPTURE_LIMIT
    try:
        with os.fdopen(read_fd, "rb", closefd=True) as source:
            while True:
                chunk = source.read(8192)
                if not chunk:
                    return
                if remaining > 0:
                    retained = chunk[:remaining]
                    destination.write(retained)
                    remaining -= len(retained)
    except (OSError, ValueError):
        return


def _install_lifetime_output_sink() -> None:
    """Permanently redirect this child process's streams to bounded sinks."""
    temporary_files = [
        tempfile.TemporaryFile(prefix="las-download-process-output-")
        for _ in range(2)
    ]
    pipes = [os.pipe() for _ in range(2)]
    threads = [
        threading.Thread(
            target=_drain_bounded,
            args=(read_fd, temporary),
            daemon=True,
        )
        for (read_fd, _), temporary in zip(pipes, temporary_files, strict=True)
    ]
    for thread in threads:
        thread.start()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    for target, (_, write_fd) in zip((1, 2), pipes, strict=True):
        os.dup2(write_fd, target)
        os.close(write_fd)
    sys.stdout = _FdText(1)
    sys.stderr = _FdText(2)
    _LIFETIME_OUTPUT_RESOURCES.extend(temporary_files)
    _LIFETIME_OUTPUT_RESOURCES.extend(threads)


def run_download_isolated(
    destination: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = "main",
    snapshot_download: Callable[..., str] | None = None,
    mp_context: Any = None,
) -> dict[str, str]:
    """Run the dependency in a child whose complete process output is discarded."""
    _validate_model_id(model_id)
    _validate_revision(revision)
    final_dir = _resolve_destination(destination)
    context = mp_context or multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_download_process_worker,
        args=(
            result_queue,
            str(final_dir),
            model_id,
            revision,
            snapshot_download,
        ),
    )
    with _controlled_parent_termination():
        start_attempted = False
        outcome: dict[str, str] = {"status": "failed"}
        pending_error: BaseException | None = None
        deferred_signals: list[tuple[int, Any]] = []
        reaped = True
        try:
            start_attempted = True
            _start_process_with_deferred_signals(process)
            process.join()
            if process.is_alive():
                outcome = {"status": "failed"}
            elif process.exitcode not in (0, None):
                outcome = {"status": "failed"}
            else:
                try:
                    value = result_queue.get(timeout=0.5)
                except (queue_module.Empty, EOFError, OSError):
                    outcome = {"status": "failed"}
                else:
                    outcome = _sanitize_download_result(value)
        except BaseException as error:
            pending_error = error
        finally:
            if start_attempted:
                with _defer_signals_during_reap() as deferred_signals:
                    reaped = _reap_download_process(process)
            _close_result_queue(result_queue)
        if not reaped:
            raise RuntimeError("download process could not be reaped")
        if pending_error is not None:
            raise pending_error
        if deferred_signals:
            _raise_deferred_signal(deferred_signals[0])
        return outcome


def _download_process_worker(
    result_queue: Any,
    destination: str,
    model_id: str,
    revision: str,
    snapshot_download: Callable[..., str] | None,
) -> None:
    try:
        _install_lifetime_output_sink()
    except BaseException:
        try:
            result_queue.put({"status": "failed"})
        except BaseException:
            pass
        return
    try:
        manifest = download_model(
            destination,
            model_id=model_id,
            revision=revision,
            snapshot_download=snapshot_download,
        )
    except DownloadSafetyError:
        result = {"status": "refused"}
    except BaseException:
        result = {"status": "failed"}
    else:
        result = {"manifest": manifest.name, "status": "completed"}
    try:
        result_queue.put(result)
    except BaseException:
        pass


def _sanitize_download_result(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) not in (
        {"status"},
        {"manifest", "status"},
    ):
        return {"status": "failed"}
    status = value.get("status")
    if status in {"failed", "refused"} and set(value) == {"status"}:
        return {"status": status}
    if (
        status == "completed"
        and value.get("manifest") == MANIFEST_NAME
        and set(value) == {"manifest", "status"}
    ):
        return {"manifest": MANIFEST_NAME, "status": "completed"}
    return {"status": "failed"}


def _reap_download_process(process: Any) -> bool:
    if not process.is_alive():
        return True
    process.terminate()
    process.join(timeout=_PROCESS_REAP_GRACE_SECONDS)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if not callable(kill):
            return False
        kill()
        process.join(timeout=_PROCESS_REAP_GRACE_SECONDS)
    return not process.is_alive()


@contextlib.contextmanager
def _defer_signals_during_reap() -> Any:
    """Record process-control signals without interrupting bounded reaping."""
    received: list[tuple[int, Any]] = []
    if threading.current_thread() is not threading.main_thread():
        yield received
        return
    signals = tuple(
        dict.fromkeys(
            value
            for value in (
                signal.SIGINT,
                signal.SIGTERM,
                getattr(signal, "SIGHUP", None),
            )
            if isinstance(value, int)
        )
    )
    previous = {value: signal.getsignal(value) for value in signals}

    def defer(signum: int, frame: Any) -> None:
        if not received:
            received.append((signum, frame))

    with _blocked_signal_transition(signals):
        for value in signals:
            signal.signal(value, defer)
    try:
        yield received
    finally:
        with _blocked_signal_transition(signals):
            for value, handler in previous.items():
                signal.signal(value, handler)


@contextlib.contextmanager
def _blocked_signal_transition(signals: Sequence[int]) -> Any:
    """Make a multi-handler install/restore indivisible to POSIX signals."""
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if not callable(pthread_sigmask):
        yield
        return
    previous_mask = pthread_sigmask(signal.SIG_BLOCK, set(signals))
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _raise_deferred_signal(received: tuple[int, Any]) -> None:
    signum, frame = received
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    handler = signal.getsignal(signum)
    if handler is signal.SIG_IGN:
        return
    if callable(handler):
        handler(signum, frame)
        return
    raise _ParentTermination(signum)


@contextlib.contextmanager
def _controlled_parent_termination() -> Any:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    signals = tuple(
        value
        for value in (signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if isinstance(value, int)
    )
    previous = {value: signal.getsignal(value) for value in signals}
    terminating = False

    def request_termination(signum: int, frame: Any) -> None:
        del frame
        nonlocal terminating
        if terminating:
            return
        terminating = True
        raise _ParentTermination(signum)

    installed = False
    try:
        with _blocked_signal_transition(signals):
            installed = True
            for value in signals:
                signal.signal(value, request_termination)
        yield
    finally:
        if installed:
            with _blocked_signal_transition(signals):
                for value, handler in previous.items():
                    signal.signal(value, handler)


def _start_process_with_deferred_signals(process: Any) -> None:
    if threading.current_thread() is not threading.main_thread():
        process.start()
        return
    signals = tuple(
        dict.fromkeys(
            value
            for value in (
                signal.SIGINT,
                signal.SIGTERM,
                getattr(signal, "SIGHUP", None),
            )
            if isinstance(value, int)
        )
    )
    previous = {value: signal.getsignal(value) for value in signals}
    received: list[tuple[int, Any]] = []

    def defer(signum: int, frame: Any) -> None:
        if not received:
            received.append((signum, frame))

    with _blocked_signal_transition(signals):
        for value in signals:
            signal.signal(value, defer)
    try:
        process.start()
    finally:
        with _blocked_signal_transition(signals):
            for value, handler in previous.items():
                signal.signal(value, handler)
        if received:
            signum, frame = received[0]
            handler = previous[signum]
            if handler is signal.SIG_IGN:
                pass
            elif callable(handler):
                handler(signum, frame)
            else:
                raise _ParentTermination(signum)


def _close_result_queue(result_queue: Any) -> None:
    try:
        close = getattr(result_queue, "close", None)
        if callable(close):
            close()
        join_thread = getattr(result_queue, "join_thread", None)
        if callable(join_thread):
            join_thread()
    except Exception:
        pass


def _manifest_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            root.rglob("*"),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    except (OSError, RuntimeError):
        raise DownloadSafetyError("downloaded snapshot cannot be enumerated") from None
    for path in candidates:
        relative = path.relative_to(root)
        if relative.parts[0] == ".cache" or relative.as_posix() in {
            MANIFEST_NAME,
            STATE_NAME,
        }:
            continue
        if path.is_symlink():
            raise DownloadSafetyError("downloaded snapshot contains a symbolic link")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            size = path.stat().st_size
        except OSError:
            raise DownloadSafetyError("downloaded snapshot file cannot be hashed") from None
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": digest.hexdigest(),
                "size": size,
            }
        )
    if not files:
        raise DownloadSafetyError("downloaded snapshot contains no model files")
    return files


def _verify_manifest(root: Path, *, model_id: str, revision: str) -> None:
    path = root / MANIFEST_NAME
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 16 * 1024 * 1024
        ):
            raise OSError
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise DownloadSafetyError("published snapshot manifest is invalid") from None
    expected = {
        "files": _manifest_files(root),
        "format": "las-repro-model-sha256-v1",
        "model_id": model_id,
        "revision": revision,
    }
    if value != expected:
        raise DownloadSafetyError("published snapshot manifest is invalid")


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_write_exclusive(
    path: Path, payload: dict[str, Any]
) -> _FileIdentity:
    """Create a transaction marker without replacing any appearing pathname."""
    claim = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.las-marker-claim-",
            dir=path.parent,
        )
    )
    claim_identity = _capture_directory_identity(claim)
    if claim_identity is None:
        raise DownloadSafetyError("publication state could not be created")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=claim
    )
    observed = os.fstat(descriptor)
    temporary_identity = _FileIdentity(
        observed.st_dev, observed.st_ino, observed.st_mode
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if not _path_matches_file_identity(temporary, temporary_identity):
            raise DownloadSafetyError("publication state changed during creation")
        _rename_exclusive(temporary, path)
        if not _path_matches_file_identity(path, temporary_identity):
            raise DownloadSafetyError("publication state changed during creation")
        _fsync_directory(path.parent)
        return temporary_identity
    except FileExistsError:
        raise DownloadSafetyError("publication state appeared during creation") from None
    except OSError:
        raise DownloadSafetyError("publication state could not be created") from None
    finally:
        _remove_owned_marker_claim(
            claim,
            claim_identity=claim_identity,
            temporary=temporary,
            temporary_identity=temporary_identity,
        )


def _remove_owned_marker_claim(
    claim: Path,
    *,
    claim_identity: _DirectoryIdentity,
    temporary: Path,
    temporary_identity: _FileIdentity,
) -> None:
    """Retain uncommitted claims because portable inode-conditional delete is absent."""
    del claim, claim_identity, temporary, temporary_identity


def _import_snapshot_download() -> Callable[..., str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is required on the connected download machine"
        ) from None
    return snapshot_download


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, "invalid command arguments\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Export an allowlisted Qwen3-VL snapshot and SHA-256 manifest."
    )
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        choices=sorted(ALLOWED_MODEL_IDS),
    )
    parser.add_argument("--revision", default="main")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_download_isolated(
            args.destination,
            model_id=args.model_id,
            revision=args.revision,
        )
    except _ParentTermination as termination:
        return 128 + termination.signum
    except DownloadSafetyError:
        print("download refused", file=sys.stderr)
        return 2
    except BaseException:
        print("model download failed", file=sys.stderr)
        return 1
    if result.get("status") == "refused":
        print("download refused", file=sys.stderr)
        return 2
    if result.get("status") != "completed":
        print("model download failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "manifest": MANIFEST_NAME,
                "model_id": args.model_id,
                "revision": args.revision,
                "status": "completed",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
