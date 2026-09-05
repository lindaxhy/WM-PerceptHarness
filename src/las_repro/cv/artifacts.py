"""Content-addressed storage for large computer-vision evidence artifacts."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import (
    AbstractContextManager,
    ExitStack,
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import threading
import time
from typing import Iterator
import weakref

from pydantic import ValidationError

from .contracts import (
    CvEvidenceArtifact,
    CvEvidenceRequest,
    EntityPrompt,
    EvidenceThresholds,
    SamplingPolicy,
)


_MANIFEST_NAME = "manifest.json"
_STREAM_BYTES = 1024 * 1024
_DEFAULT_MAX_FILES = 10_000
_DEFAULT_MAX_BYTES = 8 * 1024 * 1024 * 1024
_DEFAULT_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_KEY_LENGTH = 64
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CACHE_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "video_sha256",
        "provider",
        "model_identity",
        "checkpoint_sha256",
        "entities",
        "sampling",
        "thresholds",
    }
)
_MANIFEST_FIELDS = frozenset({"cache_identity", "artifact"})


class CvArtifactError(RuntimeError):
    """A sanitized artifact-store boundary failure."""


@dataclass(frozen=True, slots=True)
class CvArtifactHandle:
    """Immutable identity for one validated content-addressed artifact."""

    key: str
    manifest_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def cv_cache_key(request: CvEvidenceRequest) -> str:
    """Hash only immutable inputs that determine provider CV evidence."""
    return hashlib.sha256(_canonical_json(_cache_identity(request))).hexdigest()


def _cache_identity(request: CvEvidenceRequest) -> dict[str, object]:
    return {
        "schema_version": request.schema_version,
        "video_sha256": request.video_sha256,
        "provider": request.provider,
        "model_identity": request.model_identity,
        "checkpoint_sha256": request.checkpoint_sha256,
        "entities": [entity.model_dump(mode="json") for entity in request.entities],
        "sampling": request.sampling.model_dump(mode="json"),
        "thresholds": request.thresholds.model_dump(mode="json"),
    }


class CvArtifactStore:
    """Owner-controlled, content-addressed storage for CV evidence chunks."""

    _process_locks_guard = threading.Lock()
    _process_locks: dict[str, threading.RLock] = {}

    def __init__(
        self,
        root: Path,
        *,
        max_files: int = _DEFAULT_MAX_FILES,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_manifest_bytes: int = _DEFAULT_MAX_MANIFEST_BYTES,
    ) -> None:
        if max_files <= 0 or max_bytes <= 0 or max_manifest_bytes <= 0:
            raise ValueError("artifact limits must be positive")
        self._root = Path(os.path.abspath(root))
        self._max_files = max_files
        self._max_bytes = max_bytes
        self._max_manifest_bytes = max_manifest_bytes
        self._max_tree_entries = max(64, max_files * 4)
        self._staging: dict[Path, tuple[str, int, int]] = {}
        self._staging_lock = threading.RLock()
        try:
            (
                self._root_descriptor,
                self._ancestry_identities,
            ) = self._open_trusted_ancestry(self._root, create_final=True)
        except (OSError, ValueError):
            raise CvArtifactError("unsafe CV artifact directory") from None
        root_status = os.fstat(self._root_descriptor)
        self._root_identity = root_status.st_dev, root_status.st_ino
        self._lifecycle_lock = threading.Lock()
        self._active_operations = 0
        self._closed = False
        self._root_finalizer = weakref.finalize(
            self, os.close, self._root_descriptor
        )

    def __enter__(self) -> CvArtifactStore:
        self._begin_operation()
        return self

    def __exit__(self, *exc_info: object) -> None:
        try:
            self._finish_operation()
        finally:
            self.close()

    def close(self) -> None:
        """Idempotently release the trusted root after active operations finish."""
        descriptor: int | None = None
        with self._lifecycle_lock:
            if not self._closed:
                self._closed = True
            if self._active_operations == 0:
                descriptor = self._take_root_descriptor_locked()
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise CvArtifactError("unable to close CV artifact store") from None

    @contextmanager
    def _operation(self) -> Iterator[None]:
        self._begin_operation()
        try:
            yield
        finally:
            self._finish_operation()

    def _begin_operation(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise CvArtifactError("CV artifact store is closed")
            self._active_operations += 1

    def _finish_operation(self) -> None:
        descriptor: int | None = None
        with self._lifecycle_lock:
            self._active_operations -= 1
            if self._active_operations == 0 and self._closed:
                descriptor = self._take_root_descriptor_locked()
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                raise CvArtifactError("unable to close CV artifact store") from None

    def _take_root_descriptor_locked(self) -> int | None:
        if self._root_descriptor < 0:
            return None
        descriptor = self._root_descriptor
        self._root_descriptor = -1
        self._root_finalizer.detach()
        return descriptor

    def lookup(self, key: str) -> CvArtifactHandle | None:
        """Return a digest-validated immutable handle, or None on a miss."""
        with self._operation():
            return self._lookup(key)

    def _lookup(self, key: str) -> CvArtifactHandle | None:
        self._validate_key(key)
        self._validate_store_root()
        prefix_descriptor = self._open_prefix(key, create=False)
        if prefix_descriptor is None:
            return None
        os.close(prefix_descriptor)
        with self._key_lock(key):
            identity = self._entry_identity(key)
            if identity is None:
                return None
            entry = self._entry_path(key)
            try:
                artifact, manifest_digest = self._validate_entry(entry, key)
            except (
                CvArtifactError,
                OSError,
                RecursionError,
                TypeError,
                ValueError,
                ValidationError,
            ):
                self._quarantine_entry(key, entry, identity)
                return None
            del artifact
            self._validate_store_root()
            return CvArtifactHandle(key=key, manifest_sha256=manifest_digest)

    @contextmanager
    def staging(self, key: str) -> Iterator[Path]:
        """Yield an owner-only sibling directory and remove it on exit."""
        with self._operation():
            with self._staging_directory(key) as staging:
                yield staging

    @contextmanager
    def _staging_directory(self, key: str) -> Iterator[Path]:
        self._validate_key(key)
        self._validate_store_root()
        prefix = self._prefix_directory(key)
        prefix_descriptor = self._open_prefix(key, create=True)
        if prefix_descriptor is None:
            raise CvArtifactError("unable to create CV artifact staging")
        try:
            staging, identity = self._make_private_temp_directory(
                prefix, prefix_descriptor, ".staging-"
            )
        finally:
            os.close(prefix_descriptor)
        with self._staging_lock:
            self._staging[staging] = (key, identity[0], identity[1])
        try:
            yield staging
        finally:
            with self._staging_lock:
                expected = self._staging.pop(staging, None)
            if expected is not None and self._has_identity(staging, expected[1:]):
                shutil.rmtree(staging)

    def publish(
        self,
        request: CvEvidenceRequest,
        staging: Path,
        artifact: CvEvidenceArtifact,
        *,
        commit_guard: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> CvArtifactHandle:
        """Validate, fsync, and atomically publish one complete entry."""
        with self._operation():
            return self._publish(
                request,
                staging,
                artifact,
                commit_guard=commit_guard,
            )

    def _publish(
        self,
        request: CvEvidenceRequest,
        staging: Path,
        artifact: CvEvidenceArtifact,
        *,
        commit_guard: Callable[[], AbstractContextManager[object]] | None,
    ) -> CvArtifactHandle:
        try:
            request = CvEvidenceRequest.model_validate(request.model_dump(mode="python"))
            artifact = CvEvidenceArtifact.model_validate(
                artifact.model_dump(mode="python")
            )
            key = cv_cache_key(request)
            self._validate_store_root()
            stage = Path(os.path.abspath(staging))
            with self._staging_lock:
                expected = self._staging.get(stage)
            if expected is None or expected[0] != key:
                raise ValueError
            if not self._has_identity(stage, expected[1:]):
                raise ValueError
            self._validate_private_directory(stage, require_owner_only=True)
            self._validate_artifact_matches_request(request, artifact)
            self._validate_artifact_files(stage, artifact, include_manifest=False)
            manifest = _canonical_json(
                {
                    "cache_identity": _cache_identity(request),
                    "artifact": artifact.model_dump(mode="json"),
                }
            )
            if len(manifest) > self._max_manifest_bytes:
                raise ValueError
            manifest_digest = hashlib.sha256(manifest).hexdigest()
            with self._publication_staging(key) as publication:
                self._copy_artifact_files(stage, publication, artifact)
                self._write_new_file(publication / _MANIFEST_NAME, manifest)
                self._fsync_tree(publication)
                destination = self._entry_path(key)
                with self._key_lock(key):
                    destination_identity = self._entry_identity(key)
                    if destination_identity is not None:
                        try:
                            existing_artifact, existing_digest = self._validate_entry(
                                destination, key
                            )
                        except (
                            CvArtifactError,
                            OSError,
                            RecursionError,
                            TypeError,
                            ValueError,
                            ValidationError,
                        ):
                            self._quarantine_entry(
                                key, destination, destination_identity
                            )
                        else:
                            self._validate_artifact_matches_request(
                                request, existing_artifact
                            )
                            self._validate_store_root()
                            return CvArtifactHandle(
                                key=key, manifest_sha256=existing_digest
                            )
                    prefix_descriptor = self._open_prefix(key, create=False)
                    if prefix_descriptor is None:
                        raise ValueError
                    try:
                        publication_status = os.stat(
                            publication.name,
                            dir_fd=prefix_descriptor,
                            follow_symlinks=False,
                        )
                        publication_identity = (
                            publication_status.st_dev,
                            publication_status.st_ino,
                        )
                        staged_artifact, staged_digest = self._validate_entry(
                            publication, key
                        )
                        if (
                            staged_artifact != artifact
                            or staged_digest != manifest_digest
                        ):
                            raise ValueError
                        current_publication = os.stat(
                            publication.name,
                            dir_fd=prefix_descriptor,
                            follow_symlinks=False,
                        )
                        if (
                            current_publication.st_dev,
                            current_publication.st_ino,
                        ) != publication_identity:
                            raise ValueError
                        guard = (
                            nullcontext()
                            if commit_guard is None
                            else commit_guard()
                        )
                        with guard:
                            os.replace(
                                publication.name,
                                key,
                                src_dir_fd=prefix_descriptor,
                                dst_dir_fd=prefix_descriptor,
                            )
                        os.fsync(prefix_descriptor)
                        destination_status = os.stat(
                            key,
                            dir_fd=prefix_descriptor,
                            follow_symlinks=False,
                        )
                        destination_identity = (
                            destination_status.st_dev,
                            destination_status.st_ino,
                        )
                    finally:
                        os.close(prefix_descriptor)
                    try:
                        published_artifact, published_digest = self._validate_entry(
                            destination, key
                        )
                        if (
                            published_artifact != artifact
                            or published_digest != manifest_digest
                        ):
                            raise ValueError
                    except (
                        CvArtifactError,
                        OSError,
                        RecursionError,
                        TypeError,
                        ValueError,
                        ValidationError,
                    ):
                        self._quarantine_entry(
                            key, destination, destination_identity
                        )
                        raise ValueError
                    self._validate_store_root()
                    return CvArtifactHandle(
                        key=key,
                        manifest_sha256=manifest_digest,
                    )
        except (
            AttributeError,
            CvArtifactError,
            OSError,
            TypeError,
            ValueError,
            ValidationError,
        ):
            raise CvArtifactError("unable to publish CV artifact") from None

    def load(self, handle: CvArtifactHandle) -> CvEvidenceArtifact:
        """Revalidate the handle and parse its canonical manifest."""
        with self._operation():
            return self._load(handle)

    def read_overlays(
        self,
        handle: CvArtifactHandle,
        paths: tuple[str, ...],
        *,
        max_total_bytes: int = 64 * 1024 * 1024,
    ) -> dict[str, bytes]:
        """Read at most 24 registered PNGs, bound to an authenticated manifest.

        This is deliberately not a general artifact-file accessor. Returned
        bytes are checked again after load, through pinned no-follow descriptors;
        callers never need filesystem access to masks or the cache layout.
        """
        with self._operation():
            try:
                if (
                    not isinstance(paths, tuple)
                    or len(paths) > 24
                    or any(not isinstance(path, str) for path in paths)
                    or len(set(paths)) != len(paths)
                    or type(max_total_bytes) is not int
                    or not 0 < max_total_bytes <= 64 * 1024 * 1024
                ):
                    raise ValueError
                artifact = self._load(handle)
                registered = {record.path for record in artifact.overlay_records}
                files = {file.path: file for file in artifact.files}
                if not set(paths) <= registered:
                    raise ValueError
                if sum(files[path].size_bytes for path in paths) > max_total_bytes:
                    raise ValueError
                # _load releases its lock before this acquisition (flock is not
                # reentrant across independently opened lock descriptors).
                with self._key_lock(handle.key), ExitStack() as stack:
                    self._validate_store_root()
                    prefix = self._open_prefix(handle.key, create=False)
                    if prefix is None:
                        raise ValueError
                    stack.callback(os.close, prefix)
                    entry = self._open_directory_descriptor(handle.key, dir_fd=prefix)
                    stack.callback(os.close, entry)
                    entry_status = os.fstat(entry)
                    self._validate_private_directory_status(
                        entry_status, require_owner_only=True
                    )
                    pinned = []
                    output: dict[str, bytes] = {}
                    remaining = max_total_bytes
                    for path in (_MANIFEST_NAME, *paths):
                        opened = stack.enter_context(
                            self._open_relative_regular_file(
                                entry, path, nonblocking=True
                            )
                        )
                        descriptor, parent, name, status = opened
                        limit = (
                            self._max_manifest_bytes
                            if path == _MANIFEST_NAME
                            else remaining
                        )
                        content, digest = self._read_bounded_descriptor(
                            descriptor, limit
                        )
                        if path == _MANIFEST_NAME:
                            if digest != handle.manifest_sha256:
                                raise ValueError
                        else:
                            file = files[path]
                            if (
                                len(content) != file.size_bytes
                                or digest != file.sha256
                                or not content.startswith(_PNG_SIGNATURE)
                            ):
                                raise ValueError
                            output[path] = content
                            remaining -= len(content)
                        pinned.append((path, descriptor, parent, name, status))
                    for path, descriptor, parent, name, status in pinned:
                        expected = self._stat_signature(status)
                        if (
                            self._stat_signature(os.fstat(descriptor)) != expected
                            or self._stat_signature(
                                os.stat(name, dir_fd=parent, follow_symlinks=False)
                            )
                            != expected
                        ):
                            raise ValueError
                        # Rewalk ancestry: an attacker may rename an intermediate
                        # directory while the original parent descriptor stays valid.
                        with self._open_relative_regular_file(
                            entry, path, nonblocking=True
                        ) as current:
                            if self._stat_signature(current[3]) != expected:
                                raise ValueError
                    current = os.stat(handle.key, dir_fd=prefix, follow_symlinks=False)
                    if self._stat_signature(current) != self._stat_signature(
                        entry_status
                    ):
                        raise ValueError
                    self._validate_private_directory_status(
                        os.fstat(entry), require_owner_only=True
                    )
                    self._validate_store_root()
                    current_prefix = self._open_prefix(handle.key, create=False)
                    if current_prefix is None:
                        raise ValueError
                    stack.callback(os.close, current_prefix)
                    original_prefix = os.fstat(prefix)
                    reopened_prefix = os.fstat(current_prefix)
                    if (original_prefix.st_dev, original_prefix.st_ino) != (
                        reopened_prefix.st_dev,
                        reopened_prefix.st_ino,
                    ):
                        raise ValueError
                    return output
            except (CvArtifactError, OSError, KeyError, TypeError, ValueError):
                raise CvArtifactError("unable to read CV overlays") from None

    def _load(self, handle: CvArtifactHandle) -> CvEvidenceArtifact:
        try:
            if not isinstance(handle, CvArtifactHandle):
                raise ValueError
            self._validate_key(handle.key)
            self._validate_store_root()
            prefix_descriptor = self._open_prefix(handle.key, create=False)
            if prefix_descriptor is None:
                raise ValueError
            os.close(prefix_descriptor)
            with self._key_lock(handle.key):
                entry = self._entry_path(handle.key)
                identity = self._entry_identity(handle.key)
                if identity is None:
                    raise ValueError
                try:
                    artifact, manifest_digest = self._validate_entry(
                        entry, handle.key
                    )
                except (
                    CvArtifactError,
                    OSError,
                    RecursionError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ):
                    self._quarantine_entry(handle.key, entry, identity)
                    raise ValueError
                if manifest_digest != handle.manifest_sha256:
                    raise ValueError
                self._validate_store_root()
                return artifact
        except (
            CvArtifactError,
            OSError,
            RecursionError,
            TypeError,
            ValueError,
            ValidationError,
        ):
            raise CvArtifactError("unable to load CV artifact") from None

    def _entry_path(self, key: str) -> Path:
        return self._root / key[:2] / key

    def _prefix_directory(self, key: str) -> Path:
        return self._root / key[:2]

    def _validate_store_root(self) -> None:
        try:
            descriptor, identities = self._open_trusted_ancestry(
                self._root, create_final=False
            )
            current = os.fstat(self._root_descriptor)
            reopened = os.fstat(descriptor)
            os.close(descriptor)
        except (OSError, ValueError):
            raise CvArtifactError("unsafe CV artifact directory") from None
        if (
            identities != self._ancestry_identities
            or (current.st_dev, current.st_ino) != self._root_identity
            or (reopened.st_dev, reopened.st_ino) != self._root_identity
        ):
            raise CvArtifactError("unsafe CV artifact directory")

    @classmethod
    def _open_trusted_ancestry(
        cls, path: Path, *, create_final: bool
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        if not path.is_absolute():
            raise ValueError
        descriptor = cls._open_directory_descriptor(Path("/"))
        identities: list[tuple[int, int]] = []
        try:
            cls._validate_ancestor_status(os.fstat(descriptor), final=False)
            root_status = os.fstat(descriptor)
            identities.append((root_status.st_dev, root_status.st_ino))
            parts = path.parts[1:]
            for index, part in enumerate(parts):
                final = index == len(parts) - 1
                observed_missing = False
                try:
                    child = cls._open_directory_descriptor(part, dir_fd=descriptor)
                except FileNotFoundError:
                    if not (create_final and final):
                        raise
                    observed_missing = True
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    child = cls._open_directory_descriptor(part, dir_fd=descriptor)
                try:
                    status = os.fstat(child)
                    cls._validate_ancestor_status(status, final=final)
                    if observed_missing:
                        os.fsync(descriptor)
                    identities.append((status.st_dev, status.st_ino))
                    os.close(descriptor)
                except BaseException:
                    os.close(child)
                    raise
                descriptor = child
            return descriptor, tuple(identities)
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _validate_ancestor_status(status: os.stat_result, *, final: bool) -> None:
        effective_uid = os.geteuid() if hasattr(os, "geteuid") else status.st_uid
        trusted_owners = {effective_uid, 0}
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid not in trusted_owners
            or (final and status.st_uid != effective_uid)
            or status.st_mode & 0o022
        ):
            raise ValueError

    def _open_prefix(self, key: str, *, create: bool) -> int | None:
        name = key[:2]
        observed_missing = False
        try:
            descriptor = self._open_directory_descriptor(
                name, dir_fd=self._root_descriptor
            )
        except FileNotFoundError:
            if not create:
                return None
            observed_missing = True
            try:
                os.mkdir(name, mode=0o700, dir_fd=self._root_descriptor)
            except FileExistsError:
                pass
            descriptor = self._open_directory_descriptor(
                name, dir_fd=self._root_descriptor
            )
        status = os.fstat(descriptor)
        effective_uid = os.geteuid() if hasattr(os, "geteuid") else status.st_uid
        if status.st_uid != effective_uid or status.st_mode & 0o077:
            os.close(descriptor)
            raise CvArtifactError("unsafe CV artifact directory")
        if observed_missing:
            os.fsync(self._root_descriptor)
        return descriptor

    @staticmethod
    def _make_private_temp_directory(
        parent: Path, parent_descriptor: int, prefix: str
    ) -> tuple[Path, tuple[int, int]]:
        for _ in range(128):
            name = f"{prefix}{secrets.token_hex(16)}"
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            descriptor = CvArtifactStore._open_directory_descriptor(
                name, dir_fd=parent_descriptor
            )
            try:
                os.fchmod(descriptor, 0o700)
                status = os.fstat(descriptor)
                return parent / name, (status.st_dev, status.st_ino)
            finally:
                os.close(descriptor)
        raise CvArtifactError("unable to create CV artifact staging")

    @staticmethod
    def _validate_key(key: str) -> None:
        if (
            not isinstance(key, str)
            or len(key) != _KEY_LENGTH
            or any(character not in "0123456789abcdef" for character in key)
        ):
            raise CvArtifactError("invalid CV artifact key")

    @staticmethod
    def _validate_private_directory(path: Path, *, require_owner_only: bool) -> None:
        try:
            status = path.lstat()
            CvArtifactStore._validate_private_directory_status(
                status, require_owner_only=require_owner_only
            )
        except (OSError, ValueError):
            raise CvArtifactError("unsafe CV artifact directory") from None

    @staticmethod
    def _validate_private_directory_status(
        status: os.stat_result, *, require_owner_only: bool
    ) -> None:
        owner_matches = not hasattr(os, "geteuid") or status.st_uid == os.geteuid()
        forbidden = 0o077 if require_owner_only else 0o022
        if (
            not stat.S_ISDIR(status.st_mode)
            or not owner_matches
            or status.st_mode & forbidden
        ):
            raise ValueError

    @staticmethod
    def _has_identity(path: Path, identity: tuple[int, int]) -> bool:
        try:
            status = path.lstat()
        except OSError:
            return False
        return stat.S_ISDIR(status.st_mode) and (status.st_dev, status.st_ino) == identity

    @staticmethod
    def _path_exists(path: Path) -> bool:
        try:
            path.lstat()
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _path_identity(path: Path) -> tuple[int, int]:
        status = path.lstat()
        return status.st_dev, status.st_ino

    def _entry_identity(self, key: str) -> tuple[int, int] | None:
        prefix_descriptor = self._open_prefix(key, create=False)
        if prefix_descriptor is None:
            return None
        try:
            try:
                status = os.stat(
                    key, dir_fd=prefix_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                return None
            return status.st_dev, status.st_ino
        finally:
            os.close(prefix_descriptor)

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        lock_identity = f"{self._root}:{key}"
        with self._process_locks_guard:
            process_lock = self._process_locks.setdefault(
                lock_identity, threading.RLock()
            )
        with process_lock:
            lock_name = f".{key}.lock"
            flags = os.O_RDWR | os.O_CREAT
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if nofollow is None:
                raise CvArtifactError("unable to lock CV artifact")
            flags |= nofollow
            descriptor: int | None = None
            prefix_descriptor: int | None = None
            try:
                prefix_descriptor = self._open_prefix(key, create=True)
                if prefix_descriptor is None:
                    raise ValueError
                descriptor = os.open(
                    lock_name, flags, 0o600, dir_fd=prefix_descriptor
                )
                status = os.fstat(descriptor)
                owner_matches = not hasattr(os, "geteuid") or status.st_uid == os.geteuid()
                current = os.stat(
                    lock_name,
                    dir_fd=prefix_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(status.st_mode)
                    or not owner_matches
                    or status.st_mode & 0o077
                    or (current.st_dev, current.st_ino)
                    != (status.st_dev, status.st_ino)
                ):
                    raise CvArtifactError("unsafe CV artifact lock")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                current = os.stat(
                    lock_name,
                    dir_fd=prefix_descriptor,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) != (status.st_dev, status.st_ino):
                    raise CvArtifactError("unsafe CV artifact lock")
            except (CvArtifactError, OSError, ValueError):
                if descriptor is not None:
                    os.close(descriptor)
                if prefix_descriptor is not None:
                    os.close(prefix_descriptor)
                raise CvArtifactError("unable to lock CV artifact") from None
            try:
                yield
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    raise CvArtifactError("unable to unlock CV artifact") from None
                finally:
                    os.close(descriptor)
                    if prefix_descriptor is not None:
                        os.close(prefix_descriptor)

    def _quarantine_entry(
        self, key: str, entry: Path, identity: tuple[int, int]
    ) -> bool:
        prefix_descriptor: int | None = None
        quarantine_descriptor: int | None = None
        try:
            prefix_descriptor = self._open_prefix(key, create=False)
            if prefix_descriptor is None:
                return False
            try:
                current = os.stat(
                    key, dir_fd=prefix_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                return False
            if (current.st_dev, current.st_ino) != identity:
                raise ValueError
            observed_missing = False
            try:
                quarantine_descriptor = self._open_directory_descriptor(
                    "quarantine", dir_fd=self._root_descriptor
                )
            except FileNotFoundError:
                observed_missing = True
                try:
                    os.mkdir(
                        "quarantine", mode=0o700, dir_fd=self._root_descriptor
                    )
                except FileExistsError:
                    pass
                quarantine_descriptor = self._open_directory_descriptor(
                    "quarantine", dir_fd=self._root_descriptor
                )
            quarantine_status = os.fstat(quarantine_descriptor)
            effective_uid = (
                os.geteuid() if hasattr(os, "geteuid") else quarantine_status.st_uid
            )
            if quarantine_status.st_uid != effective_uid or quarantine_status.st_mode & 0o077:
                raise ValueError
            if observed_missing:
                os.fsync(self._root_descriptor)
            try:
                current = os.stat(
                    key, dir_fd=prefix_descriptor, follow_symlinks=False
                )
            except FileNotFoundError:
                return False
            if (current.st_dev, current.st_ino) != identity:
                raise ValueError
            destination = f"{key}-{time.time_ns()}-{secrets.token_hex(8)}"
            try:
                os.replace(
                    key,
                    destination,
                    src_dir_fd=prefix_descriptor,
                    dst_dir_fd=quarantine_descriptor,
                )
            except FileNotFoundError:
                try:
                    os.stat(key, dir_fd=prefix_descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    return False
                raise
            os.fsync(quarantine_descriptor)
            os.fsync(prefix_descriptor)
            return True
        except (CvArtifactError, OSError, ValueError):
            raise CvArtifactError("unable to quarantine CV artifact") from None
        finally:
            if quarantine_descriptor is not None:
                os.close(quarantine_descriptor)
            if prefix_descriptor is not None:
                os.close(prefix_descriptor)

    @contextmanager
    def _publication_staging(self, key: str) -> Iterator[Path]:
        prefix = self._prefix_directory(key)
        prefix_descriptor = self._open_prefix(key, create=True)
        if prefix_descriptor is None:
            raise CvArtifactError("unable to create CV artifact staging")
        try:
            publication, identity = self._make_private_temp_directory(
                prefix, prefix_descriptor, ".publish-"
            )
        finally:
            os.close(prefix_descriptor)
        try:
            yield publication
        finally:
            if self._has_identity(publication, identity):
                shutil.rmtree(publication)

    def _copy_artifact_files(
        self,
        source: Path,
        destination: Path,
        artifact: CvEvidenceArtifact,
    ) -> None:
        directories = sorted(
            {
                "/".join(parts[:index])
                for file in artifact.files
                for parts in (file.path.split("/"),)
                for index in range(1, len(parts))
            },
            key=lambda value: (value.count("/"), value),
        )
        for relative in directories:
            target = destination.joinpath(*relative.split("/"))
            target.mkdir(mode=0o700)
            self._validate_private_directory(target, require_owner_only=True)
        total = 0
        for file in artifact.files:
            with self._open_relative_regular_file(
                source, file.path, nonblocking=True
            ) as (
                source_descriptor,
                source_parent,
                source_name,
                source_status,
            ):
                if (
                    source_status.st_nlink != 1
                    or source_status.st_size != file.size_bytes
                ):
                    raise ValueError
                target = destination.joinpath(*file.path.split("/"))
                target_descriptor = self._open_new_regular_file(target)
                try:
                    digest = hashlib.sha256()
                    size = 0
                    while chunk := os.read(source_descriptor, _STREAM_BYTES):
                        size += len(chunk)
                        total += len(chunk)
                        if total > self._max_bytes:
                            raise ValueError
                        digest.update(chunk)
                        self._write_all(target_descriptor, chunk)
                    os.fsync(target_descriptor)
                    target_status = os.fstat(target_descriptor)
                finally:
                    os.close(target_descriptor)
                source_current = os.stat(
                    source_name, dir_fd=source_parent, follow_symlinks=False
                )
                if (
                    (source_status.st_dev, source_status.st_ino)
                    != (source_current.st_dev, source_current.st_ino)
                    or source_current.st_nlink != 1
                    or not stat.S_ISREG(target_status.st_mode)
                    or target_status.st_nlink != 1
                    or target_status.st_size != size
                    or size != file.size_bytes
                    or digest.hexdigest() != file.sha256
                ):
                    raise ValueError

    @contextmanager
    def _open_relative_regular_file(
        self, root: Path | int, relative: str, *, nonblocking: bool = False
    ) -> Iterator[tuple[int, int, str, os.stat_result]]:
        parts = relative.split("/")
        directory_descriptor = self._open_directory_descriptor(root)
        try:
            for part in parts[:-1]:
                next_descriptor = self._open_directory_descriptor(
                    part, dir_fd=directory_descriptor
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if nofollow is None:
                raise ValueError
            flags = os.O_RDONLY | nofollow
            if nonblocking:
                nonblocking_flag = getattr(os, "O_NONBLOCK", None)
                if nonblocking_flag is None:
                    raise ValueError
                flags |= nonblocking_flag
            file_descriptor = os.open(
                parts[-1], flags, dir_fd=directory_descriptor
            )
            try:
                status = os.fstat(file_descriptor)
                if not stat.S_ISREG(status.st_mode):
                    raise ValueError
                yield file_descriptor, directory_descriptor, parts[-1], status
            finally:
                os.close(file_descriptor)
        finally:
            os.close(directory_descriptor)

    @staticmethod
    def _open_directory_descriptor(
        path: Path | str | int, *, dir_fd: int | None = None
    ) -> int:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory_flag is None:
            raise ValueError
        if isinstance(path, int):
            if dir_fd is not None:
                raise ValueError
            descriptor = os.dup(path)
        else:
            descriptor = os.open(
                path,
                os.O_RDONLY | nofollow | directory_flag,
                dir_fd=dir_fd,
            )
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_new_regular_file(path: Path) -> int:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise ValueError
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        return descriptor

    @staticmethod
    def _write_all(descriptor: int, value: bytes) -> None:
        remaining = memoryview(value)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("unable to write artifact")
            remaining = remaining[written:]

    def _write_new_file(self, path: Path, value: bytes) -> None:
        descriptor = self._open_new_regular_file(path)
        try:
            self._write_all(descriptor, value)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_artifact_matches_request(
        request: CvEvidenceRequest, artifact: CvEvidenceArtifact
    ) -> None:
        if (
            artifact.provider != request.provider
            or artifact.model_identity != request.model_identity
            or artifact.video_sha256 != request.video_sha256
            or artifact.checkpoint_sha256 != request.checkpoint_sha256
            or artifact.entities != request.entities
        ):
            raise ValueError

    def _validate_artifact_files(
        self,
        directory: Path | int,
        artifact: CvEvidenceArtifact,
        *,
        include_manifest: bool,
        pinned_files: tuple[tuple[str, int, int, str, os.stat_result], ...] = (),
    ) -> None:
        if len(artifact.files) > self._max_files:
            raise ValueError
        if sum(file.size_bytes for file in artifact.files) > self._max_bytes:
            raise ValueError
        expected = {file.path for file in artifact.files}
        if _MANIFEST_NAME in expected or len(expected) != len(artifact.files):
            raise ValueError
        before = self._capture_tree_metadata(directory)
        actual = {
            relative
            for relative, metadata in before.items()
            if relative and stat.S_ISREG(metadata[2])
        }
        actual_directories = {
            relative
            for relative, metadata in before.items()
            if relative and stat.S_ISDIR(metadata[2])
        }
        allowed = expected | ({_MANIFEST_NAME} if include_manifest else set())
        if actual != allowed:
            raise ValueError
        expected_directories = {
            "/".join(parts[:index])
            for file_path in expected
            for parts in (file_path.split("/"),)
            for index in range(1, len(parts))
        }
        if actual_directories != expected_directories:
            raise ValueError
        total = 0
        with ExitStack() as stack:
            opened_files: dict[
                str, tuple[int, int, str, os.stat_result]
            ] = {
                relative: (descriptor, parent, name, status)
                for relative, descriptor, parent, name, status in pinned_files
            }
            for file in artifact.files:
                opened_files[file.path] = stack.enter_context(
                    self._open_relative_regular_file(
                        directory, file.path, nonblocking=True
                    )
                )
            for relative, (_, _, _, opened) in opened_files.items():
                if relative not in before or self._stat_signature(opened) != before[relative]:
                    raise ValueError
            for file in artifact.files:
                descriptor = opened_files[file.path][0]
                os.lseek(descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                size = 0
                signature = bytearray()
                while chunk := os.read(descriptor, _STREAM_BYTES):
                    size += len(chunk)
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise ValueError
                    if len(signature) < len(_PNG_SIGNATURE):
                        signature.extend(
                            chunk[: len(_PNG_SIGNATURE) - len(signature)]
                        )
                    digest.update(chunk)
                if size != file.size_bytes or digest.hexdigest() != file.sha256:
                    raise ValueError
                self._validate_descriptor_content(file.path, bytes(signature))
            after = self._capture_tree_metadata(directory)
            if before != after:
                raise ValueError
            for relative, (descriptor, parent, name, opened) in opened_files.items():
                current_descriptor = os.fstat(descriptor)
                current_path = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (
                    self._stat_signature(opened) != self._stat_signature(current_descriptor)
                    or self._stat_signature(opened) != self._stat_signature(current_path)
                    or self._stat_signature(opened) != after[relative]
                ):
                    raise ValueError

    @staticmethod
    def _validate_descriptor_content(relative: str, signature: bytes) -> None:
        if relative.startswith("overlays/") and signature != _PNG_SIGNATURE:
            raise ValueError

    def _capture_tree_metadata(
        self, root: Path | int
    ) -> dict[str, tuple[int, int, int, int, int, int, int, int]]:
        root_status = os.fstat(root) if isinstance(root, int) else root.lstat()
        metadata = {"": self._stat_signature(root_status)}
        file_count = 0
        for relative, status in self._walk_tree_relative(root):
            if stat.S_ISREG(status.st_mode):
                file_count += 1
                if file_count > self._max_files + 1:
                    raise ValueError
                if status.st_nlink != 1:
                    raise ValueError
            metadata[relative] = self._stat_signature(status)
        return metadata

    @staticmethod
    def _stat_signature(
        status: os.stat_result,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        return (
            status.st_dev,
            status.st_ino,
            status.st_mode,
            status.st_nlink,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
            status.st_flags if hasattr(status, "st_flags") else 0,
        )

    def _walk_tree_relative(
        self, root: Path | int
    ) -> Iterator[tuple[str, os.stat_result]]:
        entry_count = 0
        root_descriptor = self._open_directory_descriptor(root)
        stack: list[tuple[str, int, os.ScandirIterator[str]]] = []
        try:
            stack.append(("", root_descriptor, os.scandir(root_descriptor)))
            root_descriptor = -1
            while stack:
                parent, parent_descriptor, entries = stack[-1]
                try:
                    entry = next(entries)
                except StopIteration:
                    entries.close()
                    os.close(parent_descriptor)
                    stack.pop()
                    continue
                entry_count += 1
                if entry_count > self._max_tree_entries:
                    raise ValueError
                status = entry.stat(follow_symlinks=False)
                relative = f"{parent}/{entry.name}" if parent else entry.name
                if stat.S_ISLNK(status.st_mode) or not (
                    stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)
                ):
                    raise ValueError
                yield relative, status
                if stat.S_ISDIR(status.st_mode):
                    child_descriptor = self._open_directory_descriptor(
                        entry.name, dir_fd=parent_descriptor
                    )
                    opened = os.fstat(child_descriptor)
                    if (opened.st_dev, opened.st_ino) != (
                        status.st_dev,
                        status.st_ino,
                    ):
                        os.close(child_descriptor)
                        raise ValueError
                    stack.append(
                        (relative, child_descriptor, os.scandir(child_descriptor))
                    )
        finally:
            if root_descriptor >= 0:
                os.close(root_descriptor)
            while stack:
                _, descriptor, entries = stack.pop()
                entries.close()
                os.close(descriptor)

    def _walk_tree(self, root: Path) -> Iterator[tuple[Path, os.stat_result]]:
        for relative, status in self._walk_tree_relative(root):
            yield root.joinpath(*relative.split("/")), status

    def _validate_entry(
        self, entry: Path, expected_key: str
    ) -> tuple[CvEvidenceArtifact, str]:
        prefix_descriptor: int | None = None
        entry_descriptor: int | None = None
        prefix = self._prefix_directory(expected_key)
        anchored = entry.parent == prefix and (
            entry.name == expected_key or entry.name.startswith(".publish-")
        )
        try:
            if anchored:
                prefix_descriptor = self._open_prefix(expected_key, create=False)
                if prefix_descriptor is None:
                    raise ValueError
                entry_descriptor = self._open_directory_descriptor(
                    entry.name, dir_fd=prefix_descriptor
                )
            else:
                self._validate_private_directory(
                    entry.parent, require_owner_only=True
                )
                entry_descriptor = self._open_directory_descriptor(entry)
            entry_status = os.fstat(entry_descriptor)
            self._validate_private_directory_status(
                entry_status, require_owner_only=True
            )
            entry_identity = entry_status.st_dev, entry_status.st_ino
            with self._open_relative_regular_file(
                entry_descriptor, _MANIFEST_NAME, nonblocking=True
            ) as (manifest_descriptor, manifest_parent, manifest_name, manifest_status):
                if (
                    manifest_status.st_nlink != 1
                    or manifest_status.st_size > self._max_manifest_bytes
                ):
                    raise ValueError
                manifest, manifest_digest = self._read_bounded_descriptor(
                    manifest_descriptor, self._max_manifest_bytes
                )
                payload = json.loads(manifest)
                if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
                    raise ValueError
                cache_identity, identity_entities = self._validated_cache_identity(
                    payload["cache_identity"]
                )
                if (
                    hashlib.sha256(_canonical_json(cache_identity)).hexdigest()
                    != expected_key
                ):
                    raise ValueError
                artifact = CvEvidenceArtifact.model_validate(payload["artifact"])
                self._validate_artifact_matches_identity(
                    cache_identity, identity_entities, artifact
                )
                canonical = _canonical_json(
                    {
                        "cache_identity": cache_identity,
                        "artifact": artifact.model_dump(mode="json"),
                    }
                )
                if manifest != canonical:
                    raise ValueError
                self._validate_artifact_files(
                    entry_descriptor,
                    artifact,
                    include_manifest=True,
                    pinned_files=(
                        (
                            _MANIFEST_NAME,
                            manifest_descriptor,
                            manifest_parent,
                            manifest_name,
                            manifest_status,
                        ),
                    ),
                )
            current = os.fstat(entry_descriptor)
            if (current.st_dev, current.st_ino) != entry_identity:
                raise ValueError
            if anchored:
                if prefix_descriptor is None:
                    raise ValueError
                current = os.stat(
                    entry.name,
                    dir_fd=prefix_descriptor,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) != entry_identity:
                    raise ValueError
            elif self._path_identity(entry) != entry_identity:
                raise ValueError
            return artifact, manifest_digest
        finally:
            if entry_descriptor is not None:
                os.close(entry_descriptor)
            if prefix_descriptor is not None:
                os.close(prefix_descriptor)

    @staticmethod
    def _validated_cache_identity(
        value: object,
    ) -> tuple[dict[str, object], tuple[EntityPrompt, ...]]:
        if not isinstance(value, dict) or set(value) != _CACHE_IDENTITY_FIELDS:
            raise ValueError
        schema_version = value["schema_version"]
        provider = value["provider"]
        model_identity = value["model_identity"]
        if schema_version != "cv_request_v1" or provider not in {"fake", "sam31"}:
            raise ValueError
        if not isinstance(model_identity, str) or not model_identity.strip():
            raise ValueError
        for field in ("video_sha256", "checkpoint_sha256"):
            digest = value[field]
            if (
                not isinstance(digest, str)
                or len(digest) != _KEY_LENGTH
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError
        raw_entities = value["entities"]
        if not isinstance(raw_entities, list):
            raise ValueError
        entities = tuple(EntityPrompt.model_validate(item) for item in raw_entities)
        if len({entity.entity_id for entity in entities}) != len(entities):
            raise ValueError
        sampling = SamplingPolicy.model_validate(value["sampling"])
        thresholds = EvidenceThresholds.model_validate(value["thresholds"])
        normalized: dict[str, object] = {
            "schema_version": schema_version,
            "video_sha256": value["video_sha256"],
            "provider": provider,
            "model_identity": model_identity,
            "checkpoint_sha256": value["checkpoint_sha256"],
            "entities": [entity.model_dump(mode="json") for entity in entities],
            "sampling": sampling.model_dump(mode="json"),
            "thresholds": thresholds.model_dump(mode="json"),
        }
        if value != normalized:
            raise ValueError
        return normalized, entities

    @staticmethod
    def _validate_artifact_matches_identity(
        identity: dict[str, object],
        entities: tuple[EntityPrompt, ...],
        artifact: CvEvidenceArtifact,
    ) -> None:
        if (
            artifact.provider != identity["provider"]
            or artifact.model_identity != identity["model_identity"]
            or artifact.video_sha256 != identity["video_sha256"]
            or artifact.checkpoint_sha256 != identity["checkpoint_sha256"]
            or artifact.entities != entities
        ):
            raise ValueError

    @staticmethod
    def _read_bounded_descriptor(descriptor: int, limit: int) -> tuple[bytes, str]:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or opened.st_size > limit:
            raise ValueError
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, min(_STREAM_BYTES, limit + 1 - size)):
            size += len(chunk)
            if size > limit:
                raise ValueError
            chunks.append(chunk)
            digest.update(chunk)
        if size != opened.st_size:
            raise ValueError
        return b"".join(chunks), digest.hexdigest()

    @staticmethod
    def _fsync_directory(
        path: Path, expected_identity: tuple[int, int] | None = None
    ) -> None:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory_flag is None:
            raise ValueError
        descriptor = os.open(path, os.O_RDONLY | nofollow | directory_flag)
        try:
            opened = os.fstat(descriptor)
            current = path.lstat()
            identity = opened.st_dev, opened.st_ino
            if (
                not stat.S_ISDIR(opened.st_mode)
                or identity != (current.st_dev, current.st_ino)
                or (expected_identity is not None and identity != expected_identity)
            ):
                raise ValueError
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _fsync_tree(self, root: Path) -> None:
        root_status = root.lstat()
        directories = [(root, (root_status.st_dev, root_status.st_ino))]
        for path, status in self._walk_tree(root):
            if stat.S_ISDIR(status.st_mode):
                directories.append((path, (status.st_dev, status.st_ino)))
            else:
                if status.st_nlink != 1:
                    raise ValueError
                with path.open("rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                        or (opened.st_dev, opened.st_ino)
                        != (status.st_dev, status.st_ino)
                    ):
                        raise ValueError
                    os.fsync(stream.fileno())
                current = path.lstat()
                if (current.st_dev, current.st_ino) != (
                    status.st_dev,
                    status.st_ino,
                ):
                    raise ValueError
        for directory, identity in reversed(directories):
            self._fsync_directory(directory, identity)
