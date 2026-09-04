"""Content-addressed storage for large computer-vision evidence artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
import threading
import time
from typing import Iterator

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
            self._root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        self._validate_private_directory(self._root, require_owner_only=False)
        root_status = self._root.lstat()
        self._root_identity = root_status.st_dev, root_status.st_ino

    def lookup(self, key: str) -> CvArtifactHandle | None:
        """Return a digest-validated immutable handle, or None on a miss."""
        self._validate_key(key)
        self._validate_store_root()
        prefix = self._prefix_directory(key)
        if not self._path_exists(prefix):
            return None
        self._validate_private_directory(prefix, require_owner_only=True)
        with self._key_lock(key):
            entry = self._entry_path(key)
            if not self._path_exists(entry):
                return None
            identity = self._path_identity(entry)
            try:
                artifact, manifest_digest = self._validate_entry(entry, key)
            except (CvArtifactError, OSError, TypeError, ValueError, ValidationError):
                self._quarantine_entry(key, entry, identity)
                return None
            del artifact
            return CvArtifactHandle(key=key, manifest_sha256=manifest_digest)

    @contextmanager
    def staging(self, key: str) -> Iterator[Path]:
        """Yield an owner-only sibling directory and remove it on exit."""
        self._validate_key(key)
        self._validate_store_root()
        prefix = self._prefix_directory(key)
        try:
            prefix.mkdir(mode=0o700)
        except FileExistsError:
            pass
        self._validate_private_directory(prefix, require_owner_only=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=prefix))
        os.chmod(staging, 0o700)
        identity = staging.stat(follow_symlinks=False)
        with self._staging_lock:
            self._staging[staging] = (key, identity.st_dev, identity.st_ino)
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
    ) -> CvArtifactHandle:
        """Validate, fsync, and atomically publish one complete entry."""
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
                    if self._path_exists(destination):
                        destination_identity = self._path_identity(destination)
                        try:
                            existing_artifact, existing_digest = self._validate_entry(
                                destination, key
                            )
                        except (
                            CvArtifactError,
                            OSError,
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
                            return CvArtifactHandle(
                                key=key, manifest_sha256=existing_digest
                            )
                    publication_identity = self._path_identity(publication)
                    staged_artifact, staged_digest = self._validate_entry(
                        publication, key
                    )
                    if staged_artifact != artifact or staged_digest != manifest_digest:
                        raise ValueError
                    if self._path_identity(publication) != publication_identity:
                        raise ValueError
                    os.replace(publication, destination)
                    self._fsync_directory(destination.parent)
                    destination_identity = self._path_identity(destination)
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
                        TypeError,
                        ValueError,
                        ValidationError,
                    ):
                        self._quarantine_entry(
                            key, destination, destination_identity
                        )
                        raise ValueError
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
        try:
            if not isinstance(handle, CvArtifactHandle):
                raise ValueError
            self._validate_key(handle.key)
            self._validate_store_root()
            prefix = self._prefix_directory(handle.key)
            self._validate_private_directory(prefix, require_owner_only=True)
            with self._key_lock(handle.key):
                entry = self._entry_path(handle.key)
                identity = self._path_identity(entry)
                try:
                    artifact, manifest_digest = self._validate_entry(
                        entry, handle.key
                    )
                except (
                    CvArtifactError,
                    OSError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ):
                    self._quarantine_entry(handle.key, entry, identity)
                    raise ValueError
                if manifest_digest != handle.manifest_sha256:
                    raise ValueError
                return artifact
        except (CvArtifactError, OSError, TypeError, ValueError, ValidationError):
            raise CvArtifactError("unable to load CV artifact") from None

    def _entry_path(self, key: str) -> Path:
        return self._root / key[:2] / key

    def _prefix_directory(self, key: str) -> Path:
        return self._root / key[:2]

    def _validate_store_root(self) -> None:
        self._validate_private_directory(self._root, require_owner_only=False)
        if self._path_identity(self._root) != self._root_identity:
            raise CvArtifactError("unsafe CV artifact directory")

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
            owner_matches = not hasattr(os, "geteuid") or status.st_uid == os.geteuid()
            forbidden = 0o077 if require_owner_only else 0o022
            if (
                stat.S_ISLNK(status.st_mode)
                or not stat.S_ISDIR(status.st_mode)
                or not owner_matches
                or status.st_mode & forbidden
            ):
                raise ValueError
        except (OSError, ValueError):
            raise CvArtifactError("unsafe CV artifact directory") from None

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

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        lock_identity = f"{self._root}:{key}"
        with self._process_locks_guard:
            process_lock = self._process_locks.setdefault(
                lock_identity, threading.RLock()
            )
        with process_lock:
            lock_path = self._prefix_directory(key) / f".{key}.lock"
            flags = os.O_RDWR | os.O_CREAT
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if nofollow is None:
                raise CvArtifactError("unable to lock CV artifact")
            flags |= nofollow
            descriptor: int | None = None
            try:
                descriptor = os.open(lock_path, flags, 0o600)
                status = os.fstat(descriptor)
                owner_matches = not hasattr(os, "geteuid") or status.st_uid == os.geteuid()
                if (
                    not stat.S_ISREG(status.st_mode)
                    or not owner_matches
                    or status.st_mode & 0o077
                    or self._path_identity(lock_path)
                    != (status.st_dev, status.st_ino)
                ):
                    raise CvArtifactError("unsafe CV artifact lock")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                if self._path_identity(lock_path) != (status.st_dev, status.st_ino):
                    raise CvArtifactError("unsafe CV artifact lock")
            except (CvArtifactError, OSError, ValueError):
                if descriptor is not None:
                    os.close(descriptor)
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

    def _quarantine_entry(
        self, key: str, entry: Path, identity: tuple[int, int]
    ) -> None:
        try:
            if self._path_identity(entry) != identity:
                raise ValueError
            quarantine = self._root / "quarantine"
            try:
                quarantine.mkdir(mode=0o700)
            except FileExistsError:
                pass
            self._validate_private_directory(quarantine, require_owner_only=True)
            destination = quarantine / f"{key}-{time.time_ns()}-{secrets.token_hex(8)}"
            os.replace(entry, destination)
            self._fsync_directory(quarantine)
            self._fsync_directory(entry.parent)
        except (CvArtifactError, OSError, ValueError):
            raise CvArtifactError("unable to quarantine CV artifact") from None

    @contextmanager
    def _publication_staging(self, key: str) -> Iterator[Path]:
        prefix = self._prefix_directory(key)
        publication = Path(tempfile.mkdtemp(prefix=".publish-", dir=prefix))
        os.chmod(publication, 0o700)
        identity = self._path_identity(publication)
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
            with self._open_relative_regular_file(source, file.path) as (
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
        self, root: Path, relative: str
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
            file_descriptor = os.open(
                parts[-1], os.O_RDONLY | nofollow, dir_fd=directory_descriptor
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
        path: Path | str, *, dir_fd: int | None = None
    ) -> int:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if nofollow is None or directory_flag is None:
            raise ValueError
        descriptor = os.open(
            path,
            os.O_RDONLY | nofollow | directory_flag,
            dir_fd=dir_fd,
        )
        status = os.fstat(descriptor)
        if not stat.S_ISDIR(status.st_mode):
            os.close(descriptor)
            raise ValueError
        return descriptor

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
        directory: Path,
        artifact: CvEvidenceArtifact,
        *,
        include_manifest: bool,
    ) -> None:
        if len(artifact.files) > self._max_files:
            raise ValueError
        if sum(file.size_bytes for file in artifact.files) > self._max_bytes:
            raise ValueError
        expected = {file.path for file in artifact.files}
        if _MANIFEST_NAME in expected or len(expected) != len(artifact.files):
            raise ValueError
        actual: set[str] = set()
        actual_directories: set[str] = set()
        for path, status in self._walk_tree(directory):
            relative = path.relative_to(directory).as_posix()
            if stat.S_ISREG(status.st_mode):
                if status.st_nlink != 1:
                    raise ValueError
                actual.add(relative)
                if len(actual) > self._max_files + int(include_manifest):
                    raise ValueError
            elif not stat.S_ISDIR(status.st_mode):
                raise ValueError
            else:
                actual_directories.add(relative)
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
        for file in artifact.files:
            path = directory.joinpath(*file.path.split("/"))
            status_before = path.lstat()
            if (
                not stat.S_ISREG(status_before.st_mode)
                or status_before.st_size != file.size_bytes
            ):
                raise ValueError
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as stream:
                while chunk := stream.read(_STREAM_BYTES):
                    size += len(chunk)
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise ValueError
                    digest.update(chunk)
                opened = os.fstat(stream.fileno())
            status_after = path.lstat()
            identities = {
                (status_before.st_dev, status_before.st_ino),
                (opened.st_dev, opened.st_ino),
                (status_after.st_dev, status_after.st_ino),
            }
            if (
                len(identities) != 1
                or status_before.st_nlink != 1
                or opened.st_nlink != 1
                or status_after.st_nlink != 1
                or size != file.size_bytes
                or digest.hexdigest() != file.sha256
            ):
                raise ValueError

    def _walk_tree(self, root: Path) -> Iterator[tuple[Path, os.stat_result]]:
        entry_count = 0
        for path in root.rglob("*"):
            entry_count += 1
            if entry_count > self._max_tree_entries:
                raise ValueError
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode) or not (
                stat.S_ISREG(status.st_mode) or stat.S_ISDIR(status.st_mode)
            ):
                raise ValueError
            yield path, status

    def _validate_entry(
        self, entry: Path, expected_key: str
    ) -> tuple[CvEvidenceArtifact, str]:
        entry_identity = self._path_identity(entry)
        self._validate_private_directory(entry.parent, require_owner_only=True)
        self._validate_private_directory(entry, require_owner_only=True)
        manifest_path = entry / _MANIFEST_NAME
        manifest, manifest_digest = self._read_bounded_regular_file(
            manifest_path, self._max_manifest_bytes
        )
        payload = json.loads(manifest)
        if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
            raise ValueError
        cache_identity, identity_entities = self._validated_cache_identity(
            payload["cache_identity"]
        )
        if hashlib.sha256(_canonical_json(cache_identity)).hexdigest() != expected_key:
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
        self._validate_artifact_files(entry, artifact, include_manifest=True)
        if self._path_identity(entry) != entry_identity:
            raise ValueError
        return artifact, manifest_digest

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
    def _read_bounded_regular_file(path: Path, limit: int) -> tuple[bytes, str]:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise ValueError
        descriptor = os.open(path, os.O_RDONLY | nofollow)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size > limit
            ):
                raise ValueError
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            size = 0
            while chunk := os.read(descriptor, min(_STREAM_BYTES, limit + 1 - size)):
                size += len(chunk)
                if size > limit:
                    raise ValueError
                chunks.append(chunk)
                digest.update(chunk)
            current = path.lstat()
            if (
                (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
                or current.st_nlink != 1
                or size != opened.st_size
            ):
                raise ValueError
            return b"".join(chunks), digest.hexdigest()
        finally:
            os.close(descriptor)

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
