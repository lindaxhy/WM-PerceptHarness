"""Atomic, process-safe caching of CV evidence in a private local directory.

The cache is owned by the running user. Workers coordinate through per-key
locks; external processes must not modify published entries while they are read.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Iterator

from .contracts import CvEvidenceArtifact, CvEvidenceRequest

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MANIFEST_NAME = "manifest.json"


class CvArtifactError(RuntimeError):
    """An invalid or unavailable CV artifact."""


@dataclass(frozen=True, slots=True)
class CvArtifactHandle:
    key: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True,
                      separators=(",", ":"), sort_keys=True).encode("ascii")


def _cache_identity(request: CvEvidenceRequest) -> dict:
    return request.model_dump(mode="json", exclude={"video_path"})


def cv_cache_key(request: CvEvidenceRequest) -> str:
    return hashlib.sha256(_canonical_json(_cache_identity(request))).hexdigest()


class CvArtifactStore:
    """Validate files once per read and publish complete directories by rename."""

    def __init__(self, root: Path, *, max_files: int = 10_000,
                 max_bytes: int = 8 * 1024**3,
                 max_manifest_bytes: int = 8 * 1024**2) -> None:
        if any(type(value) is not int or value <= 0
               for value in (max_files, max_bytes, max_manifest_bytes)):
            raise ValueError("artifact limits must be positive integers")
        root = Path(root).absolute()
        if root.is_symlink():
            raise CvArtifactError("CV cache root must not be a symlink")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._root = root.resolve()
        self._directory(self._root, private=True)
        self._max_files = max_files
        self._max_bytes = max_bytes
        self._max_manifest_bytes = max_manifest_bytes
        self._staging: dict[Path, tuple[str, int, int]] = {}
        self._closed = False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self) -> None:
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise CvArtifactError("CV artifact store is closed")
        self._directory(self._root, private=True)

    @staticmethod
    def _directory(path: Path, *, private: bool = False) -> None:
        status = path.lstat()
        if (not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid()
                or (private and status.st_mode & 0o077)):
            raise CvArtifactError("CV cache directories must be private and owned by the user")

    def _entry_path(self, key: str) -> Path:
        return self._root / key[:2] / key

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        self._check_open()
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise CvArtifactError("invalid CV cache key")
        prefix = self._root / key[:2]
        prefix.mkdir(mode=0o700, exist_ok=True)
        self._directory(prefix, private=True)
        descriptor = os.open(prefix / f".{key}.lock",
                             os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            status = os.fstat(descriptor)
            if (not stat.S_ISREG(status.st_mode) or status.st_nlink != 1
                    or status.st_uid != os.getuid() or status.st_mode & 0o077):
                raise CvArtifactError("invalid CV cache lock")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @contextmanager
    def staging(self, key: str) -> Iterator[Path]:
        with self._key_lock(key):
            stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=self._root / key[:2]))
            status = stage.stat()
            self._staging[stage] = (key, status.st_dev, status.st_ino)
        try:
            yield stage
        finally:
            expected = self._staging.pop(stage, None)
            if expected and stage.exists():
                status = stage.lstat()
                if (status.st_dev, status.st_ino) == expected[1:] and stat.S_ISDIR(status.st_mode):
                    shutil.rmtree(stage)

    def lookup(self, key: str) -> CvArtifactHandle | None:
        with self._key_lock(key):
            entry = self._entry_path(key)
            if not entry.exists() and not entry.is_symlink():
                return None
            try:
                self._validate_entry(entry, key)
            except (OSError, ValueError, TypeError, RecursionError, CvArtifactError):
                self._discard(entry)
                return None
            return CvArtifactHandle(key)

    def publish(self, request: CvEvidenceRequest, staging: Path,
                artifact: CvEvidenceArtifact) -> CvArtifactHandle:
        try:
            request = CvEvidenceRequest.model_validate(request.model_dump(mode="python"))
            artifact = CvEvidenceArtifact.model_validate(artifact.model_dump(mode="python"))
            key = cv_cache_key(request)
            stage = Path(staging).absolute()
            expected = self._staging.get(stage)
            status = stage.lstat()
            if expected != (key, status.st_dev, status.st_ino):
                raise ValueError("unregistered staging directory")
            self._validate_artifact_matches_request(request, artifact)
            self._validate_files(stage, artifact, include_manifest=False)
            manifest = _canonical_json({"cache_identity": _cache_identity(request),
                                        "artifact": artifact.model_dump(mode="json")})
            if len(manifest) > self._max_manifest_bytes:
                raise ValueError("manifest exceeds byte budget")
            with (stage / _MANIFEST_NAME).open("xb") as stream:
                stream.write(manifest)
                stream.flush()
                os.fsync(stream.fileno())
            # Flush files and directories before publishing the complete tree.
            for file in artifact.files:
                with self._open_file(stage, file.path) as stream:
                    os.fsync(stream.fileno())
            directories = {stage, *(parent for file in artifact.files
                                    for parent in (stage / file.path).parents if parent.is_relative_to(stage))}
            for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
                self._sync_directory(directory)
            with self._key_lock(key):
                destination = self._entry_path(key)
                if destination.exists() or destination.is_symlink():
                    try:
                        existing = self._validate_entry(destination, key)
                    except (OSError, ValueError, TypeError, RecursionError, CvArtifactError):
                        self._discard(destination)
                    else:
                        self._validate_artifact_matches_request(request, existing)
                        return CvArtifactHandle(key)
                os.replace(stage, destination)
                self._sync_directory(destination.parent)
            return CvArtifactHandle(key)
        except (OSError, ValueError, TypeError, AttributeError, CvArtifactError):
            raise CvArtifactError("unable to publish CV artifact") from None

    def load(self, handle: CvArtifactHandle) -> CvEvidenceArtifact:
        try:
            with self._key_lock(handle.key):
                return self._validate_entry(self._entry_path(handle.key), handle.key)
        except (OSError, ValueError, TypeError, AttributeError, RecursionError, CvArtifactError):
            raise CvArtifactError("unable to load CV artifact") from None

    def read_overlays(self, handle: CvArtifactHandle, paths: tuple[str, ...], *,
                      max_total_bytes: int = 64 * 1024**2) -> dict[str, bytes]:
        try:
            if (type(paths) is not tuple or len(paths) > 24
                    or any(type(path) is not str for path in paths)
                    or len(set(paths)) != len(paths) or type(max_total_bytes) is not int
                    or not 0 < max_total_bytes <= 64 * 1024**2):
                raise ValueError("invalid overlay request")
            with self._key_lock(handle.key):
                entry = self._entry_path(handle.key)
                artifact = self._validate_entry(entry, handle.key)
                if not set(paths) <= {record.path for record in artifact.overlay_records}:
                    raise ValueError("unregistered overlay")
                files = {file.path: file for file in artifact.files}
                if sum(files[path].size_bytes for path in paths) > max_total_bytes:
                    raise ValueError("overlay byte budget exceeded")
                output = {}
                for path in paths:
                    with self._open_file(entry, path) as stream:
                        content = stream.read(files[path].size_bytes + 1)
                    if len(content) != files[path].size_bytes or not content.startswith(_PNG_SIGNATURE):
                        raise ValueError("invalid overlay")
                    output[path] = content
                return output
        except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError, CvArtifactError):
            raise CvArtifactError("unable to read CV overlays") from None

    def _validate_entry(self, entry: Path, key: str) -> CvEvidenceArtifact:
        self._directory(entry, private=True)
        with self._open_file(entry, _MANIFEST_NAME) as stream:
            data = stream.read(self._max_manifest_bytes + 1)
        if len(data) > self._max_manifest_bytes:
            raise ValueError("manifest exceeds byte budget")
        payload = json.loads(data)
        if type(payload) is not dict or set(payload) != {"cache_identity", "artifact"}:
            raise ValueError("invalid manifest")
        request = CvEvidenceRequest.model_validate({**payload["cache_identity"], "video_path": Path("/cache-source")})
        identity = _cache_identity(request)
        artifact = CvEvidenceArtifact.model_validate(payload["artifact"])
        if (cv_cache_key(request) != key or data != _canonical_json(
                {"cache_identity": identity, "artifact": artifact.model_dump(mode="json")})):
            raise ValueError("cache identity mismatch")
        self._validate_artifact_matches_request(request, artifact)
        self._validate_files(entry, artifact, include_manifest=True)
        return artifact

    @staticmethod
    def _validate_artifact_matches_request(request: CvEvidenceRequest, artifact: CvEvidenceArtifact) -> None:
        for field in ("provider", "model_identity", "video_identity", "checkpoint_identity", "entities"):
            if getattr(artifact, field) != getattr(request, field):
                raise ValueError("artifact does not match request")
        if artifact.processed_timeline is None:
            return
        selected = {frame.frame_index: frame.timestamp_seconds for frame in request.timeline.frames}
        references = {prompt.frame_index for prompt in request.spatial_prompts}
        processed = {frame.frame_index for frame in artifact.processed_timeline.frames}
        for frame in artifact.processed_timeline.frames:
            if selected.get(frame.frame_index) != frame.timestamp_seconds:
                raise ValueError("processed frame must belong to selected request timeline")
            if request.mode == "probe" and frame.frame_index not in references:
                raise ValueError("probe may only process reference frames")
        if not references <= processed:
            raise ValueError("spatial reference frames must be processed")

    def _validate_files(self, directory: Path, artifact: CvEvidenceArtifact, *, include_manifest: bool) -> None:
        self._directory(directory, private=True)
        files = {file.path: file for file in artifact.files}
        if (len(files) != len(artifact.files) or len(files) > self._max_files
                or _MANIFEST_NAME in files or sum(file.size_bytes for file in files.values()) > self._max_bytes):
            raise ValueError("artifact file budget exceeded")
        expected = set(files) | ({_MANIFEST_NAME} if include_manifest else set())
        directories = {parent.as_posix() for name in files for parent in Path(name).parents if parent != Path('.')}
        seen, seen_directories = set(), set()
        # os.walk does not follow symlinks; explicitly reject them and special files.
        count = 0
        for parent, dirs, names in os.walk(directory, followlinks=False):
            count += len(dirs) + len(names)
            if count > max(64, self._max_files * 4):
                raise ValueError("artifact tree budget exceeded")
            for name in dirs:
                path = Path(parent) / name
                self._directory(path)
                seen_directories.add(path.relative_to(directory).as_posix())
            for name in names:
                seen.add((Path(parent) / name).relative_to(directory).as_posix())
        if seen != expected or seen_directories != directories:
            raise ValueError("artifact files do not match manifest")
        for name, file in files.items():
            with self._open_file(directory, name) as stream:
                if os.fstat(stream.fileno()).st_size != file.size_bytes:
                    raise ValueError("artifact size mismatch")
                if name.startswith("overlays/") and stream.read(8) != _PNG_SIGNATURE:
                    raise ValueError("invalid overlay signature")

    @staticmethod
    @contextmanager
    def _open_file(root: Path, relative: str):
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
            raise ValueError("invalid artifact path")
        for parent in path.parents:
            CvArtifactStore._directory(root / parent)
        descriptor = os.open(root / path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or status.st_uid != os.getuid():
                raise ValueError("artifact must be an owned regular file")
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            yield stream

    @staticmethod
    def _discard(entry: Path) -> None:
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink()
        else:
            shutil.rmtree(entry)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
