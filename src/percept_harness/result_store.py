"""Content/configuration identity and durable storage for annotation results."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_digest() -> str:
    root = Path(__file__).parent
    return json_digest({p.relative_to(root).as_posix(): file_digest(p)
                        for p in sorted(root.rglob("*"))
                        if p.is_file() and p.suffix in {".py", ".txt", ".json"}})


def _runtime() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for name in ("wm-percept-harness", "httpx", "pydantic", "pydantic-settings",
                 "torch", "torchvision", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        try:
            result = subprocess.run([executable or name, "-version"], capture_output=True,
                                    timeout=10, check=True)
            tools[name] = {"executable": executable,
                           "version_sha256": hashlib.sha256(result.stdout).hexdigest()}
        except (OSError, subprocess.SubprocessError):
            tools[name] = None
    return {"packages": versions, "media_tools": tools}


def run_identity(runner, template: str, prompt_context: str | None,
                 query: str | None) -> dict[str, Any]:
    """Snapshot the built-in execution path; unverifiable integrations cannot resume."""
    from .models.fake import FakeVideoModel
    from .models.openai_compat import OpenAICompatVideoModel

    settings = runner.settings.model_dump(mode="json", exclude={"openai_api_key"})
    # Hash provider-supplied dictionaries/URLs rather than serialize credentials.
    for name in ("openai_base_url", "openai_extra_body", "openai_extra_headers"):
        settings[name] = {"sha256": json_digest(settings[name])}
    proxy = runner.settings.openai_proxy
    settings["openai_proxy"] = {"sha256": json_digest(proxy.get_secret_value() if proxy else None)}
    settings.pop("work_root")
    settings.pop("cv_cache_root")
    model = runner.model
    reusable = not runner.custom_registry
    if type(model) is FakeVideoModel:
        backend = {"kind": "fake"}
    elif type(model) is OpenAICompatVideoModel:
        backend = model.run_identity(runner.model_alias)
        reusable = reusable and model.supports_batch_resume
    else:
        backend = {"kind": f"{type(model).__module__}.{type(model).__qualname__}"}
        reusable = False
    # A custom executor or external CV runtime cannot be verified from Settings alone.
    # Until its loaded weights/source/runtime has an explicit identity, rerun safely.
    if runner.cv_executor is not None or runner.settings.cv_provider != "disabled":
        reusable = False
    identity = {
        "schema_version": SCHEMA_VERSION,
        "template": template,
        "model_alias": runner.model_alias,
        "backend": backend,
        "settings": settings,
        "prompt_context": prompt_context,
        "query": query,
        "code_sha256": package_digest(),
        "runtime": _runtime(),
        "reusable": reusable,
    }
    return identity


def provenance(video: Path, run: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION,
            "input": {"path": str(video.resolve()), "sha256": file_digest(video)},
            "run": run, "run_fingerprint": json_digest(run)}


def read_record(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def result_path(output: Path, video: Path) -> Path:
    """Keep the familiar name if free/owned, otherwise isolate by full source path."""
    resolved = str(video.resolve())
    suffix = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    alternate = output / f"{video.stem[:32]}--{suffix}.json"
    primary = output / f"{video.stem}.json"
    # An existing disambiguated result keeps its path even if a later batch is smaller.
    for candidate in (alternate, primary):
        record = read_record(candidate)
        if record and record.get("video_path") == resolved:
            return candidate
    if not primary.exists():
        return primary
    if not alternate.exists():
        return alternate
    raise ValueError("Result paths are occupied by other or unreadable records; use a fresh output directory")


def completed_result(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    record = read_record(path)
    if (not expected["run"]["reusable"] or not record
            or record.get("status") != "completed"
            or record.get("template") != expected["run"]["template"]
            or record.get("video_path") != expected["input"]["path"]
            or record.get("provenance") != expected
            or not isinstance(record.get("data"), dict)):
        return None
    try:
        if record.get("data_sha256") != json_digest(record["data"]):
            return None
    except (ValueError, TypeError):
        return None
    return record


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".percept-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def archive_result(path: Path, output: Path) -> None:
    if not path.exists():
        return
    payload = path.read_bytes()
    destination = output / ".percept" / "history" / (hashlib.sha256(payload).hexdigest() + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("Archived result is inconsistent; refusing to replace current output")
    else:
        atomic_bytes(destination, payload)



class OutputBusyError(RuntimeError):
    """Another batch owns this output directory, or a crashed run left a lock."""


@contextmanager
def output_lock(output: Path):
    directory = output / ".percept"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "writer.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise OutputBusyError(
            f"Output directory is already in use: {output}. "
            f"Inspect {path}; remove a stale lock only after confirming its process stopped."
        ) from None
    try:
        with stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        path.unlink()
