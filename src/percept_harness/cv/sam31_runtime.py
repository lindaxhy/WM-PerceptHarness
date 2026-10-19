"""Local SAM3.1 checkout pinning, asset validation and CUDA runtime probes."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
import hashlib
import importlib
import inspect
from numbers import Integral
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


PINNED_REPOSITORY_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
BPE_REPOSITORY_PATH = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
_HASH_READ_BYTES = 1024 * 1024
_NETWORK_PREFIX = re.compile(
    r"(?i)^(?:https?|ssh|git|ftp|s3|gs|hf)://|^[^/\\\s]+@[^/\\\s]+:"
)


def require_predictor_interface(predictor: Any) -> None:
    if not callable(getattr(predictor, "handle_request", None)):
        raise TypeError
    if not callable(getattr(predictor, "handle_stream_request", None)):
        raise TypeError


def install_pinned_multiplex_init_compatibility(predictor: Any) -> None:
    """Bridge the one known BasePredictor/multiplex init signature mismatch."""
    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    if not callable(init_state):
        return
    parameters = inspect.signature(init_state).parameters
    if "offload_state_to_cpu" in parameters:
        return

    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        if "offload_state_to_cpu" in kwargs:
            if kwargs["offload_state_to_cpu"] is not False:
                raise ValueError
            kwargs.pop("offload_state_to_cpu")
        return init_state(*args, **kwargs)

    setattr(model, "init_state", compatible_init_state)


def shutdown_predictor(predictor: Any) -> None:
    shutdown = getattr(predictor, "shutdown", None)
    if callable(shutdown):
        shutdown()


def validated_repository(value: Path | str) -> Path:
    raw = local_path(value)
    resolved = raw.resolve(strict=True)
    if (
        not stat.S_ISDIR(raw.lstat().st_mode) or resolved != raw.absolute()
        or not stat.S_ISREG((resolved / "sam3" / "__init__.py").lstat().st_mode)
    ):
        raise ValueError
    return resolved


def validated_local_asset(value: Path | str) -> Path:
    path = local_path(value)
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
        raise ValueError
    return path.resolve(strict=True)


def git_blob_digest(path: Path) -> str:
    """Check the tokenizer against the pinned Git blob without loading it all."""
    with path.open("rb") as source:
        digest = hashlib.sha1(f"blob {os.fstat(source.fileno()).st_size}\0".encode("ascii"))
        while payload := source.read(_HASH_READ_BYTES):
            digest.update(payload)
    return digest.hexdigest()


def local_path(value: Path | str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError
    if raw.startswith("//") or _NETWORK_PREFIX.search(raw):
        raise ValueError
    return Path(raw)


def verify_repository_revision(repository: Path) -> None:
    if _git(repository, "rev-parse", "--verify", "HEAD^{commit}") != PINNED_REPOSITORY_REVISION:
        raise ValueError
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all", "--", "sam3"):
        raise ValueError


def pinned_blob_object_id(repository: Path, path: str) -> str:
    return _git(repository, "rev-parse", "--verify", f"{PINNED_REPOSITORY_REVISION}:{path}")


def _git(repository: Path, *arguments: str) -> str:
    # The checkout is trusted and local; Git must not redirect to another repo.
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    completed = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repository), *arguments],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env=environment,
    )
    return completed.stdout.strip()


def sam_modules() -> dict[str, Any]:
    return {
        name: module for name, module in sys.modules.items()
        if name == "sam3" or name.startswith("sam3.")
    }


def import_local_builder(repository: Path) -> Callable[..., Any]:
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(repository))
        importlib.invalidate_caches()
        builder_module = importlib.import_module("sam3.model_builder")
        verify_loaded_sam_modules(repository)
        builder = getattr(builder_module, "build_sam3_multiplex_video_predictor", None)
        if not callable(builder):
            raise ValueError
        return builder
    finally:
        sys.path[:] = original_path


def verify_loaded_sam_modules(repository: Path) -> None:
    modules = sam_modules()
    if not modules or any(
        not module_is_beneath(module, repository) for module in modules.values()
    ):
        raise ValueError


def module_is_beneath(module: Any, repository: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        return False
    try:
        return Path(origin).resolve(strict=True).is_relative_to(repository)
    except (OSError, RuntimeError):
        return False


def restore_sam_modules(snapshot: dict[str, Any]) -> None:
    for name in sam_modules():
        sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _cuda(torch_module: Any, name: str) -> Any:
    return getattr(getattr(torch_module, "cuda", None), name, None)


def reset_peak_memory(torch_module: Any) -> None:
    reset = _cuda(torch_module, "reset_peak_memory_stats")
    if callable(reset):
        reset()


def peak_memory(torch_module: Any) -> int:
    maximum = _cuda(torch_module, "max_memory_allocated")
    if not callable(maximum):
        return 0
    value = maximum()
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError
    return int(value)


def empty_cuda_cache(torch_module: Any) -> None:
    empty = _cuda(torch_module, "empty_cache")
    if callable(empty):
        with suppress(BaseException):
            empty()


def is_cuda_oom(torch_module: Any, error: Exception) -> bool:
    oom_type = _cuda(torch_module, "OutOfMemoryError")
    return isinstance(oom_type, type) and isinstance(error, oom_type)
