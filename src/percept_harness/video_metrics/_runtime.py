"""Shared preflight, subprocess and output helpers for the metric wrappers.

Standard library only: dover.py also imports this file directly when it is run
as a script by an older upstream Python without the package installed.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

VBENCH_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"


def verify_revision(source: Path, expected: str | None, *, runner=None) -> dict[str, str | None]:
    if not (source / ".git").exists():
        raise ValueError(f"Source is not a git checkout: {source}")
    if expected is None:
        return {"revision": None}
    result = (runner or subprocess.run)(["git", "-C", str(source), "rev-parse", "HEAD"],
                                        check=True, capture_output=True, text=True)
    actual = result.stdout.strip()
    if actual != expected:
        raise ValueError(f"{source} revision mismatch: expected {expected}, got {actual}")
    return {"revision": actual}


def verify_source(root: Path, files, record: dict, revision: str) -> dict:
    for relative in files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source file: {path}")
    return {**record, **verify_revision(root, revision)}


def package_versions(names) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def subprocess_env(*, venv_path: bool = False, prepend: dict | None = None, unset=(),
                   values: dict | None = None, alloc_conf: bool = False, gpu: str | None = None) -> dict:
    env = os.environ.copy()
    prepend = dict(prepend or {})
    if venv_path:
        # VBench starts its distributed child using the literal command 'python'.
        # Do not resolve sys.executable symlinks: preserve the active virtualenv.
        prepend = {"PATH": str(Path(sys.executable).parent), **prepend}
    for key, value in prepend.items():
        env[key] = value + os.pathsep + env.get(key, "")
    for key in unset:
        env.pop(key, None)
    env.update(values or {})
    if alloc_conf:
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def run_logged(command, *, log: Path, label: str, mode: str = "x", **kwargs) -> None:
    with log.open(mode, encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
    if completed.returncode:
        raise RuntimeError(f"{label} exited with {completed.returncode}; see {log}")


def run_vbench(command, *, log: Path, output: Path, label: str, missing: Exception, **kwargs) -> Path:
    """Run an official VBench evaluation and return its single result JSON."""
    run_logged(command, log=log, label=label, **kwargs)
    results = list(output.glob("*_eval_results.json"))
    if len(results) != 1:
        raise missing
    return results[0]


def is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def finite_number(value, message: str) -> float:
    if not is_number(value) or not math.isfinite(value):
        raise ValueError(message)
    return float(value)


def single_row(rows, message: str) -> dict:
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(message)
    return rows[0]


def write_json_exclusive(path: Path, value, *, newline: bool = True, parents: bool = False) -> str:
    """Create a new, strictly valid JSON file and return its text (without the newline)."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n" if newline else text)
    return text


def run_cli(label: str, function, *, allow_nan: bool = False) -> int:
    try:
        print(json.dumps(function(), ensure_ascii=False, indent=2, allow_nan=allow_nan))
        return 0
    except Exception as error:
        print(f"{label} failed: {error}", file=sys.stderr)
        return 1
