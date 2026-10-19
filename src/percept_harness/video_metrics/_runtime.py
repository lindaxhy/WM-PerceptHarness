"""Shared preflight, subprocess and output helpers for the metric wrappers.

Standard library only: dover.py also imports this file directly when it is run
as a script by an older upstream Python without the package installed.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

VBENCH_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"


def verify_source(root: Path, files, record: dict) -> dict:
    for relative in files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source file: {path}")
    return dict(record)


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
