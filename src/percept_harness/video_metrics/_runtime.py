"""Shared preflight, subprocess and output helpers for the metric wrappers.

Standard library only: dover.py also imports this file directly when it is run
as a script by an older upstream Python without the package installed.
"""
from __future__ import annotations

from pathlib import Path

VBENCH_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"


def verify_source(root: Path, files, record: dict) -> dict:
    for relative in files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source file: {path}")
    return dict(record)
