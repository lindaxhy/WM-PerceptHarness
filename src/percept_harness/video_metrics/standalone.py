"""Locate and run the audited standalone evaluator programs.

The source packages under ``metric_packages/`` are not installed with the
harness wheel.  These helpers select the pinned runner from that source tree
and start it in the interpreter of the evaluator's environment.  No sampling,
model or scoring code is reimplemented here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

PACKAGES_ENV = "PERCEPT_METRIC_PACKAGES"


def _packages_dir(candidate: Path) -> Path:
    """Accept either the checkout root or the metric_packages directory itself."""
    try:
        root = candidate.expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Standalone metric packages not found at {candidate}; run from the source checkout, "
            f"pass --metric-packages-root, or set {PACKAGES_ENV}"
        ) from None
    nested = root / "metric_packages"
    return nested if nested.is_dir() else root


def resolve_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = _packages_dir(explicit)
    elif configured := os.environ.get(PACKAGES_ENV):
        root = _packages_dir(Path(configured))
    else:
        candidates = [parent / "metric_packages" for parent in Path(__file__).resolve().parents]
        candidates.append(Path.cwd() / "metric_packages")
        root = next((candidate for candidate in candidates if candidate.is_dir()), candidates[-1])
    if not root.is_dir():
        raise FileNotFoundError(
            f"Standalone metric packages not found at {root}; run from the source checkout, "
            f"pass --metric-packages-root, or set {PACKAGES_ENV}"
        )
    return root


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metric-packages-root",
        type=Path,
        help="Checkout root or its metric_packages directory (default: repository source tree or $PERCEPT_METRIC_PACKAGES)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        metavar="PATH",
        help="Interpreter of the evaluator environment (default: the current interpreter)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved runner command without starting it",
    )


def run(*, package: str, script: str, arguments: Sequence[str], packages_root: Path | None,
        python: str | None, dry_run: bool = False) -> int:
    root = resolve_root(packages_root)
    runner = root / package / script
    if not runner.is_file():
        raise FileNotFoundError(f"Missing standalone runner: {runner}")
    command = [python or sys.executable, str(runner), *arguments]
    display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    print("Running: " + display, file=sys.stderr)
    if dry_run:
        return 0
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as error:
        raise RuntimeError(f"Could not start {command[0]}: {error}") from error
