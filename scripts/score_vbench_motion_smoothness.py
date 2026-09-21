#!/usr/bin/env python3
"""Compatibility entry point. Prefer `percept score` after installing the package."""
import sys
from pathlib import Path

# Retain direct source-checkout execution, including `python -S ... --help`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from percept_harness.video_metrics.motion import main

if __name__ == "__main__":
    raise SystemExit(main())
