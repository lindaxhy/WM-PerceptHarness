"""Run strict, offline LAS alignment evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from las_repro.evaluation.las_alignment import main

if __name__ == "__main__":
    raise SystemExit(main())
