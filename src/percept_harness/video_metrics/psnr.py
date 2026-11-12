"""Run the standalone paired source-reference PSNR evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from percept_harness.video_metrics import standalone

PACKAGE = "06_psnr"
RUNNER = "modified/run_psnr.py"


def arguments(namespace: argparse.Namespace) -> list[str]:
    command = ["--manifest", str(namespace.manifest), "--output-dir", str(namespace.output_dir),
               "--workers", str(namespace.workers)]
    if namespace.limit is not None:
        command += ["--limit", str(namespace.limit)]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="percept score psnr", description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen source/generated pair manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="New result directory")
    parser.add_argument("--limit", type=int, help="Score at most this many sample pairs")
    parser.add_argument("--workers", type=int, default=3, help="Worker processes (default: 3)")
    standalone.add_arguments(parser)
    namespace = parser.parse_args(argv)
    try:
        return standalone.run(package=PACKAGE, script=RUNNER, arguments=arguments(namespace),
                              packages_root=namespace.metric_packages_root,
                              python=namespace.python, dry_run=namespace.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"PSNR failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
