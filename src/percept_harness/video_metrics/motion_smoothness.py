"""Run the standalone VBench 0.1.5 Motion Smoothness adapter."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from percept_harness.video_metrics import standalone

PACKAGE = "08_motion_smoothness"
RUNNER = "run_motion_smoothness.py"


def arguments(namespace: argparse.Namespace) -> list[str]:
    return ["--mode", namespace.mode, "--video", str(namespace.video),
            "--cache-dir", str(namespace.cache_dir), "--gpu", namespace.gpu,
            "--output", str(namespace.output)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="percept score motion-smoothness", description=__doc__)
    parser.add_argument("--mode", choices=("official", "cpu-staging"), required=True)
    parser.add_argument("--video", type=Path, required=True, help="One lowercase .mp4 file")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Cache containing amt_model/amt-s.pth")
    parser.add_argument("--gpu", required=True, help="One physical GPU ID; mapped to cuda:0")
    parser.add_argument("--output", type=Path, required=True, help="New single-video JSON file")
    standalone.add_arguments(parser)
    namespace = parser.parse_args(argv)
    try:
        return standalone.run(package=PACKAGE, script=RUNNER, arguments=arguments(namespace),
                              packages_root=namespace.metric_packages_root,
                              python=namespace.python, dry_run=namespace.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Motion Smoothness failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
