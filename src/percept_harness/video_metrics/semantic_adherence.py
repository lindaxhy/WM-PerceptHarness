"""Run the standalone Semantic Adherence (VideoPhy-2) evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from percept_harness.video_metrics import standalone

PACKAGE = "01_semantic_adherence"
RUNNER = "modified/sapc/run_sa.py"


def arguments(namespace: argparse.Namespace) -> list[str]:
    command = ["--input-csv", str(namespace.input_csv), "--checkpoint", str(namespace.checkpoint),
               "--output", str(namespace.output), "--sampling", namespace.sampling]
    if namespace.frame_cap is not None:
        command += ["--frame-cap", str(namespace.frame_cap)]
    if namespace.source is not None:
        command += ["--source", str(namespace.source)]
    if namespace.preprocess_only:
        command.append("--preprocess-only")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="percept score semantic-adherence",
        description=__doc__,
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV with videopath and caption columns")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local VideoPhy-2 checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--sampling", choices=("official", "half"), default="half",
                        help="official: 32 fixed centers; half: min(max(ceil(N/2),32),256) frames (default)")
    parser.add_argument("--frame-cap", type=int, choices=(256,), help="Only for --sampling half")
    parser.add_argument("--source", type=Path, help="Pinned VideoPhy-2 source directory")
    parser.add_argument("--preprocess-only", action="store_true")
    standalone.add_arguments(parser)
    namespace = parser.parse_args(argv)
    try:
        return standalone.run(package=PACKAGE, script=RUNNER, arguments=arguments(namespace),
                              packages_root=namespace.metric_packages_root,
                              python=namespace.python, dry_run=namespace.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Semantic Adherence failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
