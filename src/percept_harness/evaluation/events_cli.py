"""Score grounded event annotations with an OpenAI-compatible semantic judge."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .events import OpenAIJudge, evaluate


def _safe_url(value):
    if not value:
        return None
    try:
        parts = urlsplit(value)
        hostname = parts.hostname or ""
        port = parts.port
    except ValueError as error:
        raise ValueError("invalid judge base URL") from error
    if not parts.scheme or not hostname:
        raise ValueError("invalid judge base URL")
    host = hostname + (f":{port}" if port else "")
    return f"{parts.scheme}://{host}"


def main(argv=None, *, judge=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--system", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--out", required=True, type=Path, help="report output directory")
    parser.add_argument("--tiou-threshold", type=float, default=.5)
    parser.add_argument("--judge-base-url", default=os.getenv("PERCEPT_JUDGE_BASE_URL"))
    parser.add_argument("--judge-model", default=os.getenv("PERCEPT_JUDGE_MODEL"))
    parser.add_argument("--judge-api-key", default=os.getenv("PERCEPT_JUDGE_API_KEY"))
    args = parser.parse_args(argv)
    systems = {}
    for spec in args.system:
        name, separator, directory = spec.partition("=")
        if not separator or not name or not directory or name in systems:
            parser.error(f"expected unique NAME=DIR, got {spec!r}")
        systems[name] = Path(directory)
    if judge is None:
        try:
            judge = OpenAIJudge(args.judge_base_url, args.judge_model, args.judge_api_key)
        except ValueError as error:
            parser.error(str(error))
    try:
        safe_url = _safe_url(args.judge_base_url)
        report = evaluate(args.reference, systems, judge, args.tiou_threshold)
        report["judge"] = {"protocol": "OpenAI-compatible chat completions, strict JSON match/reason", "base_url": safe_url, "model": args.judge_model} if isinstance(judge, OpenAIJudge) else {"protocol": "injected callable"}
        failed = any(system["invalid_count"] for system in report["systems"].values())
    except (OSError, ValueError) as error:
        report = {"status": "error", "error": f"{type(error).__name__}: {error}", "systems": {name: {"samples": {}, "valid_count": 0, "invalid_count": 1, "aggregate": None, "error": f"{type(error).__name__}: {error}"} for name in systems}, "judge": {"protocol": "OpenAI-compatible chat completions", "base_url": None, "model": args.judge_model}}
        failed = True
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "events.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"event score report: {args.out / 'events.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
