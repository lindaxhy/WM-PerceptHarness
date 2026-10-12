#!/usr/bin/env python3
"""Join VideoPhy-2 PC and SA CSV outputs into the documented analysis field.

VideoPhy-2 Joint is not a third detector.  For matching rows it is simply
``SA >= 4 and PC >= 4``.  The two official inference runs must use the same
videos and preserve the original PC/SA scores; this script refuses duplicate
or unmatched keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or any(not field or not field.strip() for field in fields) or len(fields) != len(set(fields)):
                raise ValueError(f"CSV needs non-empty, unique column names: {path}")
            rows = [dict(row) for row in reader]
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError(f"CSV row width does not match its header: {path}")
            return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("rows", "results", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a list of rows in {path}")
    return payload


def _key(value: str, root: Path | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Video key must be a non-empty path string")
    path = Path(value.strip()).expanduser()
    if root and not path.is_absolute():
        path = root / path
    # Linux paths are case-sensitive. Never lowercase a pathname to join rows.
    return str(path.resolve(strict=False))


def _index(rows: Iterable[dict[str, Any]], *, video_column: str, score_column: str, root: Path | None, label: str) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    if isinstance(rows, (str, bytes, dict)) or rows is None:
        raise ValueError(f"{label} input must be a collection of score rows")
    if not isinstance(video_column, str) or not video_column.strip() or not isinstance(score_column, str) or not score_column.strip():
        raise ValueError("Column names must be non-empty strings")
    try:
        rows = iter(rows)
    except TypeError as error:
        raise ValueError(f"{label} input must be a collection of score rows") from error
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} must be an object")
        if video_column not in row or score_column not in row:
            raise ValueError(f"{label} row {index} needs columns {video_column!r} and {score_column!r}")
        raw_key = row[video_column]
        raw_score = row[score_column]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (str, int, float)):
            raise ValueError(f"{label} row {index} score must be an integer in [1,5]")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} row {index} has a non-numeric score: {row[score_column]!r}") from error
        if not math.isfinite(score) or not score.is_integer() or not 1 <= score <= 5:
            raise ValueError(f"{label} row {index} score must be a finite integer in [1,5], got {score}")
        key = _key(raw_key, root)
        if key in result:
            raise ValueError(f"Duplicate {label} video key: {raw_key}")
        result[key] = (raw_key, int(score))
    if not result:
        raise ValueError(f"{label} input must contain at least one score row")
    return result


def aggregate(
    pc_rows: Iterable[dict[str, Any]],
    sa_rows: Iterable[dict[str, Any]],
    *,
    video_column: str = "videopath",
    pc_column: str = "pc",
    sa_column: str = "sa",
    root: Path | None = None,
) -> dict[str, Any]:
    pc = _index(pc_rows, video_column=video_column, score_column=pc_column, root=root, label="PC")
    sa = _index(sa_rows, video_column=video_column, score_column=sa_column, root=root, label="SA")
    if set(pc) != set(sa):
        missing_sa = sorted(set(pc) - set(sa))
        missing_pc = sorted(set(sa) - set(pc))
        raise ValueError(f"PC/SA video sets differ; missing SA={missing_sa}, missing PC={missing_pc}")
    rows = []
    for key in sorted(pc):
        pc_path, pc_score = pc[key]
        _, sa_score = sa[key]
        rows.append(
            {
                "video": pc_path,
                "pc": pc_score,
                "sa": sa_score,
                "joint_pass": bool(pc_score >= 4 and sa_score >= 4),
            }
        )
    return {
        "metric": "video_phy_joint_analysis",
        "rule": "SA >= 4 and PC >= 4",
        "higher_is_better": True,
        "joint_rate": sum(row["joint_pass"] for row in rows) / len(rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc", type=Path, required=True)
    parser.add_argument("--sa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-column", default="videopath")
    parser.add_argument("--pc-column", default="pc")
    parser.add_argument("--sa-column", default="sa")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = aggregate(
            _read_rows(args.pc.expanduser().resolve(strict=True)),
            _read_rows(args.sa.expanduser().resolve(strict=True)),
            video_column=args.video_column,
            pc_column=args.pc_column,
            sa_column=args.sa_column,
            root=args.root,
        )
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
        print(serialized)
        return 0
    except Exception as error:
        print(f"VideoPhy-2 Joint aggregation failed: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
