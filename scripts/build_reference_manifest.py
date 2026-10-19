#!/usr/bin/env python3
"""Pair generated/reference frame directories without inventing missing GT.

PSNR, SSIM and LPIPS need aligned future reference frames.  This utility only
matches identical relative filenames and records unmatched files; it does not
copy, resize, duplicate, or synthesize frames. Matching filenames do not prove
temporal alignment, equal geometry, or that the reference is real future GT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from percept_harness.video_metrics._runtime import run_cli, write_json_exclusive


DEFAULT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})


def build_pairs(generated_root: Path, reference_root: Path, extensions: set[str] | None = None) -> dict:
    generated_root = generated_root.expanduser().resolve(strict=True)
    reference_root = reference_root.expanduser().resolve(strict=True)
    if not generated_root.is_dir() or not reference_root.is_dir():
        raise ValueError("Both frame inputs must be directories")
    if generated_root == reference_root or generated_root.samefile(reference_root):
        raise ValueError("Generated and reference roots must be different; generated frames cannot be their own GT")
    if generated_root in reference_root.parents or reference_root in generated_root.parents:
        raise ValueError("Generated and reference roots must not contain one another")
    requested_extensions = DEFAULT_EXTENSIONS if extensions is None else extensions
    if not requested_extensions or any(not isinstance(value, str) or not value.strip(". ") for value in requested_extensions):
        raise ValueError("At least one non-empty frame extension is required")
    extensions = {value.lower() if value.startswith(".") else f".{value.lower()}" for value in requested_extensions}
    generated = {
        path.relative_to(generated_root).as_posix(): path
        for path in generated_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }
    reference = {
        path.relative_to(reference_root).as_posix(): path
        for path in reference_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }
    if not generated or not reference:
        raise ValueError("Both frame directories must contain at least one supported image")
    matched_names = sorted(set(generated) & set(reference))
    if not matched_names:
        raise ValueError("No matching relative frame filenames were found")
    for name in matched_names:
        if generated[name].samefile(reference[name]):
            raise ValueError(f"Generated and reference frame refer to the same file: {name}")
    return {
        "protocol": "identical relative frame filenames; no alignment or synthesis",
        "alignment_verified": False,
        "warning": "Matching filenames do not establish temporal or geometric alignment or prove real future GT; verify these separately before scoring.",
        "generated_root": str(generated_root),
        "reference_root": str(reference_root),
        "matched": [
            {"relative_path": name, "generated": str(generated[name]), "reference": str(reference[name])}
            for name in matched_names
        ],
        "unmatched_generated": sorted(set(generated) - set(reference)),
        "unmatched_reference": sorted(set(reference) - set(generated)),
        "counts": {"generated": len(generated), "reference": len(reference), "matched": len(matched_names)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-frames", type=Path, required=True)
    parser.add_argument("--reference-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", action="append")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    def run():
        generated = args.generated_frames.expanduser().resolve(strict=True)
        reference = args.reference_frames.expanduser().resolve(strict=True)
        if not generated.is_dir() or not reference.is_dir():
            raise ValueError("Both frame inputs must be directories")
        payload = build_pairs(generated, reference, set(args.extension) if args.extension else None)
        if args.require_complete and (payload["unmatched_generated"] or payload["unmatched_reference"]):
            raise ValueError("Frame directories are not complete matches; inspect unmatched_* in the manifest")
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        write_json_exclusive(output, payload, parents=True)
        return payload
    return run_cli("Reference manifest", run)


if __name__ == "__main__":
    raise SystemExit(main())
