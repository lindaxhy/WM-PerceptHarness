"""Checkpoint paths and digests shared by metric scoring and installation checks.

Only assets with recorded trusted SHA-256 digests are managed here. Files remain
in upstream-compatible cache layouts; no weights are included in the package.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class Weight:
    path: str
    sha256: str


WEIGHTS = {
    'clipiqa': (
        Weight('pyiqa/CLIP-IQA%2B_learned_prompts-603f3273.pth',
               '603f3273b6e8a75b309297c85781170c067c6b474f66818c2f253a46a6534447'),
        Weight('clip/RN50.pt',
               'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'),
    ),
    'motion': (
        Weight('amt_model/amt-s.pth',
               '07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77'),
    ),
}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_weights(metric: str, cache: Path) -> None:
    for asset in WEIGHTS[metric]:
        path = cache / asset.path
        if not path.is_file() or sha256(path) != asset.sha256:
            raise ValueError(
                f'Missing checkpoint or SHA-256 mismatch: {path}. '
                'See docs/metrics.md#checkpoint-preparation for download instructions.'
            )
