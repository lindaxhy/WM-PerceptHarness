"""Pinned checkpoints shared by download commands and metric verification.

Only assets with recorded trusted SHA-256 digests are managed here. Files remain
in upstream-compatible cache layouts; no weights are included in the package.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import tempfile


@dataclass(frozen=True)
class Weight:
    path: str
    url: str
    sha256: str


WEIGHTS = {
    'clipiqa': (
        Weight('pyiqa/CLIP-IQA%2B_learned_prompts-603f3273.pth',
               'https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/CLIP-IQA%2B_learned_prompts-603f3273.pth',
               '603f3273b6e8a75b309297c85781170c067c6b474f66818c2f253a46a6534447'),
        Weight('clip/RN50.pt',
               'https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt',
               'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'),
    ),
    'motion': (
        Weight('amt_model/amt-s.pth',
               'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth',
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
                f'Run percept download {metric} --cache-dir "{cache}"'
            )


def download_weight(asset: Weight, cache: Path, client) -> Path:
    """Reuse verified bytes or atomically replace a missing/corrupt checkpoint."""
    target = cache / asset.path
    if target.is_file() and sha256(target) == asset.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                                         suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            with client.stream('GET', asset.url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != asset.sha256:
            raise ValueError(f'SHA-256 mismatch for {asset.url}; downloaded file discarded')
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='percept download', description=__doc__)
    parser.add_argument('metric', choices=tuple(WEIGHTS))
    parser.add_argument('--cache-dir', type=Path, default=Path('weights'),
                        help='Upstream-compatible cache directory (default: ./weights)')
    args = parser.parse_args(argv)
    import httpx
    try:
        cache = args.cache_dir.expanduser().resolve()
        with ExitStack() as stack:
            client = None
            for asset in WEIGHTS[args.metric]:
                print(f'Preparing {asset.path} …', file=sys.stderr, flush=True)
                target = cache / asset.path
                if not target.is_file() or sha256(target) != asset.sha256:
                    if client is None:
                        client = stack.enter_context(httpx.Client(follow_redirects=True, timeout=60))
                path = download_weight(asset, cache, client)
                print(f'Verified {path}', flush=True)
        return 0
    except (OSError, ValueError, ImportError, httpx.HTTPError) as error:
        print(f'Weight download failed: {error}', file=sys.stderr)
        return 1
