"""Independent source validation and loading for PSNR."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_official():
    manifest = json.loads((PACKAGE_ROOT / 'OFFICIAL_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = PACKAGE_ROOT / entry['path']
        if file_sha256(path) != entry['sha256']:
            raise ValueError('Bundled official source changed: ' + entry['path'])
    return manifest


def read_manifest(path, kind):
    path = Path(path).resolve(strict=True)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty manifest')
    seen = set()
    for row in rows:
        model, sample_id = row['model'], row['sample_id']
        if model not in ('wan', 'lingbot-dense'):
            raise ValueError('Unsupported model geometry/prompt mapping: ' + str(model))
        if not isinstance(sample_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', sample_id):
            raise ValueError('sample_id must be an ASCII filename-safe identifier')
        if (model, sample_id) in seen:
            raise ValueError('Duplicate model/sample_id')
        seen.add((model, sample_id))
        keys = ['video_path'] + (['source_video_path'] if kind == 'psnr' else [])
        for key in keys:
            p = Path(row[key]).expanduser()
            row[key] = str((path.parent/p if not p.is_absolute() else p).resolve(strict=True))
        hash_keys = ['recorded_final_sha256'] + (['source_sha256', 'first_frame_sha256'] if kind == 'psnr' else [])
        for key in hash_keys:
            if not isinstance(row.get(key), str) or not re.fullmatch('[0-9a-f]{64}', row[key]):
                raise ValueError(key + ' must be the frozen lowercase SHA-256')
        if not isinstance(row.get('recorded_evaluation_frames'), int) or row['recorded_evaluation_frames'] < 1:
            raise ValueError('recorded_evaluation_frames must be a positive integer')
    if kind == 'psnr':
        grouped = {}
        for row in rows:
            identity = (row['source_video_path'], row['source_sha256'], row['first_frame_sha256'])
            previous = grouped.setdefault(row['sample_id'], identity)
            if previous != identity:
                raise ValueError('Rows paired by sample_id must share source and conditioning-frame identity')
    return sorted(rows, key=lambda x: (x['sample_id'], x['model']))


def require_persistent_remote_session():
    # Honor the project's remote-session rule, without requiring tmux on a local
    # Windows workstation. Documentation requires tmux for every remote long run.
    if (os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY')) and not os.environ.get('TMUX'):
        raise RuntimeError('Create and enter tmux before starting a remote scoring run')
