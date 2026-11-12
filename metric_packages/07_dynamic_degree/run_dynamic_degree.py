"""Pinned VBench 0.1.5 dynamic_degree, official or CPU-staged frame storage.

This adapter does not change sampling, resolution, inference or score formulas.
Run remote inference only from an already verified persistent tmux session.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import textwrap
import time

ROOT = Path(__file__).resolve().parent
METRIC = 'dynamic_degree'
SOURCE_HASH = '2e93d3a7af38bfbe459c19c9d127894f687424a0bc534dbf7e6571d17b24dff7'
WEIGHT = ('raft_model/models/raft-things.pth', 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1')
PATCHES = [('get_frames', 'frame = frame[None].to(self.device)', 'frame = frame[None]'), ('infer', 'padder = InputPadder(image1.shape)', 'image1, image2 = image1.to(self.device), image2.to(self.device)\n            padder = InputPadder(image1.shape)')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def patch_method(module, cls, name, old, new):
    original = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    if original.count(old) != 1:
        raise ValueError('Pinned method differs: ' + name)
    changed = original.replace(old, new)
    namespace = {}
    exec(compile(changed, module.__file__ + ':cpu_staging', 'exec'), module.__dict__, namespace)
    setattr(cls, name, namespace[name])
    return {'method': name, 'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'patched_sha256': hashlib.sha256(changed.encode()).hexdigest(), 'old': old, 'new': new}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['official', 'cpu-staging'], required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='One physical GPU ID; mapped to cuda:0 in this process')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get('TMUX'):
        raise RuntimeError('Create, enter and verify tmux before remote inference; see README')
    video = args.video.resolve(strict=True)
    if video.suffix != '.mp4':
        raise ValueError('This adapter audits MP4 video only')
    if args.output.exists():
        raise FileExistsError(args.output)
    if importlib.metadata.version('vbench') != '0.1.5':
        raise ValueError('Requires the bundled official vbench 0.1.5 wheel')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Only load the fixed, hash-verified checkpoint with its official loader.
    os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD', None)
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    relative_weight, expected_weight = WEIGHT
    weight = (args.cache_dir / relative_weight).resolve(strict=True)
    if sha(weight) != expected_weight:
        raise ValueError('Checkpoint SHA256 does not match the audited original')
    video_hash = sha(video)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA device is required')
    module = importlib.import_module('vbench.' + METRIC)
    if sha(module.__file__) != SOURCE_HASH:
        raise ValueError('Installed official metric source differs from the bundled baseline')
    cls = module.DynamicDegree
    edits = []
    if args.mode == 'cpu-staging':
        edits = [patch_method(module, cls, *entry) for entry in PATCHES]
    # Preserve the official CLI's torch.device object in both execution modes.
    device = torch.device('cuda')
    started = time.time()
    obj = cls(module.edict({'model': str(weight), 'small': False,
                           'mixed_precision': False, 'alternate_corr': False}), device)
    value = float(obj.infer(str(video)))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid metric result')
    if sha(video) != video_hash:
        raise ValueError('Input video changed during scoring')
    payload = {'metric': METRIC, 'score': value, 'mode': args.mode,
               'video': str(video), 'video_sha256': video_hash,
               'source_sha256': sha(module.__file__), 'checkpoint_sha256': expected_weight,
               'adapter_sha256': sha(__file__), 'patches': edits,
               'device_argument': 'torch.device(cuda), identical to official CLI',
               'versions': {'vbench': '0.1.5', 'torch': torch.__version__},
               'elapsed_seconds': time.time() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({'metric': METRIC, 'score': value, 'mode': args.mode}))


if __name__ == '__main__':
    main()
