#!/usr/bin/env python3
"""Check a Percept installation without downloading weights or calling APIs.

Run from the repository with the Python environment you intend to use:
    python scripts/verify_install.py
    python scripts/verify_install.py --metric clipiqa --cache-dir ./weights
    python scripts/verify_install.py --metric motion --cache-dir ./weights
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import math
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


def check_python():
    if sys.version_info < (3, 12):
        raise RuntimeError('Python 3.12+ is required')
    return sys.version.split()[0]


def check_core():
    for name in ('httpx', 'pydantic', 'pydantic_settings'):
        importlib.import_module(name)
    package = importlib.import_module('percept_harness')
    return f'{package.__version__} ({Path(package.__file__).resolve()})'


def check_program(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f'{name} is missing from PATH; install FFmpeg/FFprobe')
    result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f'{name} failed to execute (exit {result.returncode})')
    return result.stdout.splitlines()[0] if result.stdout.strip() else path


def check_cli(metric):
    launcher = shutil.which('percept')
    expected = Path(sysconfig.get_path('scripts')) / ('percept.exe' if sys.platform == 'win32' else 'percept')
    if launcher is None:
        raise RuntimeError('percept is missing from PATH; activate the installed Python environment')
    if Path(launcher).resolve() != expected.resolve():
        raise RuntimeError('percept on PATH belongs to another environment; activate the environment shown above')
    commands = [['--help'], ['annotate', '--help'], ['score', 'events', '--help']]
    if metric != 'core':
        commands.append(['score', metric, '--help'])
    for args in commands:
        result = subprocess.run([launcher, *args],
                                capture_output=True, text=True, timeout=30)
        if result.returncode or 'usage:' not in result.stdout:
            raise RuntimeError(f'percept {" ".join(args)} failed; reinstall the package in this environment')
    return f'{len(commands)} command parsers available'


def check_smoke():
    from percept_harness.evaluation.event_matching import Event, match_events, temporal_iou
    from percept_harness.evaluation.events import score_pair
    from percept_harness.pipelines.event_schemas import GroundedEvents
    from percept_harness.models.base import ModelOutputError, parse_strict_json

    if parse_strict_json('{"duration": 1, "events": []}') != {'duration': 1, 'events': []}:
        raise RuntimeError('Structured output parsing failed')
    try:
        parse_strict_json('{"score": NaN}')
    except ModelOutputError:
        pass
    else:
        raise RuntimeError('Structured output accepted a non-finite number')
    reference, prediction = Event(0, 2, 'move'), Event(1, 3, 'move')
    if not math.isclose(temporal_iou(reference, prediction), 1 / 3):
        raise RuntimeError('Temporal overlap smoke check failed')
    matches = match_events([reference], [prediction])
    if len(matches) != 1 or (matches[0].reference_index, matches[0].prediction_index) != (0, 0):
        raise RuntimeError('Event matching smoke check failed')
    grounded = GroundedEvents.model_validate({'duration': 2.0, 'events': [
        {'start': 0.0, 'end': 2.0, 'description': 'A hand moves a cup.'}]})
    score = score_pair(grounded, grounded, lambda a, b: {'match': a == b, 'reason': 'same action'})
    if score['segment_f1'] != 1 or score['semantic_match_rate'] != 1 or score['joint_f1'] != 1:
        raise RuntimeError('Grounded event scoring smoke check failed')
    return 'JSON parsing, temporal matching and grounded event scoring passed'


def check_runtime(metric):
    modules = {
        'clipiqa': ('torch', 'torchvision', 'cv2', 'numpy', 'PIL', 'pyiqa'),
        'motion': ('torch', 'torchvision', 'cv2', 'numpy', 'PIL', 'tqdm',
                   'omegaconf', 'decord', 'imageio', 'scipy', 'vbench'),
    }
    for name in modules[metric]:
        importlib.import_module(name)
    if metric == 'clipiqa':
        from percept_harness.video_metrics.clipiqa import PYIQA_VERSION
        package, expected = 'pyiqa', PYIQA_VERSION
    else:
        from percept_harness.video_metrics.motion import VBENCH_VERSION
        package, expected = 'vbench', VBENCH_VERSION
        if not Path(sysconfig.get_path('scripts')).joinpath('vbench.exe' if sys.platform == 'win32' else 'vbench').is_file():
            raise RuntimeError('Official VBench CLI is missing from the active Python environment')
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise RuntimeError(f'Expected {package}=={expected}; found {actual}')
    return f'{package}=={actual}; runtime imports passed'


def check_device(metric, device):
    torch = importlib.import_module('torch')
    if metric == 'motion' and device == 'cpu':
        raise RuntimeError('Motion Smoothness requires CUDA')
    needs_cuda = metric == 'motion' or device == 'cuda'
    available = torch.cuda.is_available()
    if needs_cuda and not available:
        raise RuntimeError('CUDA is unavailable in this Python environment')
    if available and device != 'cpu':
        # Exercise allocation, rather than relying only on the driver probe.
        torch.empty(1, device='cuda')
        return f'CUDA: {torch.cuda.get_device_name(0)}'
    return 'CPU'


def check_weights(metric, cache):
    from percept_harness.video_metrics import check_weights
    check_weights(metric, cache)
    return f'Checkpoint files present: {cache}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=('core', 'clipiqa', 'motion'), default='core',
                        help='Check only the selected environment (default: core)')
    parser.add_argument('--cache-dir', type=Path, default=Path('weights'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    checks = [
        ('Python', check_python),
        ('Core package', check_core),
        ('FFmpeg', lambda: check_program('ffmpeg')),
        ('FFprobe', lambda: check_program('ffprobe')),
        ('CLI', lambda: check_cli(args.metric)),
        ('Local smoke', check_smoke),
    ]
    if args.metric != 'core':
        cache = args.cache_dir.expanduser().resolve()
        checks.extend([
            ('Metric runtime', lambda: check_runtime(args.metric)),
            ('Device', lambda: check_device(args.metric, args.device)),
            ('Weights', lambda: check_weights(args.metric, cache)),
        ])
    print(f'Percept installation check: {args.metric}\nPython: {sys.executable}')
    failed = 0
    for name, function in checks:
        try:
            detail = function()
            print(f'[PASS] {name}: {detail}')
        except Exception as error:
            failed += 1
            print(f'[FAIL] {name}: {error}')
    print(f'\n{len(checks) - failed}/{len(checks)} checks passed for {args.metric}.')
    print('Local checks only; API connectivity and real-model inference were not tested.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
