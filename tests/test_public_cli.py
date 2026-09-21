"""Public commands exercise real parsers and lightweight scoring, without models."""
import json
import subprocess
import sys

import pytest

from percept_harness.cli import main


def test_annotate_defaults_to_openai(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('PERCEPT_OPENAI_API_KEY', raising=False)
    video = tmp_path / 'input.mp4'
    video.touch()
    assert main(['annotate', '--videos', str(video), '--template',
                 'general_video_captioning', '--output', str(tmp_path / 'out')]) == 2
    assert 'PERCEPT_OPENAI_API_KEY' in capsys.readouterr().err


@pytest.mark.parametrize('metric', ['fidelity', 'clipiqa', 'motion'])
def test_score_help_without_optional_models(metric):
    result = subprocess.run([sys.executable, '-m', 'percept_harness.cli',
                             'score', metric, '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--reference' in result.stdout if metric == 'fidelity' else '--video' in result.stdout


def test_fidelity_command_scores_matching_timelines(tmp_path):
    for group in ('reference', 'generated'):
        folder = tmp_path / group / 'sample'
        folder.mkdir(parents=True)
        (folder / 'sample.json').write_text(json.dumps({'data': {
            'duration': 2, 'segments': [{'start': 0, 'end': 2}],
            'semantic_events': [{'event_type': 'move', 'start': 0, 'end': 2}],
        }}))
    output = tmp_path / 'report.json'
    assert main(['score', 'fidelity', '--reference', str(tmp_path / 'reference'),
                 '--system', f'demo={tmp_path / "generated"}', '--out', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['systems']['demo']['paired_samples'] == 1
    assert report['systems']['demo']['frame_miou_micro'] == 1


def test_unknown_score_metric_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main(['score', 'unknown'])
    assert exc.value.code == 2


def test_metric_help_does_not_load_optional_dependencies():
    # Reject even installed optional packages, so this checks laziness on GPU hosts too.
    code = '''
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'cv2', 'numpy', 'pyiqa', 'vbench'}:
            raise RuntimeError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from percept_harness.cli import main
main(['score', 'clipiqa', '--help'])
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--sample-step' in result.stdout
