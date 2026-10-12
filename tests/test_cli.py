"""Public commands exercise real parsers and lightweight scoring, without models."""
import json
import re
from pathlib import Path
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


@pytest.mark.parametrize('metric', ['fidelity', 'clipiqa', 'motion', 'vbench-clip', 'dover', 'external', 'videophy-joint'])
def test_score_help_without_optional_models(metric):
    result = subprocess.run([sys.executable, '-m', 'percept_harness.cli',
                             'score', metric, '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = {'fidelity': '--reference', 'external': '--source-dir', 'videophy-joint': '--pc'}.get(metric, '--video')
    assert expected in result.stdout


def test_unknown_score_metric_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main(['score', 'unknown'])
    assert exc.value.code == 2


@pytest.mark.parametrize('metric', ['clipiqa', 'motion', 'vbench-clip', 'dover', 'external', 'videophy-joint'])
def test_metric_help_does_not_load_optional_dependencies(metric):
    # Reject even installed optional packages, so this checks laziness on GPU hosts too.
    code = '''
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'cv2', 'numpy', 'pyiqa', 'vbench'}:
            raise RuntimeError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from percept_harness.cli import main
main(['score', sys.argv[1], '--help'])
'''
    result = subprocess.run([sys.executable, '-c', code, metric], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_fidelity_scores_matching_timelines_outside_checkout(tmp_path):
    for group in ('reference', 'generated'):
        folder = tmp_path / group / 'sample'
        folder.mkdir(parents=True)
        (folder / 'sample.json').write_text(json.dumps({'data': {
            'duration': 2,
            'segments': [{'start': 0, 'end': 2}],
            'semantic_events': [{'event_type': 'move', 'start': 0, 'end': 2}],
        }}))
    output = tmp_path / 'report.json'
    code = '''
import sys
from percept_harness.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    result = subprocess.run(
        [sys.executable, '-c', code, 'score', 'fidelity',
         '--reference', str(tmp_path / 'reference'),
         '--system', f'demo={tmp_path / "generated"}', '--out', str(output)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['systems']['demo']['paired_samples'] == 1
    assert report['systems']['demo']['frame_miou_micro'] == 1


def test_videophy_joint_command_scores_csv_outside_checkout(tmp_path):
    pc = tmp_path / 'pc.csv'
    sa = tmp_path / 'sa.csv'
    pc.write_text('videopath,pc\n/data/a.mp4,4\n/data/b.mp4,3\n')
    sa.write_text('videopath,sa\n/data/a.mp4,5\n/data/b.mp4,5\n')
    output = tmp_path / 'joint.json'
    result = subprocess.run(
        [sys.executable, '-m', 'percept_harness.cli', 'score', 'videophy-joint',
         '--pc', str(pc), '--sa', str(sa), '--output', str(output)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['joint_rate'] == 0.5
    assert [row['joint_pass'] for row in report['rows']] == [True, False]


@pytest.mark.parametrize('metric', [
    'imaging-quality', 'aesthetic-quality', 'temporal-flickering',
    'dynamic-degree', 'subject-consistency', 'background-consistency',
    'overall-consistency',
])
def test_direct_video_metric_help_has_no_dimension_selector(metric):
    result = subprocess.run([sys.executable, '-m', 'percept_harness.cli',
                             metric, '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--videos-path' in result.stdout
    assert '--dimension' not in result.stdout


_ACCESS_TOKEN_PATTERN = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|"
    rb"sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{20,})"
)


def _tracked_text_files(repository_root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        capture_output=True,
        check=True,
    )
    return [
        repository_root / Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw and (repository_root / Path(raw.decode("utf-8"))).is_file()
    ]


def test_repository_contains_no_committed_access_token_values() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    offenders = [
        path.relative_to(repository_root).as_posix()
        for path in _tracked_text_files(repository_root)
        if _ACCESS_TOKEN_PATTERN.search(path.read_bytes()) is not None
    ]

    assert offenders == []
