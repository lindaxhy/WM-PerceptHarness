"""Real media, synthetic evidence and an injected semantic judge; no hosted/GPU inference."""
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from percept_harness.config import Settings
from percept_harness.batch import archive_cv_evidence
from percept_harness.cv.artifacts import CvArtifactStore
from percept_harness.cv.base import FakeCvEvidenceProvider
from percept_harness.cv.executor import SyncCvExecutor
from percept_harness.evaluation import events_cli
from percept_harness.models.fake import FakeVideoModel
from percept_harness.runner import SyncRunner, run_batch


class OfflineEventIntegrationTests(unittest.TestCase):
    def test_nested_root_annotate_score_archives_baseline_and_local_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video = root / 'clip.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'color=c=blue:s=64x64:r=4:d=4', '-c:v', 'mpeg4', str(video)], check=True)
            review = {'concerns': [{'reason': 'Check cup movement', 'event_indices': [0],
                'start': 1.0, 'end': 2.0, 'new_entities': ['blue cup']}], 'unresolved': []}
            model = FakeVideoModel(failure_script={'evidence_review': [review, review]})
            settings = Settings(work_root=root/'work', cv_cache_root=root/'cv', cv_provider='fake')
            with CvArtifactStore(settings.cv_cache_root) as artifacts:
                runner = SyncRunner(model, settings, model_alias='fake',
                    cv_executor=SyncCvExecutor(FakeCvEvidenceProvider(), artifacts))
                reference, prediction = root/'reference', root/'prediction'
                for output in (reference, prediction):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        result = list(run_batch(runner, [video], 'embodied_action_captioning', output/'job-1', log=lambda message: None))[0]
                    self.assertEqual(result.status, 'completed', result.error)
                    self.assertEqual(result.metadata['recheck_budget']['selected_windows'], 1)
                    evidence = result.metadata['evidence']
                    self.assertEqual({item['scope'] for item in evidence}, {'baseline', 'local_1'})
                    archives = [archive_cv_evidence(item, settings.cv_cache_root, output/'.cv-archive')
                                for item in evidence]
                    self.assertEqual(len(archives), 2)
                    for archived in archives:
                        self.assertTrue((archived/'manifest.json').is_file())
                    (output/'shard-00.status.json').write_text(json.dumps({'phase': 'done'}))
                report_dir = root/'scores'
                with redirect_stdout(StringIO()):
                    code = events_cli.main(['--reference', str(reference), '--system', f'fake={prediction}',
                        '--out', str(report_dir)], judge=lambda a, b: {'match': a == b, 'reason': 'same action'})
                self.assertEqual(code, 0)
                report = json.loads((report_dir/'events.json').read_text())
                system = report['systems']['fake']
                self.assertEqual((system['valid_count'], system['invalid_count']), (1, 0))
                self.assertEqual(system['aggregate']['segment']['micro']['f1'], 1)
                self.assertEqual(system['aggregate']['joint']['micro']['f1'], 1)
                self.assertEqual(set(system['samples']), {'job-1'})
                (prediction/'job-1'/'clip.json').unlink()
                with redirect_stdout(StringIO()):
                    code = events_cli.main(['--reference', str(reference), '--system', f'fake={prediction}',
                        '--out', str(report_dir)], judge=lambda a, b: {'match': True, 'reason': 'same'})
                self.assertEqual(code, 1)
                self.assertIsNone(json.loads((report_dir/'events.json').read_text())['systems']['fake']['aggregate'])
