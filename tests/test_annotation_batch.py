import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AnnotationBatchTests(unittest.TestCase):
    def test_manifest_archives_minimal_grounded_annotation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve()
            video=root/'clip.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=64x64:r=2:d=2','-c:v','mpeg4',str(video)],check=True)
            manifest=root/'input.jsonl';manifest.write_text(json.dumps({'job_id':'one','video_path':str(video)})+'\n')
            env={**os.environ,'PYTHONPATH':str(Path('src').resolve()),'PERCEPT_CV_CACHE_ROOT':str(root/'cv'), 'PERCEPT_WORK_ROOT':str(root/'work')}
            process=subprocess.run([sys.executable,'-m','percept_harness.cli','annotate','--manifest',str(manifest),
                '--output',str(root/'out'),'--backend','fake','--model','fake','--cv','fake','--cv-devices','0',
                '--workers-per-gpu','1','--request-concurrency','1'],env=env,text=True,capture_output=True,timeout=45)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            record=json.loads((root/'out/one/clip.json').read_text())
            self.assertEqual(set(record['data']),{'duration','events'})
            self.assertTrue(record['metadata']['synthetic'])
            archives=json.loads((root/'out/one/.percept/cv-artifact.json').read_text())['artifacts']
            self.assertEqual(len(archives),1)
            self.assertTrue((Path(archives[0]['archive_path'])/'manifest.json').is_file())

class LocalRecheckArchiveTests(unittest.TestCase):
    def test_manifest_archives_baseline_and_local_recheck(self):
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from percept_harness import cli
        from percept_harness.models.fake import FakeVideoModel
        original = FakeVideoModel._fixture
        def with_review(self, request):
            if request.stage == 'evidence_review':
                return {'concerns': [{'reason': 'Check cup movement', 'event_indices': [0],
                    'start': 0.5, 'end': 1.5, 'new_entities': ['blue cup']}], 'unresolved': []}
            return original(self, request)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            video = root/'clip.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'color=c=blue:s=64x64:r=4:d=3', '-c:v', 'mpeg4', str(video)], check=True)
            manifest = root/'input.jsonl'
            manifest.write_text(json.dumps({'job_id': 'one', 'video_path': str(video)})+'\n')
            settings = {'PERCEPT_CV_CACHE_ROOT': str(root/'cv'), 'PERCEPT_WORK_ROOT': str(root/'work')}
            with patch.dict(os.environ, settings), patch.object(FakeVideoModel, '_fixture', with_review), \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = cli.main(['annotate', '--manifest', str(manifest), '--output', str(root/'out'),
                    '--backend', 'fake', '--model', 'fake', '--cv', 'fake', '--cv-devices', '0',
                    '--workers-per-gpu', '1', '--request-concurrency', '1'])
            self.assertEqual(code, 0)
            record = json.loads((root/'out/one/clip.json').read_text())
            self.assertEqual(record['metadata']['recheck_budget']['selected_windows'], 1)
            archive = json.loads((root/'out/one/.percept/cv-artifact.json').read_text())['artifacts']
            self.assertEqual({item['scope'] for item in archive}, {'baseline', 'local_1'})
            for item in archive:
                self.assertTrue((Path(item['archive_path'])/'manifest.json').is_file())
            status = json.loads((root/'out/shard-00.status.json').read_text())
            self.assertEqual(status['jobs']['one']['status'], 'completed')
            from percept_harness.evaluation import events_cli
            with redirect_stdout(StringIO()):
                scored = events_cli.main(['--reference', str(root/'out'), '--system', f"fake={root/'out'}",
                    '--out', str(root/'scores')], judge=lambda a, b: {'match': a == b, 'reason': 'same action'})
            self.assertEqual(scored, 0)
            report = json.loads((root/'scores/events.json').read_text())
            self.assertEqual(set(report['systems']['fake']['samples']), {'one'})
            self.assertEqual(report['systems']['fake']['aggregate']['joint']['micro']['f1'], 1)
