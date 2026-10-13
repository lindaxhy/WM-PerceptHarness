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
