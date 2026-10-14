import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from percept_harness import video_metrics as weights


class WeightTests(unittest.TestCase):
    def test_download_command_is_retired(self):
        from percept_harness.cli import main
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main(['download', '--help'])
        self.assertEqual(error.exception.code, 2)

    def test_cache_manifest_accepts_path_list_and_legacy_values(self):
        from percept_harness.video_metrics.vbench_backend import verify_cache
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            relative = 'clip_model/ViT-B-32.pt'
            checkpoint = cache / relative
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'local checkpoint')
            manifest = cache / 'manifest.json'
            for content in ([relative], {relative: 'legacy value ignored'}):
                manifest.write_text(json.dumps(content))
                verify_cache(cache, ['background_consistency'], manifest)
            checkpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                verify_cache(cache, ['background_consistency'], manifest)

    def test_weight_paths_accept_existing_files_and_reject_missing(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(weights.WEIGHTS, {'test': ('model.pt',)}):
            cache = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'Missing checkpoint'):
                weights.check_weights('test', cache)
            (cache / 'model.pt').write_bytes(b'locally prepared checkpoint')
            weights.check_weights('test', cache)

    def test_cache_without_manifest_checks_required_files(self):
        from percept_harness.video_metrics.vbench_backend import verify_cache
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                verify_cache(cache, ['background_consistency'], None)
            model = cache / 'clip_model/ViT-B-32.pt'
            model.parent.mkdir()
            model.write_bytes(b'local checkpoint')
            verify_cache(cache, ['background_consistency'], None)

    def test_cache_manifest_rejects_paths_outside_cache(self):
        from percept_harness.video_metrics.vbench_backend import verify_cache
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            manifest = cache / 'manifest.json'
            for relative in ('../outside.pt', '/outside.pt'):
                manifest.write_text(json.dumps([relative]))
                with self.assertRaisesRegex(ValueError, 'cache-relative'):
                    verify_cache(cache, ['temporal_flickering'], manifest)
