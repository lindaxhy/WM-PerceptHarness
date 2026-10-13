import hashlib
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from percept_harness import weights
from percept_harness.cli import main


class WeightTests(unittest.TestCase):
    def test_download_command_is_retired(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main(['download', '--help'])
        self.assertEqual(error.exception.code, 2)

    def test_verification_accepts_valid_bytes_and_rejects_missing_or_corrupt(self):
        data = b'test checkpoint'
        asset = weights.Weight(path='model/checkpoint.pt', sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as temporary, patch.dict(weights.WEIGHTS, {'test': (asset,)}):
            cache = Path(temporary)
            for content in (None, b'corrupt', data):
                if content is not None:
                    path = cache / asset.path
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(content)
                if content == data:
                    weights.verify_weights('test', cache)
                else:
                    with self.assertRaisesRegex(ValueError, 'docs/metrics.md#checkpoint-preparation'):
                        weights.verify_weights('test', cache)
