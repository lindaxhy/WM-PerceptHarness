import base64
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
from percept_harness.models.openai_compat import OpenAICompatVideoModel, OpenAICompatBackendError


class RequestImageTests(unittest.TestCase):
    def payload(self):
        # Uncompressed PNG models bulky mandatory overlays; no resolution loss.
        image = Image.new('RGB', (256, 256), 'red')
        out = io.BytesIO()
        image.save(out, format='PNG', compress_level=0)
        return [{'type': 'text', 'text': 'timestamp=2; track=original'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()}}]

    def backend(self, limit):
        backend = OpenAICompatVideoModel(base_url='https://example.invalid', api_key='test', model_name='test',
            frame_extractor=lambda *args: [object()])
        backend.MAX_REQUEST_BYTES = limit
        backend._content = Mock(return_value=self.payload())
        backend._post_with_retries = Mock(return_value=b'{}')
        backend._parse = lambda raw: json.loads(raw)
        return backend

    def request(self):
        return SimpleNamespace(span=SimpleNamespace(start=0,end=4), visual_evidence=(), max_frames=128,
            fps=1, video_path=Path('unused'), max_output_tokens=20, stage='draft', response_contract=None)

    def test_oversized_overlay_compacted_without_losing_metadata_or_dimensions(self):
        backend = self.backend(10000)
        self.assertEqual(backend.generate(self.request()), {})
        raw = backend._post_with_retries.call_args.args[0]
        self.assertLessEqual(len(raw), 10000)
        content = json.loads(raw)['messages'][0]['content']
        self.assertEqual(content[0]['text'], 'timestamp=2; track=original')
        self.assertEqual(len(content), 2)
        image = Image.open(io.BytesIO(base64.b64decode(content[1]['image_url']['url'].split(',')[1])))
        self.assertEqual(image.size, (256,256))
        self.assertEqual(image.format, 'JPEG')

    def test_still_oversized_never_sent(self):
        backend = self.backend(10)
        with self.assertRaises(OpenAICompatBackendError):
            backend.generate(self.request())
        backend._post_with_retries.assert_not_called()

    def test_small_payload_keeps_original_encoding(self):
        backend = self.backend(1000000)
        backend.generate(self.request())
        raw = backend._post_with_retries.call_args.args[0]
        self.assertIn(b'data:image/png;base64,', raw)
