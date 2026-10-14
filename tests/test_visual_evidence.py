import base64
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import httpx
from percept_harness.models import base
from percept_harness.models.openai_compat import OpenAICompatVideoModel
from percept_harness.runner import SemanticResultCache
from percept_harness.media import TimeSpan, FrameRef


class VisualEvidenceTests(unittest.TestCase):
    def test_http_transmits_original_and_overlay_and_cache_tracks_content(self):
        self.assertTrue(hasattr(base, 'VisualEvidence'), 'mandatory visual attachment contract is missing')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / 'video'; video.write_bytes(b'video')
            frame = root / 'frame.jpg'; frame.write_bytes(b'jpeg-test')
            png = b'\x89PNG\r\n\x1a\nfixture-a'
            ev = base.VisualEvidence(png_bytes=png, timestamp=1.25, entity_id='cup', track_id='baseline:cup_1', source='baseline')
            from percept_harness.config import Settings
            registry = Settings(openai_model='vendor/model-v1', openai_model_registry={}).openai_model_aliases
            request = base.ModelRequest(stage='grounded_draft', video_path=video, span=TimeSpan(0, 4), fps=2,
                prompt='inspect', schema_name='GroundedEvents', model_name='vendor-model-v1', visual_evidence=(ev,))
            received = []
            def handle(req):
                received.append(json.loads(req.content))
                return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': '{"duration":4,"events":[]}'}}]})
            model = OpenAICompatVideoModel(base_url='https://example.test/v1', api_key='test', model_registry=registry,
                transport=httpx.MockTransport(handle), frame_extractor=lambda *a: [FrameRef(timestamp=0.0, path=frame)])
            self.addCleanup(model.close)
            self.assertEqual(model.generate(request)['events'], [])
            self.assertEqual(received[0]['model'], 'vendor/model-v1')
            content = received[0]['messages'][0]['content']
            images = [x['image_url']['url'] for x in content if x['type'] == 'image_url']
            self.assertEqual(images, ['data:image/jpeg;base64,' + base64.b64encode(b'jpeg-test').decode(),
                                      'data:image/png;base64,' + base64.b64encode(png).decode()])
            self.assertTrue(any('1.25' in x.get('text', '') and 'baseline:cup_1' in x.get('text', '') for x in content))
            cache = SemanticResultCache(root / 'cache')
            key = cache.key(request, {}, {})
            changed = replace(ev, png_bytes=png+b'b')
            self.assertNotEqual(key, cache.key(replace(request, visual_evidence=(changed,)), {}, {}))
            self.assertNotEqual(key, cache.key(replace(request, visual_evidence=(replace(ev, timestamp=2.0),)), {}, {}))

    def test_visual_budget_never_silently_discards_evidence(self):
        self.assertTrue(hasattr(base, 'VisualEvidence'), 'mandatory visual attachment contract is missing')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); video=root/'video'; video.write_bytes(b'v'); frame=root/'f.jpg'; frame.write_bytes(b'jpg')
            ev=base.VisualEvidence(png_bytes=b'\x89PNG\r\n\x1a\n'+b'x'*4000, timestamp=0.0, entity_id='cup', track_id='b:cup', source='b')
            request=base.ModelRequest(stage='grounded_draft',video_path=video,span=TimeSpan(0,4),fps=1,prompt='',schema_name='GroundedEvents',model_name='test',visual_evidence=(ev,))
            model=OpenAICompatVideoModel(base_url='https://example.test/v1',api_key='test',model_registry={'test':'test'},max_request_bytes=1500,
                transport=httpx.MockTransport(lambda r: self.fail('oversize request reached HTTP')),frame_extractor=lambda *a:[FrameRef(timestamp=0.0,path=frame)])
            self.addCleanup(model.close)
            with self.assertRaisesRegex(RuntimeError, 'size|budget|limit'):
                model.generate(request)
