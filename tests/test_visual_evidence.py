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
            model = OpenAICompatVideoModel(base_url='https://example.test/v1', api_key='test', model_name=registry[request.model_name],
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
            model=OpenAICompatVideoModel(base_url='https://example.test/v1',api_key='test',model_name='test',
                transport=httpx.MockTransport(lambda r: self.fail('oversize request reached HTTP')),frame_extractor=lambda *a:[FrameRef(timestamp=0.0,path=frame)])
            self.addCleanup(model.close)
            model.MAX_REQUEST_BYTES = 1500
            with self.assertRaisesRegex(RuntimeError, 'size|budget|limit'):
                model.generate(request)


def test_call_budgets_cache_identity_and_fixed_timeout(tmp_path):
    from unittest.mock import patch
    video = tmp_path / 'clip'; video.write_bytes(b'video')
    frame = tmp_path / 'frame.jpg'; frame.write_bytes(b'jpeg')
    requests = []
    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={'choices': [{'message': {
            'role': 'assistant', 'content': '{"events":[]}'}}]})
    model = OpenAICompatVideoModel(base_url='https://example.test/v1', api_key='test',
        model_name='vendor/model', timeout_seconds=17,
        transport=httpx.MockTransport(respond),
        frame_extractor=lambda *args: [FrameRef(timestamp=0.0, path=frame)])
    request = base.ModelRequest(stage='grounded_draft', video_path=video,
        span=TimeSpan(0, 4), fps=2, prompt='inspect', schema_name='GroundedEvents',
        model_name='local-alias', max_frames=4, max_output_tokens=512)
    try:
        identity = model.semantic_cache_identity(request)
        assert identity['max_frames'] == 4
        assert identity['max_output_tokens'] == 512
        cache = SemanticResultCache(tmp_path / 'cache')
        key = cache.key(request, identity, {})
        for changed in (replace(request, max_frames=8), replace(request, max_output_tokens=1024)):
            assert cache.key(changed, model.semantic_cache_identity(changed), {}) != key
        with patch('percept_harness.models.openai_compat.time.sleep') as sleep:
            assert model.generate(request) == {'events': []}
            sleep.assert_called_once()
        assert len(requests) == 2
        assert all(set(item.extensions['timeout'].values()) == {17} for item in requests)
        payload = json.loads(requests[-1].content)
        assert payload['model'] == 'vendor/model'
        assert payload['max_tokens'] == 512
        assert model.semantic_cache_identity(replace(request, max_output_tokens=None))['max_output_tokens'] == 16384
    finally:
        model.close()


def test_configuration_and_request_boundaries_reject_invalid_values(tmp_path):
    import pytest
    from percept_harness.config import Settings
    from percept_harness.runner import _model_request
    for config in ({'openai_base_url': 'ftp://host'},
                   {'openai_response_format': 'unsupported'},
                   {'openai_timeout_seconds': float('inf')},
                   {'openai_timeout_seconds': 0},
                   {'openai_model_registry': {'valid': ' '}},
                   {'openai_model_registry': {'bad alias': 'model'}}):
        with pytest.raises(ValueError):
            Settings(**config)
    video = tmp_path / 'video'; video.write_bytes(b'video')
    payload = {'video_path': str(video), 'start': 0, 'end': 1, 'fps': 2}
    for field in ('start', 'end', 'fps'):
        for value in (True, '1', None, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                _model_request('grounded_draft', {**payload, field: value}, 'model')
    for key in ('model', 'messages', 'max_tokens', 'response_format'):
        with pytest.raises(RuntimeError, match='reserved keys'):
            OpenAICompatVideoModel(base_url='https://example.test/v1', api_key='test',
                model_name='model', extra_body={key: 'override'})
