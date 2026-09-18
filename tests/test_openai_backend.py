from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from percept_harness.media import FrameRef, TimeSpan
from percept_harness.models.base import ModelOutputError, ModelRequest
from percept_harness.models.openai_compat import (
    STAGE_MAX_OUTPUT_TOKENS,
    OpenAICompatBackendError,
    OpenAICompatVideoModel,
)
from percept_harness.models.response_contract import (
    CHOICE_CONTRACT,
    ModelResponseContract,
    canonical,
)

BASE_URL = "https://provider.example/v1"


def _request(tmp_path: Path, **changes):
    video = tmp_path / "private-video.mp4"
    video.write_bytes(b"video")
    values = dict(stage="active_objects", video_path=video, span=TimeSpan(2, 4), fps=1,
                  prompt="original prompt", schema_name="active_objects", video_session_id=None,
                  model_name="doubao-pro")
    values.update(changes)
    return ModelRequest(**values)


def _frames(path, span, fps, output):
    assert (span, fps) == (TimeSpan(2, 4), 1)
    first, second = output / "secret-a.jpg", output / "secret-b.jpg"
    first.write_bytes(b"jpeg-a"); second.write_bytes(b"jpeg-b")
    return [FrameRef(first, 2.0), FrameRef(second, 3.0)]


def _envelope(text='{"objects": []}', **changes):
    value = {"choices": [{"finish_reason": "stop",
                          "message": {"role": "assistant", "content": text}}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}
    value.update(changes)
    return value


def _model(handler, **changes):
    values = dict(base_url=BASE_URL, api_key="top-secret",
                  model_registry={"doubao-pro": "doubao-seed-2-1-pro-260628"},
                  transport=httpx.MockTransport(handler), frame_extractor=_frames)
    values.update(changes)
    return OpenAICompatVideoModel(**values)


def _contract():
    schema = canonical({"type": "object", "properties": {"objects": {"type": "array"}},
                        "required": ["objects"], "additionalProperties": False})
    return ModelResponseContract(
        name=CHOICE_CONTRACT, schema_json=schema,
        schema_sha256=hashlib.sha256(schema.encode()).hexdigest())


def test_request_is_visual_only_bounded_and_strict(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        assert request.url == "https://provider.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer top-secret"
        assert "top-secret" not in request.content.decode()
        return httpx.Response(200, json=_envelope())
    model = _model(handler)
    assert model.generate(_request(tmp_path)) == {"objects": []}
    assert model.request_metrics() == {"input_tokens": 12, "output_tokens": 8}
    assert captured["model"] == "doubao-seed-2-1-pro-260628"
    assert captured["stream"] is False
    assert captured["response_format"] == {"type": "json_object"}
    serialized = json.dumps(captured)
    assert "private-video" not in serialized and "secret-a" not in serialized and "audio" not in serialized
    (message,) = captured["messages"]
    assert message["role"] == "user"
    assert [part["text"] for part in message["content"] if part["type"] == "text"] == [
        "original prompt", "Frame timestamp: 2.000000 seconds", "Frame timestamp: 3.000000 seconds"]
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,")
               for part in message["content"] if part["type"] == "image_url")


@pytest.mark.parametrize(("stage", "expected_cap"), sorted(STAGE_MAX_OUTPUT_TOKENS.items()))
def test_request_and_cache_identity_use_the_same_stage_budget(tmp_path, stage, expected_cap):
    """A request/cache cap mismatch can replay output generated under another budget."""
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    model = _model(handler)
    request = _request(tmp_path, stage=stage)
    assert model.generate(request) == {"objects": []}
    identity = model.semantic_cache_identity(request)
    assert captured["max_tokens"] == expected_cap
    assert identity["max_output_tokens"] == expected_cap
    assert identity["adapter_contract_version"] == "openai-compat-chat-completions-v1"
    assert identity["endpoint"] == "https://provider.example/v1/chat/completions"


def test_base_url_trailing_slash_is_normalized(tmp_path):
    def handler(request):
        assert request.url == "https://provider.example/v1/chat/completions"
        return httpx.Response(200, json=_envelope())
    model = _model(handler, base_url=BASE_URL + "/")
    assert model.generate(_request(tmp_path)) == {"objects": []}


def test_json_schema_mode_forwards_the_contract(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    contract = _contract()
    model = _model(handler, response_format="json_schema")
    result = model.generate(_request(tmp_path, response_contract=contract))
    assert result == {"objects": []}
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["name"] == CHOICE_CONTRACT
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"]["type"] == "object"
    identity = model.semantic_cache_identity(_request(tmp_path, response_contract=contract))
    assert identity["response_format"]["mode"] == "json_schema"
    assert identity["response_format"]["contract"] == contract.cache_identity()


def test_json_schema_mode_without_contract_falls_back_to_json_object(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    model = _model(handler, response_format="json_schema")
    assert model.generate(_request(tmp_path)) == {"objects": []}
    assert captured["response_format"] == {"type": "json_object"}


def test_none_mode_sends_no_response_format(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    model = _model(handler, response_format="none")
    assert model.generate(_request(tmp_path)) == {"objects": []}
    assert "response_format" not in captured


def test_extra_body_and_headers_reach_the_provider(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        assert request.headers["x-provider-tenant"] == "team-a"
        return httpx.Response(200, json=_envelope())
    model = _model(handler, extra_body={"thinking": {"type": "disabled"}},
                   extra_headers={"X-Provider-Tenant": "team-a"})
    assert model.generate(_request(tmp_path)) == {"objects": []}
    assert captured["thinking"] == {"type": "disabled"}
    identity = model.semantic_cache_identity(_request(tmp_path))
    assert json.loads(identity["extra_body"]) == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("key", ["model", "messages", "stream", "max_tokens", "response_format"])
def test_extra_body_must_not_override_reserved_keys(key):
    with pytest.raises(OpenAICompatBackendError):
        _model(lambda request: httpx.Response(200, json=_envelope()),
               extra_body={key: "override"})


@pytest.mark.parametrize("kwargs", [
    dict(api_key=" "),
    dict(base_url="ftp://provider.example"),
    dict(base_url=""),
    dict(response_format="yaml"),
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(OpenAICompatBackendError):
        _model(lambda request: httpx.Response(200, json=_envelope()), **kwargs)


def test_unknown_stage_and_unlisted_alias_fail_before_transport(tmp_path):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_envelope())
    model = _model(handler)
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_request(tmp_path, stage="mystery_stage"))
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_request(tmp_path, model_name="unlisted"))
    assert calls == 0


def test_service_failure_is_sanitized(tmp_path):
    model = _model(lambda request: httpx.Response(500, text="secret internals"))
    with pytest.raises(OpenAICompatBackendError) as info:
        model.generate(_request(tmp_path))
    assert "secret internals" not in str(info.value)
    assert "500" in str(info.value)


@pytest.mark.parametrize("envelope", [
    {"choices": []},
    {"choices": [{"finish_reason": "length",
                  "message": {"role": "assistant", "content": '{"objects": []}'}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "user", "content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None}}]},
    _envelope(usage={"prompt_tokens": -1, "completion_tokens": 8}),
])
def test_malformed_envelopes_are_rejected(tmp_path, envelope):
    model = _model(lambda request: httpx.Response(200, json=envelope))
    with pytest.raises(ModelOutputError):
        model.generate(_request(tmp_path))


def test_partial_usage_is_tolerated(tmp_path):
    model = _model(lambda request: httpx.Response(
        200, json=_envelope(usage={"completion_tokens": 8})))
    assert model.generate(_request(tmp_path)) == {"objects": []}
    assert model.request_metrics() == {"output_tokens": 8}


def test_missing_usage_is_tolerated(tmp_path):
    envelope = _envelope()
    del envelope["usage"]
    model = _model(lambda request: httpx.Response(200, json=envelope))
    assert model.generate(_request(tmp_path)) == {"objects": []}
    assert model.request_metrics() == {}


def test_duplicate_keys_in_model_output_are_rejected(tmp_path):
    model = _model(lambda request: httpx.Response(
        200, json=_envelope(text='{"objects": [], "objects": []}')))
    with pytest.raises(ModelOutputError):
        model.generate(_request(tmp_path))


def test_fenced_output_is_accepted(tmp_path):
    model = _model(lambda request: httpx.Response(
        200, json=_envelope(text='```json\n{"objects": []}\n```')))
    assert model.generate(_request(tmp_path)) == {"objects": []}


def test_frame_bounds_are_enforced(tmp_path):
    def no_frames(path, span, fps, output):
        return []
    model = _model(lambda request: httpx.Response(200, json=_envelope()),
                   frame_extractor=no_frames)
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_request(tmp_path))

    def too_many(path, span, fps, output):
        frames = []
        for index in range(3):
            frame = output / f"frame-{index}.jpg"
            frame.write_bytes(b"jpeg")
            frames.append(FrameRef(frame, 2.0 + index * 0.5))
        return frames
    model = _model(lambda request: httpx.Response(200, json=_envelope()),
                   frame_extractor=too_many, max_frames=2)
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_request(tmp_path))


def test_oversized_request_is_rejected_before_transport(tmp_path):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_envelope())
    model = _model(handler, max_request_bytes=64)
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_request(tmp_path))
    assert calls == 0
