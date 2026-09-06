from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from las_repro.media import FrameRef, TimeSpan
from las_repro.models.base import ModelOutputError, ModelRequest


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
    value = {"status": "completed", "model": "doubao-seed-2-1-pro-260628",
             "output": [{"type": "reasoning", "summary": []},
                        {"type": "message", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": text}]}],
             "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}}
    value.update(changes)
    return value


def _model(handler, **changes):
    from las_repro.models.ark import ArkVideoModel
    values = dict(api_key="top-secret", model_registry={"doubao-pro": "doubao-seed-2-1-pro-260628"},
                  transport=httpx.MockTransport(handler), frame_extractor=_frames)
    values.update(changes)
    return ArkVideoModel(**values)


def test_request_is_visual_only_bounded_and_strict(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        assert request.url == "https://ark.cn-beijing.volces.com/api/v3/responses"
        assert request.headers["authorization"] == "Bearer top-secret"
        assert "top-secret" not in request.content.decode()
        return httpx.Response(200, json=_envelope())
    model = _model(handler)
    assert model.generate(_request(tmp_path)) == {"objects": []}
    assert model.request_metrics() == {"input_tokens": 12, "output_tokens": 8}
    assert captured["store"] is False
    assert captured["model"] == "doubao-seed-2-1-pro-260628"
    assert captured["thinking"] == {"type": "disabled"}
    serialized = json.dumps(captured)
    assert "private-video" not in serialized and "secret-a" not in serialized and "audio" not in serialized
    content = captured["input"][0]["content"]
    assert [part["text"] for part in content if part["type"] == "input_text"] == [
        "original prompt", "Frame timestamp: 2.000000 seconds", "Frame timestamp: 3.000000 seconds"]
    assert all(part["image_url"].startswith("data:image/jpeg;base64,") for part in content if part["type"] == "input_image")
    assert all("image_pixel_limit" not in part for part in content)


@pytest.mark.parametrize(
    ("stage", "expected_cap"),
    [
        ("active_objects", 1_024),
        ("general_segment", 4_096),
        ("general_summary", 2_048),
        ("embodied_pass_a", 4_096),
        ("embodied_pass_b", 8_192),
        ("embodied_enrichment", 4_096),
        ("scene_semantics", 8_192),
        ("occlusion_semantics", 4_096),
    ],
)
def test_ark_request_and_cache_identity_use_the_same_stage_budget(
    tmp_path, stage, expected_cap
):
    """A request/cache cap mismatch can replay output generated under another budget."""
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())

    model = _model(handler)
    request = _request(tmp_path, stage=stage)
    assert model.generate(request) == {"objects": []}
    identity = model.semantic_cache_identity(request)

    assert captured["max_output_tokens"] == expected_cap
    assert identity["max_output_tokens"] == expected_cap


def test_ark_cache_identity_uses_the_scene_budget_contract(tmp_path):
    """A stale adapter contract can replay entries created before the budget change."""
    model = _model(lambda request: httpx.Response(200, json=_envelope()))
    identity = model.semantic_cache_identity(
        _request(tmp_path, stage="scene_semantics")
    )

    assert identity["adapter_contract_version"] == "ark-responses-scene-budget-v2"


def test_unknown_stage_fails_before_visual_extraction_or_transport(tmp_path):
    """An absent finite stage budget must not leak work into external boundaries."""
    extraction_calls = transport_calls = 0

    def extract(*args):
        nonlocal extraction_calls
        extraction_calls += 1
        return _frames(*args)

    def handler(request):
        nonlocal transport_calls
        transport_calls += 1
        return httpx.Response(200, json=_envelope())

    from las_repro.models.ark import ArkBackendError

    model = _model(handler, frame_extractor=extract)
    with pytest.raises(ArkBackendError, match="no configured output budget"):
        model.generate(_request(tmp_path, stage="unsupported_stage"))
    assert extraction_calls == 0
    assert transport_calls == 0


@pytest.mark.parametrize("resolution,maximum", [
    ("low", 65536), ("medium", 131072), ("high", 262144),
])
def test_requested_resolution_reaches_ark_without_changing_frames(tmp_path, resolution, maximum):
    """Ignoring the caller's resolution silently sends unrestricted image tokens."""
    import base64

    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())

    model = _model(handler)
    try:
        model.generate(_request(tmp_path, media_resolution=resolution))
    finally:
        model.close()
    content = captured["input"][0]["content"]
    images = [part for part in content if part["type"] == "input_image"]
    assert [part.get("image_pixel_limit") for part in images] == [
        {"min_pixels": 4096, "max_pixels": maximum},
        {"min_pixels": 4096, "max_pixels": maximum},
    ]
    assert [base64.b64decode(part["image_url"].split(",", 1)[1]) for part in images] == [
        b"jpeg-a", b"jpeg-b",
    ]
    assert [part["text"] for part in content if part["type"] == "input_text"] == [
        "original prompt", "Frame timestamp: 2.000000 seconds", "Frame timestamp: 3.000000 seconds",
    ]


@pytest.mark.parametrize("envelope", [
    _envelope("not json"), _envelope('{"x":1,"x":2}'), _envelope(status="incomplete"),
    _envelope(model="foreign"), _envelope(output=[]),
    _envelope(output=[_envelope()["output"][1], _envelope()["output"][1]]),
    _envelope(usage={"input_tokens": -1, "output_tokens": 8}),
])
def test_invalid_provider_output_is_model_output_error_and_clears_metrics(tmp_path, envelope):
    model = _model(lambda request: httpx.Response(200, json=envelope))
    with pytest.raises(ModelOutputError): model.generate(_request(tmp_path))
    assert model.request_metrics() == {}


def test_length_incomplete_response_is_never_accepted_or_repaired_posthoc(tmp_path):
    """Partial JSON at the provider token limit must remain a generation failure."""
    envelope = _envelope(
        '{"objects":[],"locations":[',
        status="incomplete",
        incomplete_details={"reason": "length"},
    )
    model = _model(lambda request: httpx.Response(200, json=envelope))

    with pytest.raises(ModelOutputError, match="identity or status is invalid"):
        model.generate(_request(tmp_path, stage="scene_semantics"))

    assert model.request_metrics() == {}


@pytest.mark.parametrize("response", [
    httpx.Response(401, text="secret body"), httpx.Response(429, text="secret body"),
    httpx.Response(500, text="secret body"), httpx.Response(302, headers={"location": "https://evil.test"}),
])
def test_service_failures_are_sanitized(tmp_path, response):
    from las_repro.models.ark import ArkBackendError
    model = _model(lambda request: response)
    with pytest.raises(ArkBackendError) as caught: model.generate(_request(tmp_path))
    message = str(caught.value)
    assert "secret" not in message and "private-video" not in message and "top-secret" not in message


def test_transport_failure_is_sanitized(tmp_path):
    from las_repro.models.ark import ArkBackendError
    def fail(request): raise httpx.ReadTimeout("top-secret /private-video.mp4")
    with pytest.raises(ArkBackendError, match="transport failed"):
        _model(fail).generate(_request(tmp_path))


@pytest.mark.parametrize("frames", [[], [FrameRef(Path("a"), 1.0)],
    [FrameRef(Path("a"), 3.0), FrameRef(Path("b"), 2.0)]])
def test_invalid_frames_fail_before_transport(tmp_path, frames):
    calls = 0
    def handler(request):
        nonlocal calls; calls += 1
        return httpx.Response(200, json=_envelope())
    model = _model(handler, frame_extractor=lambda *args: frames)
    with pytest.raises(Exception): model.generate(_request(tmp_path))
    assert calls == 0


def test_frame_and_request_size_limits_fail_before_transport(tmp_path):
    calls = 0
    def handler(request):
        nonlocal calls; calls += 1
        return httpx.Response(200, json=_envelope())
    with pytest.raises(Exception): _model(handler, max_frames=1).generate(_request(tmp_path))
    with pytest.raises(Exception): _model(handler, max_request_bytes=10).generate(_request(tmp_path))
    assert calls == 0


def test_close_closes_owned_client_and_is_idempotent(tmp_path):
    model = _model(lambda request: httpx.Response(200, json=_envelope()))
    model.generate(_request(tmp_path)); model.close(); model.close()
    with pytest.raises(Exception): model.generate(_request(tmp_path))


@pytest.mark.parametrize("text", [
    '```\n{"x":1,"x":2}\n```',
    '```JSON\n{"x":1,"x":2}\n```',
])
def test_every_supported_fence_rejects_duplicate_keys(tmp_path, text):
    model = _model(lambda request: httpx.Response(200, json=_envelope(text)))
    with pytest.raises(ModelOutputError, match="duplicate"):
        model.generate(_request(tmp_path))


@pytest.mark.parametrize("changes", [
    {"output": None}, {"output": 7},
    {"output": [{"type": "message", "role": "assistant", "content": None}]},
    {"output": [{"type": "message", "role": "assistant", "content": 7}]},
])
def test_malformed_output_containers_are_repairable_model_errors(tmp_path, changes):
    model = _model(lambda request: httpx.Response(200, json=_envelope(**changes)))
    with pytest.raises(ModelOutputError):
        model.generate(_request(tmp_path))


def test_injected_client_still_uses_configured_timeout_and_refuses_redirect(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(302, headers={"location": "https://evil.test"})
        return httpx.Response(200, json=_envelope())
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True,
                          timeout=999, trust_env=False)
    model = _model(handler, client=client, transport=None, timeout_seconds=3.5)
    from las_repro.models.ark import ArkBackendError
    with pytest.raises(ArkBackendError): model.generate(_request(tmp_path))
    assert len(calls) == 1
    assert calls[0].extensions["timeout"]["read"] == 3.5
    client.close()


def test_deeply_nested_json_is_a_repairable_model_output_error(tmp_path):
    text = '{"value":' + "[" * 997 + "0" + "]" * 997 + "}"
    model = _model(lambda request: httpx.Response(200, json=_envelope(text)))
    with pytest.raises(ModelOutputError, match="valid JSON"):
        model.generate(_request(tmp_path))
