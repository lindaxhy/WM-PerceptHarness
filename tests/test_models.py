"""Models behavior and regression tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from percept_harness.config import Settings
from percept_harness.media import FrameRef, TimeSpan
from percept_harness.models.base import (
    ModelOutputError,
    ModelRequest,
    ModelRequestError,
    parse_strict_json,
)
from percept_harness.models.fake import (
    FakeVideoModel,
    UnknownModelStageError,
    _split_at_visible_changes,
)
from percept_harness.models.openai_compat import (
    OpenAICompatBackendError,
    OpenAICompatVideoModel,
    STAGE_MAX_OUTPUT_TOKENS,
)
from percept_harness.models.response_contract import (
    CHOICE_CONTRACT,
    ModelResponseContract,
    canonical,
)
from percept_harness.pipelines.embodied import PromptRenderer


@pytest.mark.parametrize(
    "text",
    [
        '{"segments": []}',
        '```json\n{"segments": []}\n```',
    ],
)
def test_parser_accepts_one_json_object_with_an_optional_complete_fence(text):
    """Removing the optional complete fence must still leave one object."""
    assert parse_strict_json(text) == {"segments": []}


@pytest.mark.parametrize(
    "text",
    [
        'before {"segments": []}',
        '{"segments": []} after',
        '{"a": 1}{"b": 2}',
        '```json\n{"segments": []}',
        '[{"segments": []}]',
    ],
)
def test_parser_rejects_prose_multiple_objects_incomplete_fences_and_arrays(text):
    """Accepting any of these would let unvalidated model prose reach a pipeline."""
    with pytest.raises(ModelOutputError):
        parse_strict_json(text)


@pytest.mark.parametrize("text", ['{"score": NaN}', '{"score": Infinity}'])
def test_parser_rejects_non_finite_json_numbers(text):
    """A numeric confidence or timestamp must never become NaN or infinity."""
    with pytest.raises(ModelOutputError, match="finite"):
        parse_strict_json(text)


def test_parser_normalizes_the_json_integer_digit_limit_error():
    """A decoder numeric limit must not escape as a raw implementation error."""
    text = '{"integer": ' + "9" * 5_000 + "}"

    with pytest.raises(ModelOutputError, match=r"^model output is not valid JSON$"):
        parse_strict_json(text)


def test_parser_normalizes_deep_json_nesting_errors():
    """A below-limit nested model object must use the parser's public error type."""
    depth = 998
    text = '{"value":' * depth + "0" + "}" * depth

    with pytest.raises(ModelOutputError, match=r"^model output is not valid JSON$"):
        parse_strict_json(text)


def test_parser_enforces_the_configurable_output_size_bound():
    """Oversized output must fail before the JSON decoder allocates more work."""
    settings = Settings(max_model_output_chars=5)
    with pytest.raises(ModelOutputError, match="maximum"):
        parse_strict_json('{"segments": []}', max_chars=settings.max_model_output_chars)


def test_model_request_is_typed_and_immutable():
    """Workers must not be able to mutate a claimed request's path or session."""
    request = _request("general_segment")

    assert request.video_path == Path("/media/demo.mp4")
    assert request.video_session_id == "task-1"
    with pytest.raises(FrozenInstanceError):
        request.video_session_id = "other"  # type: ignore[misc]
    with pytest.raises(ModelRequestError, match="video_path"):
        ModelRequest(
            stage="general_segment",
            video_path="/media/demo.mp4",  # type: ignore[arg-type]
            span=TimeSpan(0.0, 1.0),
            fps=2.0,
            prompt="return JSON",
            schema_name="general_segment",
            video_session_id="task-1",
        )


def test_model_request_carries_validated_local_inference_tuning() -> None:
    """Persisted local controls must survive reconstruction at the model boundary."""
    request = replace(
        _request("active_objects"),
        media_resolution="high",
        reasoning_effort="low",
        clip_context="medium",
    )

    assert request.media_resolution == "high"
    assert request.reasoning_effort == "low"
    assert request.clip_context == "medium"
    assert FakeVideoModel().generate(request)["objects"]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("media_resolution", "ultra"),
        ("reasoning_effort", "maximum"),
        ("clip_context", 1),
        ("clip_context", ["low"]),
    ],
)
def test_model_request_rejects_invalid_local_inference_tuning(
    field: str, invalid: object
) -> None:
    """Corrupt durable rows must not widen the canonical tuning value set."""
    with pytest.raises(ModelRequestError, match=field):
        replace(_request("active_objects"), **{field: invalid})


def test_fake_general_segment_is_deterministic_for_the_requested_span():
    """Changing the requested interval must change the emitted event bounds."""
    model = FakeVideoModel()

    result = model.generate(_request("general_segment", start=1.0, end=2.5))

    assert result == {
        "segments": [
            {
                "start_time": 1.0,
                "end_time": 2.5,
                "scene": ["deterministic indoor scene"],
                "subjects": ["deterministic visible subject"],
                "actions": ["deterministic visible action"],
                "visible_text": [],
                "uncertainty": [],
                "description": "deterministic visual event",
                "warnings": [],
            }
        ],
        "warnings": [],
    }
    assert model.generate(_request("general_segment", start=1.0, end=2.5)) == result


def test_fake_embodied_stages_emit_complete_deterministic_shapes():
    """The offline backend must exercise every later embodied pipeline stage."""
    model = FakeVideoModel()
    full_video = _request("embodied_pass_a", end=2.0)

    active_objects = model.generate(_request("active_objects", end=2.0))
    coarse = model.generate(full_video)
    boundaries = model.generate(_request("embodied_pass_b", end=2.0))
    enrichment = model.generate(
        _request(
            "embodied_enrichment",
            end=2.0,
            prompt='{"segments": [{"segment_index": 3}, {"segment_index": 7}]}',
        )
    )
    summary = model.generate(_request("general_summary", end=2.0))

    assert active_objects == {
        "objects": [
            {
                "category": "container",
                "instances": [
                    {
                        "name": "red container",
                        "description": "a visible red rectangular container",
                    }
                ],
            }
        ]
    }
    assert [(action["start"], action["end"]) for action in coarse["actions"]] == [
        (0.0, 1.0),
        (1.0, 2.0),
    ]
    assert all(
        segment["end"] - segment["start"] <= 1.0
        for action in boundaries["actions"]
        for segment in action["fine_segments"]
    )
    assert [item["segment_index"] for item in enrichment["segments"]] == [3, 7]
    assert all(
        set(item) == {
            "segment_index",
            "actor",
            "actor_state",
            "skill",
            "target",
            "visual_motion_state",
            "confidence",
        }
        for item in enrichment["segments"]
    )
    assert summary["summary"]
    assert summary["timeline"] == sorted(summary["timeline"], key=lambda item: item["start_time"])


def test_fake_enrichment_prefers_trusted_nonzero_gapped_skeleton_indices() -> None:
    """The later one-row schema example must not append index zero to trusted rows."""
    rows = [
        {
            "segment_index": index,
            "start": float(position),
            "end": float(position + 1),
            "description": "right hand moves red container",
        }
        for position, index in enumerate((2, 7, 11))
    ]
    prompt = PromptRenderer().enrichment(
        rows,
        expected_indices=[2, 7, 11],
    )

    result = FakeVideoModel().generate(
        _request("embodied_enrichment", prompt=prompt)
    )

    assert [record["segment_index"] for record in result["segments"]] == [
        2,
        7,
        11,
    ]


def test_fake_enrichment_rejects_a_present_but_malformed_trusted_skeleton() -> None:
    """Legacy regex fallback must not hide corruption in a trusted prompt block."""
    prompt = """[trusted enrichment output requirements JSON data]
trusted data follows
{"exact_record_count":2,"expected_indices":[2,7],"record_skeleton":[]}

[task]
schema example only: {"segment_index":0}
"""

    with pytest.raises(ModelRequestError, match="enrichment requirements"):
        FakeVideoModel().generate(
            _request("embodied_enrichment", prompt=prompt)
        )


def test_fake_enrichment_rejects_boolean_confidence_in_trusted_skeleton() -> None:
    """JSON bool/number equality must not make a malformed trusted row look valid."""
    prompt = PromptRenderer().enrichment(
        [
            {
                "segment_index": 2,
                "start": 0.0,
                "end": 1.0,
                "description": "right hand moves red container",
            }
        ],
        expected_indices=[2],
    ).replace('"confidence":0.0', '"confidence":false', 1)

    with pytest.raises(ModelRequestError, match="enrichment requirements"):
        FakeVideoModel().generate(
            _request("embodied_enrichment", prompt=prompt)
        )


def test_fake_failure_script_is_fifo_and_does_not_mutate_caller_data():
    """A repair test needs exactly one scripted bad response before the good one."""
    invalid = {"actions": []}
    valid = {"actions": [{"action_index": 0}]}
    script = {"embodied_pass_b": [invalid, valid]}
    model = FakeVideoModel(failure_script=script)

    first = model.generate(_request("embodied_pass_b"))
    first["actions"].append({"mutated": True})
    second = model.generate(_request("embodied_pass_b"))
    third = model.generate(_request("embodied_pass_b"))

    assert script == {"embodied_pass_b": [invalid, valid]}
    assert second == valid
    assert third["actions"] != first["actions"]


def test_fake_rejects_unknown_stages_with_a_clear_error():
    """A misspelled pipeline stage must not silently receive an unrelated fixture."""
    with pytest.raises(UnknownModelStageError, match="unsupported fake model stage: unknown"):
        FakeVideoModel().generate(_request("unknown"))


def test_fake_pass_b_uses_deterministic_nonuniform_visible_change_boundaries():
    result = FakeVideoModel().generate(
        _request("embodied_pass_b", start=0.0, end=2.0)
    )
    intervals = [
        (segment["start"], segment["end"])
        for action in result["actions"]
        for segment in action["fine_segments"]
    ]

    assert intervals == [
        (0.0, 0.41),
        (0.41, 1.0),
        (1.0, 1.63),
        (1.63, 2.0),
    ]
    durations = [end - start for start, end in intervals]
    assert len({round(duration, 9) for duration in durations}) > 1
    assert all(duration <= 1.0 for duration in durations)


@pytest.mark.parametrize("duration", [5e-324, 0.2, 1.0, 1.000001, 2.2, 7.4])
def test_fake_pass_b_partitions_long_spans_without_gaps_or_grid_padding(
    duration: float,
) -> None:
    """A shifted zip or rounded grid would lose an endpoint or raise here."""
    first = _split_at_visible_changes(0.0, duration, phase=0)
    second = _split_at_visible_changes(0.0, duration, phase=0)
    intervals = first

    assert first == second
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == duration
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert all(0.0 < end - start <= 1.0 for start, end in intervals)
    if duration > 1.0:
        assert any(
            boundary not in {0.0, duration} and boundary % 1.0 != 0.0
            for _, boundary in intervals
        )


def _request(
    stage: str,
    *,
    start: float = 0.0,
    end: float = 1.0,
    prompt: str = "return JSON",
) -> ModelRequest:
    return ModelRequest(
        stage=stage,
        video_path=Path("/media/demo.mp4"),
        span=TimeSpan(start, end),
        fps=2.0,
        prompt=prompt,
        schema_name=stage,
        video_session_id="task-1",
    )


BASE_URL = "https://provider.example/v1"


def _openai_request(tmp_path: Path, **changes):
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
                  transport=httpx.MockTransport(handler), frame_extractor=_frames,
                  transport_retries=0)
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
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
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
    request = _openai_request(tmp_path, stage=stage)
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
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}


def test_json_schema_mode_forwards_the_contract(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    contract = _contract()
    model = _model(handler, response_format="json_schema")
    result = model.generate(_openai_request(tmp_path, response_contract=contract))
    assert result == {"objects": []}
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["name"] == CHOICE_CONTRACT
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"]["type"] == "object"
    identity = model.semantic_cache_identity(_openai_request(tmp_path, response_contract=contract))
    assert identity["response_format"]["mode"] == "json_schema"
    assert identity["response_format"]["contract"] == contract.cache_identity()


def test_json_schema_mode_without_contract_falls_back_to_json_object(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    model = _model(handler, response_format="json_schema")
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
    assert captured["response_format"] == {"type": "json_object"}


def test_none_mode_sends_no_response_format(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    model = _model(handler, response_format="none")
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
    assert "response_format" not in captured


def test_extra_body_and_headers_reach_the_provider(tmp_path):
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        assert request.headers["x-provider-tenant"] == "team-a"
        return httpx.Response(200, json=_envelope())
    model = _model(handler, extra_body={"thinking": {"type": "disabled"}},
                   extra_headers={"X-Provider-Tenant": "team-a"})
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
    assert captured["thinking"] == {"type": "disabled"}
    identity = model.semantic_cache_identity(_openai_request(tmp_path))
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
        model.generate(_openai_request(tmp_path, stage="mystery_stage"))
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_openai_request(tmp_path, model_name="unlisted"))
    assert calls == 0


def test_service_failure_is_sanitized(tmp_path):
    model = _model(lambda request: httpx.Response(500, text="secret internals"))
    with pytest.raises(OpenAICompatBackendError) as info:
        model.generate(_openai_request(tmp_path))
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
        model.generate(_openai_request(tmp_path))


def test_partial_usage_is_tolerated(tmp_path):
    model = _model(lambda request: httpx.Response(
        200, json=_envelope(usage={"completion_tokens": 8})))
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
    assert model.request_metrics() == {"output_tokens": 8}


def test_missing_usage_is_tolerated(tmp_path):
    envelope = _envelope()
    del envelope["usage"]
    model = _model(lambda request: httpx.Response(200, json=envelope))
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
    assert model.request_metrics() == {}


def test_duplicate_keys_in_model_output_are_rejected(tmp_path):
    model = _model(lambda request: httpx.Response(
        200, json=_envelope(text='{"objects": [], "objects": []}')))
    with pytest.raises(ModelOutputError):
        model.generate(_openai_request(tmp_path))


def test_fenced_output_is_accepted(tmp_path):
    model = _model(lambda request: httpx.Response(
        200, json=_envelope(text='```json\n{"objects": []}\n```')))
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}


def test_frame_bounds_are_enforced(tmp_path):
    def no_frames(path, span, fps, output):
        return []
    model = _model(lambda request: httpx.Response(200, json=_envelope()),
                   frame_extractor=no_frames)
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_openai_request(tmp_path))

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
        model.generate(_openai_request(tmp_path))


def test_oversized_request_is_rejected_before_transport(tmp_path):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_envelope())
    model = _model(handler, max_request_bytes=64)
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_openai_request(tmp_path))
    assert calls == 0


def test_transient_status_is_retried_until_success(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr("percept_harness.models.openai_compat.time.sleep", delays.append)
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, text="throttled")
        return httpx.Response(200, json=_envelope())
    model = _model(handler, transport_retries=3, retry_backoff_seconds=0.01)
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
    assert calls == 3
    assert len(delays) == 2
    assert delays[1] > delays[0]  # exponential backoff


def test_retries_exhausted_raises_with_attempt_count(tmp_path, monkeypatch):
    monkeypatch.setattr("percept_harness.models.openai_compat.time.sleep", lambda _: None)
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="secret internals")
    model = _model(handler, transport_retries=2, retry_backoff_seconds=0.01)
    with pytest.raises(OpenAICompatBackendError) as info:
        model.generate(_openai_request(tmp_path))
    assert calls == 3
    assert "3 attempts" in str(info.value)
    assert "503" in str(info.value)
    assert "secret internals" not in str(info.value)


def test_timeout_is_retried_and_cause_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("percept_harness.models.openai_compat.time.sleep", lambda _: None)
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("read deadline")
    model = _model(handler, transport_retries=1, retry_backoff_seconds=0.01)
    with pytest.raises(OpenAICompatBackendError) as info:
        model.generate(_openai_request(tmp_path))
    assert calls == 2
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


def test_deterministic_status_is_never_retried(tmp_path):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="bad request")
    model = _model(handler, transport_retries=3)
    with pytest.raises(OpenAICompatBackendError):
        model.generate(_openai_request(tmp_path))
    assert calls == 1


def test_timeout_scales_with_span_and_is_capped(tmp_path):
    captured = {}
    def handler(request):
        captured["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json=_envelope())
    model = _model(handler, timeout_seconds=10.0,
                   timeout_seconds_per_video_second=2.0, max_timeout_seconds=100.0)
    model.generate(_openai_request(tmp_path))  # span is (2, 4): 10 + 2*2 = 14
    assert captured["timeout"]["read"] == pytest.approx(14.0)

    model = _model(handler, timeout_seconds=10.0,
                   timeout_seconds_per_video_second=100.0, max_timeout_seconds=50.0)
    model.generate(_openai_request(tmp_path))  # 10 + 100*2 = 210, capped at 50
    assert captured["timeout"]["read"] == pytest.approx(50.0)


def test_run_identity_includes_retry_and_timeout_settings(tmp_path):
    model = _model(lambda request: httpx.Response(200, json=_envelope()),
                   transport_retries=2, timeout_seconds_per_video_second=1.5)
    identity = model.run_identity("doubao-pro")
    assert identity["transport_retries"] == 2
    assert identity["timeout_seconds_per_video_second"] == 1.5


@pytest.mark.parametrize("kwargs", [
    {"transport_retries": -1},
    {"transport_retries": True},
    {"retry_backoff_seconds": 0},
    {"timeout_seconds_per_video_second": -0.5},
    {"max_timeout_seconds": 1.0, "timeout_seconds": 2.0},
])
def test_invalid_retry_and_timeout_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _model(lambda request: httpx.Response(200, json=_envelope()), **kwargs)


def test_long_span_downsamples_to_frame_budget_instead_of_failing(tmp_path):
    """fps * span above max_frames spreads the budget instead of erroring."""
    seen = {}
    def extractor(path, span, fps, output):
        seen["fps"] = fps
        frames = []
        count = int((span.end - span.start) * fps) + 1
        for index in range(count):
            frame = output / f"frame-{index}.jpg"
            frame.write_bytes(b"jpeg")
            frames.append(FrameRef(frame, span.start + index / fps))
        return [f for f in frames if f.timestamp < span.end]
    model = _model(lambda request: httpx.Response(200, json=_envelope()),
                   frame_extractor=extractor, max_frames=16)
    # span (0, 60) at 3 fps would need 180 frames; budget is 16.
    request = _openai_request(tmp_path, span=TimeSpan(0, 60), fps=3)
    assert model.generate(request) == {"objects": []}
    assert seen["fps"] * 60 <= 16


def test_oversized_payload_thins_frames_instead_of_failing(tmp_path):
    """A dense clip over the byte budget resamples frames evenly, keeping order."""
    def many_frames(path, span, fps, output):
        frames = []
        for index in range(64):
            frame = output / f"frame-{index}.jpg"
            frame.write_bytes(b"j" * 4096)  # 64 * ~5.5KiB base64 ≈ 470KiB payload
            frames.append(FrameRef(frame, span.start + index * 0.03))
        return frames
    captured = {}
    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_envelope())
    model = _model(handler, frame_extractor=many_frames, max_request_bytes=120_000)
    assert model.generate(_openai_request(tmp_path)) == {"objects": []}
    images = [part for part in captured["messages"][0]["content"]
              if part.get("type") == "image_url"]
    assert 8 <= len(images) < 64
    stamps = [part["text"] for part in captured["messages"][0]["content"]
              if part.get("type") == "text" and part["text"].startswith("Frame timestamp")]
    assert stamps == sorted(stamps)


def test_truly_oversized_request_still_fails(tmp_path):
    """Thinning stops at the frame floor; a still-oversized request errors."""
    def big_frames(path, span, fps, output):
        frames = []
        for index in range(9):
            frame = output / f"frame-{index}.jpg"
            frame.write_bytes(b"j" * 65536)
            frames.append(FrameRef(frame, span.start + index * 0.1))
        return frames
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_envelope())
    model = _model(handler, frame_extractor=big_frames, max_request_bytes=4096)
    with pytest.raises(OpenAICompatBackendError, match="size limit"):
        model.generate(_openai_request(tmp_path))
    assert calls == 0
