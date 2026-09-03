"""Behavioral tests for the local video-model boundary and offline fake."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from las_repro.config import Settings
from las_repro.media import TimeSpan
from las_repro.models.base import (
    ModelOutputError,
    ModelRequest,
    ModelRequestError,
    parse_strict_json,
)
from las_repro.models.fake import (
    FakeVideoModel,
    UnknownModelStageError,
    _split_at_visible_changes,
)
from las_repro.pipelines.embodied import PromptRenderer


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
