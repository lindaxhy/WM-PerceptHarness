from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from las_repro.media import FrameRef, TimeSpan, VideoMetadata
from las_repro.models.base import ModelOutputError, ModelRequest, VideoSession


class _Factory:
    def __init__(self, result: Any, *, reject_torch_dtype: bool = False) -> None:
        self.result = result
        self.reject_torch_dtype = reject_torch_dtype
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, path: str, **kwargs: Any) -> Any:
        self.calls.append((path, kwargs))
        if self.reject_torch_dtype and "torch_dtype" in kwargs:
            raise TypeError("got an unexpected keyword argument 'torch_dtype'")
        return self.result


class _LoadedModel:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.hf_device_map = {"": "cuda:2"}

    def eval(self) -> _LoadedModel:
        self.eval_calls += 1
        return self


class _InferenceMode:
    def __init__(self, torch: _FakeTorch) -> None:
        self.torch = torch

    def __enter__(self) -> None:
        self.torch.inference_entries += 1

    def __exit__(self, *args: object) -> None:
        self.torch.inference_exits += 1


class _CudaDeviceContext:
    def __init__(self, cuda: _FakeCuda, device: str) -> None:
        self.cuda = cuda
        self.device = device

    def __enter__(self) -> None:
        self.cuda.device_contexts.append(self.device)

    def __exit__(self, *args: object) -> None:
        pass


class _FakeCuda:
    def __init__(self, reserved_bytes: int = 0) -> None:
        self.reserved_bytes = reserved_bytes
        self.memory_queries: list[str] = []
        self.device_contexts: list[str] = []
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def memory_reserved(self, device: str) -> int:
        self.memory_queries.append(device)
        return self.reserved_bytes

    def device(self, device: str) -> _CudaDeviceContext:
        return _CudaDeviceContext(self, device)

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self, reserved_bytes: int = 0) -> None:
        self.inference_entries = 0
        self.inference_exits = 0
        self.cuda = _FakeCuda(reserved_bytes)

    def inference_mode(self) -> _InferenceMode:
        return _InferenceMode(self)


class _Inputs(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__(input_ids=[[11, 12, 13]], attention_mask=[[1, 1, 1]])
        self.moved_to: list[str] = []

    @property
    def input_ids(self) -> list[list[int]]:
        return self["input_ids"]

    def to(self, device: str) -> _Inputs:
        self.moved_to.append(device)
        return self


class _GeneratingModel:
    def __init__(self, generated: Any = None) -> None:
        self.generated = generated if generated is not None else [[11, 12, 13, 91, 92]]
        self.calls: list[dict[str, Any]] = []
        self.hf_device_map = {"": "cuda:1"}

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.generated


class _Processor:
    def __init__(self, decoded: str = '{"ok":true}') -> None:
        self.image_processor = SimpleNamespace(patch_size=16)
        self.decoded = decoded
        self.template_calls: list[tuple[Any, dict[str, Any]]] = []
        self.processor_calls: list[dict[str, Any]] = []
        self.decode_calls: list[tuple[Any, dict[str, Any]]] = []
        self.inputs = _Inputs()

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.template_calls.append((messages, kwargs))
        return "rendered prompt"

    def __call__(self, **kwargs: Any) -> _Inputs:
        self.processor_calls.append(deepcopy(kwargs))
        return self.inputs

    def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
        self.decode_calls.append((token_ids, kwargs))
        return [self.decoded]


class _VisionProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.opened_paths: list[Path] = []

    def __call__(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, kwargs))
        for message in messages:
            for item in message.get("content", []):
                video = item.get("video") if isinstance(item, dict) else None
                if isinstance(video, list):
                    for value in video:
                        path = Path(value)
                        assert path.open("rb").read(1)
                        self.opened_paths.append(path)
        return (
            ["image-tensor"],
            [("video-tensor", {"sample_fps": 2.0})],
            {"do_sample_frames": False, "fps": [2.0]},
        )


def _fake_frame_extractor(
    video_path: Path,
    span: TimeSpan,
    fps: float,
    output_dir: Path,
) -> list[FrameRef]:
    del video_path
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = (span.start, min(span.end - 0.001, span.start + 1.0 / fps))
    frames: list[FrameRef] = []
    for index, timestamp in enumerate(dict.fromkeys(timestamps)):
        path = output_dir / f"frame #{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(FrameRef(path.resolve(), timestamp))
    return frames


def _fake_video_probe(video_path: Path) -> VideoMetadata:
    del video_path
    return VideoMetadata(duration=10.0, width=320, height=180, fps=10.0)


def _request(
    video_path: Path,
    *,
    stage: str = "embodied_pass_b",
    media_resolution: str | None = "high",
    reasoning_effort: str | None = "low",
    clip_context: str | None = "medium",
) -> ModelRequest:
    return ModelRequest(
        stage=stage,
        video_path=video_path,
        span=TimeSpan(1.25, 4.75),
        fps=2.5,
        prompt="return one strict JSON object",
        schema_name="BoundaryPlan",
        video_session_id="task-1",
        media_resolution=media_resolution,  # type: ignore[arg-type]
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        clip_context=clip_context,  # type: ignore[arg-type]
    )


def test_alias_resolution_uses_only_the_configured_registry(tmp_path: Path) -> None:
    from las_repro.models.qwen3_vl import ModelAliasError, resolve_model_alias

    model_dir = tmp_path / "official-snapshot"
    model_dir.mkdir()
    registry = {"qwen3-vl-8b-instruct": model_dir}

    assert resolve_model_alias("qwen3-vl-8b-instruct", registry) == model_dir.resolve()

    disguised_path = str(model_dir)
    with pytest.raises(ModelAliasError, match="not allowlisted"):
        resolve_model_alias(disguised_path, registry)
    with pytest.raises(ModelAliasError, match="not allowlisted"):
        resolve_model_alias("Qwen/Qwen3-VL-8B-Instruct", registry)


def test_load_is_local_only_single_device_and_calls_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import las_repro.models.qwen3_vl as qwen

    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    loaded_model = _LoadedModel()
    model_factory = _Factory(loaded_model)
    processor = object()
    processor_factory = _Factory(processor)
    fake_torch = SimpleNamespace()
    runtime = SimpleNamespace(
        torch=fake_torch,
        model_class=model_factory,
        processor_class=processor_factory,
        process_vision_info=lambda messages, **kwargs: (None, None, {}),
    )
    monkeypatch.setattr(qwen, "_import_runtime_dependencies", lambda: runtime)

    backend = qwen.Qwen3VLModel.load(model_dir, "cuda:2", "auto")

    common = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert model_factory.calls == [
        (
            str(model_dir.resolve()),
            {
                **common,
                "torch_dtype": "auto",
                "device_map": {"": "cuda:2"},
            },
        )
    ]
    assert processor_factory.calls == [(str(model_dir.resolve()), common)]
    assert loaded_model.eval_calls == 1
    assert backend.device == "cuda:2"
    assert backend.processor is processor


def test_load_disables_hub_network_and_telemetry_only_for_the_load_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import las_repro.models.qwen3_vl as qwen

    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    observed: list[dict[str, str | None]] = []

    class Factory(_Factory):
        def from_pretrained(self, path: str, **kwargs: Any) -> Any:
            observed.append(
                {
                    key: os.environ.get(key)
                    for key in (
                        "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                        "HF_HUB_DISABLE_TELEMETRY",
                    )
                }
            )
            return super().from_pretrained(path, **kwargs)

    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        monkeypatch.delenv(key, raising=False)
    loaded_model = _LoadedModel()
    runtime = SimpleNamespace(
        torch=SimpleNamespace(),
        model_class=Factory(loaded_model),
        processor_class=Factory(object()),
        process_vision_info=lambda messages, **kwargs: (None, None, {}),
    )
    monkeypatch.setattr(qwen, "_import_runtime_dependencies", lambda: runtime)

    qwen.Qwen3VLModel.load(model_dir, "cuda:2")

    assert observed == [
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        },
    ]
    assert all(os.environ.get(key) is None for key in observed[0])


def test_load_alias_composes_allowlist_resolution_with_local_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import las_repro.models.qwen3_vl as qwen

    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    loaded_model = _LoadedModel()
    model_factory = _Factory(loaded_model)
    runtime = SimpleNamespace(
        torch=SimpleNamespace(),
        model_class=model_factory,
        processor_class=_Factory(object()),
        process_vision_info=lambda messages, **kwargs: (None, None, {}),
    )
    monkeypatch.setattr(qwen, "_import_runtime_dependencies", lambda: runtime)

    backend = qwen.Qwen3VLModel.load_alias(
        "approved",
        {"approved": model_dir},
        "cuda:2",
        "auto",
    )

    assert backend.device == "cuda:2"
    assert model_factory.calls[0][0] == str(model_dir.resolve())
    with pytest.raises(qwen.ModelAliasError, match="not allowlisted"):
        qwen.Qwen3VLModel.load_alias(
            str(model_dir),
            {"approved": model_dir},
            "cuda:2",
            "auto",
        )


def test_load_supports_transformers_dtype_rename_without_weakening_local_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import las_repro.models.qwen3_vl as qwen

    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    loaded_model = _LoadedModel()
    model_factory = _Factory(loaded_model, reject_torch_dtype=True)
    runtime = SimpleNamespace(
        torch=SimpleNamespace(),
        model_class=model_factory,
        processor_class=_Factory(object()),
        process_vision_info=lambda messages, **kwargs: (None, None, {}),
    )
    monkeypatch.setattr(qwen, "_import_runtime_dependencies", lambda: runtime)

    qwen.Qwen3VLModel.load(model_dir, "cuda:2", "bfloat16")

    assert len(model_factory.calls) == 2
    _, first = model_factory.calls[0]
    _, second = model_factory.calls[1]
    assert first["torch_dtype"] == "bfloat16"
    assert "dtype" not in first
    assert second["dtype"] == "bfloat16"
    assert "torch_dtype" not in second
    for call in (first, second):
        assert call["local_files_only"] is True
        assert call["trust_remote_code"] is False
        assert call["device_map"] == {"": "cuda:2"}


def test_load_refuses_missing_remote_or_local_identifiers_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import las_repro.models.qwen3_vl as qwen

    imported = False

    def fail_if_imported() -> Any:
        nonlocal imported
        imported = True
        raise AssertionError("optional runtime must not be imported")

    monkeypatch.setattr(qwen, "_import_runtime_dependencies", fail_if_imported)

    for unsafe in (
        Path("Qwen/Qwen3-VL-8B-Instruct"),
        tmp_path / "missing",
        tmp_path / "model.bin",
    ):
        if unsafe.name == "model.bin":
            unsafe.write_bytes(b"not a directory")
        with pytest.raises(qwen.LocalModelError):
            qwen.Qwen3VLModel.load(unsafe, "cuda:0", "auto")
    assert imported is False


def test_load_rejects_a_runtime_that_ignored_the_assigned_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import las_repro.models.qwen3_vl as qwen

    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    misplaced = _LoadedModel()
    misplaced.hf_device_map = {"": "cuda:3"}
    runtime = SimpleNamespace(
        torch=SimpleNamespace(),
        model_class=_Factory(misplaced),
        processor_class=_Factory(object()),
        process_vision_info=lambda messages, **kwargs: (None, None, {}),
    )
    monkeypatch.setattr(qwen, "_import_runtime_dependencies", lambda: runtime)

    with pytest.raises(qwen.QwenBackendError, match="assigned CUDA device"):
        qwen.Qwen3VLModel.load(model_dir, "cuda:2", "auto")


def test_generate_uses_official_qwen3_video_flow_and_strips_prompt_tokens(
    tmp_path: Path,
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    video = tmp_path / "clip name.mp4"
    video.write_bytes(b"video")
    torch = _FakeTorch()
    processor = _Processor()
    model = _GeneratingModel()
    vision = _VisionProcessor()
    backend = Qwen3VLModel(
        processor=processor,
        model=model,
        torch_module=torch,
        process_vision_info=vision,
        device="cuda:1",
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    result = backend.generate(_request(video))

    assert result == {"ok": True}
    messages, template_kwargs = processor.template_calls[0]
    assert template_kwargs == {"tokenize": False, "add_generation_prompt": True}
    video_item = messages[0]["content"][0]
    assert video_item["type"] == "video"
    assert video_item["fps"] == 2.5
    assert video_item["min_pixels"] == 4 * 32 * 32
    assert video_item["max_pixels"] == 256 * 32 * 32
    assert video_item["total_pixels"] == 8_192 * 32 * 32
    assert all(
        isinstance(path, str) and not path.startswith("file:")
        for path in video_item["video"]
    )
    assert all("frame #" in path for path in video_item["video"])
    assert messages[0]["content"][1] == {
        "type": "text",
        "text": "return one strict JSON object",
    }
    visual_messages, _ = vision.calls[0]
    assert visual_messages == [{"role": "user", "content": [video_item]}]
    assert vision.opened_paths == [Path(path) for path in video_item["video"]]
    assert vision.calls == [
        (
            visual_messages,
            {
                "image_patch_size": 16,
                "return_video_kwargs": True,
                "return_video_metadata": True,
            },
        )
    ]
    assert processor.processor_calls == [
        {
            "text": "rendered prompt",
            "images": ["image-tensor"],
            "videos": ["video-tensor"],
            "video_metadata": [
                {
                    "fps": 10.0,
                    "frames_indices": [13, 17],
                    "total_num_frames": 100,
                    "video_backend": "las_ffmpeg_video_only",
                }
            ],
            "padding": True,
            "return_tensors": "pt",
            "do_resize": False,
            "do_sample_frames": False,
            "fps": [2.0],
        }
    ]
    assert processor.inputs.moved_to == ["cuda:1"]
    assert model.calls == [
        {
            "input_ids": [[11, 12, 13]],
            "attention_mask": [[1, 1, 1]],
            "do_sample": False,
            "max_new_tokens": 4_096,
        }
    ]
    assert processor.decode_calls == [
        (
            [[91, 92]],
            {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            },
        )
    ]
    assert torch.inference_entries == torch.inference_exits == 1


def test_qwen_uses_real_video_only_frames_for_a_nonzero_silent_clip(
    short_video: Path,
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    processor = _Processor()
    vision = _VisionProcessor()
    backend = Qwen3VLModel(
        processor=processor,
        model=_GeneratingModel(),
        torch_module=_FakeTorch(),
        process_vision_info=vision,
        device="cuda:1",
        video_probe=lambda path: VideoMetadata(
            duration=2.0,
            width=320,
            height=180,
            fps=10.0,
        ),
    )
    request = replace(
        _request(short_video),
        span=TimeSpan(0.5, 1.5),
        fps=2.0,
    )

    assert backend.generate(request) == {"ok": True}

    assert len(vision.opened_paths) == 2
    assert all(path.suffix == ".jpg" for path in vision.opened_paths)
    assert processor.processor_calls[0]["video_metadata"] == [
        {
            "fps": 10.0,
            "frames_indices": [5, 10],
            "total_num_frames": 20,
            "video_backend": "las_ffmpeg_video_only",
        }
    ]


def test_absolute_video_metadata_clamps_rounding_at_the_source_frame_boundary(
    tmp_path: Path,
) -> None:
    from las_repro.models.qwen3_vl import _absolute_video_metadata

    metadata = _absolute_video_metadata(
        VideoMetadata(duration=2.0, width=320, height=180, fps=10.0),
        (FrameRef(path=tmp_path / "frame.jpg", timestamp=1.99),),
    )

    assert metadata["total_num_frames"] == 20
    assert metadata["frames_indices"] == [19]


def test_visual_preparation_is_video_only_absolute_and_reused_per_session(
    tmp_path: Path,
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    video = tmp_path / "clip space #1.mp4"
    video.write_bytes(b"video without audio")
    extracted: list[tuple[Path, TimeSpan, float, Path]] = []

    def extract_video_frames(
        path: Path, span: TimeSpan, fps: float, output_dir: Path
    ) -> list[FrameRef]:
        extracted.append((path, span, fps, output_dir))
        return _fake_frame_extractor(path, span, fps, output_dir)

    processor = _Processor()
    vision = _VisionProcessor()
    backend = Qwen3VLModel(
        processor=processor,
        model=_GeneratingModel(),
        torch_module=_FakeTorch(),
        process_vision_info=vision,
        device="cuda:1",
        frame_extractor=extract_video_frames,
        video_probe=lambda path: pytest.fail("session metadata must be reused"),
    )
    session = VideoSession(
        metadata=VideoMetadata(duration=10.0, width=320, height=180, fps=10.0)
    )
    first = replace(
        _request(video),
        span=TimeSpan(2.0, 3.0),
        fps=2.0,
        prompt="first request prompt",
        video_session=session,
    )
    second = replace(first, prompt="second request prompt")

    assert backend.generate(first) == {"ok": True}
    prepared_paths = tuple(
        Path(path)
        for path in processor.template_calls[0][0][0]["content"][0]["video"]
    )
    assert backend.generate(second) == {"ok": True}

    assert len(extracted) == 1
    assert len(vision.calls) == 1
    assert all(path.is_absolute() and path.is_file() for path in prepared_paths)
    assert processor.processor_calls[0]["video_metadata"] == [
        {
            "fps": 10.0,
            "frames_indices": [20, 25],
            "total_num_frames": 100,
            "video_backend": "las_ffmpeg_video_only",
        }
    ]
    assert (
        processor.template_calls[1][0][0]["content"][1]["text"]
        == "second request prompt"
    )
    assert "first request prompt" not in repr(session.backend_cache)
    assert "second request prompt" not in repr(session.backend_cache)

    backend.release_video_session("task-1")
    assert session.backend_cache == {}
    assert session.sampled_frames == ()
    assert all(not path.exists() for path in prepared_paths)

    replacement = replace(first, video_session=VideoSession(metadata=session.metadata))
    assert backend.generate(replacement) == {"ok": True}
    assert len(extracted) == 2


def test_session_cache_fails_closed_if_a_visual_tensor_leaves_cpu(
    tmp_path: Path,
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel, QwenDependencyError

    class Tensor:
        def __init__(self) -> None:
            self.device = "cpu"

    tensor = Tensor()

    def vision(messages: Any, **kwargs: Any) -> Any:
        del messages, kwargs
        return None, [(tensor, {"fps": 2.0})], {"fps": [2.0]}

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    session = VideoSession(
        metadata=VideoMetadata(duration=10.0, width=320, height=180, fps=10.0)
    )
    request = replace(_request(video), video_session=session)
    backend = Qwen3VLModel(
        processor=_Processor(),
        model=_GeneratingModel(),
        torch_module=_FakeTorch(),
        process_vision_info=vision,
        device="cuda:1",
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    assert backend.generate(request) == {"ok": True}
    tensor.device = "cuda:1"

    with pytest.raises(QwenDependencyError, match="remain on CPU"):
        backend.generate(request)
    backend.release_video_session("task-1")


def test_qwen_fails_closed_and_cleans_up_when_video_only_extraction_is_unavailable(
    tmp_path: Path,
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel, QwenDependencyError

    output_dirs: list[Path] = []

    def unavailable(
        video_path: Path, span: TimeSpan, fps: float, output_dir: Path
    ) -> list[FrameRef]:
        del video_path, span, fps
        output_dirs.append(output_dir)
        raise RuntimeError("decoder diagnostic token=must-not-survive")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    vision = _VisionProcessor()
    backend = Qwen3VLModel(
        processor=_Processor(),
        model=_GeneratingModel(),
        torch_module=_FakeTorch(),
        process_vision_info=vision,
        device="cuda:1",
        frame_extractor=unavailable,
        video_probe=_fake_video_probe,
    )

    with pytest.raises(QwenDependencyError, match="video-only frame extraction failed"):
        backend.generate(_request(video))

    assert vision.calls == []
    assert len(output_dirs) == 1
    assert output_dirs[0].exists() is False


@pytest.mark.parametrize(
    ("stage", "expected_cap"),
    [
        ("active_objects", 1_024),
        ("general_segment", 4_096),
        ("general_summary", 2_048),
        ("embodied_pass_a", 4_096),
        ("embodied_pass_b", 8_192),
        ("embodied_enrichment", 4_096),
        ("scene_semantics", 4_096),
    ],
)
def test_generation_token_budget_never_exceeds_stage_cap(
    tmp_path: Path, stage: str, expected_cap: int
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    processor = _Processor()
    model = _GeneratingModel()
    backend = Qwen3VLModel(
        processor=processor,
        model=model,
        torch_module=_FakeTorch(),
        process_vision_info=_VisionProcessor(),
        device="cuda:1",
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    backend.generate(
        _request(video, stage=stage, reasoning_effort="high")
    )

    assert model.calls[0]["max_new_tokens"] == expected_cap


def test_generate_rejects_unknown_stage_and_nonlocal_video_before_processing(
    tmp_path: Path,
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel, QwenStageError

    processor = _Processor()
    vision = _VisionProcessor()
    backend = Qwen3VLModel(
        processor=processor,
        model=_GeneratingModel(),
        torch_module=_FakeTorch(),
        process_vision_info=vision,
        device="cuda:1",
    )
    missing = tmp_path / "missing.mp4"

    with pytest.raises(QwenStageError, match="unsupported"):
        backend.generate(_request(missing, stage="unbounded"))
    with pytest.raises(ValueError, match="video path"):
        backend.generate(_request(missing, stage="general_segment"))
    assert processor.template_calls == []
    assert vision.calls == []


def test_generate_uses_strict_json_parser_and_output_size_limit(tmp_path: Path) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    backend = Qwen3VLModel(
        processor=_Processor('{"ok":true} trailing'),
        model=_GeneratingModel(),
        torch_module=_FakeTorch(),
        process_vision_info=_VisionProcessor(),
        device="cuda:1",
        max_output_chars=12,
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    with pytest.raises(ModelOutputError):
        backend.generate(_request(video))


def test_worker_schema_gate_replaces_invalid_qwen_object_before_persistence(
    tmp_path: Path,
) -> None:
    from las_repro.domain import InferenceJobSpec, InferenceStatus
    from las_repro.models.qwen3_vl import Qwen3VLModel
    from las_repro.store import SQLiteTaskStore
    from las_repro.workers import GPUWorker

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    store.initialize()
    task = store.create_task({"task_template": "embodied_active_object_detection"})
    [job] = store.create_inference_jobs(
        task.task_id,
        [
            InferenceJobSpec(
                stage="active_objects",
                ordinal=0,
                payload={
                    "video_path": str(video.resolve()),
                    "start": 0.0,
                    "end": 1.0,
                    "fps": 1.0,
                    "prompt": "return object inventory JSON",
                    "schema_name": "ObjectInventory",
                    "video_session_id": task.task_id,
                },
            )
        ],
    )
    backend = Qwen3VLModel(
        processor=_Processor('{"objects":"unvalidated model value"}'),
        model=_GeneratingModel(),
        torch_module=_FakeTorch(),
        process_vision_info=_VisionProcessor(),
        device="cuda:1",
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    assert GPUWorker(
        store,
        backend,
        worker_id="gpu-1",
        device="cuda:1",
        lease_seconds=10.0,
    ).run_once()

    persisted = store.get_inference_job(job.job_id)
    assert persisted is not None
    assert persisted.status is InferenceStatus.COMPLETED
    assert persisted.result == {
        "_schema_validation": {
            "schema_name": "ObjectInventory",
            "status": "invalid",
            "issue_codes": ["OBJECT_INVENTORY_LIST_TYPE"],
        }
    }
    assert "unvalidated model value" not in json.dumps(persisted.result)


def test_general_schema_gate_replaces_invalid_raw_evidence_before_persistence() -> None:
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    invalid = {
        "segments": [
            {
                "start_time": 0.5,
                "end_time": 2.5,
                "scene": ["private token=must-not-persist"],
                "subjects": [],
                "actions": [],
                "visible_text": [],
                "uncertainty": [],
                "description": "unsafe",
                "warnings": [],
            }
        ],
        "warnings": [],
    }

    sanitized = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "general_segment",
        invalid,
        {"span": {"start": 0.0, "end": 1.0}},
    )

    assert sanitized == {
        "_schema_validation": {
            "schema_name": "general_segment",
            "status": "invalid",
            "issue_codes": ["GENERAL_SEGMENT_TIMESTAMP_OUT_OF_SPAN"],
        }
    }
    assert "must-not-persist" not in json.dumps(sanitized)


def test_general_summary_schema_gate_canonicalizes_and_checks_exact_timeline() -> None:
    from las_repro.pipelines.output_validation import DEFAULT_OUTPUT_SCHEMAS

    event = {
        "start_time": 0.0,
        "end_time": 1.0,
        "scene": ["room"],
        "subjects": ["person"],
        "actions": ["moves"],
        "visible_text": [],
        "uncertainty": [],
        "description": "visible event",
        "warnings": [],
    }
    valid = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "general_summary",
        {"summary": "  visible summary  ", "timeline": [event], "warnings": []},
        {
            "span": {"start": 0.0, "end": 1.0},
            "expected_timeline": [event],
        },
    )
    mismatch = DEFAULT_OUTPUT_SCHEMAS.sanitize(
        "general_summary",
        {"summary": "visible summary", "timeline": [], "warnings": []},
        {
            "span": {"start": 0.0, "end": 1.0},
            "expected_timeline": [event],
        },
    )

    assert valid["summary"] == "visible summary"
    assert valid["timeline"] == [event]
    assert mismatch == {
        "_schema_validation": {
            "schema_name": "general_summary",
            "status": "invalid",
            "issue_codes": ["GENERAL_SUMMARY_TIMELINE_MISMATCH"],
        }
    }


@pytest.mark.parametrize(
    ("reserved", "threshold", "expected_clears"),
    [(99, 100, 0), (100, 100, 1)],
)
def test_cache_is_cleared_only_when_the_configured_threshold_is_reached(
    tmp_path: Path, reserved: int, threshold: int, expected_clears: int
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    torch = _FakeTorch(reserved)
    backend = Qwen3VLModel(
        processor=_Processor(),
        model=_GeneratingModel(),
        torch_module=torch,
        process_vision_info=_VisionProcessor(),
        device="cuda:1",
        cache_clear_threshold_bytes=threshold,
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    backend.generate(_request(video))

    assert torch.cuda.memory_queries == ["cuda:1"]
    assert torch.cuda.empty_cache_calls == expected_clears
    assert torch.cuda.device_contexts == (["cuda:1"] if expected_clears else [])


def test_request_tensors_are_released_before_memory_pressure_is_measured(
    tmp_path: Path,
) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    class Tensor:
        pass

    references: list[weakref.ReferenceType[Tensor]] = []

    def vision(messages: Any, **kwargs: Any) -> Any:
        del messages, kwargs
        tensor = Tensor()
        references.append(weakref.ref(tensor))
        return None, [(tensor, {"fps": 1.0})], {"do_sample_frames": False}

    class Processor(_Processor):
        def __init__(self) -> None:
            super().__init__()
            del self.inputs

        def __call__(self, **kwargs: Any) -> _Inputs:
            del kwargs
            inputs = _Inputs()
            tensor = Tensor()
            references.append(weakref.ref(tensor))
            inputs["pixel_values_videos"] = tensor
            return inputs

    class Model(_GeneratingModel):
        def generate(self, **kwargs: Any) -> Any:
            del kwargs
            return self.generated

    class Cuda(_FakeCuda):
        def memory_reserved(self, device: str) -> int:
            assert device == "cuda:1"
            assert references and all(reference() is None for reference in references)
            return 0

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    torch = _FakeTorch()
    torch.cuda = Cuda()
    backend = Qwen3VLModel(
        processor=Processor(),
        model=Model(),
        torch_module=torch,
        process_vision_info=vision,
        device="cuda:1",
        cache_clear_threshold_bytes=1,
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    assert backend.generate(_request(video)) == {"ok": True}


def test_cache_probe_failure_does_not_mask_strict_output_error(tmp_path: Path) -> None:
    from las_repro.models.qwen3_vl import Qwen3VLModel

    class FailingCuda(_FakeCuda):
        def memory_reserved(self, device: str) -> int:
            del device
            raise RuntimeError("driver diagnostics that must not escape cleanup")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    torch = _FakeTorch()
    torch.cuda = FailingCuda()
    backend = Qwen3VLModel(
        processor=_Processor("not JSON"),
        model=_GeneratingModel(),
        torch_module=torch,
        process_vision_info=_VisionProcessor(),
        device="cuda:1",
        cache_clear_threshold_bytes=1,
        frame_extractor=_fake_frame_extractor,
        video_probe=_fake_video_probe,
    )

    with pytest.raises(ModelOutputError, match="not valid JSON"):
        backend.generate(_request(video))


def test_download_exporter_uses_only_official_model_and_writes_stable_manifest(
    tmp_path: Path,
) -> None:
    from scripts import download_model

    destination = tmp_path / "snapshot"
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "subdir").mkdir(parents=True)
        (local_dir / "config.json").write_bytes(b'{"model":"qwen3-vl"}\n')
        (local_dir / "subdir" / "weights.bin").write_bytes(b"weights")
        cache = local_dir / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "download.lock").write_text("ephemeral", encoding="utf-8")
        return str(local_dir)

    manifest_path = download_model.download_model(
        destination,
        revision="main",
        snapshot_download=fake_snapshot_download,
    )

    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
            "repo_type": "model",
            "revision": "main",
            "local_dir": str((tmp_path / ".snapshot.las-incomplete").resolve()),
        }
    ]
    assert manifest_path == destination / "sha256-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "files": [
            {
                "path": "config.json",
                "sha256": hashlib.sha256(b'{"model":"qwen3-vl"}\n').hexdigest(),
                "size": 21,
            },
            {
                "path": "subdir/weights.bin",
                "sha256": hashlib.sha256(b"weights").hexdigest(),
                "size": 7,
            },
        ],
        "format": "las-repro-model-sha256-v1",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "main",
    }
    assert not list(destination.glob(".sha256-manifest.json.*.tmp"))
    assert "token" not in json.dumps(manifest, sort_keys=True).lower()
    assert json.loads(
        (destination / ".las-download-state.json").read_text(encoding="utf-8")
    ) == {
        "format": "las-repro-download-state-v1",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "main",
    }


def test_download_resume_survives_dependency_failure_without_leaking_streams(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    from scripts.download_model import download_model

    destination = tmp_path / "snapshot"
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("owned by user", encoding="utf-8")
    attempts = 0

    def flaky_snapshot(**kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        staging = Path(kwargs["local_dir"])
        if attempts == 1:
            (staging / "partial.bin").write_bytes(b"partial")
            print("token=python-stream-secret")
            os.write(1, b"authorization=native-stdout-secret\n")
            os.write(2, b"password=native-stderr-secret\n")
            raise RuntimeError("dependency detail")
        assert (staging / "partial.bin").read_bytes() == b"partial"
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    with pytest.raises(RuntimeError, match="dependency detail"):
        download_model(destination, snapshot_download=flaky_snapshot)
    captured = capfd.readouterr()
    assert "secret" not in captured.out.casefold() + captured.err.casefold()
    assert destination.exists() is False
    assert unrelated.read_text(encoding="utf-8") == "owned by user"

    manifest = download_model(destination, snapshot_download=flaky_snapshot)

    assert attempts == 2
    assert manifest == destination / "sha256-manifest.json"
    assert destination.is_dir()
    assert not (tmp_path / ".snapshot.las-incomplete").exists()
    assert unrelated.read_text(encoding="utf-8") == "owned by user"


def test_download_retry_recovers_from_interrupted_staging_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    real_write = module._atomic_json_write
    interrupted = False

    def interrupt_after_marker(path: Path, payload: dict[str, Any]) -> None:
        nonlocal interrupted
        if not interrupted and path.name == module.STATE_NAME:
            interrupted = True
            raise KeyboardInterrupt("claim marker interruption")
        real_write(path, payload)

    def downloader(**kwargs: Any) -> str:
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    monkeypatch.setattr(module, "_atomic_json_write", interrupt_after_marker)
    with pytest.raises(KeyboardInterrupt, match="claim marker interruption"):
        module.download_model(destination, snapshot_download=downloader)

    manifest = module.download_model(destination, snapshot_download=downloader)

    assert manifest == destination / module.MANIFEST_NAME
    assert not (tmp_path / ".snapshot.las-incomplete").exists()
    [retained_claim] = list(
        tmp_path.glob(".snapshot.las-incomplete.las-claim-*")
    )
    assert list(retained_claim.iterdir()) == []


def test_download_never_adopts_markerless_deterministic_staging(
    tmp_path: Path,
) -> None:
    from scripts.download_model import DownloadSafetyError, download_model

    destination = tmp_path / "snapshot"
    staging = tmp_path / ".snapshot.las-incomplete"
    for contents in (None, b"unowned"):
        staging.mkdir()
        if contents is not None:
            (staging / "partial.bin").write_bytes(contents)
        with pytest.raises(DownloadSafetyError, match="state"):
            download_model(
                destination,
                snapshot_download=lambda **kwargs: pytest.fail("must not download"),
            )
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()


def test_download_exclusive_rename_never_replaces_an_existing_directory(
    tmp_path: Path,
) -> None:
    from scripts.download_model import _rename_exclusive

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    with pytest.raises(FileExistsError):
        _rename_exclusive(source, target)

    assert source.is_dir()
    assert target.is_dir()


def test_download_retry_recovers_if_interrupted_after_atomic_staging_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    real_rename = module._rename_exclusive
    interrupted = False

    def interrupt_after_claim(source: Path, target: Path) -> None:
        nonlocal interrupted
        real_rename(source, target)
        if not interrupted and target.name == ".snapshot.las-incomplete":
            interrupted = True
            raise KeyboardInterrupt("after staging claim")

    def downloader(**kwargs: Any) -> str:
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    monkeypatch.setattr(module, "_rename_exclusive", interrupt_after_claim)
    with pytest.raises(KeyboardInterrupt, match="after staging claim"):
        module.download_model(destination, snapshot_download=downloader)

    assert module.download_model(destination, snapshot_download=downloader).is_file()


def test_download_claim_cleanup_never_deletes_a_replacement_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    real_rename = module._rename_exclusive

    def replace_claim_after_rename(source: Path, target: Path) -> None:
        real_rename(source, target)
        if Path(target).name == ".snapshot.las-incomplete":
            Path(source).mkdir()
            (Path(source) / module.STATE_NAME).write_text(
                "unrelated marker", encoding="utf-8"
            )
            (Path(source) / "keep.txt").write_text("keep", encoding="utf-8")
            raise KeyboardInterrupt("claim path replaced")

    monkeypatch.setattr(module, "_rename_exclusive", replace_claim_after_rename)
    with pytest.raises(KeyboardInterrupt, match="claim path replaced"):
        module.download_model(
            destination,
            snapshot_download=lambda **kwargs: pytest.fail("must not download"),
        )

    [replacement] = list(tmp_path.glob(".snapshot.las-incomplete.las-claim-*"))
    assert (replacement / module.STATE_NAME).read_text(encoding="utf-8") == (
        "unrelated marker"
    )
    assert (replacement / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_download_claim_cleanup_never_unlinks_even_exact_owned_paths(
    tmp_path: Path,
) -> None:
    from scripts import download_model as module

    staging_claim = tmp_path / "staging-claim"
    staging_claim.mkdir()
    staging_marker = staging_claim / module.STATE_NAME
    staging_marker.write_text("owned", encoding="utf-8")
    staging_identity = staging_claim.stat()

    module._remove_empty_claim(
        staging_claim,
        device=staging_identity.st_dev,
        inode=staging_identity.st_ino,
    )

    assert staging_marker.read_text(encoding="utf-8") == "owned"

    marker_claim = tmp_path / "marker-claim"
    marker_claim.mkdir()
    temporary = marker_claim / "publication-state.tmp"
    temporary.write_text("owned", encoding="utf-8")
    claim_identity = module._capture_directory_identity(marker_claim)
    temporary_identity = module._capture_regular_file_identity(temporary)
    assert claim_identity is not None
    assert temporary_identity is not None

    module._remove_owned_marker_claim(
        marker_claim,
        claim_identity=claim_identity,
        temporary=temporary,
        temporary_identity=temporary_identity,
    )

    assert temporary.read_text(encoding="utf-8") == "owned"


def test_concurrent_download_starts_publish_one_deterministic_snapshot(
    tmp_path: Path,
) -> None:
    from scripts.download_model import download_model

    destination = tmp_path / "snapshot"
    calls = 0
    calls_lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()

    def downloader(**kwargs: Any) -> str:
        nonlocal calls
        with calls_lock:
            calls += 1
        first_started.set()
        assert release_first.wait(timeout=5.0)
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            download_model, destination, snapshot_download=downloader
        )
        assert first_started.wait(timeout=5.0)
        second = executor.submit(
            download_model, destination, snapshot_download=downloader
        )
        time.sleep(0.05)
        release_first.set()
        manifests = [first.result(timeout=5.0), second.result(timeout=5.0)]

    assert manifests == [
        destination / "sha256-manifest.json",
        destination / "sha256-manifest.json",
    ]
    assert calls == 1


def test_download_accepts_existing_empty_destination_and_preserves_it_on_failure(
    tmp_path: Path,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    original = destination.stat()
    attempts = 0

    def downloader(**kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        if attempts == 1:
            raise RuntimeError("interrupted download")
        return str(staging)

    with pytest.raises(RuntimeError, match="interrupted download"):
        module.download_model(destination, snapshot_download=downloader)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []

    manifest = module.download_model(destination, snapshot_download=downloader)

    assert manifest == destination / module.MANIFEST_NAME
    assert attempts == 2
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    assert retired.is_dir()
    assert retired.stat().st_ino == original.st_ino
    assert list(retired.iterdir()) == []


def test_download_rolls_back_empty_destination_when_publication_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    real_rename = module._rename_exclusive
    failed = False

    def fail_staging_publication(source: str | Path, target: str | Path) -> None:
        nonlocal failed
        source_path = Path(source)
        target_path = Path(target)
        if (
            not failed
            and source_path.name == ".snapshot.las-incomplete"
            and target_path == destination
        ):
            failed = True
            raise OSError("injected publication failure")
        real_rename(source, target)

    def downloader(**kwargs: Any) -> str:
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    monkeypatch.setattr(module, "_rename_exclusive", fail_staging_publication)
    with pytest.raises(module.DownloadSafetyError, match="publish"):
        module.download_model(destination, snapshot_download=downloader)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []

    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    assert module.download_model(destination, snapshot_download=downloader).is_file()


def test_download_retry_recovers_after_each_completed_publication_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    for interrupted_target, expected_download_calls in (
        (".snapshot.las-empty-destination", 2),
        ("snapshot", 1),
    ):
        case = tmp_path / interrupted_target.removeprefix(".").replace(".", "-")
        case.mkdir()
        destination = case / "snapshot"
        destination.mkdir()
        real_rename = module._rename_exclusive
        interrupted = False
        download_calls = 0

        def interrupt_after_rename(source: Path, target: Path) -> None:
            nonlocal interrupted
            real_rename(source, target)
            if not interrupted and Path(target).name == interrupted_target:
                interrupted = True
                raise KeyboardInterrupt("publication interruption")

        def downloader(**kwargs: Any) -> str:
            nonlocal download_calls
            download_calls += 1
            staging = Path(kwargs["local_dir"])
            (staging / "config.json").write_text("{}", encoding="utf-8")
            return str(staging)

        monkeypatch.setattr(module, "_rename_exclusive", interrupt_after_rename)
        with pytest.raises(KeyboardInterrupt, match="publication interruption"):
            module.download_model(destination, snapshot_download=downloader)

        monkeypatch.setattr(module, "_rename_exclusive", real_rename)
        assert module.download_model(destination, snapshot_download=downloader).is_file()
        assert destination.is_dir()
        assert not (case / ".snapshot.las-empty-destination").exists()
        assert not (case / ".snapshot.las-publish-state.json").exists()
        assert download_calls == expected_download_calls


def test_download_retry_recovers_when_empty_destination_rollback_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    real_rename = module._rename_exclusive
    publish_failed = False
    rollback_failed = False

    def interrupt_publish_and_rollback(source: Path, target: Path) -> None:
        nonlocal publish_failed, rollback_failed
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name == ".snapshot.las-incomplete" and not publish_failed:
            publish_failed = True
            raise OSError("publish failed")
        if source_path.name == ".snapshot.las-empty-destination" and not rollback_failed:
            rollback_failed = True
            raise OSError("rollback failed")
        real_rename(source_path, target_path)

    def downloader(**kwargs: Any) -> str:
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    monkeypatch.setattr(module, "_rename_exclusive", interrupt_publish_and_rollback)
    with pytest.raises(module.DownloadSafetyError, match="publish"):
        module.download_model(destination, snapshot_download=downloader)
    assert not destination.exists()
    assert (tmp_path / ".snapshot.las-empty-destination").is_dir()

    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    assert module.download_model(destination, snapshot_download=downloader).is_file()
    assert destination.is_dir()
    assert not (tmp_path / ".snapshot.las-empty-destination").exists()


@pytest.mark.parametrize(
    "interposition",
    [
        "content-before-handoff",
        "content-after-handoff",
        "content-before-publish",
        "path-swap-before-handoff",
        "path-swap-after-handoff",
        "path-swap-before-publish",
        "content-before-cleanup-handoff",
        "content-after-cleanup-handoff",
        "path-swap-before-cleanup-handoff",
    ],
)
def test_download_empty_destination_handoff_never_hides_concurrent_user_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interposition: str,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    original = destination.stat()
    swapped_original = tmp_path / "swapped-original"
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    real_rename = module._rename_exclusive
    interposed = False

    def interpose(source: Path, target: Path) -> None:
        nonlocal interposed
        source_path = Path(source)
        target_path = Path(target)
        first_handoff = source_path == destination and target_path == placeholder
        publication = source_path == staging and target_path == destination
        cleanup_handoff = (
            source_path == placeholder
            and target_path.name == ".snapshot.las-retired-empty-destination"
        )
        if not interposed and first_handoff:
            if interposition == "content-before-handoff":
                (destination / "late-user-file.txt").write_text(
                    "preserve me", encoding="utf-8"
                )
                interposed = True
            elif interposition == "path-swap-before-handoff":
                real_rename(destination, swapped_original)
                destination.mkdir()
                interposed = True
        if not interposed and publication and interposition == "content-before-publish":
            (placeholder / "late-user-file.txt").write_text(
                "preserve me", encoding="utf-8"
            )
            interposed = True
        if not interposed and publication and interposition == "path-swap-before-publish":
            real_rename(placeholder, swapped_original)
            placeholder.mkdir()
            interposed = True
        if not interposed and cleanup_handoff:
            if interposition == "content-before-cleanup-handoff":
                (placeholder / "late-user-file.txt").write_text(
                    "preserve me", encoding="utf-8"
                )
                interposed = True
            elif interposition == "path-swap-before-cleanup-handoff":
                real_rename(placeholder, swapped_original)
                placeholder.mkdir()
                interposed = True
        real_rename(source_path, target_path)
        if not interposed and first_handoff and interposition == "content-after-handoff":
            (placeholder / "late-user-file.txt").write_text(
                "preserve me", encoding="utf-8"
            )
            interposed = True
        if not interposed and first_handoff and interposition == "path-swap-after-handoff":
            real_rename(placeholder, swapped_original)
            placeholder.mkdir()
            interposed = True
        if (
            not interposed
            and cleanup_handoff
            and interposition == "content-after-cleanup-handoff"
        ):
            (target_path / "late-user-file.txt").write_text(
                "preserve me", encoding="utf-8"
            )
            interposed = True

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", interpose)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(destination, snapshot_download=downloader)

    assert destination.is_dir()
    assert not (destination / module.MANIFEST_NAME).exists()
    assert staging.is_dir()
    assert not placeholder.exists()
    assert not (tmp_path / ".snapshot.las-retired-empty-destination").exists()
    assert not (tmp_path / ".snapshot.las-publish-state.json").exists()
    if interposition.startswith("path-swap"):
        assert destination.stat().st_ino != original.st_ino
        assert swapped_original.stat().st_ino == original.st_ino
    else:
        assert destination.stat().st_ino == original.st_ino
        assert (destination / "late-user-file.txt").read_text(encoding="utf-8") == (
            "preserve me"
        )
        (destination / "late-user-file.txt").unlink()

    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    assert module.download_model(destination, snapshot_download=downloader).is_file()
    if interposition.startswith("path-swap"):
        assert swapped_original.stat().st_ino == original.st_ino


def test_download_competing_writer_cannot_hide_content_during_empty_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    original = destination.stat()
    real_rename = module._rename_exclusive
    handoff_complete = threading.Event()
    writer_complete = threading.Event()

    def competing_writer() -> None:
        assert handoff_complete.wait(timeout=5.0)
        (placeholder / "competing-user-file.txt").write_text(
            "do not hide", encoding="utf-8"
        )
        writer_complete.set()

    writer = threading.Thread(target=competing_writer)
    writer.start()

    def pause_after_handoff(source: Path, target: Path) -> None:
        real_rename(source, target)
        if Path(source) == destination and Path(target) == placeholder:
            handoff_complete.set()
            assert writer_complete.wait(timeout=5.0)

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", pause_after_handoff)
    try:
        with pytest.raises(module.DownloadSafetyError, match="destination changed"):
            module.download_model(destination, snapshot_download=downloader)
    finally:
        writer.join(timeout=5.0)

    assert writer.is_alive() is False
    assert destination.stat().st_ino == original.st_ino
    assert (destination / "competing-user-file.txt").read_text(
        encoding="utf-8"
    ) == "do not hide"
    assert not placeholder.exists()


@pytest.mark.parametrize("phase", ["initial-handoff", "cleanup-handoff"])
@pytest.mark.parametrize(
    "replacement_type",
    [
        "file",
        "symlink",
        "dangling-symlink",
        "empty-directory",
        "content-directory",
    ],
)
def test_download_changed_handoff_restores_exact_user_entry_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    replacement_type: str,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    original = destination.stat()
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    committed_state = tmp_path / ".snapshot.las-retired-empty-state.json"
    moved_original = tmp_path / "writer-moved-original"
    symlink_target = tmp_path / "user-target.txt"
    dangling_target = tmp_path / "missing-user-target.txt"
    symlink_target.write_text("target remains", encoding="utf-8")
    real_rename = module._rename_exclusive
    interposed = False
    replacement_identity: os.stat_result | None = None

    def install_replacement(path: Path) -> None:
        if replacement_type == "file":
            path.write_bytes(b"exact user bytes")
        elif replacement_type == "symlink":
            path.symlink_to(symlink_target)
        elif replacement_type == "dangling-symlink":
            path.symlink_to(dangling_target)
        else:
            path.mkdir()
            path.chmod(0o751)
            if replacement_type == "content-directory":
                (path / "user-content.txt").write_bytes(b"exact directory content")

    def interpose(source: Path, target: Path) -> None:
        nonlocal interposed, replacement_identity
        source_path = Path(source)
        target_path = Path(target)
        initial_handoff = source_path == destination and target_path == placeholder
        cleanup_handoff = source_path == placeholder and target_path == retired
        if not interposed and (
            (phase == "initial-handoff" and initial_handoff)
            or (phase == "cleanup-handoff" and cleanup_handoff)
        ):
            real_rename(source_path, moved_original)
            install_replacement(source_path)
            replacement_identity = source_path.lstat()
            interposed = True
        real_rename(source_path, target_path)

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", interpose)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(destination, snapshot_download=downloader)

    assert interposed is True
    assert replacement_identity is not None
    assert moved_original.stat().st_ino == original.st_ino
    observed_replacement = destination.lstat()
    assert (
        observed_replacement.st_dev,
        observed_replacement.st_ino,
        observed_replacement.st_mode,
    ) == (
        replacement_identity.st_dev,
        replacement_identity.st_ino,
        replacement_identity.st_mode,
    )
    if replacement_type == "file":
        assert destination.is_file()
        assert destination.read_bytes() == b"exact user bytes"
    elif replacement_type == "symlink":
        assert destination.is_symlink()
        assert destination.readlink() == symlink_target
        assert symlink_target.read_text(encoding="utf-8") == "target remains"
    elif replacement_type == "dangling-symlink":
        assert destination.is_symlink()
        assert destination.readlink() == dangling_target
        assert not dangling_target.exists()
    else:
        assert destination.is_dir()
        assert stat.S_IMODE(destination.lstat().st_mode) == 0o751
        if replacement_type == "content-directory":
            assert (destination / "user-content.txt").read_bytes() == (
                b"exact directory content"
            )
    assert not placeholder.exists() and not placeholder.is_symlink()
    assert not retired.exists() and not retired.is_symlink()
    assert (staging / module.MANIFEST_NAME).is_file()
    assert not (destination / module.MANIFEST_NAME).exists()
    assert not (tmp_path / ".snapshot.las-publish-state.json").exists()
    assert not committed_state.exists()

    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    else:
        if replacement_type == "content-directory":
            (destination / "user-content.txt").unlink()
        destination.rmdir()
    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    manifest = module.download_model(destination, snapshot_download=downloader)

    assert manifest == destination / module.MANIFEST_NAME
    assert (destination / "config.json").is_file()
    assert moved_original.stat().st_ino == original.st_ino
    assert symlink_target.read_text(encoding="utf-8") == "target remains"


@pytest.mark.parametrize(
    "interrupt_rename", ["snapshot-to-staging", "entry-to-destination"]
)
def test_download_changed_directory_handoff_recovers_after_rollback_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_rename: str,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    original = destination.stat()
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    publish_state = tmp_path / ".snapshot.las-publish-state.json"
    committed_state = tmp_path / ".snapshot.las-retired-empty-state.json"
    moved_original = tmp_path / "writer-moved-original"
    real_rename = module._rename_exclusive
    replacement_identity: os.stat_result | None = None
    replacement_installed = False
    interrupted = False

    def interpose(source: Path, target: Path) -> None:
        nonlocal replacement_identity, replacement_installed, interrupted
        source_path = Path(source)
        target_path = Path(target)
        cleanup_handoff = source_path == placeholder and target_path == retired
        if cleanup_handoff and not replacement_installed:
            real_rename(source_path, moved_original)
            source_path.mkdir()
            source_path.chmod(0o751)
            (source_path / "user-content.txt").write_bytes(
                b"exact interrupted directory content"
            )
            replacement_identity = source_path.lstat()
            replacement_installed = True
        real_rename(source_path, target_path)
        snapshot_to_staging = (
            source_path == destination
            and target_path == staging
            and replacement_installed
        )
        entry_to_destination = source_path == retired and target_path == destination
        if not interrupted and (
            (interrupt_rename == "snapshot-to-staging" and snapshot_to_staging)
            or (
                interrupt_rename == "entry-to-destination"
                and entry_to_destination
            )
        ):
            interrupted = True
            raise KeyboardInterrupt(f"interrupted {interrupt_rename}")

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", interpose)
    with pytest.raises(KeyboardInterrupt, match=interrupt_rename):
        module.download_model(destination, snapshot_download=downloader)

    assert interrupted is True
    assert replacement_identity is not None
    assert moved_original.stat().st_ino == original.st_ino

    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    with pytest.raises(module.DownloadSafetyError):
        module.download_model(
            destination,
            snapshot_download=lambda **kwargs: pytest.fail("must not redownload"),
        )

    observed_replacement = destination.lstat()
    assert (
        observed_replacement.st_dev,
        observed_replacement.st_ino,
        observed_replacement.st_mode,
    ) == (
        replacement_identity.st_dev,
        replacement_identity.st_ino,
        replacement_identity.st_mode,
    )
    assert (destination / "user-content.txt").read_bytes() == (
        b"exact interrupted directory content"
    )
    assert not (destination / module.MANIFEST_NAME).exists()
    assert (staging / module.MANIFEST_NAME).is_file()
    assert not placeholder.exists()
    assert not retired.exists()
    assert not publish_state.exists()
    assert not committed_state.exists()

    (destination / "user-content.txt").unlink()
    destination.rmdir()

    def resume_downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        assert (local_dir / "config.json").is_file()
        return str(local_dir)

    manifest = module.download_model(
        destination,
        snapshot_download=resume_downloader,
    )

    assert manifest == destination / module.MANIFEST_NAME
    assert (destination / "config.json").is_file()
    assert moved_original.stat().st_ino == original.st_ino


@pytest.mark.parametrize("phase", ["initial-handoff", "cleanup-handoff"])
@pytest.mark.parametrize("post_restore", ["replacement", "absent"])
def test_download_changed_handoff_keeps_snapshot_staged_after_post_restore_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    post_restore: str,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    original = destination.stat()
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    publish_state = tmp_path / ".snapshot.las-publish-state.json"
    moved_original = tmp_path / "writer-moved-original"
    moved_first_replacement = tmp_path / "writer-moved-first-replacement"
    real_rename = module._rename_exclusive
    first_replacement_identity: os.stat_result | None = None
    second_replacement_identity: os.stat_result | None = None
    handoff_swapped = False
    restore_swapped = False

    def interpose(source: Path, target: Path) -> None:
        nonlocal first_replacement_identity, second_replacement_identity
        nonlocal handoff_swapped, restore_swapped
        source_path = Path(source)
        target_path = Path(target)
        initial_handoff = source_path == destination and target_path == placeholder
        cleanup_handoff = source_path == placeholder and target_path == retired
        selected_handoff = (
            phase == "initial-handoff" and initial_handoff
        ) or (phase == "cleanup-handoff" and cleanup_handoff)
        if selected_handoff and not handoff_swapped:
            real_rename(source_path, moved_original)
            source_path.mkdir()
            (source_path / "first-user-content.txt").write_bytes(b"first entry")
            first_replacement_identity = source_path.lstat()
            handoff_swapped = True

        real_rename(source_path, target_path)

        rollback_restore = (
            phase == "initial-handoff"
            and source_path == placeholder
            and target_path == destination
        ) or (
            phase == "cleanup-handoff"
            and source_path == retired
            and target_path == destination
        )
        if rollback_restore and handoff_swapped and not restore_swapped:
            real_rename(destination, moved_first_replacement)
            if post_restore == "replacement":
                destination.mkdir()
                destination.chmod(0o751)
                (destination / "second-user-content.txt").write_bytes(
                    b"second entry must stay visible"
                )
                second_replacement_identity = destination.lstat()
            restore_swapped = True

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", interpose)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(destination, snapshot_download=downloader)

    assert handoff_swapped is True
    assert restore_swapped is True
    assert first_replacement_identity is not None
    assert moved_original.stat().st_ino == original.st_ino
    assert moved_first_replacement.lstat().st_ino == first_replacement_identity.st_ino
    if post_restore == "replacement":
        assert second_replacement_identity is not None
        observed = destination.lstat()
        assert (observed.st_dev, observed.st_ino, observed.st_mode) == (
            second_replacement_identity.st_dev,
            second_replacement_identity.st_ino,
            second_replacement_identity.st_mode,
        )
        assert (destination / "second-user-content.txt").read_bytes() == (
            b"second entry must stay visible"
        )
    else:
        assert second_replacement_identity is None
        assert not destination.exists()
    assert not (destination / module.MANIFEST_NAME).exists()
    assert (staging / module.MANIFEST_NAME).is_file()
    assert not placeholder.exists()
    assert not retired.exists()
    assert not publish_state.exists()

    if post_restore == "replacement":
        (destination / "second-user-content.txt").unlink()
        destination.rmdir()
    monkeypatch.setattr(module, "_rename_exclusive", real_rename)

    def resume_downloader(**kwargs: Any) -> str:
        return str(Path(kwargs["local_dir"]))

    manifest = module.download_model(
        destination,
        snapshot_download=resume_downloader,
    )

    assert manifest == destination / module.MANIFEST_NAME
    assert (destination / "config.json").is_file()
    assert not publish_state.exists()
    assert (moved_first_replacement / "first-user-content.txt").read_bytes() == (
        b"first entry"
    )


def test_download_marker_commit_failure_after_empty_inode_handoff_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    real_commit = module._commit_publication_marker
    failed = False

    def fail_once(*args: Any, **kwargs: Any) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("marker commit interruption")
        real_commit(*args, **kwargs)

    def downloader(**kwargs: Any) -> str:
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    monkeypatch.setattr(module, "_commit_publication_marker", fail_once)
    with pytest.raises(module.DownloadSafetyError, match="publish"):
        module.download_model(destination, snapshot_download=downloader)
    assert (destination / module.MANIFEST_NAME).is_file()
    assert not (tmp_path / ".snapshot.las-empty-destination").exists()

    monkeypatch.setattr(module, "_commit_publication_marker", real_commit)
    manifest = module.download_model(
        destination,
        snapshot_download=lambda **kwargs: pytest.fail("must not redownload"),
    )
    assert manifest == destination / module.MANIFEST_NAME
    assert not (tmp_path / ".snapshot.las-publish-state.json").exists()


@pytest.mark.parametrize("interruption", ["after-cleanup-rename", "after-cleanup"])
def test_download_cleanup_interruption_leaves_a_recoverable_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    real_rename = module._rename_exclusive
    real_remove = module._remove_published_placeholder
    interrupted = False
    download_calls = 0

    def interrupt_cleanup_rename(source: Path, target: Path) -> None:
        nonlocal interrupted
        real_rename(source, target)
        if (
            interruption == "after-cleanup-rename"
            and not interrupted
            and Path(source).name == ".snapshot.las-empty-destination"
            and Path(target).name != "snapshot"
        ):
            interrupted = True
            raise KeyboardInterrupt("cleanup rename interruption")

    def interrupt_after_cleanup(*args: Any, **kwargs: Any) -> None:
        nonlocal interrupted
        real_remove(*args, **kwargs)
        if interruption == "after-cleanup" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("cleanup completion interruption")

    def downloader(**kwargs: Any) -> str:
        nonlocal download_calls
        download_calls += 1
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)

    monkeypatch.setattr(module, "_rename_exclusive", interrupt_cleanup_rename)
    monkeypatch.setattr(module, "_remove_published_placeholder", interrupt_after_cleanup)
    with pytest.raises(KeyboardInterrupt, match="cleanup"):
        module.download_model(destination, snapshot_download=downloader)

    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    monkeypatch.setattr(module, "_remove_published_placeholder", real_remove)
    manifest = module.download_model(
        destination,
        snapshot_download=lambda **kwargs: pytest.fail("must recover, not redownload"),
    )

    assert manifest == destination / module.MANIFEST_NAME
    assert download_calls == 1
    assert not (tmp_path / ".snapshot.las-publish-state.json").exists()
    assert not (tmp_path / ".snapshot.las-empty-destination").exists()


@pytest.mark.parametrize("replacement", ["directory", "absent"])
def test_download_retires_stale_publication_marker_after_completed_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    original = destination.stat()
    marker = tmp_path / ".snapshot.las-publish-state.json"
    staging = tmp_path / ".snapshot.las-incomplete"
    real_rollback = module._rollback_changed_publication

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rollback_changed_publication", lambda *a, **k: False)
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    real_rename = module._rename_exclusive

    def force_changed_publication(source: Path, target: Path) -> None:
        real_rename(source, target)
        if Path(source) == staging and Path(target) == destination:
            (placeholder / "late.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(module, "_rename_exclusive", force_changed_publication)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(destination, snapshot_download=downloader)
    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    monkeypatch.setattr(module, "_rollback_changed_publication", real_rollback)

    # Emulate a completed rollback whose final marker cleanup was interrupted.
    real_rename(destination, staging)
    (placeholder / "late.txt").unlink()
    placeholder.rmdir()
    if replacement == "directory":
        destination.mkdir()
        (destination / "user.txt").write_text("preserve", encoding="utf-8")

    if replacement == "directory":
        with pytest.raises(module.DownloadSafetyError):
            module.download_model(
                destination,
                snapshot_download=lambda **kwargs: pytest.fail("recover first"),
            )
        assert staging.is_dir()
        assert (destination / "user.txt").read_text(encoding="utf-8") == "preserve"
    else:
        assert module.download_model(destination, snapshot_download=downloader).is_file()

    assert not marker.exists()
    assert destination.stat().st_ino != original.st_ino


def test_download_rollback_never_adopts_a_swapped_destination_as_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    stolen_snapshot = tmp_path / "stolen-published-snapshot"
    real_rename = module._rename_exclusive
    published = False
    swapped = False

    def interpose(source: Path, target: Path) -> None:
        nonlocal published, swapped
        source_path = Path(source)
        target_path = Path(target)
        if (
            published
            and not swapped
            and source_path == destination
            and target_path == staging
        ):
            real_rename(destination, stolen_snapshot)
            destination.mkdir()
            (destination / "replacement-user.txt").write_text(
                "never adopt", encoding="utf-8"
            )
            swapped = True
        real_rename(source_path, target_path)
        if source_path == staging and target_path == destination:
            published = True
            (placeholder / "late-user.txt").write_text("keep", encoding="utf-8")

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", interpose)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(destination, snapshot_download=downloader)

    assert swapped is True
    assert (destination / "replacement-user.txt").read_text(encoding="utf-8") == (
        "never adopt"
    )
    assert not (staging / "replacement-user.txt").exists()
    assert (stolen_snapshot / module.MANIFEST_NAME).is_file()


def test_download_marker_creation_is_exclusive_and_preserves_a_racing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    marker = tmp_path / ".snapshot.las-publish-state.json"
    real_mkstemp = module.tempfile.mkstemp
    raced = False

    def racing_mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        nonlocal raced
        result = real_mkstemp(*args, **kwargs)
        if not raced and "las-publish-state" in str(kwargs.get("prefix", "")):
            marker.write_text("user-owned marker content", encoding="utf-8")
            raced = True
        return result

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module.tempfile, "mkstemp", racing_mkstemp)
    with pytest.raises(module.DownloadSafetyError):
        module.download_model(destination, snapshot_download=downloader)

    assert raced is True
    assert marker.read_text(encoding="utf-8") == "user-owned marker content"
    assert list(destination.iterdir()) == []
    assert not tuple(tmp_path.glob("..snapshot.las-publish-state.json.*.tmp"))
    [retained_claim] = tuple(
        tmp_path.glob("..snapshot.las-publish-state.json.las-marker-claim-*")
    )
    [owned_temporary] = tuple(retained_claim.iterdir())
    assert owned_temporary.is_file()


def test_download_marker_creation_rejects_a_post_handoff_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    marker = tmp_path / ".snapshot.las-publish-state.json"
    saved_marker = tmp_path / "saved-owned-marker"
    real_rename = module._rename_exclusive
    swapped = False

    def swap_after_marker_handoff(source: Path, target: Path) -> None:
        nonlocal swapped
        real_rename(source, target)
        if not swapped and Path(target) == marker:
            real_rename(marker, saved_marker)
            marker.write_text("post-handoff user marker", encoding="utf-8")
            swapped = True

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", swap_after_marker_handoff)
    with pytest.raises(module.DownloadSafetyError):
        module.download_model(destination, snapshot_download=downloader)

    assert swapped is True
    assert marker.read_text(encoding="utf-8") == "post-handoff user marker"
    assert saved_marker.is_file()
    assert list(destination.iterdir()) == []


def test_download_marker_cleanup_preserves_a_swapped_user_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    marker = tmp_path / ".snapshot.las-publish-state.json"
    saved_marker = tmp_path / "saved-owned-marker"
    real_commit = module._commit_publication_marker
    swapped = False

    def swap_before_commit(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal swapped
        if not swapped:
            path.rename(saved_marker)
            path.write_text("user marker replacement", encoding="utf-8")
            swapped = True
        real_commit(path, *args, **kwargs)

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_commit_publication_marker", swap_before_commit)
    with pytest.raises(module.DownloadSafetyError):
        module.download_model(destination, snapshot_download=downloader)

    assert swapped is True
    assert marker.read_text(encoding="utf-8") == "user marker replacement"
    assert saved_marker.is_file()


def test_download_late_write_before_commit_is_exposed_and_snapshot_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    real_commit = module._commit_publication_marker
    interposed = False

    def write_before_commit(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal interposed
        if not interposed:
            (retired / "late-user-file.txt").write_text("expose", encoding="utf-8")
            interposed = True
        real_commit(path, *args, **kwargs)

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_commit_publication_marker", write_before_commit)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(destination, snapshot_download=downloader)

    assert interposed is True
    assert (destination / "late-user-file.txt").read_text(encoding="utf-8") == (
        "expose"
    )
    assert (staging / module.MANIFEST_NAME).is_file()
    assert not retired.exists()


def test_download_committed_marker_recovery_exposes_changed_retained_destination(
    tmp_path: Path,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    assert module.download_model(destination, snapshot_download=downloader).is_file()
    (retired / "post-commit-user-file.txt").write_text("expose", encoding="utf-8")

    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(
            destination,
            snapshot_download=lambda **kwargs: pytest.fail("recover first"),
        )

    assert (destination / "post-commit-user-file.txt").read_text(
        encoding="utf-8"
    ) == "expose"
    assert (staging / module.MANIFEST_NAME).is_file()
    assert not retired.exists()


def test_download_does_not_adopt_retired_path_without_durable_provenance(
    tmp_path: Path,
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    assert module.download_model(destination, snapshot_download=downloader).is_file()
    retired.mkdir()
    (retired / "unowned-user-file.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(module.DownloadSafetyError, match="publication state"):
        module.download_model(
            destination,
            snapshot_download=lambda **kwargs: pytest.fail("must not download"),
        )

    assert (destination / module.MANIFEST_NAME).is_file()
    assert (retired / "unowned-user-file.txt").read_text(encoding="utf-8") == (
        "preserve"
    )
    assert not staging.exists()


def test_download_committed_rollback_rechecks_recorded_inode_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    retired = tmp_path / ".snapshot.las-retired-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    moved_original = tmp_path / "writer-moved-original"

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    assert module.download_model(destination, snapshot_download=downloader).is_file()
    (retired / "original-user-file.txt").write_text("original", encoding="utf-8")
    real_rollback = module._rollback_changed_publication
    interposed = False

    def swap_before_rollback(*args: Any, **kwargs: Any) -> bool:
        nonlocal interposed
        if not interposed:
            retired.rename(moved_original)
            retired.mkdir()
            (retired / "replacement-user-file.txt").write_text(
                "replacement", encoding="utf-8"
            )
            interposed = True
        return real_rollback(*args, **kwargs)

    monkeypatch.setattr(module, "_rollback_changed_publication", swap_before_rollback)
    with pytest.raises(module.DownloadSafetyError):
        module.download_model(
            destination,
            snapshot_download=lambda **kwargs: pytest.fail("must not download"),
        )

    assert interposed is True
    assert (destination / module.MANIFEST_NAME).is_file()
    assert not staging.exists()
    assert (retired / "replacement-user-file.txt").read_text(encoding="utf-8") == (
        "replacement"
    )
    assert (moved_original / "original-user-file.txt").read_text(
        encoding="utf-8"
    ) == "original"


def test_download_commit_does_not_relocate_a_post_handoff_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    publish_state = tmp_path / ".snapshot.las-publish-state.json"
    committed_state = tmp_path / ".snapshot.las-retired-empty-state.json"
    moved_owned_marker = tmp_path / "writer-moved-owned-marker"
    real_rename = module._rename_exclusive
    swapped = False

    def swap_after_commit_handoff(source: Path, target: Path) -> None:
        nonlocal swapped
        real_rename(source, target)
        if not swapped and Path(source) == publish_state and Path(target) == committed_state:
            real_rename(committed_state, moved_owned_marker)
            committed_state.write_text("user replacement", encoding="utf-8")
            swapped = True

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", swap_after_commit_handoff)
    with pytest.raises(module.DownloadSafetyError):
        module.download_model(destination, snapshot_download=downloader)

    assert swapped is True
    assert committed_state.read_text(encoding="utf-8") == "user replacement"
    assert not publish_state.exists()
    assert moved_owned_marker.is_file()
    assert (destination / module.MANIFEST_NAME).is_file()


def test_download_retry_unwinds_a_crashed_publication_with_user_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    destination.mkdir()
    placeholder = tmp_path / ".snapshot.las-empty-destination"
    staging = tmp_path / ".snapshot.las-incomplete"
    real_rename = module._rename_exclusive
    real_rollback = module._rollback_changed_publication
    interposed = False

    def add_content_before_publish(source: Path, target: Path) -> None:
        nonlocal interposed
        if not interposed and Path(source) == staging and Path(target) == destination:
            (placeholder / "user-file.txt").write_text("keep", encoding="utf-8")
            interposed = True
        real_rename(source, target)

    def downloader(**kwargs: Any) -> str:
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(module, "_rename_exclusive", add_content_before_publish)
    monkeypatch.setattr(module, "_rollback_changed_publication", lambda *a, **k: False)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(destination, snapshot_download=downloader)
    assert (destination / module.MANIFEST_NAME).is_file()
    assert (placeholder / "user-file.txt").is_file()

    monkeypatch.setattr(module, "_rename_exclusive", real_rename)
    monkeypatch.setattr(module, "_rollback_changed_publication", real_rollback)
    with pytest.raises(module.DownloadSafetyError, match="destination changed"):
        module.download_model(
            destination,
            snapshot_download=lambda **kwargs: pytest.fail("must recover first"),
        )
    assert (destination / "user-file.txt").read_text(encoding="utf-8") == "keep"
    assert staging.is_dir()
    assert not placeholder.exists()

    (destination / "user-file.txt").unlink()
    assert module.download_model(destination, snapshot_download=downloader).is_file()


def test_download_wrong_return_and_hash_failure_are_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    outside = tmp_path / "outside"
    outside.mkdir()
    calls = 0

    def downloader(**kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(outside if calls == 1 else staging)

    with pytest.raises(module.DownloadSafetyError, match="escaped"):
        module.download_model(destination, snapshot_download=downloader)
    assert destination.exists() is False
    assert module.download_model(destination, snapshot_download=downloader).is_file()

    second_destination = tmp_path / "snapshot-two"
    real_manifest_files = module._manifest_files
    hash_calls = 0

    def fail_once(root: Path) -> list[dict[str, Any]]:
        nonlocal hash_calls
        hash_calls += 1
        if hash_calls == 1:
            raise module.DownloadSafetyError("downloaded snapshot file cannot be hashed")
        return real_manifest_files(root)

    monkeypatch.setattr(module, "_manifest_files", fail_once)
    with pytest.raises(module.DownloadSafetyError, match="cannot be hashed"):
        module.download_model(second_destination, snapshot_download=downloader)
    assert second_destination.exists() is False
    assert module.download_model(second_destination, snapshot_download=downloader).is_file()


def test_download_recovers_if_interrupted_immediately_after_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "snapshot"
    staging = tmp_path / ".snapshot.las-incomplete"
    real_replace = module._rename_exclusive
    download_calls = 0

    def downloader(**kwargs: Any) -> str:
        nonlocal download_calls
        download_calls += 1
        local_dir = Path(kwargs["local_dir"])
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    def interrupt_after_publish(source: str | Path, target: str | Path) -> None:
        real_replace(source, target)
        if Path(source) == staging and Path(target) == destination:
            raise KeyboardInterrupt("after atomic publish")

    monkeypatch.setattr(module, "_rename_exclusive", interrupt_after_publish)
    with pytest.raises(KeyboardInterrupt, match="after atomic publish"):
        module.download_model(destination, snapshot_download=downloader)
    assert destination.is_dir()

    monkeypatch.setattr(module, "_rename_exclusive", real_replace)
    manifest = module.download_model(
        destination,
        snapshot_download=lambda **kwargs: pytest.fail("must not redownload"),
    )

    assert manifest == destination / "sha256-manifest.json"
    assert download_calls == 1


def test_download_rejects_mismatched_or_symlinked_incomplete_state(
    tmp_path: Path,
) -> None:
    from scripts.download_model import DownloadSafetyError, download_model

    destination = tmp_path / "snapshot"
    staging = tmp_path / ".snapshot.las-incomplete"
    staging.mkdir()
    (staging / ".las-download-state.json").write_text(
        json.dumps(
            {
                "format": "las-repro-download-state-v1",
                "model_id": "other/model",
                "revision": "main",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DownloadSafetyError, match="does not match"):
        download_model(destination, snapshot_download=lambda **kwargs: kwargs["local_dir"])

    (staging / ".las-download-state.json").unlink()
    staging.rmdir()
    real = tmp_path / "other-partial"
    real.mkdir()
    staging.symlink_to(real, target_is_directory=True)
    with pytest.raises(DownloadSafetyError, match="incomplete"):
        download_model(destination, snapshot_download=lambda **kwargs: kwargs["local_dir"])


def test_download_cli_success_output_is_stable_and_contains_no_user_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import download_model as module

    destination = tmp_path / "customer token=secret" / "snapshot"
    monkeypatch.setattr(
        module,
        "run_download_isolated",
        lambda *args, **kwargs: {
            "manifest": "sha256-manifest.json",
            "status": "completed",
        },
    )

    assert module.main(["--destination", str(destination)]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "manifest": "sha256-manifest.json",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "main",
        "status": "completed",
    }
    assert str(tmp_path) not in captured.out
    assert "secret" not in captured.out.casefold()


def test_download_cli_returns_conventional_code_after_controlled_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import download_model as module

    def terminate(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise module._ParentTermination(signal.SIGTERM)

    monkeypatch.setattr(module, "run_download_isolated", terminate)

    assert module.main(["--destination", str(tmp_path / "snapshot")]) == 143
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def _delayed_streaming_download(**kwargs: Any) -> str:
    staging = Path(kwargs["local_dir"])
    (staging / "config.json").write_text("{}", encoding="utf-8")

    def delayed_output() -> None:
        time.sleep(0.05)
        print("DOWNLOAD_POST_RETURN_PYTHON_SECRET", flush=True)
        os.write(1, b"DOWNLOAD_POST_RETURN_STDOUT_SECRET\n")
        os.write(2, b"DOWNLOAD_POST_RETURN_STDERR_SECRET\n")

    threading.Thread(target=delayed_output, daemon=False).start()
    return str(staging)


class _SigtermIgnoringDownload:
    def __init__(self, child_pid: Path, survival_marker: Path) -> None:
        self.child_pid = child_pid
        self.survival_marker = survival_marker

    def __call__(self, **kwargs: Any) -> str:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        self.child_pid.write_text(str(os.getpid()), encoding="ascii")
        time.sleep(3.0)
        self.survival_marker.write_text("orphan survived", encoding="utf-8")
        staging = Path(kwargs["local_dir"])
        (staging / "config.json").write_text("{}", encoding="utf-8")
        return str(staging)


def _download_parent_for_termination_test(
    destination: str, child_pid: str, survival_marker: str
) -> None:
    from scripts.download_model import run_download_isolated

    try:
        run_download_isolated(
            destination,
            snapshot_download=_SigtermIgnoringDownload(
                Path(child_pid), Path(survival_marker)
            ),
        )
    except BaseException:
        raise SystemExit(143) from None


def test_download_parent_sigterm_reaps_resistant_child_and_releases_lock(
    tmp_path: Path,
) -> None:
    from scripts.download_model import download_model

    destination = tmp_path / "snapshot"
    child_pid_path = tmp_path / "child.pid"
    survival_marker = tmp_path / "child-survived.txt"
    context = __import__("multiprocessing").get_context("spawn")
    parent = context.Process(
        target=_download_parent_for_termination_test,
        args=(str(destination), str(child_pid_path), str(survival_marker)),
    )
    child_pid: int | None = None
    try:
        parent.start()
        deadline = time.monotonic() + 5.0
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text(encoding="ascii"))

        os.kill(parent.pid, signal.SIGTERM)
        parent.join(timeout=8.0)

        assert parent.is_alive() is False
        assert parent.exitcode not in (0, None)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert not survival_marker.exists()
        assert not destination.exists()

        def complete_resume(**kwargs: Any) -> str:
            staging = Path(kwargs["local_dir"])
            (staging / "config.json").write_text("{}", encoding="utf-8")
            return str(staging)

        assert download_model(
            destination, snapshot_download=complete_resume
        ).is_file()
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join(timeout=2.0)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP is unavailable")
def test_download_parent_sighup_requests_unwind_and_restores_handler() -> None:
    from scripts import download_model as module

    previous = signal.getsignal(signal.SIGHUP)
    with pytest.raises(module._ParentTermination) as captured:
        with module._controlled_parent_termination():
            os.kill(os.getpid(), signal.SIGHUP)

    assert captured.value.signum == signal.SIGHUP
    assert signal.getsignal(signal.SIGHUP) is previous


def test_download_process_isolates_delayed_python_and_native_streams(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    from scripts.download_model import run_download_isolated

    result = run_download_isolated(
        tmp_path / "snapshot",
        snapshot_download=_delayed_streaming_download,
    )

    captured = capfd.readouterr()
    assert result == {"manifest": "sha256-manifest.json", "status": "completed"}
    assert "DOWNLOAD_POST_RETURN" not in captured.out
    assert "DOWNLOAD_POST_RETURN" not in captured.err


@pytest.mark.parametrize(
    ("model_id", "revision"),
    [
        ("other/model", "main"),
        ("Qwen/Qwen3-VL-8B-Instruct", "feature-safe-name"),
        ("Qwen/Qwen3-VL-8B-Instruct", "../../main"),
        ("Qwen/Qwen3-VL-8B-Instruct", "refs/heads/main.lock"),
        ("Qwen/Qwen3-VL-8B-Instruct", "branch@{1}"),
    ],
)
def test_download_exporter_rejects_unallowlisted_id_and_unsafe_revision(
    tmp_path: Path, model_id: str, revision: str
) -> None:
    from scripts.download_model import DownloadSafetyError, download_model

    called = False

    def must_not_download(**kwargs: Any) -> str:
        nonlocal called
        called = True
        return kwargs["local_dir"]

    with pytest.raises(DownloadSafetyError):
        download_model(
            tmp_path / "snapshot",
            model_id=model_id,
            revision=revision,
            snapshot_download=must_not_download,
        )
    assert called is False


def test_download_exporter_refuses_broad_symlink_and_nonempty_destinations(
    tmp_path: Path,
) -> None:
    from scripts.download_model import DownloadSafetyError, download_model

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    for destination in (Path("/"), Path.home(), nonempty, linked):
        with pytest.raises(DownloadSafetyError):
            download_model(
                destination,
                snapshot_download=lambda **kwargs: kwargs["local_dir"],
            )


def test_download_exporter_accepts_an_immutable_commit_revision(tmp_path: Path) -> None:
    from scripts.download_model import download_model

    revision = "a" * 40
    destination = tmp_path / "snapshot"
    seen_revision: list[str] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        seen_revision.append(kwargs["revision"])
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(exist_ok=True)
        (local_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    manifest = download_model(
        destination,
        revision=revision,
        snapshot_download=fake_snapshot_download,
    )

    assert seen_revision == [revision]
    assert json.loads(manifest.read_text(encoding="utf-8"))["revision"] == revision


class _SmokeCuda:
    def __init__(self, *, current_device: int = 2) -> None:
        self.current = current_device
        self.set_calls: list[int] = []
        self.reset_calls: list[int] = []

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 4

    def set_device(self, device: int) -> None:
        self.current = device
        self.set_calls.append(device)

    def current_device(self) -> int:
        return self.current

    def reset_peak_memory_stats(self, device: int) -> None:
        self.reset_calls.append(device)

    def max_memory_allocated(self, device: int) -> int:
        return 123_456

    def get_device_name(self, device: int) -> str:
        return f"Fake GPU {device}"


class _SmokeBackend:
    def __init__(self, *, correct_device: bool = True) -> None:
        self.correct_device = correct_device
        self.requests: list[ModelRequest] = []
        self.model = SimpleNamespace(
            __class__=SimpleNamespace(__name__="FakeQwen3VL"),
            config=SimpleNamespace(model_type="qwen3_vl"),
        )

    def loaded_on_assigned_device(self) -> bool:
        return self.correct_device

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        return {"ok": True}


class _InlineQueue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)

    def get(self, timeout: float | None = None) -> Any:
        del timeout
        if not self.items:
            raise __import__("queue").Empty
        return self.items.pop(0)


class _InlineProcess:
    def __init__(self, *, target: Any, args: tuple[Any, ...]) -> None:
        self.target = target
        self.args = args
        self.exitcode: int | None = None
        self.started = False
        self.terminated = False

    def start(self) -> None:
        self.started = True
        try:
            self.target(*self.args)
        except BaseException:
            self.exitcode = 1
        else:
            self.exitcode = 0

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminated = True


class _InlineContext:
    def __init__(self) -> None:
        self.processes: list[_InlineProcess] = []

    def Queue(self) -> _InlineQueue:
        return _InlineQueue()

    def Process(self, *, target: Any, args: tuple[Any, ...]) -> _InlineProcess:
        process = _InlineProcess(target=target, args=args)
        self.processes.append(process)
        return process


class _InterruptingProcess(_InlineProcess):
    def __init__(self, *, target: Any, args: tuple[Any, ...]) -> None:
        super().__init__(target=target, args=args)
        self.alive = False

    def start(self) -> None:
        self.started = True
        self.alive = True

    def join(self, timeout: float | None = None) -> None:
        del timeout
        if self.alive:
            raise KeyboardInterrupt

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


class _InterruptingContext(_InlineContext):
    def Process(self, *, target: Any, args: tuple[Any, ...]) -> _InterruptingProcess:
        process = _InterruptingProcess(target=target, args=args)
        self.processes.append(process)
        return process


class _StartInterruptingProcess(_InterruptingProcess):
    def start(self) -> None:
        self.started = True
        self.alive = True
        raise KeyboardInterrupt("interrupted during process start")


class _StartInterruptingContext(_InlineContext):
    def Process(self, *, target: Any, args: tuple[Any, ...]) -> _StartInterruptingProcess:
        process = _StartInterruptingProcess(target=target, args=args)
        self.processes.append(process)
        return process


class _HiddenHandleSigintProcess(_InterruptingProcess):
    def __init__(self, *, target: Any, args: tuple[Any, ...]) -> None:
        super().__init__(target=target, args=args)
        self.child_alive = False
        self.handle_assigned = False

    def start(self) -> None:
        self.started = True
        self.child_alive = True
        os.kill(os.getpid(), signal.SIGINT)
        self.handle_assigned = True

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def is_alive(self) -> bool:
        return self.child_alive if self.handle_assigned else False

    def terminate(self) -> None:
        self.terminated = True
        self.child_alive = False


class _HiddenHandleSigintContext(_InlineContext):
    def Process(self, *, target: Any, args: tuple[Any, ...]) -> _HiddenHandleSigintProcess:
        process = _HiddenHandleSigintProcess(target=target, args=args)
        self.processes.append(process)
        return process


class _SecondSignalDuringReapProcess(_InlineProcess):
    def __init__(self, *, target: Any, args: tuple[Any, ...]) -> None:
        super().__init__(target=target, args=args)
        self.alive = False
        self.timed_join_calls = 0
        self.killed = False

    def start(self) -> None:
        self.started = True
        self.alive = True

    def join(self, timeout: float | None = None) -> None:
        if timeout is None:
            raise KeyboardInterrupt("first interruption")
        self.timed_join_calls += 1
        if self.timed_join_calls == 1:
            os.kill(os.getpid(), signal.SIGTERM)

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False


class _SecondSignalDuringReapContext(_InlineContext):
    def Process(
        self, *, target: Any, args: tuple[Any, ...]
    ) -> _SecondSignalDuringReapProcess:
        process = _SecondSignalDuringReapProcess(target=target, args=args)
        self.processes.append(process)
        return process


def test_download_process_is_reaped_if_start_is_interrupted_after_child_creation(
    tmp_path: Path,
) -> None:
    from scripts.download_model import run_download_isolated

    context = _StartInterruptingContext()
    with pytest.raises(KeyboardInterrupt, match="during process start"):
        run_download_isolated(
            tmp_path / "snapshot",
            snapshot_download=_delayed_streaming_download,
            mp_context=context,
        )

    [process] = context.processes
    assert process.terminated is True
    assert process.is_alive() is False


def test_download_defers_sigint_until_a_new_child_handle_is_visible(
    tmp_path: Path,
) -> None:
    from scripts.download_model import run_download_isolated

    context = _HiddenHandleSigintContext()
    with pytest.raises(KeyboardInterrupt):
        run_download_isolated(
            tmp_path / "snapshot",
            snapshot_download=_delayed_streaming_download,
            mp_context=context,
        )

    [process] = context.processes
    assert process.handle_assigned is True
    assert process.terminated is True
    assert process.child_alive is False


def test_download_process_is_reaped_when_parent_is_interrupted(
    tmp_path: Path,
) -> None:
    from scripts.download_model import run_download_isolated

    context = _InterruptingContext()
    with pytest.raises(KeyboardInterrupt):
        run_download_isolated(
            tmp_path / "snapshot",
            snapshot_download=_delayed_streaming_download,
            mp_context=context,
        )

    [process] = context.processes
    assert process.terminated is True
    assert process.is_alive() is False


def test_download_reap_defers_a_second_termination_signal_until_child_is_dead(
    tmp_path: Path,
) -> None:
    from scripts.download_model import run_download_isolated

    context = _SecondSignalDuringReapContext()
    with pytest.raises(KeyboardInterrupt, match="first interruption"):
        run_download_isolated(
            tmp_path / "snapshot",
            snapshot_download=_delayed_streaming_download,
            mp_context=context,
        )

    [process] = context.processes
    assert process.terminated is True
    assert process.killed is True
    assert process.is_alive() is False


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"), reason="atomic POSIX masking unavailable"
)
def test_download_reap_atomically_installs_all_signal_deferrals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    context = _SecondSignalDuringReapContext()
    real_signal = module.signal.signal
    injected = False

    def interrupt_handler_install(signum: int, handler: Any) -> Any:
        nonlocal injected
        previous = real_signal(signum, handler)
        if (
            not injected
            and signum == signal.SIGINT
            and "_defer_signals_during_reap" in getattr(handler, "__qualname__", "")
        ):
            injected = True
            os.kill(os.getpid(), signal.SIGTERM)
        return previous

    monkeypatch.setattr(module.signal, "signal", interrupt_handler_install)
    with pytest.raises(KeyboardInterrupt, match="first interruption"):
        module.run_download_isolated(
            tmp_path / "snapshot",
            snapshot_download=_delayed_streaming_download,
            mp_context=context,
        )

    [process] = context.processes
    assert injected is True
    assert process.terminated is True
    assert process.killed is True
    assert process.is_alive() is False


def test_download_child_fails_closed_if_lifetime_stream_isolation_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import download_model as module

    result_queue = _InlineQueue()
    monkeypatch.setattr(
        module,
        "_install_lifetime_output_sink",
        lambda: (_ for _ in ()).throw(RuntimeError("token=must-not-escape")),
    )

    module._download_process_worker(
        result_queue,
        str(tmp_path / "snapshot"),
        module.DEFAULT_MODEL_ID,
        "main",
        lambda **kwargs: pytest.fail("dependency must not run"),
    )

    assert result_queue.items == [{"status": "failed"}]


def test_gpu_child_fails_closed_if_lifetime_stream_isolation_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import gpu_smoke as module

    result_queue = _InlineQueue()
    monkeypatch.setattr(
        module.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="SpawnProcess-1"),
    )
    monkeypatch.setattr(
        module,
        "_install_lifetime_output_sink",
        lambda: (_ for _ in ()).throw(RuntimeError("secret")),
    )

    module._isolated_worker(
        lambda *args: pytest.fail("worker must not run"),
        result_queue,
        str(tmp_path / "model"),
        str(tmp_path / "video.mp4"),
        0,
    )

    assert result_queue.items == []


def _passing_smoke_worker(
    queue: _InlineQueue, model_dir: str, video: str, device: int
) -> None:
    del model_dir, video
    queue.put(
        {
            "status": "passed",
            "assigned_device": device,
            "observed_device": device,
            "gpu_name": f"Fake GPU {device}",
            "model_class": "Qwen3VLForConditionalGeneration",
            "model_type": "qwen3_vl",
            "torch_version": "2.10.0",
            "transformers_version": "4.57.1",
            "qwen_vl_utils_version": "0.0.14",
            "device_scope": "unmasked-logical-ordinal",
            "latency_seconds": 1.25,
            "peak_allocated_bytes": 1000 + device,
        }
    )


def _unsafe_failure_worker(
    queue: _InlineQueue, model_dir: str, video: str, device: int
) -> None:
    del model_dir, video
    queue.put(
        {
            "status": "failed",
            "assigned_device": device,
            "observed_device": device + 1,
            "error_code": "api_token=do-not-print",
            "detail": "authorization: Bearer secret",
        }
    )


def _report_then_fail_worker(
    queue: _InlineQueue, model_dir: str, video: str, device: int
) -> None:
    _passing_smoke_worker(queue, model_dir, video, device)
    raise RuntimeError("crashed after report")


def _streaming_smoke_worker(
    queue: _InlineQueue, model_dir: str, video: str, device: int
) -> None:
    print("token=python-child-secret")
    os.write(1, b"authorization=native-child-stdout-secret\n")
    os.write(2, b"password=native-child-stderr-secret\n")
    _passing_smoke_worker(queue, model_dir, video, device)


def _delayed_streaming_smoke_worker(
    queue: _InlineQueue, model_dir: str, video: str, device: int
) -> None:
    _passing_smoke_worker(queue, model_dir, video, device)

    def delayed_output() -> None:
        time.sleep(0.05)
        print("SMOKE_POST_RETURN_PYTHON_SECRET", flush=True)
        os.write(1, b"SMOKE_POST_RETURN_STDOUT_SECRET\n")
        os.write(2, b"SMOKE_POST_RETURN_STDERR_SECRET\n")

    threading.Thread(target=delayed_output, daemon=False).start()


def test_gpu_probe_pins_device_and_records_sanitized_runtime_metrics(
    tmp_path: Path,
) -> None:
    from scripts.gpu_smoke import _probe_device

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    cuda = _SmokeCuda(current_device=0)
    torch = SimpleNamespace(cuda=cuda, __version__="2.10.0+cu128")
    backend = _SmokeBackend()
    load_calls: list[tuple[Path, str, str]] = []

    def load_backend(path: Path, device: str, dtype: str) -> _SmokeBackend:
        load_calls.append((path, device, dtype))
        return backend

    moments = iter((10.0, 12.5))
    report = _probe_device(
        model_dir,
        video,
        2,
        torch_module=torch,
        backend_loader=load_backend,
        version_lookup=lambda package: {
            "transformers": "4.57.1",
            "qwen-vl-utils": "0.0.14",
        }[package],
        clock=lambda: next(moments),
        visible_devices=None,
    )

    assert cuda.set_calls == [2]
    assert cuda.reset_calls == [2]
    assert load_calls == [(model_dir.resolve(), "cuda:2", "auto")]
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.stage == "general_segment"
    assert request.video_path == video.resolve()
    assert request.span == TimeSpan(0.0, 1.0)
    assert request.fps == 1.0
    assert request.prompt == 'Return exactly this JSON object: {"ok":true}'
    assert report == {
        "status": "passed",
        "assigned_device": 2,
        "observed_device": 2,
        "gpu_name": "Fake GPU 2",
        "model_class": "SimpleNamespace",
        "model_type": "qwen3_vl",
        "torch_version": "2.10.0+cu128",
        "transformers_version": "4.57.1",
        "qwen_vl_utils_version": "0.0.14",
        "device_scope": "unmasked-logical-ordinal",
        "latency_seconds": 2.5,
        "peak_allocated_bytes": 123_456,
    }


def test_gpu_smoke_starts_one_spawnable_worker_per_unique_device(
    tmp_path: Path,
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    context = _InlineContext()

    reports = run_gpu_smoke(
        model_dir,
        video,
        (0, 1, 2, 3),
        timeout_seconds=10.0,
        mp_context=context,
        worker_target=_passing_smoke_worker,
    )

    assert [report["assigned_device"] for report in reports] == [0, 1, 2, 3]
    assert all(report["status"] == "passed" for report in reports)
    assert len(context.processes) == 4
    assert all(process.started for process in context.processes)


def test_gpu_smoke_replaces_child_errors_with_allowlisted_failure_codes(
    tmp_path: Path,
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    reports = run_gpu_smoke(
        model_dir,
        video,
        (0,),
        timeout_seconds=10.0,
        mp_context=_InlineContext(),
        worker_target=_unsafe_failure_worker,
    )

    assert reports == [
        {
            "status": "failed",
            "assigned_device": 0,
            "error_code": "WRONG_DEVICE",
        }
    ]
    assert "secret" not in json.dumps(reports).lower()
    assert "token" not in json.dumps(reports).lower()


def test_gpu_smoke_rejects_success_report_from_nonzero_child(tmp_path: Path) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    reports = run_gpu_smoke(
        model_dir,
        video,
        (0,),
        timeout_seconds=10.0,
        mp_context=_InlineContext(),
        worker_target=_report_then_fail_worker,
    )

    assert reports == [
        {
            "status": "failed",
            "assigned_device": 0,
            "error_code": "PROCESS_EXITED",
        }
    ]


def test_gpu_smoke_terminates_all_spawned_children_when_parent_is_interrupted(
    tmp_path: Path,
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    context = _InterruptingContext()

    with pytest.raises(KeyboardInterrupt):
        run_gpu_smoke(
            model_dir,
            video,
            (0, 1),
            timeout_seconds=10.0,
            mp_context=context,
            worker_target=_passing_smoke_worker,
        )

    assert len(context.processes) == 2
    assert all(process.terminated for process in context.processes)


def test_gpu_smoke_defers_sigint_until_a_new_child_handle_is_visible(
    tmp_path: Path,
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    context = _HiddenHandleSigintContext()

    with pytest.raises(KeyboardInterrupt):
        run_gpu_smoke(
            model_dir,
            video,
            (0,),
            timeout_seconds=10.0,
            mp_context=context,
            worker_target=_passing_smoke_worker,
        )

    [process] = context.processes
    assert process.handle_assigned is True
    assert process.terminated is True
    assert process.child_alive is False


class _TerminateResistantProcess(_InterruptingProcess):
    def __init__(self, *, target: Any, args: tuple[Any, ...]) -> None:
        super().__init__(target=target, args=args)
        self.killed = False

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        self.exitcode = -9


class _TerminateResistantContext(_InlineContext):
    def Process(
        self, *, target: Any, args: tuple[Any, ...]
    ) -> _TerminateResistantProcess:
        process = _TerminateResistantProcess(target=target, args=args)
        self.processes.append(process)
        return process


class _InterruptThenResistProcess(_TerminateResistantProcess):
    def __init__(self, *, target: Any, args: tuple[Any, ...]) -> None:
        super().__init__(target=target, args=args)
        self.joins = 0

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.joins += 1
        if self.joins == 1:
            raise KeyboardInterrupt


class _InterruptThenResistContext(_InlineContext):
    def Process(
        self, *, target: Any, args: tuple[Any, ...]
    ) -> _InterruptThenResistProcess:
        process = _InterruptThenResistProcess(target=target, args=args)
        self.processes.append(process)
        return process


def test_download_ctrl_c_kills_and_rechecks_a_terminate_resistant_child(
    tmp_path: Path,
) -> None:
    from scripts.download_model import run_download_isolated

    context = _InterruptThenResistContext()
    with pytest.raises(KeyboardInterrupt):
        run_download_isolated(
            tmp_path / "snapshot",
            snapshot_download=_delayed_streaming_download,
            mp_context=context,
        )

    [process] = context.processes
    assert process.terminated is True
    assert process.killed is True
    assert process.is_alive() is False


def test_gpu_smoke_kills_and_rechecks_a_terminate_resistant_timed_out_child(
    tmp_path: Path,
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    context = _TerminateResistantContext()

    reports = run_gpu_smoke(
        model_dir,
        video,
        (0,),
        timeout_seconds=0.001,
        mp_context=context,
        worker_target=_passing_smoke_worker,
    )

    assert reports == [
        {"status": "failed", "assigned_device": 0, "error_code": "PROCESS_TIMEOUT"}
    ]
    [process] = context.processes
    assert process.terminated is True
    assert process.killed is True
    assert process.is_alive() is False


def test_gpu_smoke_kills_a_terminate_resistant_child_during_parent_interrupt(
    tmp_path: Path,
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    context = _InterruptThenResistContext()

    with pytest.raises(KeyboardInterrupt):
        run_gpu_smoke(
            model_dir,
            video,
            (0,),
            timeout_seconds=10.0,
            mp_context=context,
            worker_target=_passing_smoke_worker,
        )

    [process] = context.processes
    assert process.terminated is True
    assert process.killed is True
    assert process.is_alive() is False


def test_gpu_smoke_suppresses_complete_python_and_native_child_streams(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    reports = run_gpu_smoke(
        model_dir,
        video,
        (0,),
        timeout_seconds=10.0,
        mp_context=_InlineContext(),
        worker_target=_streaming_smoke_worker,
    )

    captured = capfd.readouterr()
    assert reports[0]["status"] == "passed"
    assert "secret" not in captured.out.casefold() + captured.err.casefold()


def test_gpu_smoke_real_spawn_isolates_and_sanitizes_child_output(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    reports = run_gpu_smoke(
        model_dir,
        video,
        (0,),
        timeout_seconds=10.0,
        worker_target=_streaming_smoke_worker,
    )

    captured = capfd.readouterr()
    assert reports[0]["status"] == "passed"
    assert "secret" not in captured.out.casefold() + captured.err.casefold()


def test_gpu_smoke_real_spawn_isolates_delayed_child_output_for_process_lifetime(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    from scripts.gpu_smoke import run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    reports = run_gpu_smoke(
        model_dir,
        video,
        (0,),
        timeout_seconds=10.0,
        worker_target=_delayed_streaming_smoke_worker,
    )

    captured = capfd.readouterr()
    assert reports[0]["status"] == "passed"
    assert "SMOKE_POST_RETURN" not in captured.out
    assert "SMOKE_POST_RETURN" not in captured.err


def test_gpu_smoke_acceptance_rejects_a_mask_without_exposing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.gpu_smoke import SmokeSafetyError, run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-customer-secret,GPU-other-secret")

    with pytest.raises(SmokeSafetyError, match="visibility mask") as captured:
        run_gpu_smoke(
            model_dir,
            video,
            (0,),
            require_unmasked=True,
            mp_context=_InlineContext(),
            worker_target=_passing_smoke_worker,
        )
    assert "customer" not in str(captured.value).casefold()


def test_gpu_smoke_distinguishes_an_empty_visibility_mask_from_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.gpu_smoke import SmokeSafetyError, _device_scope, run_gpu_smoke

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert _device_scope("") == "empty-mask-no-visible-devices"
    with pytest.raises(SmokeSafetyError, match="visibility mask"):
        run_gpu_smoke(
            model_dir,
            video,
            (0,),
            require_unmasked=True,
            mp_context=_InlineContext(),
            worker_target=_passing_smoke_worker,
        )


def test_gpu_smoke_cli_reports_an_unreapable_child_with_only_a_stable_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import gpu_smoke

    model_dir = tmp_path / "model token=secret"
    model_dir.mkdir()
    video = tmp_path / "clip token=secret.mp4"
    video.write_bytes(b"video")

    def fail(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        del args, kwargs
        raise gpu_smoke.SmokeProcessError(
            "private child diagnostic token=must-not-survive"
        )

    monkeypatch.setattr(gpu_smoke, "run_gpu_smoke", fail)

    assert gpu_smoke.main(
        [
            "--model-dir",
            str(model_dir),
            "--video",
            str(video),
            "--devices",
            "0",
        ]
    ) == 1

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "error_code": "PROCESS_UNREAPABLE",
        "status": "failed",
    }
    assert "secret" not in captured.out.casefold()


@pytest.mark.parametrize("value", ["", "0,0", "-1", "0,two", "0,,1"])
def test_gpu_device_list_rejects_empty_duplicate_and_malformed_values(value: str) -> None:
    from scripts.gpu_smoke import SmokeSafetyError, parse_devices

    with pytest.raises(SmokeSafetyError):
        parse_devices(value)


def test_model_utility_scripts_are_import_safe_and_help_needs_no_optional_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    for script in ("download_model.py", "gpu_smoke.py"):
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / script), "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize(
    "script",
    [
        "download_model.py",
        "gpu_smoke.py",
    ],
)
def test_model_utility_cli_parse_errors_do_not_echo_untrusted_arguments(
    script: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    arguments = (
        ["--destination", "unused", "--model-id", "token=cli-secret"]
        if script == "download_model.py"
        else [
            "--model-dir",
            str(root),
            "--video",
            str(root / "pyproject.toml"),
            "--devices",
            "token=cli-secret",
        ]
    )
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / script), *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "secret" not in (completed.stdout + completed.stderr).casefold()
