"""Offline, single-device Qwen3-VL video inference backend."""

from __future__ import annotations

import importlib
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ..media import FrameRef, TimeSpan, VideoMetadata, extract_frames, probe_video
from ..model_alias import validate_model_alias
from .base import (
    DEFAULT_MAX_MODEL_OUTPUT_CHARS,
    ModelRequest,
    VideoSession,
    parse_strict_json,
)


_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)\Z")
_DTYPES = frozenset({"auto", "bfloat16", "float16", "float32"})
STAGE_MAX_NEW_TOKENS = {
    "active_objects": 1_024,
    "general_segment": 4_096,
    "general_summary": 2_048,
    "embodied_pass_a": 4_096,
    "embodied_pass_b": 8_192,
    "embodied_enrichment": 4_096,
}
_MEDIA_RESOLUTION_PIXELS = {
    "low": (4 * 32 * 32, 64 * 32 * 32),
    "medium": (4 * 32 * 32, 128 * 32 * 32),
    "high": (4 * 32 * 32, 256 * 32 * 32),
}
_CLIP_CONTEXT_TOTAL_PIXELS = {
    "low": 4_096 * 32 * 32,
    "medium": 8_192 * 32 * 32,
    "high": 20_480 * 32 * 32,
}
_REASONING_TOKEN_FRACTION = {"low": 0.5, "medium": 0.75, "high": 1.0}
_CACHE_PREFIX = "qwen3_vl:prepared:"

FrameExtractor = Callable[[Path, TimeSpan, float, Path], list[FrameRef]]
VideoProbe = Callable[[Path], VideoMetadata]


@dataclass
class _PreparedVisual:
    video_element: dict[str, Any]
    image_inputs: Any
    video_inputs: Any
    video_metadata: list[dict[str, Any]]
    video_kwargs: dict[str, Any]
    frames: tuple[FrameRef, ...]
    temporary_dir: Path


class QwenBackendError(RuntimeError):
    """Base class for safe Qwen backend configuration failures."""


class QwenDependencyError(QwenBackendError):
    """Optional GPU dependencies are not installed or are incompatible."""


class LocalModelError(QwenBackendError):
    """A configured model snapshot is not one existing local directory."""


class ModelAliasError(QwenBackendError):
    """A request model name is absent from the configured alias allowlist."""


class QwenStageError(QwenBackendError):
    """The request stage has no finite generation budget."""


def resolve_model_alias(
    model_name: str,
    model_registry: Mapping[str, Path],
) -> Path:
    """Resolve an exact configured alias without treating the alias as a path."""
    try:
        model_name = validate_model_alias(model_name)
    except ValueError:
        raise ModelAliasError("model alias is not allowlisted") from None
    try:
        configured_path = model_registry[model_name]
    except (KeyError, TypeError):
        raise ModelAliasError("model alias is not allowlisted") from None
    return _local_model_directory(configured_path)


class Qwen3VLModel:
    """Generate strict JSON from one local video clip on one CUDA device."""

    def __init__(
        self,
        *,
        processor: Any,
        model: Any,
        torch_module: Any,
        process_vision_info: Any,
        device: str,
        max_output_chars: int = DEFAULT_MAX_MODEL_OUTPUT_CHARS,
        cache_clear_threshold_bytes: int | None = None,
        frame_extractor: FrameExtractor = extract_frames,
        video_probe: VideoProbe = probe_video,
    ) -> None:
        self.processor = processor
        self.model = model
        self._torch = torch_module
        self._process_vision_info = process_vision_info
        self.device = _assigned_cuda_device(device)
        if (
            isinstance(max_output_chars, bool)
            or not isinstance(max_output_chars, int)
            or max_output_chars <= 0
        ):
            raise ValueError("max_output_chars must be a positive integer")
        if cache_clear_threshold_bytes is not None and (
            isinstance(cache_clear_threshold_bytes, bool)
            or not isinstance(cache_clear_threshold_bytes, int)
            or cache_clear_threshold_bytes <= 0
        ):
            raise ValueError(
                "cache_clear_threshold_bytes must be a positive integer or None"
            )
        self.max_output_chars = max_output_chars
        self.cache_clear_threshold_bytes = cache_clear_threshold_bytes
        if not callable(frame_extractor) or not callable(video_probe):
            raise TypeError("frame_extractor and video_probe must be callable")
        self._frame_extractor = frame_extractor
        self._video_probe = video_probe
        self._sessions: dict[str, VideoSession] = {}

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        device: str,
        dtype: str = "auto",
        *,
        max_output_chars: int = DEFAULT_MAX_MODEL_OUTPUT_CHARS,
        cache_clear_threshold_bytes: int | None = None,
    ) -> Qwen3VLModel:
        """Load one pre-downloaded snapshot without remote-code or network fallback."""
        local_path = _local_model_directory(model_path)
        assigned_device = _assigned_cuda_device(device)
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise ValueError("dtype must be auto, bfloat16, float16, or float32")

        with _offline_load_environment():
            runtime = _import_runtime_dependencies()
            shared = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            model_kwargs = {
                **shared,
                "torch_dtype": dtype,
                "device_map": {"": assigned_device},
            }
            try:
                model = runtime.model_class.from_pretrained(
                    str(local_path), **model_kwargs
                )
            except TypeError as error:
                if not _is_torch_dtype_rename(error):
                    raise
                model_kwargs.pop("torch_dtype")
                model_kwargs["dtype"] = dtype
                model = runtime.model_class.from_pretrained(
                    str(local_path), **model_kwargs
                )
            processor = runtime.processor_class.from_pretrained(
                str(local_path), **shared
            )
        model.eval()
        backend = cls(
            processor=processor,
            model=model,
            torch_module=runtime.torch,
            process_vision_info=runtime.process_vision_info,
            device=assigned_device,
            max_output_chars=max_output_chars,
            cache_clear_threshold_bytes=cache_clear_threshold_bytes,
        )
        if not backend.loaded_on_assigned_device():
            raise QwenBackendError("model was not loaded on the assigned CUDA device")
        return backend

    @classmethod
    def load_alias(
        cls,
        model_name: str,
        model_registry: Mapping[str, Path],
        device: str,
        dtype: str = "auto",
        *,
        max_output_chars: int = DEFAULT_MAX_MODEL_OUTPUT_CHARS,
        cache_clear_threshold_bytes: int | None = None,
    ) -> Qwen3VLModel:
        """Load only the local path selected by an exact configured alias."""
        return cls.load(
            resolve_model_alias(model_name, model_registry),
            device,
            dtype,
            max_output_chars=max_output_chars,
            cache_clear_threshold_bytes=cache_clear_threshold_bytes,
        )

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        """Run deterministic video generation and parse exactly one JSON object."""
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")
        if request.stage not in STAGE_MAX_NEW_TOKENS:
            raise QwenStageError(f"unsupported Qwen model stage: {request.stage}")
        video_path = _local_video_file(request.video_path)
        prepared: _PreparedVisual | None = None
        retained = False

        image_inputs: Any = None
        video_inputs: Any = None
        video_metadata: Any = None
        vision_result: Any = None
        video_kwargs: Any = None
        processor_kwargs: Any = None
        inputs: Any = None
        generation_kwargs: Any = None
        generated_ids: Any = None
        input_ids: Any = None
        trimmed_ids: Any = None
        decoded: Any = None
        try:
            prepared, retained = self._prepared_visual(request, video_path)
            messages = _messages(request, prepared.video_element)
            rendered_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs = prepared.image_inputs
            video_inputs = prepared.video_inputs
            video_metadata = prepared.video_metadata
            video_kwargs = prepared.video_kwargs
            processor_kwargs = {
                "text": rendered_prompt,
                "images": image_inputs,
                "videos": video_inputs,
                "padding": True,
                "return_tensors": "pt",
                "do_resize": False,
            }
            if video_metadata is not None:
                processor_kwargs["video_metadata"] = video_metadata
            _merge_video_kwargs(processor_kwargs, video_kwargs)
            inputs = self.processor(**processor_kwargs)
            if not isinstance(inputs, Mapping) or not callable(
                getattr(inputs, "to", None)
            ):
                raise QwenDependencyError("Qwen processor returned incompatible inputs")
            inputs = inputs.to(self.device)
            if not isinstance(inputs, Mapping):
                raise QwenDependencyError("Qwen inputs cannot be moved to the CUDA device")
            generation_kwargs = dict(inputs)
            generation_kwargs.update(
                do_sample=False,
                max_new_tokens=_max_new_tokens(request),
            )
            with self._torch.inference_mode():
                generated_ids = self.model.generate(**generation_kwargs)
            generated_ids = getattr(generated_ids, "sequences", generated_ids)
            input_ids = _input_ids(inputs)
            trimmed_ids = _strip_prompt_tokens(input_ids, generated_ids)
            decoded = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if (
                not isinstance(decoded, (list, tuple))
                or len(decoded) != 1
                or not isinstance(decoded[0], str)
            ):
                raise QwenDependencyError("Qwen processor returned incompatible text")
            return parse_strict_json(decoded[0], max_chars=self.max_output_chars)
        finally:
            image_inputs = None
            video_inputs = None
            video_metadata = None
            vision_result = None
            video_kwargs = None
            processor_kwargs = None
            inputs = None
            generation_kwargs = None
            generated_ids = None
            input_ids = None
            trimmed_ids = None
            decoded = None
            if prepared is not None and not retained:
                _release_prepared(prepared)
            prepared = None
            self._clear_cache_under_pressure()

    def release_request(self, request: ModelRequest) -> None:
        """Release request-local state; session state remains intentionally reusable."""
        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a ModelRequest")

    def release_video_session(self, session_id: str) -> None:
        """Release only CPU visual state owned by one worker-local video session."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-blank string")
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        for key, value in tuple(session.backend_cache.items()):
            if key.startswith(_CACHE_PREFIX) and isinstance(value, _PreparedVisual):
                _release_prepared(value)
                del session.backend_cache[key]
        session.sampled_frames = ()

    def _prepared_visual(
        self, request: ModelRequest, video_path: Path
    ) -> tuple[_PreparedVisual, bool]:
        session = request.video_session
        key = _visual_cache_key(request, video_path)
        if session is not None:
            cached = session.backend_cache.get(key)
            if cached is not None:
                if not isinstance(cached, _PreparedVisual):
                    raise QwenDependencyError("video session cache is incompatible")
                _validate_prepared_files(cached)
                _require_cpu_visual_inputs(cached.image_inputs)
                _require_cpu_visual_inputs(cached.video_inputs)
                self._remember_session(request, session)
                return cached, True

        temporary_dir = Path(tempfile.mkdtemp(prefix="las-qwen-video-frames-"))
        try:
            try:
                extracted_frames = self._frame_extractor(
                    video_path,
                    request.span,
                    request.fps,
                    temporary_dir,
                )
            except Exception:
                raise QwenDependencyError(
                    "video-only frame extraction failed"
                ) from None
            frames = _validated_frames(
                extracted_frames,
                temporary_dir,
                request.span,
            )
            try:
                metadata = (
                    session.metadata
                    if session is not None
                    else self._video_probe(video_path)
                )
            except Exception:
                raise QwenDependencyError("video metadata is unavailable") from None
            _validate_video_metadata(metadata, request.span)
            video_element = _video_element(request, frames)
            visual_messages = [{"role": "user", "content": [video_element]}]
            vision_result = self._process_vision_info(
                visual_messages,
                image_patch_size=_image_patch_size(self.processor),
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            image_inputs, video_inputs, video_kwargs = _vision_result(vision_result)
            video_inputs, _ = _split_video_metadata(video_inputs)
            if video_inputs is None:
                raise QwenDependencyError("qwen-vl-utils returned no video inputs")
            _require_cpu_visual_inputs(image_inputs)
            _require_cpu_visual_inputs(video_inputs)
            prepared = _PreparedVisual(
                video_element=video_element,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
                video_metadata=[_absolute_video_metadata(metadata, frames)],
                video_kwargs=video_kwargs,
                frames=frames,
                temporary_dir=temporary_dir,
            )
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        if session is not None:
            session.backend_cache[key] = prepared
            session.sampled_frames = frames
            self._remember_session(request, session)
            return prepared, True
        return prepared, False

    def _remember_session(self, request: ModelRequest, session: VideoSession) -> None:
        session_id = request.video_session_id
        if session_id is None:
            return
        previous = self._sessions.get(session_id)
        if previous is not None and previous is not session:
            for key, value in tuple(previous.backend_cache.items()):
                if key.startswith(_CACHE_PREFIX) and isinstance(value, _PreparedVisual):
                    _release_prepared(value)
                    del previous.backend_cache[key]
            previous.sampled_frames = ()
        self._sessions[session_id] = session

    def loaded_on_assigned_device(self) -> bool:
        """Return whether model placement resolves exclusively to this backend's GPU."""
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, Mapping) and device_map:
            devices = {_normalize_model_device(value) for value in device_map.values()}
            return devices == {self.device}
        model_device = getattr(self.model, "device", None)
        return _normalize_model_device(model_device) == self.device

    def _clear_cache_under_pressure(self) -> None:
        threshold = self.cache_clear_threshold_bytes
        if threshold is None:
            return
        try:
            cuda = getattr(self._torch, "cuda", None)
            if cuda is None or not callable(getattr(cuda, "is_available", None)):
                return
            if not cuda.is_available():
                return
            reserved = cuda.memory_reserved(self.device)
            if (
                isinstance(reserved, bool)
                or not isinstance(reserved, (int, float))
                or not math.isfinite(float(reserved))
                or reserved < threshold
            ):
                return
            with cuda.device(self.device):
                cuda.empty_cache()
        except Exception:
            # Best-effort cleanup must not replace the inference/schema outcome.
            return


def _local_model_directory(model_path: str | Path) -> Path:
    if not isinstance(model_path, (str, Path)) or not str(model_path):
        raise LocalModelError("model path must name an existing local directory")
    try:
        resolved = Path(model_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise LocalModelError("model path must name an existing local directory") from None
    if not resolved.is_dir():
        raise LocalModelError("model path must name an existing local directory")
    return resolved


def _local_video_file(video_path: Path) -> Path:
    if not isinstance(video_path, Path):
        raise ValueError("video path must be a pathlib.Path")
    try:
        resolved = video_path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("video path must name an existing local file") from None
    if not resolved.is_file():
        raise ValueError("video path must name an existing local file")
    return resolved


def _video_element(
    request: ModelRequest, frames: tuple[FrameRef, ...]
) -> dict[str, Any]:
    video: dict[str, Any] = {
        "type": "video",
        # qwen-vl-utils passes strings to its decoder. Plain canonical paths
        # preserve spaces and '#' rather than making a percent-encoded URI.
        "video": [str(frame.path) for frame in frames],
        "fps": request.fps,
    }
    if request.media_resolution is not None:
        minimum, maximum = _MEDIA_RESOLUTION_PIXELS[request.media_resolution]
        video.update(min_pixels=minimum, max_pixels=maximum)
    if request.clip_context is not None:
        video["total_pixels"] = _CLIP_CONTEXT_TOTAL_PIXELS[request.clip_context]
    return video


def _messages(
    request: ModelRequest, video_element: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [dict(video_element), {"type": "text", "text": request.prompt}],
        }
    ]


def _visual_cache_key(request: ModelRequest, video_path: Path) -> str:
    values = (
        str(video_path),
        repr(request.span.start),
        repr(request.span.end),
        repr(request.fps),
        request.media_resolution or "",
        request.clip_context or "",
    )
    return _CACHE_PREFIX + "\x1f".join(values)


def _validated_frames(
    value: Any, temporary_dir: Path, span: TimeSpan
) -> tuple[FrameRef, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise QwenDependencyError("video-only frame extraction returned no frames")
    try:
        root = temporary_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        raise QwenDependencyError("video-only frame storage is unavailable") from None
    frames: list[FrameRef] = []
    previous = -math.inf
    for item in value:
        if not isinstance(item, FrameRef):
            raise QwenDependencyError("video-only frame extraction is incompatible")
        try:
            if item.path.is_symlink():
                raise OSError
            path = item.path.resolve(strict=True)
            path.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            raise QwenDependencyError("video-only frame escaped owned storage") from None
        timestamp = item.timestamp
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
            or timestamp < span.start
            or timestamp >= span.end
            or timestamp <= previous
            or not path.is_file()
        ):
            raise QwenDependencyError("video-only frame metadata is invalid")
        previous = float(timestamp)
        frames.append(FrameRef(path=path, timestamp=float(timestamp)))
    return tuple(frames)


def _validate_prepared_files(prepared: _PreparedVisual) -> None:
    try:
        root = prepared.temporary_dir.resolve(strict=True)
        if not root.is_dir():
            raise OSError
        for frame in prepared.frames:
            if frame.path.is_symlink():
                raise OSError
            path = frame.path.resolve(strict=True)
            path.relative_to(root)
            if not path.is_file():
                raise OSError
    except (OSError, RuntimeError, ValueError):
        raise QwenDependencyError("cached video-only frames are unavailable") from None


def _validate_video_metadata(metadata: Any, span: TimeSpan) -> None:
    if not isinstance(metadata, VideoMetadata) or span.end > metadata.duration:
        raise QwenDependencyError(
            "video metadata is incompatible with the requested span"
        )


def _absolute_video_metadata(
    metadata: VideoMetadata, frames: tuple[FrameRef, ...]
) -> dict[str, Any]:
    source_fps = float(metadata.fps)
    total_frames = max(1, int(math.ceil(metadata.duration * source_fps)))
    return {
        "fps": source_fps,
        "frames_indices": [
            min(total_frames - 1, int(frame.timestamp * source_fps + 0.5))
            for frame in frames
        ],
        "total_num_frames": total_frames,
        "video_backend": "las_ffmpeg_video_only",
    }


def _require_cpu_visual_inputs(value: Any, *, _seen: set[int] | None = None) -> None:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    device = getattr(value, "device", None)
    if device is not None and str(device) != "cpu":
        raise QwenDependencyError("prepared visual inputs must remain on CPU")
    if isinstance(value, Mapping):
        for child in value.values():
            _require_cpu_visual_inputs(child, _seen=seen)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_cpu_visual_inputs(child, _seen=seen)


def _release_prepared(prepared: _PreparedVisual) -> None:
    prepared.image_inputs = None
    prepared.video_inputs = None
    prepared.video_metadata.clear()
    prepared.video_kwargs.clear()
    shutil.rmtree(prepared.temporary_dir, ignore_errors=True)


def _image_patch_size(processor: Any) -> int:
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", 16)
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise QwenDependencyError("Qwen processor has an invalid image patch size")
    return patch_size


def _vision_result(result: Any) -> tuple[Any, Any, dict[str, Any]]:
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise QwenDependencyError("qwen-vl-utils returned incompatible video inputs")
    images, videos, raw_kwargs = result
    if not isinstance(raw_kwargs, Mapping) or any(
        not isinstance(key, str) for key in raw_kwargs
    ):
        raise QwenDependencyError("qwen-vl-utils returned invalid video arguments")
    return images, videos, dict(raw_kwargs)


def _split_video_metadata(videos: Any) -> tuple[Any, Any]:
    if videos is None:
        return None, None
    if not isinstance(videos, (list, tuple)):
        return videos, None
    tensors: list[Any] = []
    metadata: list[Any] = []
    for item in videos:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return videos, None
        tensor, item_metadata = item
        tensors.append(tensor)
        metadata.append(item_metadata)
    return tensors, metadata


def _merge_video_kwargs(
    processor_kwargs: dict[str, Any], video_kwargs: Mapping[str, Any]
) -> None:
    collisions = set(processor_kwargs).intersection(video_kwargs)
    if collisions:
        raise QwenDependencyError("qwen-vl-utils returned conflicting video arguments")
    processor_kwargs.update(video_kwargs)


def _max_new_tokens(request: ModelRequest) -> int:
    cap = STAGE_MAX_NEW_TOKENS[request.stage]
    fraction = _REASONING_TOKEN_FRACTION.get(request.reasoning_effort, 1.0)
    return max(1, int(cap * fraction))


def _input_ids(inputs: Mapping[str, Any]) -> Any:
    try:
        input_ids = inputs["input_ids"]
    except (KeyError, TypeError):
        raise QwenDependencyError("Qwen processor inputs contain no input_ids") from None
    return input_ids


def _strip_prompt_tokens(input_ids: Any, generated_ids: Any) -> list[Any]:
    try:
        if len(input_ids) != len(generated_ids):
            raise QwenDependencyError("Qwen generated batch size does not match its prompt")
        return [
            output_row[len(input_row) :]
            for input_row, output_row in zip(input_ids, generated_ids, strict=True)
        ]
    except QwenDependencyError:
        raise
    except (TypeError, ValueError, IndexError):
        raise QwenDependencyError("Qwen generated incompatible token sequences") from None


def _normalize_model_device(device: Any) -> str | None:
    if isinstance(device, bool) or device is None:
        return None
    if isinstance(device, int):
        return f"cuda:{device}"
    value = str(device)
    if value.isdecimal():
        value = f"cuda:{value}"
    return value if _CUDA_DEVICE.fullmatch(value) is not None else None


def _assigned_cuda_device(device: str) -> str:
    if not isinstance(device, str) or _CUDA_DEVICE.fullmatch(device) is None:
        raise ValueError("device must be an explicit CUDA device such as cuda:0")
    return device


def _is_torch_dtype_rename(error: TypeError) -> bool:
    message = str(error)
    return "unexpected keyword argument" in message and "torch_dtype" in message


@contextmanager
def _offline_load_environment() -> Any:
    keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    )
    missing = object()
    previous = {key: os.environ.get(key, missing) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _import_runtime_dependencies() -> SimpleNamespace:
    """Import optional heavyweight packages only when a local model is loaded."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        qwen_utils = importlib.import_module("qwen_vl_utils")
        model_class = getattr(transformers, "Qwen3VLForConditionalGeneration")
        processor_class = getattr(transformers, "AutoProcessor")
        process_vision_info = getattr(qwen_utils, "process_vision_info")
    except (ImportError, AttributeError):
        raise QwenDependencyError(
            "Qwen3-VL GPU dependencies are unavailable; install the gpu extra"
        ) from None
    return SimpleNamespace(
        torch=torch,
        model_class=model_class,
        processor_class=processor_class,
        process_vision_info=process_vision_info,
    )
