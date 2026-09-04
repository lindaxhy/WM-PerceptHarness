from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
import zipfile

import pytest

from las_repro.cv.base import CvOutOfMemoryError, CvProviderError
from las_repro.cv.contracts import (
    CvEvidenceRequest,
    EntityPrompt,
    EntityRole,
    EvidenceThresholds,
    FrameTimeline,
    FrameTimestamp,
    SamplingPolicy,
)
from las_repro.cv.sam31 import Sam31EvidenceProvider
from las_repro.cv.timeline import SampledFrame, SampledFrameSet


PINNED_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
SAM31_MODULE = sys.modules[Sam31EvidenceProvider.__module__]


class ArrayDouble:
    def __init__(self, values: Any, shape: tuple[int, ...], dtype: str) -> None:
        self._values = values
        self.shape = shape
        self.dtype = SimpleNamespace(kind=dtype)

    def tolist(self) -> Any:
        return self._values


def array(values: Any, shape: tuple[int, ...], dtype: str) -> ArrayDouble:
    return ArrayDouble(values, shape, dtype)


def frame_output(
    frame_index: int,
    *,
    object_ids: list[int] | None = None,
    probabilities: list[float] | None = None,
    boxes: list[list[float]] | None = None,
    masks: list[list[list[bool]]] | None = None,
    mask_shape: tuple[int, ...] | None = None,
    mask_dtype: str = "b",
) -> dict[str, Any]:
    object_ids = [1] if object_ids is None else object_ids
    probabilities = [0.875] if probabilities is None else probabilities
    boxes = [[0.1, 0.2, 0.4, 0.5]] if boxes is None else boxes
    masks = [[[True, False], [False, True]]] if masks is None else masks
    count = len(object_ids)
    return {
        "frame_index": frame_index,
        "outputs": {
            "out_obj_ids": array(object_ids, (count,), "i"),
            "out_probs": array(probabilities, (len(probabilities),), "f"),
            "out_boxes_xywh": array(boxes, (len(boxes), 4), "f"),
            "out_binary_masks": array(
                masks,
                mask_shape if mask_shape is not None else (len(masks), 2, 2),
                mask_dtype,
            ),
        },
    }


def empty_frame_output(frame_index: int) -> dict[str, Any]:
    return frame_output(
        frame_index,
        object_ids=[],
        probabilities=[],
        boxes=[],
        masks=[],
        mask_shape=(0, 2, 2),
    )


class PredictorDouble:
    def __init__(
        self,
        stream: Callable[[int, str, dict[str, Any]], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.stream_requests: list[dict[str, Any]] = []
        self.resources: list[Path] = []
        self.resource_files: list[tuple[str, ...]] = []
        self._session_prompts: dict[str, str] = {}
        self._stream = stream or self._default_stream
        self.shutdown_calls = 0

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        copied = dict(request)
        self.requests.append(copied)
        request_type = request["type"]
        if request_type == "start_session":
            session_id = f"session-{len(self.resources)}"
            resource = Path(request["resource_path"])
            self.resources.append(resource)
            self.resource_files.append(
                tuple(path.name for path in sorted(resource.glob("*.jpg")))
            )
            self._session_prompts[session_id] = ""
            return {"session_id": session_id}
        if request_type == "reset_session":
            self._session_prompts[request["session_id"]] = ""
            return {"is_success": True}
        if request_type == "add_prompt":
            self._session_prompts[request["session_id"]] = request["text"]
            return {"frame_index": request["frame_index"], "outputs": {}}
        if request_type == "close_session":
            self._session_prompts.pop(request["session_id"], None)
            return {"is_success": True}
        raise AssertionError(f"unexpected request type {request_type}")

    def handle_stream_request(self, request: dict[str, Any]):
        copied = dict(request)
        self.stream_requests.append(copied)
        session_id = request["session_id"]
        prompt = self._session_prompts[session_id]
        session_number = int(session_id.rsplit("-", 1)[1])
        yield from self._stream(session_number, prompt, copied)

    def _default_stream(
        self, session_number: int, prompt: str, request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        del session_number, prompt
        start = request["start_frame_index"]
        stop = start + request["max_frame_num_to_track"] + 1
        outputs = []
        for frame_index in range(start, stop):
            outputs.append(
                frame_output(
                    frame_index,
                    object_ids=[2, 1],
                    probabilities=[0.625, 0.875],
                    boxes=[[0.5, 0.25, 0.25, 0.5], [0.1, 0.2, 0.4, 0.5]],
                    masks=[
                        [[False, False], [False, True]],
                        [[True, False], [False, True]],
                    ],
                )
            )
        return outputs

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class MaterializerDouble:
    def __init__(
        self,
        *,
        corrupt_mapping: bool = False,
        add_undeclared_file: bool = False,
    ) -> None:
        self.calls: list[tuple[Path, tuple[int, ...], Path]] = []
        self.corrupt_mapping = corrupt_mapping
        self.add_undeclared_file = add_undeclared_file

    def __call__(
        self,
        video_path: Path,
        timeline: FrameTimeline,
        indices: tuple[int, ...],
        destination: Path,
    ) -> SampledFrameSet:
        self.calls.append((Path(video_path), tuple(indices), Path(destination)))
        destination.mkdir(parents=True, exist_ok=False)
        by_index = {point.frame_index: point for point in timeline.frames}
        frames = []
        for sam_index, source_index in enumerate(indices):
            path = destination / f"{sam_index:06d}.jpg"
            path.write_bytes(b"\xff\xd8\xff\xd9")
            frames.append(
                SampledFrame(
                    sam_index=0 if self.corrupt_mapping else sam_index,
                    source_frame_index=source_index,
                    source_timestamp_seconds=by_index[source_index].timestamp_seconds,
                    path=path,
                )
            )
        if self.add_undeclared_file:
            (destination / "source.mp4").write_bytes(b"undeclared source")
        return SampledFrameSet(frames=tuple(frames))


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.reset_calls = 0
        self.empty_cache_calls = 0
        self.peak = 123_456

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def max_memory_allocated(self) -> int:
        return self.peak

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


@pytest.fixture
def fake_torch() -> SimpleNamespace:
    return SimpleNamespace(cuda=FakeCuda())


def make_request(
    tmp_path: Path,
    *,
    duration_seconds: float = 2.0,
    timestamps: tuple[float, ...] = (0.0, 1.0, 2.0),
    entities: tuple[EntityPrompt, ...] | None = None,
) -> CvEvidenceRequest:
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"local video")
    if entities is None:
        entities = (
            EntityPrompt(
                entity_id="right_hand",
                canonical_label="right hand",
                aliases=("hand",),
                role=EntityRole.ACTOR,
            ),
            EntityPrompt(
                entity_id="red_cup",
                canonical_label="red cup",
                aliases=("cup",),
                role=EntityRole.MANIPULATED_OBJECT,
            ),
        )
    return CvEvidenceRequest(
        schema_version="cv_request_v1",
        provider="sam31",
        model_identity=f"sam31:{PINNED_REVISION}",
        video_path=video_path,
        video_sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(),
        duration_seconds=duration_seconds,
        frame_count=len(timestamps),
        checkpoint_sha256="a" * 64,
        timeline=FrameTimeline(
            frames=tuple(
                FrameTimestamp(frame_index=index, timestamp_seconds=timestamp)
                for index, timestamp in enumerate(timestamps)
            )
        ),
        entities=entities,
        sampling=SamplingPolicy(
            short_video_seconds=30.0,
            scan_fps=8.0,
            max_fps=30.0,
            refinement_radius_seconds=1.0,
        ),
        thresholds=EvidenceThresholds(
            min_confidence=0.5,
            min_area_fraction=0.01,
            occlusion_visibility_drop=0.5,
        ),
    )


def make_provider(
    predictor: PredictorDouble,
    fake_torch: SimpleNamespace,
    materializer: MaterializerDouble,
    **kwargs: Any,
) -> Sam31EvidenceProvider:
    return Sam31EvidenceProvider(
        predictor=predictor,
        torch_module=fake_torch,
        materialize_frames=materializer,
        **kwargs,
    )


def test_adapter_uses_official_sequence_and_numbered_sample_directory(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    predictor = PredictorDouble()
    materializer = MaterializerDouble()
    provider = make_provider(predictor, fake_torch, materializer)

    artifact = provider.analyze(request, tmp_path / "staging")

    assert artifact.provider == "sam31"
    assert artifact.entities == request.entities
    assert predictor.requests[0]["type"] == "start_session"
    assert predictor.requests[-1]["type"] == "close_session"
    assert [item["type"] for item in predictor.requests] == [
        "start_session",
        "reset_session",
        "add_prompt",
        "reset_session",
        "add_prompt",
        "close_session",
    ]
    assert [item["text"] for item in predictor.requests if item["type"] == "add_prompt"] == [
        "right hand",
        "red cup",
    ]
    assert all(
        item["frame_index"] == 0
        for item in predictor.requests
        if item["type"] == "add_prompt"
    )
    assert all(item["type"] == "propagate_in_video" for item in predictor.stream_requests)
    resource = Path(predictor.requests[0]["resource_path"])
    assert resource != request.video_path
    assert predictor.resource_files == [(
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
    )]
    assert not resource.exists()
    assert materializer.calls[0][0] == request.video_path
    assert materializer.calls[0][1] == (0, 1, 2)
    assert len(materializer.calls) == 1


def test_adapter_namespaces_instances_and_derives_validated_geometry(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    predictor = PredictorDouble()
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    artifact = provider.analyze(request, tmp_path / "staging")

    assert [track.track_id for track in artifact.tracks] == [
        "right_hand_1",
        "right_hand_2",
        "red_cup_1",
        "red_cup_2",
    ]
    first = artifact.tracks[0].observations[0]
    assert first.bbox_xyxy == (0.1, 0.2, 0.5, 0.7)
    assert first.confidence == 0.875
    assert first.area_fraction == 0.5
    assert first.center_xy == (0.25, 0.25)
    second = artifact.tracks[1].observations[0]
    assert second.bbox_xyxy == (0.5, 0.25, 0.75, 0.75)
    assert second.confidence == 0.625
    assert second.area_fraction == 0.25
    assert second.center_xy == (0.5, 0.5)
    assert all(
        observation.frame_index in {0, 1, 2}
        for track in artifact.tracks
        for observation in track.observations
    )


def test_adapter_writes_deterministic_compressed_per_prompt_mask_chunks(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    first_provider = make_provider(
        PredictorDouble(), fake_torch, MaterializerDouble()
    )
    second_torch = SimpleNamespace(cuda=FakeCuda())
    second_provider = make_provider(
        PredictorDouble(), second_torch, MaterializerDouble()
    )

    first = first_provider.analyze(request, tmp_path / "first")
    second = second_provider.analyze(request, tmp_path / "second")

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [file.path for file in first.files if file.path.endswith(".npz")] == [
        "masks/red_cup.npz",
        "masks/right_hand.npz",
    ]
    for artifact_file in first.files:
        first_payload = (tmp_path / "first" / artifact_file.path).read_bytes()
        second_payload = (tmp_path / "second" / artifact_file.path).read_bytes()
        assert first_payload == second_payload
        assert hashlib.sha256(first_payload).hexdigest() == artifact_file.sha256
        assert len(first_payload) == artifact_file.size_bytes
        if artifact_file.path.endswith(".npz"):
            with zipfile.ZipFile(tmp_path / "first" / artifact_file.path) as archive:
                assert archive.namelist() == ["masks.npy"]
                assert archive.read("masks.npy").startswith(b"\x93NUMPY")


def test_artifact_contains_only_relative_digested_files_and_no_private_values(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    provider = make_provider(PredictorDouble(), fake_torch, MaterializerDouble())

    artifact = provider.analyze(request, tmp_path / "staging")
    encoded = json.dumps(artifact.model_dump(mode="json"), sort_keys=True)

    assert str(tmp_path) not in encoded
    assert str(request.video_path) not in encoded
    assert "out_binary_masks" not in encoded
    assert all(not Path(file.path).is_absolute() for file in artifact.files)
    assert all(len(file.sha256) == 64 for file in artifact.files)
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        "masks"
    ]


def test_empty_detections_are_valid_and_still_close_the_session(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path, entities=(make_request(tmp_path).entities[0],))

    def empty_stream(
        session_number: int, prompt: str, stream_request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        del session_number, prompt
        start = stream_request["start_frame_index"]
        stop = start + stream_request["max_frame_num_to_track"] + 1
        return [empty_frame_output(index) for index in range(start, stop)]

    predictor = PredictorDouble(empty_stream)
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    artifact = provider.analyze(request, tmp_path / "staging")

    assert artifact.tracks == ()
    assert [file.path for file in artifact.files] == ["masks/right_hand.npz"]
    assert predictor.requests[-1]["type"] == "close_session"


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_probs", array([], (0,), "f")
            ),
            id="mismatched-lengths",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_boxes_xywh", array([[0.1, 0.2, 0.3]], (1, 3), "f")
            ),
            id="wrong-box-shape",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_probs", array([float("nan")], (1,), "f")
            ),
            id="non-finite-probability",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_probs", array([1.01], (1,), "f")
            ),
            id="out-of-range-probability",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_boxes_xywh",
                array([[0.1, 0.2, float("inf"), 0.5]], (1, 4), "f"),
            ),
            id="non-finite-box",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_boxes_xywh", array([[0.8, 0.2, 0.3, 0.5]], (1, 4), "f")
            ),
            id="out-of-bounds-box",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_boxes_xywh", array([[0.1, 0.2, 0.0, 0.5]], (1, 4), "f")
            ),
            id="zero-width-box",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_binary_masks",
                array([[[True, False], [False, True]]], (1, 1, 2, 2), "b"),
            ),
            id="wrong-mask-rank",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_binary_masks",
                array([[[1, 0], [0, 1]]], (1, 2, 2), "u"),
            ),
            id="non-boolean-mask",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_binary_masks",
                array([[[True, False]]], (1, 2, 2), "b"),
            ),
            id="mask-data-shape-mismatch",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_binary_masks",
                array([[[True]]], (1, 1, 1), "b"),
            ),
            id="inconsistent-mask-dimensions",
        ),
        pytest.param(
            lambda response: response["outputs"].__setitem__(
                "out_obj_ids", array([True], (1,), "b")
            ),
            id="non-integer-object-id",
        ),
        pytest.param(
            lambda response: response["outputs"].pop("out_probs"),
            id="missing-array",
        ),
    ],
)
def test_adapter_jointly_rejects_malformed_output_arrays_and_closes(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    mutate: Callable[[dict[str, Any]], Any],
) -> None:
    request = make_request(tmp_path, entities=(make_request(tmp_path).entities[0],))
    response = frame_output(0)
    mutate(response)
    predictor = PredictorDouble(
        lambda *_: [response, frame_output(1), frame_output(2)]
    )
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests[-1]["type"] == "close_session"
    assert not (tmp_path / "staging" / "masks").exists()


@pytest.mark.parametrize(
    "responses",
    [
        pytest.param([frame_output(3)], id="out-of-range-frame"),
        pytest.param([frame_output(0), frame_output(0)], id="duplicate-frame"),
        pytest.param(
            [frame_output(0), frame_output(1)],
            id="missing-frame",
        ),
        pytest.param(
            [frame_output(1), frame_output(0), frame_output(2)],
            id="out-of-order-frame",
        ),
        pytest.param(
            [
                frame_output(
                    0,
                    object_ids=[1, 1],
                    probabilities=[0.8, 0.9],
                    boxes=[[0.1, 0.1, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]],
                    masks=[
                        [[True, False], [False, False]],
                        [[False, False], [False, True]],
                    ],
                )
            ],
            id="duplicate-object-id",
        ),
    ],
)
def test_adapter_rejects_invalid_or_duplicate_indices(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    responses: list[dict[str, Any]],
) -> None:
    request = make_request(tmp_path, entities=(make_request(tmp_path).entities[0],))
    predictor = PredictorDouble(lambda *_: responses)
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests[-1]["type"] == "close_session"


def test_adapter_rejects_unstable_local_to_source_mapping(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    predictor = PredictorDouble()
    provider = make_provider(
        predictor,
        fake_torch,
        MaterializerDouble(corrupt_mapping=True),
    )

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests == []


def test_adapter_rejects_undeclared_files_in_numbered_frame_directory(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    predictor = PredictorDouble()
    provider = make_provider(
        predictor,
        fake_torch,
        MaterializerDouble(add_undeclared_file=True),
    )

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests == []


def test_adapter_chunks_one_final_mapping_without_resampling(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(
        tmp_path,
        duration_seconds=4.0,
        timestamps=(0.0, 1.0, 2.0, 3.0, 4.0),
    )
    predictor = PredictorDouble()
    materializer = MaterializerDouble()
    provider = make_provider(
        predictor,
        fake_torch,
        materializer,
        execution_chunk_frames=2,
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    per_prompt = [
        (item["start_frame_index"], item["max_frame_num_to_track"])
        for item in predictor.stream_requests[:3]
    ]
    assert per_prompt == [(0, 1), (2, 1), (4, 0)]
    assert materializer.calls[0][1] == (0, 1, 2, 3, 4)
    assert [item.frame_index for item in artifact.tracks[0].observations] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_execution_chunk_interface_is_positive_and_worker_mutable(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    del tmp_path
    provider = Sam31EvidenceProvider(
        predictor=PredictorDouble(),
        torch_module=fake_torch,
        execution_chunk_frames=8,
    )

    provider.execution_chunk_frames = 4
    assert provider.execution_chunk_frames == 4
    provider.set_execution_chunk_frames(2)
    assert provider.execution_chunk_frames == 2
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            provider.set_execution_chunk_frames(invalid)  # type: ignore[arg-type]


def test_long_video_scans_refines_union_and_discards_preliminary_outputs(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(
        tmp_path,
        duration_seconds=31.0,
        timestamps=tuple(float(index) for index in range(10)),
        entities=(make_request(tmp_path).entities[0],),
    )
    initial_calls: list[FrameTimeline] = []
    refinement_calls: list[tuple[int, ...]] = []

    def initial(timeline: FrameTimeline, policy: SamplingPolicy) -> tuple[int, ...]:
        del policy
        initial_calls.append(timeline)
        return (0, 4, 8)

    def refine(
        timeline: FrameTimeline,
        changes: tuple[int, ...],
        policy: SamplingPolicy,
    ) -> tuple[int, ...]:
        del timeline, policy
        refinement_calls.append(changes)
        return (3, 4, 5, 7, 8, 9)

    def stream(
        session_number: int, prompt: str, stream_request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        del prompt
        start = stream_request["start_frame_index"]
        stop = start + stream_request["max_frame_num_to_track"] + 1
        if session_number == 0:
            scan = {
                0: frame_output(0),
                1: empty_frame_output(1),
                2: frame_output(2),
            }
            return [scan[index] for index in range(start, stop)]
        return [frame_output(index) for index in range(start, stop)]

    predictor = PredictorDouble(stream)
    materializer = MaterializerDouble()
    provider = make_provider(
        predictor,
        fake_torch,
        materializer,
        initial_sampler=initial,
        refinement_sampler=refine,
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    assert len(initial_calls) == 1
    assert refinement_calls == [(4, 8)]
    assert [call[1] for call in materializer.calls] == [
        (0, 4, 8),
        (0, 3, 4, 5, 7, 8, 9),
    ]
    assert len(predictor.resources) == 2
    assert [item["type"] for item in predictor.requests].count("close_session") == 2
    assert [observation.frame_index for observation in artifact.tracks[0].observations] == [
        0,
        3,
        4,
        5,
        7,
        8,
        9,
    ]
    assert all(frame != 2 for frame in (
        observation.frame_index
        for track in artifact.tracks
        for observation in track.observations
    ))
    assert all(not path.exists() for path in predictor.resources)


def test_visibility_overlay_files_are_bounded_to_twenty_four(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    timestamps = tuple(index / 10.0 for index in range(30))
    request = make_request(
        tmp_path,
        duration_seconds=2.9,
        timestamps=timestamps,
    )

    def flicker(
        session_number: int, prompt: str, stream_request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        del session_number, prompt
        start = stream_request["start_frame_index"]
        stop = start + stream_request["max_frame_num_to_track"] + 1
        return [
            frame_output(index) if index % 2 == 0 else empty_frame_output(index)
            for index in range(start, stop)
        ]

    provider = make_provider(
        PredictorDouble(flicker),
        fake_torch,
        MaterializerDouble(),
        execution_chunk_frames=30,
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    overlays = [file for file in artifact.files if file.path.endswith(".png")]
    assert 0 < len(overlays) <= 24
    assert len({file.path for file in overlays}) == len(overlays)


def test_oom_is_translated_once_without_internal_retry_and_releases_resources(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)

    def oom(*_: Any) -> list[dict[str, Any]]:
        raise fake_torch.cuda.OutOfMemoryError(
            f"secret allocator state {request.video_path} right hand"
        )

    predictor = PredictorDouble(oom)
    materializer = MaterializerDouble()
    provider = make_provider(predictor, fake_torch, materializer)

    with pytest.raises(
        CvOutOfMemoryError,
        match="^SAM3.1 CV evidence inference ran out of memory$",
    ) as raised:
        provider.analyze(request, tmp_path / "staging")

    assert str(request.video_path) not in str(raised.value)
    assert "right hand" not in str(raised.value)
    assert [item["type"] for item in predictor.requests].count("start_session") == 1
    assert predictor.requests[-1]["type"] == "close_session"
    assert len(materializer.calls) == 1
    assert fake_torch.cuda.empty_cache_calls == 1
    assert not (tmp_path / "staging" / "masks").exists()
    assert all(not path.exists() for path in predictor.resources)


def test_stream_cleanup_failure_does_not_mask_primary_cuda_oom(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)

    class OomStream:
        def __iter__(self) -> OomStream:
            return self

        def __next__(self) -> dict[str, Any]:
            raise fake_torch.cuda.OutOfMemoryError("private allocator detail")

        def close(self) -> None:
            raise RuntimeError("stream cleanup detail")

    class PredictorWithFailingStreamCleanup(PredictorDouble):
        def handle_stream_request(self, request: dict[str, Any]) -> OomStream:
            self.stream_requests.append(dict(request))
            return OomStream()

    predictor = PredictorWithFailingStreamCleanup()
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(
        CvOutOfMemoryError,
        match="^SAM3.1 CV evidence inference ran out of memory$",
    ):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests[-1]["type"] == "close_session"
    assert fake_torch.cuda.empty_cache_calls == 1


def test_non_oom_failure_is_sanitized_closed_and_not_retried(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)

    def fail(*_: Any) -> list[dict[str, Any]]:
        raise RuntimeError(
            f"raw model output at {request.video_path}: {request.entities[0].canonical_label}"
        )

    predictor = PredictorDouble(fail)
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(
        CvProviderError,
        match="^SAM3.1 CV evidence inference failed$",
    ) as raised:
        provider.analyze(request, tmp_path / "staging")

    assert not isinstance(raised.value, CvOutOfMemoryError)
    assert str(request.video_path) not in str(raised.value)
    assert request.entities[0].canonical_label not in str(raised.value)
    assert [item["type"] for item in predictor.requests].count("start_session") == 1
    assert predictor.requests[-1]["type"] == "close_session"


def test_peak_memory_is_reset_recorded_and_provider_close_is_idempotent(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    predictor = PredictorDouble()
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    provider.analyze(request, tmp_path / "staging")

    assert fake_torch.cuda.reset_calls == 1
    assert provider.request_metrics() == {
        "processed_frames": 3,
        "entity_prompts": 2,
        "track_count": 4,
        "peak_allocated_bytes": 123_456,
    }
    provider.close()
    provider.close()
    assert predictor.shutdown_calls == 1


def test_importing_adapter_does_not_import_an_installed_sam_package() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockSam(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sam3' or fullname.startswith('sam3.'):
            raise AssertionError('SAM import attempted')
        return None

sys.meta_path.insert(0, BlockSam())
from las_repro.cv.sam31 import Sam31EvidenceProvider
assert Sam31EvidenceProvider is not None
assert not any(name == 'sam3' or name.startswith('sam3.') for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.fixture(autouse=True)
def restore_sam_modules():
    before = {
        name: module
        for name, module in sys.modules.items()
        if name == "sam3" or name.startswith("sam3.")
    }
    yield
    for name in tuple(sys.modules):
        if name == "sam3" or name.startswith("sam3."):
            sys.modules.pop(name, None)
    sys.modules.update(before)


def local_runtime_assets(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = tmp_path / "sam-repository"
    package = repository / "sam3"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    checkpoint = tmp_path / "sam31.pt"
    checkpoint.write_bytes((b"checkpoint block" * 100_000) + b"end")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return repository, checkpoint, digest


def install_fake_torch_module(
    monkeypatch: pytest.MonkeyPatch, fake_torch: SimpleNamespace
) -> None:
    module = ModuleType("torch")
    module.cuda = fake_torch.cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", module)


def successful_git_run(
    calls: list[tuple[list[str], dict[str, Any]]]
) -> Callable[..., SimpleNamespace]:
    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=PINNED_REVISION + "\n")

    return run


def test_load_verifies_local_revision_and_hash_then_calls_official_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    git_calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run(git_calls))
    install_fake_torch_module(monkeypatch, fake_torch)
    factory_calls: list[dict[str, Any]] = []
    predictor = PredictorDouble()

    def factory(**kwargs: Any) -> PredictorDouble:
        factory_calls.append(kwargs)
        return predictor

    provider = Sam31EvidenceProvider.load(
        repository,
        checkpoint,
        digest,
        compile_model=True,
        predictor_factory=factory,
    )

    assert git_calls == [
        (
            [
                "git",
                "-C",
                str(repository.resolve()),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "shell": False,
            },
        )
    ]
    assert factory_calls == [
        {
            "checkpoint_path": str(checkpoint.resolve()),
            "load_from_HF": False,
            "multiplex_count": 16,
            "gpus_to_use": [0],
            "compile": True,
        }
    ]
    assert provider.checkpoint_sha256 == digest
    assert provider.repository_revision == PINNED_REVISION


@pytest.mark.parametrize(
    "repository_value",
    [
        "https://example.invalid/sam3",
        "ssh://host/sam3",
        "git@host:sam3.git",
        "hf://facebook/sam3",
        "//server/share/sam3",
    ],
)
def test_load_rejects_network_looking_repository_identifiers(
    tmp_path: Path, repository_value: str
) -> None:
    checkpoint = tmp_path / "sam31.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository_value,
            checkpoint,
            hashlib.sha256(b"checkpoint").hexdigest(),
            predictor_factory=lambda **_: PredictorDouble(),
        )


@pytest.mark.parametrize(
    "checkpoint_value",
    [
        "https://example.invalid/sam31.pt",
        "hf://facebook/sam31.pt",
        "user@host:sam31.pt",
        "//server/share/sam31.pt",
    ],
)
def test_load_rejects_network_looking_checkpoint_identifiers(
    tmp_path: Path, checkpoint_value: str
) -> None:
    repository, _, _ = local_runtime_assets(tmp_path)

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository,
            checkpoint_value,
            "a" * 64,
            predictor_factory=lambda **_: PredictorDouble(),
        )


def test_load_rejects_missing_nonregular_symlinked_or_mismatched_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    checkpoint_directory = tmp_path / "checkpoint-directory"
    checkpoint_directory.mkdir()
    checkpoint_symlink = tmp_path / "checkpoint-link"
    checkpoint_symlink.symlink_to(checkpoint)
    repository_symlink = tmp_path / "repository-link"
    repository_symlink.symlink_to(repository, target_is_directory=True)
    cases = [
        (tmp_path / "missing-repository", checkpoint, digest),
        (repository, tmp_path / "missing-checkpoint", digest),
        (repository, checkpoint_directory, digest),
        (repository, checkpoint_symlink, digest),
        (repository_symlink, checkpoint, digest),
        (repository, checkpoint, "0" * 64),
        (repository, checkpoint, digest.upper()),
    ]

    for repository_value, checkpoint_value, expected_digest in cases:
        with pytest.raises(
            CvProviderError,
            match="^Unable to load local SAM3.1 runtime$",
        ):
            Sam31EvidenceProvider.load(
                repository_value,
                checkpoint_value,
                expected_digest,
                predictor_factory=lambda **_: PredictorDouble(),
            )


def test_load_rejects_wrong_repository_revision_before_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(
        SAM31_MODULE.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="b" * 40 + "\n"),
    )
    install_fake_torch_module(monkeypatch, fake_torch)
    factory_called = False

    def factory(**_: Any) -> PredictorDouble:
        nonlocal factory_called
        factory_called = True
        return PredictorDouble()

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository,
            checkpoint,
            digest,
            predictor_factory=factory,
        )

    assert factory_called is False


def test_load_shuts_down_a_built_predictor_when_interface_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)

    class InvalidPredictor:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    predictor = InvalidPredictor()

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository,
            checkpoint,
            digest,
            predictor_factory=lambda **_: predictor,
        )

    assert predictor.shutdown_calls == 1


def test_load_imports_sam_only_from_configured_repository_and_restores_sys_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    builder = repository / "sam3" / "model_builder.py"
    builder.write_text(
        "calls = []\n"
        "class Predictor:\n"
        "    def handle_request(self, request):\n"
        "        return {}\n"
        "    def handle_stream_request(self, request):\n"
        "        return iter(())\n"
        "def build_sam3_multiplex_video_predictor(**kwargs):\n"
        "    calls.append(kwargs)\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    path_before = list(sys.path)

    provider = Sam31EvidenceProvider.load(repository, checkpoint, digest)

    imported_package = sys.modules["sam3"]
    imported_builder = sys.modules["sam3.model_builder"]
    assert Path(imported_package.__file__).resolve().is_relative_to(repository.resolve())
    assert Path(imported_builder.__file__).resolve().is_relative_to(repository.resolve())
    assert sys.path == path_before
    assert imported_builder.calls == [
        {
            "checkpoint_path": str(checkpoint.resolve()),
            "load_from_HF": False,
            "multiplex_count": 16,
            "gpus_to_use": [0],
            "compile": False,
        }
    ]
    assert provider is not None


def test_load_rejects_foreign_preloaded_sam_without_mutating_import_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    foreign = ModuleType("sam3")
    foreign.__file__ = str(tmp_path / "foreign" / "sam3" / "__init__.py")
    monkeypatch.setitem(sys.modules, "sam3", foreign)
    path_before = list(sys.path)

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(repository, checkpoint, digest)

    assert sys.modules["sam3"] is foreign
    assert "sam3.model_builder" not in sys.modules
    assert sys.path == path_before


def test_predictor_factory_cannot_bypass_foreign_sam_origin_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    foreign = ModuleType("sam3")
    foreign.__file__ = str(tmp_path / "foreign" / "sam3" / "__init__.py")
    monkeypatch.setitem(sys.modules, "sam3", foreign)
    factory_called = False

    def factory(**_: Any) -> PredictorDouble:
        nonlocal factory_called
        factory_called = True
        return PredictorDouble()

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository,
            checkpoint,
            digest,
            predictor_factory=factory,
        )

    assert factory_called is False
    assert sys.modules["sam3"] is foreign


def test_failed_local_builder_restores_sys_path_and_new_sam_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    (repository / "sam3" / "model_builder.py").write_text(
        "def build_sam3_multiplex_video_predictor(**kwargs):\n"
        "    raise RuntimeError('private checkpoint path and allocator state')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    path_before = list(sys.path)

    with pytest.raises(
        CvProviderError,
        match="^Unable to load local SAM3.1 runtime$",
    ) as raised:
        Sam31EvidenceProvider.load(repository, checkpoint, digest)

    assert str(checkpoint) not in str(raised.value)
    assert "allocator" not in str(raised.value)
    assert not any(name == "sam3" or name.startswith("sam3.") for name in sys.modules)
    assert sys.path == path_before


def test_failed_builder_restores_preloaded_local_package_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    (repository / "sam3" / "model_builder.py").write_text(
        "import sam3\n"
        "sam3.injected_during_import = 'private import state'\n"
        "def build_sam3_multiplex_video_predictor(**kwargs):\n"
        "    raise RuntimeError(kwargs['checkpoint_path'])\n",
        encoding="utf-8",
    )
    package = ModuleType("sam3")
    package.__file__ = str(repository / "sam3" / "__init__.py")
    package.__path__ = [str(repository / "sam3")]  # type: ignore[attr-defined]
    package.original_marker = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sam3", package)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(repository, checkpoint, digest)

    assert sys.modules["sam3"] is package
    assert package.original_marker is not None  # type: ignore[attr-defined]
    assert not hasattr(package, "injected_during_import")
    assert not hasattr(package, "model_builder")
    assert "sam3.model_builder" not in sys.modules


def test_interrupted_local_builder_restores_import_state_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    (repository / "sam3" / "model_builder.py").write_text(
        "import sam3\n"
        "def build_sam3_multiplex_video_predictor(**kwargs):\n"
        "    sam3.poisoned_during_load = kwargs['checkpoint_path']\n"
        "    raise KeyboardInterrupt('interrupted local load')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    path_before = list(sys.path)

    with pytest.raises(KeyboardInterrupt, match="interrupted local load"):
        Sam31EvidenceProvider.load(repository, checkpoint, digest)

    assert not any(name == "sam3" or name.startswith("sam3.") for name in sys.modules)
    assert sys.path == path_before


def test_load_rejects_checkpoint_path_swap_during_stream_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    original_read = SAM31_MODULE.os.read
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(checkpoint.read_bytes())
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        payload = original_read(descriptor, size)
        if not swapped:
            swapped = True
            checkpoint.rename(tmp_path / "old.pt")
            replacement.rename(checkpoint)
        return payload

    monkeypatch.setattr(SAM31_MODULE.os, "read", swapping_read)

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository,
            checkpoint,
            digest,
            predictor_factory=lambda **_: PredictorDouble(),
        )


def test_invalid_staging_symlink_is_rejected_without_touching_target(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    staging = tmp_path / "staging-link"
    staging.symlink_to(target, target_is_directory=True)
    provider = make_provider(PredictorDouble(), fake_torch, MaterializerDouble())

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, staging)

    assert list(target.iterdir()) == []


def test_nonprivate_existing_staging_directory_is_rejected_before_inference(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    staging = tmp_path / "shared-staging"
    staging.mkdir(mode=0o755)
    staging.chmod(0o755)
    predictor = PredictorDouble()
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, staging)

    assert predictor.requests == []


def test_mask_and_overlay_files_are_owner_only(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    provider = make_provider(PredictorDouble(), fake_torch, MaterializerDouble())

    artifact = provider.analyze(request, tmp_path / "staging")

    for artifact_file in artifact.files:
        mode = stat.S_IMODE((tmp_path / "staging" / artifact_file.path).stat().st_mode)
        assert mode == 0o600
