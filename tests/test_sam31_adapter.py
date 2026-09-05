from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import struct
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
import zipfile

import pytest

from las_repro.cv.base import CvOutOfMemoryError, CvProviderError
from las_repro.cv.contracts import (
    CvEvidenceRequest,
    EntityPrompt,
    EntityRole,
    EvidenceStatus,
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

    def __getitem__(self, index: Any) -> Any:
        indices = index if isinstance(index, tuple) else (index,)
        value = self._values
        for item in indices:
            value = value[item]
        return value


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
        stop = start + request["max_frame_num_to_track"]
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
        width: int = 2,
        height: int = 2,
    ) -> None:
        self.calls: list[tuple[Path, tuple[int, ...], Path]] = []
        self.corrupt_mapping = corrupt_mapping
        self.add_undeclared_file = add_undeclared_file
        self.width = width
        self.height = height

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
            path.write_bytes(minimal_jpeg(self.width, self.height))
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


def minimal_jpeg(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0\x00\x0b\x08"
        + struct.pack(">HHB", height, width, 1)
        + b"\x01\x11\x00"
        + b"\xff\xd9"
    )


def read_npy(payload: bytes) -> tuple[tuple[int, ...], bytes]:
    assert payload.startswith(b"\x93NUMPY\x01\x00")
    header_length = struct.unpack("<H", payload[8:10])[0]
    header = ast.literal_eval(payload[10 : 10 + header_length].decode("latin1").strip())
    return tuple(header["shape"]), payload[10 + header_length :]


def read_int64_array(archive: zipfile.ZipFile, name: str) -> tuple[int, ...]:
    shape, payload = read_npy(archive.read(name))
    assert len(shape) == 1
    values = tuple(item[0] for item in struct.iter_unpack("<q", payload))
    assert len(values) == shape[0]
    return values


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
    assert artifact.processed_timeline == request.timeline
    assert artifact.overlay_records == ()
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


def test_request_without_entities_reports_consistently_unprocessed_evidence(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    """A provider short-circuit cannot claim available evidence without a clock."""
    request = make_request(tmp_path, entities=())
    predictor = PredictorDouble()
    materializer = MaterializerDouble()
    provider = make_provider(predictor, fake_torch, materializer)

    artifact = provider.analyze(request, tmp_path / "staging")

    assert artifact.status is EvidenceStatus.DISABLED
    assert artifact.processed_timeline is None
    assert artifact.tracks == ()
    assert artifact.files == ()
    assert artifact.overlay_records == ()
    assert predictor.requests == []
    assert materializer.calls == []


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
    assert first.center_xy == (0.5, 0.5)
    second = artifact.tracks[1].observations[0]
    assert second.bbox_xyxy == (0.5, 0.25, 0.75, 0.75)
    assert second.confidence == 0.625
    assert second.area_fraction == 0.25
    assert second.center_xy == (0.75, 0.75)
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
                assert archive.namelist() == [
                    "masks/00000000.npy",
                    "masks/00000001.npy",
                    "masks/00000002.npy",
                    "masks/00000003.npy",
                    "masks/00000004.npy",
                    "masks/00000005.npy",
                    "local_frame_indices.npy",
                    "frame_indices.npy",
                    "object_ids.npy",
                    "mask_indices.npy",
                    "sampled_frame_indices.npy",
                ]
                for mask_index in range(6):
                    shape, mask_payload = read_npy(
                        archive.read(f"masks/{mask_index:08d}.npy")
                    )
                    assert shape == (2, 2)
                    assert len(mask_payload) == 4


def test_every_observation_resolves_through_npz_frame_object_mapping(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    provider = make_provider(PredictorDouble(), fake_torch, MaterializerDouble())

    artifact = provider.analyze(request, tmp_path / "staging")

    for track in artifact.tracks:
        assert track.observations
        object_id = int(track.track_id.removeprefix(f"{track.entity_id}_"))
        archive_path = tmp_path / "staging" / track.observations[0].mask_ref
        with zipfile.ZipFile(archive_path) as archive:
            frame_indices = read_int64_array(archive, "frame_indices.npy")
            object_ids = read_int64_array(archive, "object_ids.npy")
            mask_indices = read_int64_array(archive, "mask_indices.npy")
            sampled = read_int64_array(archive, "sampled_frame_indices.npy")
            assert sampled == (0, 1, 2)
            mapping = {
                (frame_index, mapped_object): mask_index
                for frame_index, mapped_object, mask_index in zip(
                    frame_indices, object_ids, mask_indices
                )
            }
            for observation in track.observations:
                mask_index = mapping[(observation.frame_index, object_id)]
                assert f"masks/{mask_index:08d}.npy" in archive.namelist()


def test_mask_rows_stream_without_tolist_or_full_mask_materialization(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    width = 1920
    height = 1080

    class StreamingMaskArray:
        shape = (1, height, width)
        dtype = SimpleNamespace(kind="b")

        def __init__(self) -> None:
            self.tolist_calls = 0
            self.row_reads = 0
            self.max_materialized_bytes = 0

        def tolist(self) -> Any:
            self.tolist_calls += 1
            raise AssertionError("full-mask tolist is forbidden")

        def __getitem__(self, index: Any) -> Any:
            assert isinstance(index, tuple) and len(index) == 2
            object_index, row_index = index
            assert object_index == 0
            assert 0 <= row_index < height
            self.row_reads += 1
            row = (b"\x01" if row_index == 0 else b"\x00") + b"\x00" * (
                width - 1
            )
            self.max_materialized_bytes = max(self.max_materialized_bytes, len(row))
            return ArrayRowDouble(row)

    class ArrayRowDouble:
        def __init__(self, payload: bytes) -> None:
            self.shape = (len(payload),)
            self._payload = payload

        def item(self) -> Any:
            raise AssertionError("a non-scalar ndarray row must not call item()")

        def tobytes(self, *, order: str) -> bytes:
            assert order == "C"
            return self._payload

    mask_array = StreamingMaskArray()
    response = frame_output(0)
    response["outputs"]["out_binary_masks"] = mask_array
    predictor = PredictorDouble(lambda *_: [response])
    request = make_request(
        tmp_path,
        duration_seconds=1.0,
        timestamps=(0.0,),
        entities=(make_request(tmp_path).entities[0],),
    )
    provider = make_provider(
        predictor,
        fake_torch,
        MaterializerDouble(width=width, height=height),
        max_artifact_bytes=8 * 1024 * 1024,
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    assert artifact.tracks[0].observations[0].area_fraction == pytest.approx(
        1 / (width * height)
    )
    assert mask_array.tolist_calls == 0
    assert mask_array.row_reads == height
    assert mask_array.max_materialized_bytes == width
    assert sum(item.size_bytes for item in artifact.files) < 100_000


def test_adapter_rejects_projected_mask_output_before_reading_rows(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    class OversizedMaskArray:
        shape = (1, 1080, 1920)
        dtype = SimpleNamespace(kind="b")

        def __init__(self) -> None:
            self.row_reads = 0

        def __getitem__(self, index: Any) -> bytes:
            del index
            self.row_reads += 1
            return b""

        def tolist(self) -> Any:
            raise AssertionError("full-mask tolist is forbidden")

    masks = OversizedMaskArray()
    response = frame_output(0)
    response["outputs"]["out_binary_masks"] = masks
    request = make_request(
        tmp_path,
        duration_seconds=1.0,
        timestamps=(0.0,),
        entities=(make_request(tmp_path).entities[0],),
    )
    provider = make_provider(
        PredictorDouble(lambda *_: [response]),
        fake_torch,
        MaterializerDouble(width=1920, height=1080),
        max_artifact_bytes=1024,
    )

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert masks.row_reads == 0


def test_adapter_rejects_mask_dimensions_that_do_not_match_sampled_jpeg(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(
        tmp_path,
        duration_seconds=1.0,
        timestamps=(0.0,),
        entities=(make_request(tmp_path).entities[0],),
    )
    provider = make_provider(
        PredictorDouble(lambda *_: [frame_output(0)]),
        fake_torch,
        MaterializerDouble(width=3, height=2),
    )

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")


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
        stop = start + stream_request["max_frame_num_to_track"]
        return [empty_frame_output(index) for index in range(start, stop)]

    predictor = PredictorDouble(empty_stream)
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    artifact = provider.analyze(request, tmp_path / "staging")

    assert artifact.tracks == ()
    assert artifact.processed_timeline == request.timeline
    assert [file.path for file in artifact.files] == ["masks/right_hand.npz"]
    assert predictor.requests[-1]["type"] == "close_session"


def test_removed_sentinel_does_not_publish_detection(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(
        tmp_path, entities=(make_request(tmp_path).entities[0],)
    )

    def stream(
        session_number: int, prompt: str, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        del session_number, prompt
        return [
            frame_output(index, probabilities=[-10000.0])
            for index in range(
                payload["start_frame_index"],
                payload["start_frame_index"]
                + payload["max_frame_num_to_track"],
            )
        ]

    provider = make_provider(
        PredictorDouble(stream), fake_torch, MaterializerDouble()
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    assert artifact.tracks == ()
    assert artifact.overlay_records == ()
    assert artifact.processed_timeline == request.timeline
    with zipfile.ZipFile(tmp_path / "staging" / "masks/right_hand.npz") as archive:
        assert not any(name.startswith("masks/") for name in archive.namelist())
        assert read_int64_array(archive, "frame_indices.npy") == ()
        assert read_int64_array(archive, "object_ids.npy") == ()
        assert read_int64_array(archive, "mask_indices.npy") == ()


def test_removed_sentinel_mixed_rows_preserve_only_valid_detection_and_mask(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(
        tmp_path,
        duration_seconds=1.0,
        timestamps=(0.0,),
        entities=(make_request(tmp_path).entities[0],),
    )
    response = frame_output(
        0,
        object_ids=[2, 1],
        probabilities=[-10000.0, 0.875],
        boxes=[[0.5, 0.25, 0.25, 0.5], [0.1, 0.2, 0.4, 0.5]],
        masks=[
            [[False, False], [False, True]],
            [[True, False], [False, True]],
        ],
    )
    provider = make_provider(
        PredictorDouble(lambda *_: [response]), fake_torch, MaterializerDouble()
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    assert [track.track_id for track in artifact.tracks] == ["right_hand_1"]
    observation = artifact.tracks[0].observations[0]
    assert observation.frame_index == 0
    assert observation.bbox_xyxy == (0.1, 0.2, 0.5, 0.7)
    assert observation.confidence == 0.875
    assert artifact.overlay_records == ()
    with zipfile.ZipFile(tmp_path / "staging" / observation.mask_ref) as archive:
        assert read_int64_array(archive, "frame_indices.npy") == (0,)
        assert read_int64_array(archive, "object_ids.npy") == (1,)
        assert read_int64_array(archive, "mask_indices.npy") == (0,)
        mask_members = [
            name for name in archive.namelist() if name.startswith("masks/")
        ]
        assert mask_members == ["masks/00000000.npy"]
        assert read_npy(archive.read(mask_members[0])) == (
            (2, 2),
            b"\x01\x00\x00\x01",
        )


def test_removed_sentinel_between_visible_frames_adds_no_observation(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(
        tmp_path, entities=(make_request(tmp_path).entities[0],)
    )
    responses = [
        frame_output(0),
        frame_output(1, probabilities=[-10000.0]),
        frame_output(2),
    ]

    def render_overlay(*, destination: Path, **kwargs: Any) -> None:
        del kwargs
        destination.write_bytes(b"\x89PNG\r\n\x1a\nremoved-sentinel-test")

    provider = make_provider(
        PredictorDouble(lambda *_: responses),
        fake_torch,
        MaterializerDouble(),
        overlay_renderer=render_overlay,
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    assert [
        observation.frame_index
        for observation in artifact.tracks[0].observations
    ] == [0, 2]
    assert all(
        observation.visible
        for observation in artifact.tracks[0].observations
    )
    assert all(record.frame_index != 1 for record in artifact.overlay_records)


@pytest.mark.parametrize(
    "probability",
    [
        pytest.param(-9999.0, id="near-sentinel"),
        pytest.param(-10000.001, id="below-sentinel"),
        pytest.param(-0.1, id="negative"),
        pytest.param(1.001, id="above-one"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(True, id="boolean-true"),
        pytest.param(False, id="boolean-false"),
    ],
)
def test_removed_sentinel_does_not_admit_other_invalid_scores_and_closes(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    probability: Any,
) -> None:
    request = make_request(
        tmp_path, entities=(make_request(tmp_path).entities[0],)
    )
    predictor = PredictorDouble(
        lambda *_: [
            frame_output(0, probabilities=[probability]),
            frame_output(1),
            frame_output(2),
        ]
    )
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests[-1]["type"] == "close_session"
    assert not (tmp_path / "staging" / "masks").exists()


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            frame_output(
                0,
                probabilities=[-10000.0],
                boxes=[[0.8, 0.2, 0.3, 0.5]],
            ),
            id="invalid-geometry",
        ),
        pytest.param(
            frame_output(
                0,
                probabilities=[-10000.0],
                masks=[[[True, 0], [False, True]]],
            ),
            id="invalid-mask-content",
        ),
        pytest.param(
            frame_output(
                0,
                object_ids=[1, 1],
                probabilities=[-10000.0, 0.875],
                boxes=[[0.1, 0.2, 0.4, 0.5], [0.1, 0.2, 0.4, 0.5]],
                masks=[
                    [[True, False], [False, True]],
                    [[True, False], [False, True]],
                ],
            ),
            id="duplicate-id",
        ),
    ],
)
def test_removed_sentinel_does_not_bypass_row_safety_validation(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    response: dict[str, Any],
) -> None:
    request = make_request(
        tmp_path,
        duration_seconds=1.0,
        timestamps=(0.0,),
        entities=(make_request(tmp_path).entities[0],),
    )
    predictor = PredictorDouble(lambda *_: [response])
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests[-1]["type"] == "close_session"
    assert not (tmp_path / "staging" / "masks").exists()


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


def test_adapter_rejects_out_of_range_observation_before_publication(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path, entities=(make_request(tmp_path).entities[0],))
    predictor = PredictorDouble(lambda *_: [frame_output(request.frame_count)])
    provider = make_provider(predictor, fake_torch, MaterializerDouble())

    with pytest.raises(CvProviderError, match="^SAM3.1 CV evidence inference failed$"):
        provider.analyze(request, tmp_path / "staging")

    assert predictor.requests[-1]["type"] == "close_session"
    assert not (tmp_path / "staging" / "masks").exists()


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


@pytest.mark.parametrize("frame_count", [1, 5, 32])
def test_adapter_matches_pinned_tracker_and_detector_propagation_bounds(
    tmp_path: Path, fake_torch: SimpleNamespace, frame_count: int
) -> None:
    class PinnedBoundaryPredictor(PredictorDouble):
        detector_batch_size = 16

        def __init__(self) -> None:
            super().__init__()
            self.detector_chunks: list[tuple[tuple[int, ...], ...]] = []

        def handle_stream_request(self, stream_request: dict[str, Any]):
            self.stream_requests.append(dict(stream_request))
            session_number = int(stream_request["session_id"].rsplit("-", 1)[1])
            num_frames = len(self.resource_files[session_number])
            start = stream_request["start_frame_index"]
            count = stream_request["max_frame_num_to_track"]
            tracker_end = min(start + count, num_frames - 1)
            detector_valid_end = start + count
            chunks: list[tuple[int, ...]] = []
            for chunk_start in range(start, tracker_end + 1, self.detector_batch_size):
                chunk_end = min(
                    chunk_start + self.detector_batch_size, detector_valid_end
                )
                chunks.append(tuple(range(chunk_start, chunk_end)))
            self.detector_chunks.append(tuple(chunks))
            consumed = tuple(index for chunk in chunks for index in chunk)
            expected = tuple(range(start, tracker_end + 1))
            if consumed != expected:
                raise RuntimeError("empty final feature batch")
            for index in consumed:
                yield frame_output(index)

    request = make_request(
        tmp_path,
        duration_seconds=max(0.001, (frame_count - 1) / 30),
        timestamps=tuple(index / 30 for index in range(frame_count)),
    )
    predictor = PinnedBoundaryPredictor()
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
        for item in predictor.stream_requests
    ]
    assert per_prompt == [(0, frame_count), (0, frame_count)]
    assert [
        tuple(index for chunk in prompt_chunks for index in chunk)
        for prompt_chunks in predictor.detector_chunks
    ] == [tuple(range(frame_count)), tuple(range(frame_count))]
    assert materializer.calls[0][1] == tuple(range(frame_count))
    assert [item.frame_index for item in artifact.tracks[0].observations] == list(
        range(frame_count)
    )
    assert all(
        observation.frame_index < frame_count
        for track in artifact.tracks
        for observation in track.observations
    )


def test_pinned_base_start_session_compatibility_and_execution_chunk_control(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    class PinnedMultiplexModel:
        def __init__(self) -> None:
            self.postprocess_batch_size = 16
            self.batched_grounding_batch_size = 16
            self.init_calls: list[dict[str, Any]] = []

        def init_state(
            self,
            resource_path: str,
            offload_video_to_cpu: bool = False,
            async_loading_frames: bool = False,
            use_torchcodec: bool = False,
            use_cv2: bool = False,
            input_is_mp4: bool = False,
        ) -> dict[str, Any]:
            self.init_calls.append(
                {
                    "resource_path": resource_path,
                    "offload_video_to_cpu": offload_video_to_cpu,
                    "async_loading_frames": async_loading_frames,
                    "use_torchcodec": use_torchcodec,
                    "use_cv2": use_cv2,
                    "input_is_mp4": input_is_mp4,
                }
            )
            return {}

    class PinnedBasePredictor(PredictorDouble):
        def __init__(self) -> None:
            super().__init__()
            self.model = PinnedMultiplexModel()

        def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
            if request["type"] == "start_session":
                # This reproduces the pinned BasePredictor call which otherwise
                # raises because multiplex init_state has no such keyword.
                self.model.init_state(
                    resource_path=request["resource_path"],
                    offload_video_to_cpu=False,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )
            return super().handle_request(request)

    predictor = PinnedBasePredictor()
    provider = make_provider(
        predictor,
        fake_torch,
        MaterializerDouble(),
        execution_chunk_frames=8,
    )

    assert predictor.model.postprocess_batch_size == 8
    assert predictor.model.batched_grounding_batch_size == 8
    provider.set_execution_chunk_frames(3)
    assert predictor.model.postprocess_batch_size == 3
    assert predictor.model.batched_grounding_batch_size == 3

    artifact = provider.analyze(make_request(tmp_path), tmp_path / "staging")

    assert artifact.tracks
    assert len(predictor.model.init_calls) == 1
    assert len(predictor.stream_requests) == 2


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
        stop = start + stream_request["max_frame_num_to_track"]
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
    assert [
        frame.frame_index for frame in artifact.processed_timeline.frames
    ] == [0, 3, 4, 5, 7, 8, 9]
    assert [
        frame.timestamp_seconds for frame in artifact.processed_timeline.frames
    ] == [0.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0]
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
        stop = start + stream_request["max_frame_num_to_track"]
        return [
            frame_output(index) if index % 2 == 0 else empty_frame_output(index)
            for index in range(start, stop)
        ]

    rendered: list[tuple[Path, Path, str, tuple[float, ...]]] = []

    def render_context_overlay(
        *,
        frame_path: Path,
        archive_path: Path,
        mask_member: str,
        bbox_xyxy: tuple[float, float, float, float],
        destination: Path,
    ) -> None:
        assert frame_path.read_bytes().startswith(b"\xff\xd8")
        with zipfile.ZipFile(archive_path) as archive:
            assert mask_member in archive.namelist()
        rendered.append((frame_path, archive_path, mask_member, bbox_xyxy))
        destination.write_bytes(b"\x89PNG\r\n\x1a\ncontextual-overlay")

    provider = make_provider(
        PredictorDouble(flicker),
        fake_torch,
        MaterializerDouble(),
        execution_chunk_frames=30,
        overlay_renderer=render_context_overlay,
    )

    artifact = provider.analyze(request, tmp_path / "staging")

    overlays = [file for file in artifact.files if file.path.endswith(".png")]
    assert 0 < len(overlays) <= 24
    assert {record.path for record in artifact.overlay_records} == {
        file.path for file in overlays
    }
    observations = {
        (track.track_id, observation.frame_index)
        for track in artifact.tracks
        for observation in track.observations
        if observation.visible
    }
    assert {
        (record.track_id, record.frame_index)
        for record in artifact.overlay_records
    } <= observations
    assert len(rendered) == len(overlays)
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


@pytest.mark.parametrize("cleanup_type", [KeyboardInterrupt, SystemExit])
def test_cuda_cache_baseexception_cannot_mask_oom_or_discard_retry_plan(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    cleanup_type: type[BaseException],
) -> None:
    request = make_request(
        tmp_path,
        entities=(make_request(tmp_path).entities[0],),
    )
    attempts = 0

    def oom_once(
        _session_number: int, _prompt: str, stream_request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise fake_torch.cuda.OutOfMemoryError("private allocator state")
        return [
            frame_output(index)
            for index in range(stream_request["max_frame_num_to_track"])
        ]

    empty_cache_calls = 0

    def exploding_empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1
        raise cleanup_type("cache cleanup interrupt")

    fake_torch.cuda.empty_cache = exploding_empty_cache
    materializer = MaterializerDouble()
    provider = make_provider(PredictorDouble(oom_once), fake_torch, materializer)

    raised: BaseException | None = None
    try:
        provider.analyze(request, tmp_path / "first-staging")
    except BaseException as error:
        raised = error

    assert isinstance(raised, CvOutOfMemoryError)
    assert str(raised) == "SAM3.1 CV evidence inference ran out of memory"
    artifact = provider.analyze(request, tmp_path / "retry-staging")
    assert artifact.tracks
    assert [call[1] for call in materializer.calls] == [(0, 1, 2), (0, 1, 2)]
    assert empty_cache_calls == 1


def test_worker_owned_oom_retry_reuses_identical_long_video_final_samples(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(
        tmp_path,
        duration_seconds=31.0,
        timestamps=tuple(float(index) for index in range(10)),
        entities=(make_request(tmp_path).entities[0],),
    )
    refinement_calls = 0

    def initial(*_: Any) -> tuple[int, ...]:
        return (0, 4, 8)

    def refine(*_: Any) -> tuple[int, ...]:
        nonlocal refinement_calls
        refinement_calls += 1
        return (3, 4, 5) if refinement_calls == 1 else (7, 8, 9)

    def stream(
        session_number: int, prompt: str, stream_request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        del prompt
        count = stream_request["max_frame_num_to_track"]
        if session_number == 0:
            return [frame_output(0), empty_frame_output(1), frame_output(2)]
        if session_number == 1:
            raise fake_torch.cuda.OutOfMemoryError("private allocator state")
        return [frame_output(index) for index in range(count)]

    predictor = PredictorDouble(stream)
    materializer = MaterializerDouble()
    provider = make_provider(
        predictor,
        fake_torch,
        materializer,
        initial_sampler=initial,
        refinement_sampler=refine,
    )

    with pytest.raises(CvOutOfMemoryError):
        provider.analyze(request, tmp_path / "first-staging")
    provider.set_execution_chunk_frames(4)
    artifact = provider.analyze(request, tmp_path / "retry-staging")

    assert artifact.tracks
    assert refinement_calls == 1
    assert [call[1] for call in materializer.calls] == [
        (0, 4, 8),
        (0, 3, 4, 5, 8),
        (0, 3, 4, 5, 8),
    ]
    assert provider.request_metrics()["execution_chunk_frames"] == 4


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


@pytest.mark.parametrize(
    "failure_factory",
    [
        pytest.param(
            lambda: OSError("private stream close path"), id="oserror"
        ),
        pytest.param(
            lambda: KeyboardInterrupt("private stream close interrupt"),
            id="keyboard",
        ),
        pytest.param(
            lambda: SystemExit("private stream close exit"), id="system-exit"
        ),
    ],
)
def test_successful_stream_cleanup_failure_is_sanitized_and_cleans_outputs(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    failure_factory: Callable[[], BaseException],
) -> None:
    request = make_request(tmp_path)
    stream_closed = False

    class CompleteStream:
        def __init__(self, frame_count: int) -> None:
            self._outputs = iter(frame_output(index) for index in range(frame_count))

        def __iter__(self) -> CompleteStream:
            return self

        def __next__(self) -> dict[str, Any]:
            return next(self._outputs)

        def close(self) -> None:
            nonlocal stream_closed
            stream_closed = True
            raise failure_factory()

    class PredictorWithFailingStreamClose(PredictorDouble):
        def handle_stream_request(self, request: dict[str, Any]) -> CompleteStream:
            self.stream_requests.append(dict(request))
            return CompleteStream(request["max_frame_num_to_track"])

    predictor = PredictorWithFailingStreamClose()
    provider = make_provider(predictor, fake_torch, MaterializerDouble())
    staging = tmp_path / "staging"
    raised: BaseException | None = None

    try:
        provider.analyze(request, staging)
    except BaseException as error:
        raised = error

    assert type(raised) is CvProviderError
    assert str(raised) == "SAM3.1 CV evidence inference failed"
    assert stream_closed is True
    assert predictor.requests[-1]["type"] == "close_session"
    assert list(staging.glob(".sam31-frames-*")) == []
    assert not (staging / "masks").exists()
    assert "private" not in str(raised)


@pytest.mark.parametrize(
    "failure_factory",
    [
        pytest.param(
            lambda: OSError("private session close path"), id="oserror"
        ),
        pytest.param(
            lambda: KeyboardInterrupt("private session close interrupt"),
            id="keyboard",
        ),
        pytest.param(
            lambda: SystemExit("private session close exit"), id="system-exit"
        ),
    ],
)
def test_successful_session_cleanup_failure_is_sanitized_and_cleans_outputs(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    failure_factory: Callable[[], BaseException],
) -> None:
    request = make_request(tmp_path)
    close_attempts = 0

    class PredictorWithFailingSessionClose(PredictorDouble):
        def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
            nonlocal close_attempts
            response = super().handle_request(request)
            if request["type"] == "close_session":
                close_attempts += 1
                raise failure_factory()
            return response

    predictor = PredictorWithFailingSessionClose()
    provider = make_provider(predictor, fake_torch, MaterializerDouble())
    staging = tmp_path / "staging"
    raised: BaseException | None = None

    try:
        provider.analyze(request, staging)
    except BaseException as error:
        raised = error

    assert type(raised) is CvProviderError
    assert str(raised) == "SAM3.1 CV evidence inference failed"
    assert close_attempts == 1
    assert list(staging.glob(".sam31-frames-*")) == []
    assert not (staging / "masks").exists()
    assert "private" not in str(raised)


@pytest.mark.parametrize(
    "primary_kind", ["ordinary", "oom", "keyboard", "system-exit"]
)
def test_inference_primary_wins_over_stream_and_session_baseexception_cleanup(
    tmp_path: Path,
    fake_torch: SimpleNamespace,
    primary_kind: str,
) -> None:
    request = make_request(tmp_path)
    if primary_kind == "ordinary":
        primary: BaseException = RuntimeError("private inference failure")
    elif primary_kind == "oom":
        primary = fake_torch.cuda.OutOfMemoryError("private allocator failure")
    elif primary_kind == "keyboard":
        primary = KeyboardInterrupt("primary inference interrupt")
    else:
        primary = SystemExit("primary inference exit")
    stream_close_attempts = 0
    session_close_attempts = 0

    class FailingStream:
        def __iter__(self) -> FailingStream:
            return self

        def __next__(self) -> dict[str, Any]:
            raise primary

        def close(self) -> None:
            nonlocal stream_close_attempts
            stream_close_attempts += 1
            raise KeyboardInterrupt("secondary stream cleanup")

    class PredictorWithFailingCleanup(PredictorDouble):
        def handle_stream_request(self, request: dict[str, Any]) -> FailingStream:
            self.stream_requests.append(dict(request))
            return FailingStream()

        def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
            nonlocal session_close_attempts
            if request["type"] == "close_session":
                session_close_attempts += 1
                raise SystemExit("secondary session cleanup")
            return super().handle_request(request)

    provider = make_provider(
        PredictorWithFailingCleanup(), fake_torch, MaterializerDouble()
    )
    staging = tmp_path / "staging"
    raised: BaseException | None = None

    try:
        provider.analyze(request, staging)
    except BaseException as error:
        raised = error

    if primary_kind == "ordinary":
        assert type(raised) is CvProviderError
        assert str(raised) == "SAM3.1 CV evidence inference failed"
    elif primary_kind == "oom":
        assert type(raised) is CvOutOfMemoryError
        assert str(raised) == "SAM3.1 CV evidence inference ran out of memory"
    else:
        assert raised is primary
    assert stream_close_attempts == 1
    assert session_close_attempts == 1
    assert list(staging.glob(".sam31-frames-*")) == []
    assert not (staging / "masks").exists()


def test_baseexception_cleanup_cannot_mask_primary_stream_interrupt(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    primary = KeyboardInterrupt("primary interrupt")

    class InterruptingStream:
        def __iter__(self) -> InterruptingStream:
            return self

        def __next__(self) -> dict[str, Any]:
            raise primary

        def close(self) -> None:
            raise SystemExit("stream cleanup")

    class CleanupExplodingPredictor(PredictorDouble):
        def handle_stream_request(self, request: dict[str, Any]) -> InterruptingStream:
            self.stream_requests.append(dict(request))
            return InterruptingStream()

        def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
            if request["type"] == "close_session":
                raise SystemExit("session cleanup")
            return super().handle_request(request)

    provider = make_provider(
        CleanupExplodingPredictor(), fake_torch, MaterializerDouble()
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        provider.analyze(request, tmp_path / "staging")

    assert raised.value is primary


def test_session_cleanup_interrupt_cannot_mask_primary_system_exit(
    tmp_path: Path, fake_torch: SimpleNamespace
) -> None:
    request = make_request(tmp_path)
    primary = SystemExit("primary exit")

    class ExplodingPredictor(PredictorDouble):
        def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
            if request["type"] == "add_prompt":
                raise primary
            if request["type"] == "close_session":
                raise KeyboardInterrupt("session cleanup")
            return super().handle_request(request)

    provider = make_provider(ExplodingPredictor(), fake_torch, MaterializerDouble())

    with pytest.raises(SystemExit) as raised:
        provider.analyze(request, tmp_path / "staging")

    assert raised.value is primary


@pytest.mark.parametrize("primary_kind", ["keyboard", "system-exit", "ordinary"])
@pytest.mark.parametrize(
    "cleanup_factory",
    [
        pytest.param(lambda: OSError("cleanup path"), id="cleanup-oserror"),
        pytest.param(
            lambda: KeyboardInterrupt("cleanup interrupt"), id="cleanup-keyboard"
        ),
        pytest.param(lambda: SystemExit("cleanup exit"), id="cleanup-system-exit"),
    ],
)
def test_analysis_cleanup_baseexception_preserves_primary_and_attempts_all_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
    primary_kind: str,
    cleanup_factory: Callable[[], BaseException],
) -> None:
    request = make_request(tmp_path)
    primary: BaseException
    if primary_kind == "keyboard":
        primary = KeyboardInterrupt("primary keyboard interrupt")
    elif primary_kind == "system-exit":
        primary = SystemExit("primary system exit")
    else:
        primary = RuntimeError("private ordinary failure")

    def fail(*_: Any) -> list[dict[str, Any]]:
        raise primary

    cleanup_paths: list[Path] = []

    def exploding_cleanup(path: Path) -> None:
        cleanup_paths.append(path)
        raise cleanup_factory()

    monkeypatch.setattr(SAM31_MODULE, "_remove_tree", exploding_cleanup)
    provider = make_provider(PredictorDouble(fail), fake_torch, MaterializerDouble())

    raised: BaseException | None = None
    try:
        provider.analyze(request, tmp_path / "staging")
    except BaseException as error:
        raised = error

    if primary_kind == "ordinary":
        assert type(raised) is CvProviderError
        assert str(raised) == "SAM3.1 CV evidence inference failed"
    else:
        assert raised is primary
    assert len(cleanup_paths) == 2
    assert cleanup_paths[0].name.startswith(".sam31-frames-")
    assert cleanup_paths[1] == tmp_path / "staging" / "masks"


def test_successful_analysis_with_cleanup_baseexception_fails_sanitized_and_cleans_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    request = make_request(tmp_path)
    cleanup_paths: list[Path] = []

    def exploding_cleanup(path: Path) -> None:
        cleanup_paths.append(path)
        raise SystemExit("cleanup exit detail")

    monkeypatch.setattr(SAM31_MODULE, "_remove_tree", exploding_cleanup)
    provider = make_provider(PredictorDouble(), fake_torch, MaterializerDouble())

    raised: BaseException | None = None
    try:
        provider.analyze(request, tmp_path / "staging")
    except BaseException as error:
        raised = error

    assert type(raised) is CvProviderError
    assert str(raised) == "SAM3.1 CV evidence inference failed"
    assert len(cleanup_paths) == 2
    assert cleanup_paths[0].name.startswith(".sam31-frames-")
    assert cleanup_paths[1] == tmp_path / "staging" / "masks"


@pytest.mark.parametrize(
    "cleanup_factory",
    [
        pytest.param(lambda: OSError("private cleanup path"), id="oserror"),
        pytest.param(
            lambda: KeyboardInterrupt("private cleanup interrupt"), id="keyboard"
        ),
        pytest.param(
            lambda: SystemExit("private cleanup exit"), id="system-exit"
        ),
    ],
)
def test_handled_outer_exception_cannot_hide_successful_analysis_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
    cleanup_factory: Callable[[], BaseException],
) -> None:
    request = make_request(tmp_path)
    staging = tmp_path / "staging"
    cleanup_paths: list[Path] = []
    real_remove_tree = SAM31_MODULE._remove_tree

    def remove_then_fail(path: Path) -> None:
        cleanup_paths.append(path)
        real_remove_tree(path)
        raise cleanup_factory()

    monkeypatch.setattr(SAM31_MODULE, "_remove_tree", remove_then_fail)
    provider = make_provider(PredictorDouble(), fake_torch, MaterializerDouble())
    raised: BaseException | None = None

    try:
        raise RuntimeError("already handled by caller")
    except RuntimeError:
        try:
            provider.analyze(request, staging)
        except BaseException as error:
            raised = error

    assert type(raised) is CvProviderError
    assert str(raised) == "SAM3.1 CV evidence inference failed"
    assert "private" not in str(raised)
    assert len(cleanup_paths) == 2
    assert cleanup_paths[0].name.startswith(".sam31-frames-")
    assert cleanup_paths[1] == staging / "masks"
    assert list(staging.iterdir()) == []


def test_successful_analysis_surfaces_real_rmtree_oserror_as_sanitized_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    request = make_request(tmp_path)
    removed_paths: list[Path] = []

    def failing_rmtree(path: Path) -> None:
        removed_paths.append(Path(path))
        raise OSError("private filesystem detail")

    monkeypatch.setattr(SAM31_MODULE.shutil, "rmtree", failing_rmtree)
    provider = make_provider(PredictorDouble(), fake_torch, MaterializerDouble())

    raised: BaseException | None = None
    try:
        provider.analyze(request, tmp_path / "staging")
    except BaseException as error:
        raised = error

    assert type(raised) is CvProviderError
    assert str(raised) == "SAM3.1 CV evidence inference failed"
    assert len(removed_paths) == 2
    assert removed_paths[0].name.startswith(".sam31-frames-")
    assert removed_paths[1] == tmp_path / "staging" / "masks"


@pytest.mark.parametrize(
    "primary_kind",
    ["oom", "keyboard", "system-exit", "ordinary"],
)
def test_real_rmtree_oserror_preserves_primary_and_attempts_all_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
    primary_kind: str,
) -> None:
    request = make_request(tmp_path)
    if primary_kind == "oom":
        primary: BaseException = fake_torch.cuda.OutOfMemoryError("private OOM")
    elif primary_kind == "keyboard":
        primary = KeyboardInterrupt("primary keyboard")
    elif primary_kind == "system-exit":
        primary = SystemExit("primary exit")
    else:
        primary = RuntimeError("private ordinary failure")

    def fail(*_: Any) -> list[dict[str, Any]]:
        raise primary

    removed_paths: list[Path] = []

    def failing_rmtree(path: Path) -> None:
        removed_paths.append(Path(path))
        raise OSError("private cleanup failure")

    monkeypatch.setattr(SAM31_MODULE.shutil, "rmtree", failing_rmtree)
    provider = make_provider(PredictorDouble(fail), fake_torch, MaterializerDouble())

    raised: BaseException | None = None
    try:
        provider.analyze(request, tmp_path / "staging")
    except BaseException as error:
        raised = error

    if primary_kind == "oom":
        assert type(raised) is CvOutOfMemoryError
        assert str(raised) == "SAM3.1 CV evidence inference ran out of memory"
    elif primary_kind == "ordinary":
        assert type(raised) is CvProviderError
        assert str(raised) == "SAM3.1 CV evidence inference failed"
    else:
        assert raised is primary
    assert len(removed_paths) == 2
    assert removed_paths[0].name.startswith(".sam31-frames-")
    assert removed_paths[1] == tmp_path / "staging" / "masks"


def test_provider_close_surfaces_real_runtime_cleanup_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    runtime_directory = tmp_path / "private-runtime"
    runtime_directory.mkdir(mode=0o700)

    def failing_rmtree(_path: Path) -> None:
        raise OSError("private runtime cleanup detail")

    monkeypatch.setattr(SAM31_MODULE.shutil, "rmtree", failing_rmtree)
    provider = make_provider(
        PredictorDouble(),
        fake_torch,
        MaterializerDouble(),
        runtime_directory=runtime_directory,
    )

    with pytest.raises(
        CvProviderError, match="^Unable to close SAM3.1 runtime$"
    ):
        provider.close()

    assert runtime_directory.exists()


@pytest.mark.parametrize("failure_stage", ["shutdown", "restore", "remove"])
@pytest.mark.parametrize(
    "failure_factory",
    [
        pytest.param(
            lambda: OSError("private close filesystem path"), id="oserror"
        ),
        pytest.param(
            lambda: KeyboardInterrupt("private close interrupt"), id="keyboard"
        ),
        pytest.param(
            lambda: SystemExit("private close exit"), id="system-exit"
        ),
    ],
)
def test_handled_outer_exception_cannot_hide_direct_provider_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
    failure_stage: str,
    failure_factory: Callable[[], BaseException],
) -> None:
    runtime_directory = tmp_path / "private-runtime"
    runtime_directory.mkdir(mode=0o700)
    failure = failure_factory()
    cleanup_steps: list[str] = []

    class PredictorWithRecordedShutdown(PredictorDouble):
        def shutdown(self) -> None:
            cleanup_steps.append("shutdown")
            if failure_stage == "shutdown":
                raise failure

    def restore_imports(_snapshot: Any) -> None:
        cleanup_steps.append("restore")
        if failure_stage == "restore":
            raise failure

    real_remove_tree = SAM31_MODULE._remove_tree

    def remove_runtime(path: Path) -> None:
        cleanup_steps.append("remove")
        real_remove_tree(path)
        if failure_stage == "remove":
            raise failure

    monkeypatch.setattr(SAM31_MODULE, "_restore_sam_modules", restore_imports)
    monkeypatch.setattr(SAM31_MODULE, "_remove_tree", remove_runtime)
    provider = make_provider(
        PredictorWithRecordedShutdown(),
        fake_torch,
        MaterializerDouble(),
        runtime_directory=runtime_directory,
        import_snapshot=object(),
    )
    raised: BaseException | None = None

    try:
        raise RuntimeError("already handled by caller")
    except RuntimeError:
        try:
            provider.close()
        except BaseException as error:
            raised = error

    assert type(raised) is CvProviderError
    assert str(raised) == "Unable to close SAM3.1 runtime"
    assert "private" not in str(raised)
    assert cleanup_steps == ["shutdown", "restore", "remove"]
    assert not runtime_directory.exists()


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
        "execution_chunk_frames": 8,
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
    assets = package / "assets"
    assets.mkdir()
    (assets / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"local bpe fixture")
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
        if "status" in command:
            return SimpleNamespace(stdout="")
        if "ls-files" in command:
            repository = Path(command[2])
            tracked = sorted(
                path.relative_to(repository).as_posix()
                for path in (repository / "sam3").rglob("*")
                if path.is_file()
            )
            return SimpleNamespace(stdout="\0".join(tracked) + "\0")
        return SimpleNamespace(stdout=PINNED_REVISION + "\n")

    return run


PINNED_BUILDER_SOURCE = (
    "SOURCE_MARKER = 'pinned commit blob'\n"
    "calls = []\n"
    "class Predictor:\n"
    "    def handle_request(self, request):\n"
    "        return {}\n"
    "    def handle_stream_request(self, request):\n"
    "        return iter(())\n"
    "def build_sam3_multiplex_video_predictor(**kwargs):\n"
    "    calls.append(kwargs)\n"
    "    return Predictor()\n"
)


def pin_runtime_repository(
    monkeypatch: pytest.MonkeyPatch, repository: Path
) -> str:
    subprocess.run(
        ["git", "init", "-q", str(repository)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "add", "sam3"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=SAM Test",
            "-c",
            "user.email=sam-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "pinned fixture",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(SAM31_MODULE, "_PINNED_REPOSITORY_REVISION", revision)
    return revision


@pytest.mark.parametrize(
    "attack",
    ["assume-unchanged", "check-copy-restore"],
)
def test_source_snapshot_uses_pinned_commit_blobs_not_worktree_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
    attack: str,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    builder = repository / "sam3" / "model_builder.py"
    lazy = repository / "sam3" / "lazy_component.py"
    builder.write_text(PINNED_BUILDER_SOURCE, encoding="utf-8")
    lazy.write_text("VALUE = 'pinned commit blob'\n", encoding="utf-8")
    pin_runtime_repository(monkeypatch, repository)
    install_fake_torch_module(monkeypatch, fake_torch)
    malicious_builder = PINNED_BUILDER_SOURCE.replace(
        "pinned commit blob", "malicious worktree bytes"
    )

    if attack == "assume-unchanged":
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "update-index",
                "--assume-unchanged",
                "sam3/model_builder.py",
                "sam3/lazy_component.py",
            ],
            check=True,
            capture_output=True,
        )
        builder.write_text(malicious_builder, encoding="utf-8")
        lazy.write_text("VALUE = 'malicious worktree bytes'\n", encoding="utf-8")
    else:
        real_run = subprocess.run
        status_calls = 0

        def race_between_status_checks(
            command: list[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[Any]:
            nonlocal status_calls
            if "status" not in command:
                return real_run(command, **kwargs)
            status_calls += 1
            if status_calls == 1:
                result = real_run(command, **kwargs)
                builder.write_text(malicious_builder, encoding="utf-8")
                lazy.write_text(
                    "VALUE = 'malicious worktree bytes'\n", encoding="utf-8"
                )
                return result
            builder.write_text(PINNED_BUILDER_SOURCE, encoding="utf-8")
            lazy.write_text("VALUE = 'pinned commit blob'\n", encoding="utf-8")
            return real_run(command, **kwargs)

        monkeypatch.setattr(SAM31_MODULE.subprocess, "run", race_between_status_checks)

    provider = Sam31EvidenceProvider.load(repository, checkpoint, digest)

    imported_builder = sys.modules["sam3.model_builder"]
    imported_lazy = importlib.import_module("sam3.lazy_component")
    imported_root = Path(imported_builder.__file__).resolve().parents[1]
    assert imported_builder.SOURCE_MARKER == "pinned commit blob"
    assert imported_lazy.VALUE == "pinned commit blob"
    assert "malicious worktree bytes" not in (
        imported_root / "sam3" / "model_builder.py"
    ).read_text(encoding="utf-8")
    provider.close()


def test_source_snapshot_ignores_git_replacement_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    builder = repository / "sam3" / "model_builder.py"
    lazy = repository / "sam3" / "lazy_component.py"
    builder.write_text(PINNED_BUILDER_SOURCE, encoding="utf-8")
    lazy.write_text("VALUE = 'pinned commit blob'\n", encoding="utf-8")
    pinned_revision = pin_runtime_repository(monkeypatch, repository)
    malicious_builder = PINNED_BUILDER_SOURCE.replace(
        "pinned commit blob", "malicious replacement commit"
    )
    builder.write_text(malicious_builder, encoding="utf-8")
    lazy.write_text("VALUE = 'malicious replacement commit'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "add", "sam3"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=SAM Test",
            "-c",
            "user.email=sam-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "malicious replacement",
        ],
        check=True,
        capture_output=True,
    )
    malicious_revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "checkout", "-q", "--detach", pinned_revision],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "replace", pinned_revision, malicious_revision],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "reset", "--hard", "HEAD"],
        check=True,
        capture_output=True,
    )
    apparent_revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    apparent_status = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "sam3",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    replaced_source = subprocess.run(
        ["git", "-C", str(repository), "show", "HEAD:sam3/model_builder.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert apparent_revision == pinned_revision
    assert apparent_status == ""
    assert "malicious replacement commit" in replaced_source
    install_fake_torch_module(monkeypatch, fake_torch)

    load_error: BaseException | None = None
    provider: Sam31EvidenceProvider | None = None
    try:
        provider = Sam31EvidenceProvider.load(repository, checkpoint, digest)
    except BaseException as error:
        load_error = error
    if provider is not None:
        provider.close()

    subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(repository),
            "reset",
            "--hard",
            "HEAD",
        ],
        check=True,
        capture_output=True,
    )
    inherited_redirect = tmp_path / "inherited-git-redirect"
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_REPLACE_REF_BASE",
    ):
        monkeypatch.setenv(name, str(inherited_redirect))
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    direct_snapshot = SAM31_MODULE._snapshot_repository_source(
        repository, tmp_path / "direct-source"
    )
    snapshotted_builder = (direct_snapshot / "sam3/model_builder.py").read_text(
        encoding="utf-8"
    )
    snapshotted_lazy = (direct_snapshot / "sam3/lazy_component.py").read_text(
        encoding="utf-8"
    )

    assert type(load_error) is CvProviderError
    assert str(load_error) == "Unable to load local SAM3.1 runtime"
    assert "pinned commit blob" in snapshotted_builder
    assert "malicious replacement commit" not in snapshotted_builder
    assert snapshotted_lazy == "VALUE = 'pinned commit blob'\n"


@pytest.mark.parametrize(
    "tree_output",
    [
        b"malformed tree record\0",
        b"120000 blob " + (b"a" * 40) + b"       1\tsam3/link.py\0",
        b"160000 commit " + (b"a" * 40) + b"       1\tsam3/vendor\0",
        (
            b"100644 blob "
            + (b"a" * 40)
            + b"       1\tsam3/duplicate.py\0"
        )
        * 2,
        b"100644 blob " + (b"a" * 40) + b"       1\tsam3/../escape.py\0",
        b"100644 blob " + (b"a" * 40) + b"       1\tsam3/invalid-\xff.py\0",
        b"100644 blob " + (b"a" * 40) + b"       1\tsam3//double.py\0",
        b"100644 blob " + (b"a" * 40) + b"       1\tsam3/./dot.py\0",
        b"100644 blob " + (b"a" * 40) + b"       1\tsam3/c1-\xc2\x85.py\0",
    ],
    ids=(
        "malformed",
        "symlink",
        "submodule",
        "duplicate",
        "noncanonical-path",
        "non-utf8-path",
        "double-separator",
        "dot-segment",
        "c1-control",
    ),
)
def test_pinned_tree_rejects_malformed_duplicate_and_special_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tree_output: bytes,
) -> None:
    monkeypatch.setattr(
        SAM31_MODULE,
        "_bounded_git_output",
        lambda *_args: tree_output,
    )

    with pytest.raises(ValueError):
        SAM31_MODULE._pinned_sam_blobs(tmp_path)


def test_pinned_tree_output_is_bounded_and_git_command_failure_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _, _ = local_runtime_assets(tmp_path)
    (repository / "sam3" / "model_builder.py").write_text(
        PINNED_BUILDER_SOURCE, encoding="utf-8"
    )
    pin_runtime_repository(monkeypatch, repository)
    monkeypatch.setattr(SAM31_MODULE, "_MAX_GIT_TREE_BYTES", 32)

    with pytest.raises(ValueError):
        SAM31_MODULE._pinned_sam_blobs(repository)

    non_repository = tmp_path / "not-a-repository"
    non_repository.mkdir()
    with pytest.raises(ValueError):
        SAM31_MODULE._bounded_git_output(
            non_repository,
            ("ls-tree", "HEAD"),
            1024,
        )


def test_cat_file_process_is_not_started_when_exclusive_destination_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "already-exists.py"
    destination.write_bytes(b"existing private file")
    blob = SAM31_MODULE._PinnedBlob(
        "sam3/already-exists.py",
        "a" * 40,
        1,
    )
    process_started = False

    def unexpected_process_start(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal process_started
        process_started = True
        raise AssertionError("cat-file must start after exclusive destination open")

    monkeypatch.setattr(SAM31_MODULE.subprocess, "Popen", unexpected_process_start)

    with pytest.raises(FileExistsError):
        SAM31_MODULE._copy_pinned_blob(tmp_path, blob, destination)

    assert process_started is False
    assert destination.read_bytes() == b"existing private file"


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_source_snapshot_rejects_incomplete_or_changed_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    payloads = {
        "sam3/__init__.py": b"",
        "sam3/model_builder.py": PINNED_BUILDER_SOURCE.encode("utf-8"),
    }

    def blob_id(payload: bytes) -> str:
        return hashlib.sha1(
            f"blob {len(payload)}\0".encode("ascii") + payload
        ).hexdigest()

    blobs = tuple(
        SAM31_MODULE._PinnedBlob(name, blob_id(payload), len(payload))
        for name, payload in payloads.items()
    )
    monkeypatch.setattr(SAM31_MODULE, "_pinned_sam_blobs", lambda _: blobs)

    def damaged_copy(
        _repository: Path, blob: Any, destination: Path
    ) -> None:
        if damage == "missing" and blob.path == "sam3/model_builder.py":
            return
        payload = payloads[blob.path]
        if damage == "changed" and blob.path == "sam3/model_builder.py":
            payload = b"X" + payload[1:]
        destination.write_bytes(payload)
        destination.chmod(0o400)

    monkeypatch.setattr(SAM31_MODULE, "_copy_pinned_blob", damaged_copy)
    destination = tmp_path / "private-source"

    with pytest.raises(ValueError):
        SAM31_MODULE._snapshot_repository_source(tmp_path, destination)

    assert not destination.exists()


@pytest.mark.parametrize("validation_step", ["chmod", "stat"])
@pytest.mark.parametrize(
    "failure_kind",
    ["oserror", "keyboard", "system-exit"],
)
def test_load_cleans_owned_runtime_directory_when_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_step: str,
    failure_kind: str,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    real_mkdtemp = SAM31_MODULE.tempfile.mkdtemp
    created_paths: list[Path] = []

    def create_real_runtime_directory(*, prefix: str) -> str:
        created = Path(real_mkdtemp(prefix=prefix, dir=tmp_path))
        created_paths.append(created)
        return str(created)

    monkeypatch.setattr(
        SAM31_MODULE.tempfile, "mkdtemp", create_real_runtime_directory
    )
    if failure_kind == "oserror":
        primary: BaseException = OSError("private validation path")
    elif failure_kind == "keyboard":
        primary = KeyboardInterrupt("private validation interrupt")
    else:
        primary = SystemExit("private validation exit")
    original_validation = getattr(Path, validation_step)
    validation_failures = 0

    def fail_runtime_path_operation(
        path: Path, *args: Any, **kwargs: Any
    ) -> Any:
        nonlocal validation_failures
        if (
            path.parent == tmp_path
            and path.name.startswith(".las-sam31-runtime-")
        ):
            validation_failures += 1
            if validation_failures == 1:
                raise primary
            raise KeyboardInterrupt("secondary path cleanup failure")
        return original_validation(path, *args, **kwargs)

    monkeypatch.setattr(Path, validation_step, fail_runtime_path_operation)
    real_remove_tree = SAM31_MODULE._remove_tree
    cleanup_attempts: list[Path] = []

    def remove_then_interrupt(path: Path) -> None:
        cleanup_attempts.append(path)
        real_remove_tree(path)
        raise KeyboardInterrupt("secondary cleanup failure")

    monkeypatch.setattr(SAM31_MODULE, "_remove_tree", remove_then_interrupt)

    raised: BaseException | None = None
    try:
        Sam31EvidenceProvider.load(
            repository,
            checkpoint,
            digest,
            predictor_factory=lambda **_: PredictorDouble(),
        )
    except BaseException as error:
        raised = error

    if failure_kind == "oserror":
        assert type(raised) is CvProviderError
        assert str(raised) == "Unable to load local SAM3.1 runtime"
    else:
        assert raised is primary
    assert validation_failures >= 1
    assert cleanup_attempts == created_paths
    assert list(tmp_path.glob(".las-sam31-runtime-*")) == []


def test_load_cleans_owned_runtime_directory_when_path_conversion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    real_mkdtemp = SAM31_MODULE.tempfile.mkdtemp
    created_paths: list[Path] = []

    def create_bytes_runtime_directory(*, prefix: str) -> bytes:
        del prefix
        created = real_mkdtemp(
            prefix=b".las-sam31-runtime-", dir=os.fsencode(tmp_path)
        )
        created_paths.append(Path(os.fsdecode(created)))
        return created

    monkeypatch.setattr(
        SAM31_MODULE.tempfile, "mkdtemp", create_bytes_runtime_directory
    )

    with pytest.raises(
        CvProviderError, match="^Unable to load local SAM3.1 runtime$"
    ):
        Sam31EvidenceProvider.load(
            repository,
            checkpoint,
            digest,
            predictor_factory=lambda **_: PredictorDouble(),
        )

    assert len(created_paths) == 1
    assert list(tmp_path.glob(".las-sam31-runtime-*")) == []


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

    def factory(
        *,
        checkpoint_path: str,
        bpe_path: str,
        max_num_objects: int,
        multiplex_count: int,
        compile: bool,
        use_fa3: bool,
    ) -> PredictorDouble:
        factory_calls.append(
            {
                "checkpoint_path": checkpoint_path,
                "bpe_path": bpe_path,
                "max_num_objects": max_num_objects,
                "multiplex_count": multiplex_count,
                "compile": compile,
                "use_fa3": use_fa3,
            }
        )
        return predictor

    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_REPLACE_REF_BASE",
    ):
        monkeypatch.setenv(name, str(tmp_path / "inherited-git-redirect"))
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")

    provider = Sam31EvidenceProvider.load(
        repository,
        checkpoint,
        digest,
        compile_model=True,
        predictor_factory=factory,
        max_artifact_bytes=12_345,
        max_artifact_files=7,
    )

    assert [call[0] for call in git_calls] == [
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(repository.resolve()),
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ],
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(repository.resolve()),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "sam3",
        ],
    ]
    for _, kwargs in git_calls:
        environment = kwargs["env"]
        assert kwargs | {"env": None} == {
            "check": True,
            "capture_output": True,
            "text": True,
            "shell": False,
            "env": None,
        }
        assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        assert environment["LC_ALL"] == "C"
        assert {
            name for name in environment if name.startswith("GIT_")
        } == {"GIT_NO_REPLACE_OBJECTS"}
    checkpoint_argument = Path(factory_calls[0]["checkpoint_path"])
    bpe_argument = Path(factory_calls[0]["bpe_path"])
    assert checkpoint_argument != checkpoint.resolve()
    assert bpe_argument != (
        repository / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    ).resolve()
    assert checkpoint_argument.read_bytes() == checkpoint.read_bytes()
    assert bpe_argument.read_bytes() == b"local bpe fixture"
    assert stat.S_IMODE(checkpoint_argument.stat().st_mode) == 0o600
    assert checkpoint_argument.stat().st_nlink == 1
    assert factory_calls == [
        {
            "checkpoint_path": str(checkpoint_argument),
            "bpe_path": str(bpe_argument),
            "max_num_objects": 16,
            "multiplex_count": 16,
            "compile": True,
            "use_fa3": False,
        }
    ]
    assert provider.checkpoint_sha256 == digest
    assert provider.repository_revision == PINNED_REVISION
    assert provider._max_artifact_bytes == 12_345
    assert provider._max_artifact_files == 7
    provider.close()
    assert not checkpoint_argument.exists()
    assert not bpe_argument.exists()


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


@pytest.mark.parametrize(
    "dirty_status",
    [
        " M sam3/model_builder.py\n",
        "?? sam3/untracked_payload.py\n",
    ],
    ids=("tracked-dirty", "untracked"),
)
def test_load_rejects_dirty_or_untracked_sam_package_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
    dirty_status: str,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    install_fake_torch_module(monkeypatch, fake_torch)
    factory_called = False

    def git_run(command: list[str], **_: Any) -> SimpleNamespace:
        if "status" in command:
            return SimpleNamespace(stdout=dirty_status)
        return SimpleNamespace(stdout=PINNED_REVISION + "\n")

    def factory(**_: Any) -> PredictorDouble:
        nonlocal factory_called
        factory_called = True
        return PredictorDouble()

    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", git_run)

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository, checkpoint, digest, predictor_factory=factory
        )

    assert factory_called is False


@pytest.mark.parametrize("unsafe_kind", ["hardlink", "permissions"])
def test_load_rejects_checkpoint_hardlinks_and_writable_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
    unsafe_kind: str,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    if unsafe_kind == "hardlink":
        os.link(checkpoint, tmp_path / "checkpoint-hardlink.pt")
    else:
        checkpoint.chmod(0o666)

    with pytest.raises(CvProviderError, match="^Unable to load local SAM3.1 runtime$"):
        Sam31EvidenceProvider.load(
            repository,
            checkpoint,
            digest,
            predictor_factory=lambda **_: PredictorDouble(),
        )


def test_builder_receives_private_asset_snapshots_immune_to_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    original_checkpoint = checkpoint.read_bytes()
    original_bpe = repository / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    monkeypatch.setattr(SAM31_MODULE.subprocess, "run", successful_git_run([]))
    install_fake_torch_module(monkeypatch, fake_torch)
    builder_paths: list[tuple[Path, Path]] = []

    def factory(
        *,
        checkpoint_path: str,
        bpe_path: str,
        max_num_objects: int,
        multiplex_count: int,
        compile: bool,
        use_fa3: bool,
    ) -> PredictorDouble:
        del max_num_objects, multiplex_count, compile, use_fa3
        checkpoint.rename(tmp_path / "replaced-original.pt")
        checkpoint.write_bytes(b"attacker replacement")
        original_bpe.rename(tmp_path / "replaced-bpe.txt")
        original_bpe.write_bytes(b"attacker bpe")
        private_checkpoint = Path(checkpoint_path)
        private_bpe = Path(bpe_path)
        assert private_checkpoint.read_bytes() == original_checkpoint
        assert private_bpe.read_bytes() == b"local bpe fixture"
        builder_paths.append((private_checkpoint, private_bpe))
        return PredictorDouble()

    provider = Sam31EvidenceProvider.load(
        repository, checkpoint, digest, predictor_factory=factory
    )

    assert builder_paths
    assert all(path.exists() for pair in builder_paths for path in pair)
    provider.close()
    assert all(not path.exists() for pair in builder_paths for path in pair)


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
    lazy_module = repository / "sam3" / "lazy_component.py"
    lazy_module.write_text("VALUE = 'pinned bytes'\n", encoding="utf-8")
    pin_runtime_repository(monkeypatch, repository)
    install_fake_torch_module(monkeypatch, fake_torch)
    path_before = list(sys.path)

    provider = Sam31EvidenceProvider.load(repository, checkpoint, digest)

    imported_package = sys.modules["sam3"]
    imported_builder = sys.modules["sam3.model_builder"]
    imported_root = Path(imported_package.__file__).resolve().parents[1]
    assert not imported_root.is_relative_to(repository.resolve())
    assert Path(imported_builder.__file__).resolve().is_relative_to(imported_root)
    assert sys.path == path_before
    lazy_module.write_text("VALUE = 'replacement bytes'\n", encoding="utf-8")
    imported_lazy = importlib.import_module("sam3.lazy_component")
    assert imported_lazy.VALUE == "pinned bytes"
    assert Path(imported_lazy.__file__).resolve().is_relative_to(imported_root)
    checkpoint_argument = Path(imported_builder.calls[0]["checkpoint_path"])
    bpe_argument = Path(imported_builder.calls[0]["bpe_path"])
    assert imported_builder.calls == [
        {
            "checkpoint_path": str(checkpoint_argument),
            "bpe_path": str(bpe_argument),
            "max_num_objects": 16,
            "multiplex_count": 16,
            "compile": False,
            "use_fa3": False,
        }
    ]
    assert checkpoint_argument.read_bytes() == checkpoint.read_bytes()
    assert bpe_argument.read_bytes() == b"local bpe fixture"
    provider.close()
    assert "sam3" not in sys.modules
    assert "sam3.model_builder" not in sys.modules
    assert "sam3.lazy_component" not in sys.modules
    assert not imported_root.exists()


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
    pin_runtime_repository(monkeypatch, repository)
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
    pin_runtime_repository(monkeypatch, repository)
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
    pin_runtime_repository(monkeypatch, repository)
    install_fake_torch_module(monkeypatch, fake_torch)
    path_before = list(sys.path)

    with pytest.raises(KeyboardInterrupt, match="interrupted local load"):
        Sam31EvidenceProvider.load(repository, checkpoint, digest)

    assert not any(name == "sam3" or name.startswith("sam3.") for name in sys.modules)
    assert sys.path == path_before


def test_load_shutdown_interrupt_cannot_mask_primary_or_skip_import_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    (repository / "sam3" / "model_builder.py").write_text(
        "class Predictor:\n"
        "    @property\n"
        "    def handle_request(self):\n"
        "        raise SystemExit('primary interface exit')\n"
        "    def handle_stream_request(self, request):\n"
        "        return iter(())\n"
        "    def shutdown(self):\n"
        "        raise KeyboardInterrupt('shutdown interrupt')\n"
        "def build_sam3_multiplex_video_predictor(**kwargs):\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )
    pin_runtime_repository(monkeypatch, repository)
    install_fake_torch_module(monkeypatch, fake_torch)

    with pytest.raises(SystemExit, match="primary interface exit"):
        Sam31EvidenceProvider.load(repository, checkpoint, digest)

    assert not any(name == "sam3" or name.startswith("sam3.") for name in sys.modules)


def test_provider_close_restores_imports_and_assets_after_shutdown_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: SimpleNamespace,
) -> None:
    repository, checkpoint, digest = local_runtime_assets(tmp_path)
    (repository / "sam3" / "model_builder.py").write_text(
        "class Predictor:\n"
        "    def handle_request(self, request):\n"
        "        return {}\n"
        "    def handle_stream_request(self, request):\n"
        "        return iter(())\n"
        "    def shutdown(self):\n"
        "        raise RuntimeError('private shutdown failure')\n"
        "def build_sam3_multiplex_video_predictor(**kwargs):\n"
        "    return Predictor()\n",
        encoding="utf-8",
    )
    pin_runtime_repository(monkeypatch, repository)
    install_fake_torch_module(monkeypatch, fake_torch)
    provider = Sam31EvidenceProvider.load(repository, checkpoint, digest)
    imported_root = Path(sys.modules["sam3"].__file__).resolve().parents[1]

    with pytest.raises(CvProviderError, match="^Unable to close SAM3.1 runtime$"):
        provider.close()

    assert "sam3" not in sys.modules
    assert "sam3.model_builder" not in sys.modules
    assert not imported_root.exists()


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
