import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

numpy = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from percept_harness.cv.artifacts import CvArtifactHandle, CvArtifactStore, cv_cache_key
from percept_harness.cv.base import CvOutOfMemoryError, CvProviderError
from percept_harness.cv.contracts import CvEvidenceArtifact, CvEvidenceRequest
from percept_harness.cv.executor import SyncCvExecutor
from percept_harness.cv.sam31 import Sam31EvidenceProvider, _git_blob_digest, _local_path
from percept_harness.cv.timeline import SampledFrame, SampledFrameSet

PNG = b"\x89PNG\r\n\x1a\nfake"
HEIGHT, WIDTH = 6, 8


def request_payload(**overrides):
    return {
        "schema_version": "cv_request_v1", "provider": "sam31", "model_identity": "sam3.1",
        "video_path": "/videos/clip.mp4", "video_identity": "/videos/clip.mp4:1000:42",
        "duration_seconds": 2.0, "frame_count": 5, "checkpoint_identity": "/ckpt/sam3.1.pt:9:7",
        "timeline": {"frames": [{"frame_index": i, "timestamp_seconds": i * 0.25} for i in range(5)]},
        "entities": [{"entity_id": "target", "canonical_label": "red cup"}],
        "sampling": {"max_fps": 2.0}, "mode": "track",
        "spatial_prompts": [{"entity_id": "target", "frame_index": 2, "bbox_xyxy": [0.125, 0.25, 0.5, 0.75]}],
        **overrides,
    }


class FakeOOM(Exception):
    pass


def fake_torch():
    calls = []
    cuda = SimpleNamespace(
        OutOfMemoryError=FakeOOM, max_memory_allocated=lambda: 4096,
        reset_peak_memory_stats=lambda: calls.append("reset"), empty_cache=lambda: calls.append("empty"))
    return SimpleNamespace(cuda=cuda, calls=calls)


def materialize(video_path, timeline, indices, destination):
    destination.mkdir(parents=True)
    clock = {p.frame_index: p.timestamp_seconds for p in timeline.frames}
    frames = []
    for number, index in enumerate(indices):
        path = destination / f"{number:06d}.jpg"
        Image.new("RGB", (WIDTH, HEIGHT), (number, 0, 0)).save(path, format="PNG")
        frames.append(SampledFrame(number, index, clock[index], path))
    return SampledFrameSet(tuple(frames))


def mask(top, left, bottom, right):
    value = numpy.zeros((HEIGHT, WIDTH), dtype=bool)
    value[top:bottom, left:right] = True
    return value


# Local frame -> (object id, probability, xywh box, mask); local 1 is the reference.
OBJECTS = {
    0: [(0, 0.75, (0.0, 0.0, 0.5, 0.5), mask(0, 0, 3, 4))],
    1: [(3, -10000.0, (0.0, 0.0, 0.5, 0.5), mask(0, 0, 1, 1)),
        (0, 0.875, (0.125, 0.25, 0.375, 0.5), mask(1, 1, 4, 4))],
    2: [(2, 0.5, (0.5, 0.5, 0.25, 0.25), mask(0, 0, 0, 0)),
        (1, 0.625, (0.5, 0.5, 0.5, 0.5), mask(3, 4, 6, 8)),
        (0, 0.5, (0.25, 0.25, 0.25, 0.25), mask(2, 2, 4, 5))],
}


def response(local_index, objects=OBJECTS):
    items = objects[local_index]
    return {"frame_index": local_index, "outputs": {
        "out_obj_ids": numpy.asarray([item[0] for item in items], dtype=numpy.int64),
        "out_probs": numpy.asarray([item[1] for item in items], dtype=numpy.float32),
        "out_boxes_xywh": numpy.asarray([item[2] for item in items], dtype=numpy.float32).reshape(-1, 4),
        "out_binary_masks": numpy.asarray([item[3] for item in items], dtype=bool).reshape(-1, HEIGHT, WIDTH)}}


class FakePredictor:
    def __init__(self, oom_once=False):
        self.requests, self.oom_once, self.shutdowns = [], oom_once, 0
        self.model = SimpleNamespace(postprocess_batch_size=0, init_state=lambda video, other=None: (video, other))

    def handle_request(self, request):
        self.requests.append({k: v for k, v in request.items() if k != "resource_path"})
        if request["type"] == "start_session":
            return {"session_id": "session-1"}
        if request["type"] == "add_prompt":
            if self.oom_once:
                self.oom_once = False
                raise FakeOOM
            return response(request["frame_index"])
        return {}

    def handle_stream_request(self, request):
        self.requests.append(dict(request))
        start = request["start_frame_index"]
        return (response(index) for index in (*range(start, 3), *range(start - 1, -1, -1)))

    def shutdown(self):
        self.shutdowns += 1


def renderer(calls):
    def render(*, frame_path, archive_path, mask_member, bbox_xyxy, destination):
        with zipfile.ZipFile(archive_path) as archive:
            pixels = numpy.load(io.BytesIO(archive.read(mask_member)))
        calls.append({"frame": frame_path.name, "mask": mask_member, "pixels": int(pixels.sum()),
                      "bbox": list(bbox_xyxy), "destination": destination.name})
        destination.write_bytes(PNG)
    return render


def provider(predictor, torch, calls):
    return Sam31EvidenceProvider(predictor=predictor, torch_module=torch, materialize_frames=materialize,
                                 overlay_renderer=renderer(calls))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def archive_summary(path):
    with zipfile.ZipFile(path) as archive:
        return {name: numpy.load(io.BytesIO(archive.read(name))).astype(int).tolist()
                for name in archive.namelist()}


@pytest.mark.parametrize("mode, golden", [
    ("track", "ec2fca523f6e1a44e66cc0047cf8c1fe8602fdb2867431845247c2be45586d0b"),
    ("probe", "d6240e60459d840754f3a3e8e01a6f8bf394092aa52a6862555e60e502bf090b"),
])
def test_analyze_golden(tmp_path, mode, golden):
    payload = request_payload(mode=mode) if mode == "track" else request_payload(
        mode=mode, timeline={"frames": [{"frame_index": 2, "timestamp_seconds": 0.5}]})
    request = CvEvidenceRequest.model_validate(payload)
    predictor, torch, calls = FakePredictor(), fake_torch(), []
    sam = provider(predictor, torch, calls)
    assert predictor.model.postprocess_batch_size == 8
    assert predictor.model.init_state("v", offload_state_to_cpu=False) == ("v", None)
    staging = tmp_path / "staging"
    artifact = sam.analyze(request, staging)
    dump = artifact.model_dump(mode="json")
    sizes = {item["path"]: item.pop("size_bytes") for item in dump["files"]}
    assert sizes["masks/target.npz"] == (staging / "masks/target.npz").stat().st_size
    assert all(sizes[path] == len(PNG) for path in sizes if path.startswith("overlays/"))
    assert sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()) == sorted(sizes)
    metrics = sam.request_metrics()
    assert {key: metrics[key] for key in ("processed_frames", "execution_chunk_frames", "peak_allocated_bytes")} == {
        "processed_frames": len(artifact.processed_timeline.frames), "execution_chunk_frames": 8,
        "peak_allocated_bytes": 4096}
    summary = {"artifact": dump, "requests": predictor.requests,
               "renders": calls, "archive": archive_summary(staging / "masks/target.npz"), "torch": torch.calls}
    assert canonical_sha(summary) == golden, json.dumps(summary, sort_keys=True)
    assert CvEvidenceArtifact.model_validate(artifact.model_dump()) == artifact
    sam.close()
    sam.close()
    assert predictor.shutdowns == 1
    with pytest.raises(CvProviderError):
        sam.analyze(request, tmp_path / "again")


def test_analyze_failures_are_sanitized(tmp_path):
    request = CvEvidenceRequest.model_validate(request_payload())
    torch = fake_torch()
    sam = provider(FakePredictor(oom_once=True), torch, [])
    with pytest.raises(CvOutOfMemoryError) as raised:
        sam.analyze(request, tmp_path / "oom")
    assert raised.value.__cause__ is None and torch.calls == ["reset", "empty"]
    assert list((tmp_path / "oom").iterdir()) == []
    broken = FakePredictor()
    broken.handle_stream_request = lambda request: iter([response(1), response(0)])
    with pytest.raises(CvProviderError, match="inference failed"):
        provider(broken, torch, []).analyze(request, tmp_path / "order")
    assert list((tmp_path / "order").iterdir()) == []
    with pytest.raises(TypeError):
        Sam31EvidenceProvider(predictor=object(), torch_module=torch)


class ScriptedProvider:
    def __init__(self, failures):
        self.execution_chunk_frames, self.failures, self.calls = 8, failures, []

    def analyze(self, request, staging):
        self.calls.append(self.execution_chunk_frames)
        if len(self.calls) <= self.failures:
            raise CvOutOfMemoryError("oom")
        return CvEvidenceArtifact(
            schema_version="cv_evidence_v1", status="unavailable", provider="sam31",
            model_identity=request.model_identity, video_identity=request.video_identity,
            checkpoint_identity=request.checkpoint_identity, processed_timeline=None,
            entities=request.entities, tracks=(), files=(), overlay_records=())

    def request_metrics(self):
        return {"processed_frames": 3, "execution_chunk_frames": self.execution_chunk_frames,
                "peak_allocated_bytes": 7}


def test_executor_halves_chunk_once_on_oom(tmp_path):
    scripted = ScriptedProvider(failures=1)
    with CvArtifactStore(tmp_path / "cache") as store:
        executor = SyncCvExecutor(scripted, store)
        result, metrics = executor(request_payload())
        assert scripted.calls == [8, 4]
        key = cv_cache_key(CvEvidenceRequest.model_validate(request_payload()))
        assert result == {"artifact_key": key, "cache_hit": False, "status": "unavailable"}
        assert isinstance(metrics.pop("inference_seconds"), float)
        assert metrics == {"cache_hit": False, "oom_retry": True, "processed_frames": 3,
                           "execution_chunk_frames": 4, "entity_prompts": 1, "track_count": 0,
                           "peak_allocated_bytes": 7}
        assert executor(request_payload()) == ({**result, "cache_hit": True}, {
            "cache_hit": True, "oom_retry": False, "processed_frames": 0, "execution_chunk_frames": 0,
            "entity_prompts": 1, "track_count": 0, "peak_allocated_bytes": 0, "inference_seconds": 0.0})
        assert scripted.calls == [8, 4]
        twice = ScriptedProvider(failures=2)
        with pytest.raises(CvOutOfMemoryError):
            SyncCvExecutor(twice, store)(request_payload(model_identity="other"))
        assert twice.calls == [8, 4] and twice.execution_chunk_frames == 4


def test_executor_with_sam_provider_retries_oom(tmp_path):
    predictor, calls = FakePredictor(oom_once=True), []
    sam = provider(predictor, fake_torch(), calls)
    with CvArtifactStore(tmp_path / "cache") as store:
        result, metrics = SyncCvExecutor(sam, store)(request_payload())
        assert predictor.model.postprocess_batch_size == 4
        assert metrics["oom_retry"] is True and metrics["execution_chunk_frames"] == 4
        assert metrics["track_count"] == 2 and metrics["peak_allocated_bytes"] == 4096
        handle = CvArtifactHandle(result["artifact_key"])
        artifact = store.load(handle)
        overlays = store.read_overlays(handle, tuple(r.path for r in artifact.overlay_records))
        assert len(overlays) == len(calls) == 2


def test_runtime_path_helpers(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"hello\n")
    assert _git_blob_digest(blob) == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert _local_path("relative/file") == Path("relative/file")
    for value in ("", "//host/share", "https://example.com/x", "git@host:repo", "a\x00b"):
        with pytest.raises(ValueError):
            _local_path(value)
