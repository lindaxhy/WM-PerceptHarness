"""CPU-only coverage of SAM array, archive, loader and cleanup boundaries."""

from pathlib import Path
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

numpy = pytest.importorskip("numpy")

from percept_harness.cv import sam31
from percept_harness.cv.base import CvOutOfMemoryError, CvProviderError
from percept_harness.cv.timeline import SampledFrame, SampledFrameSet
from test_targeted_cv import Predictor, materialize, request, spatial


def arrays(index=0, ids=(9, 2), probabilities=(0.8, 0.9)):
    # Noncontiguous masks exercise serialization order independently of strides.
    masks = numpy.zeros((len(ids), 4, 8), dtype=numpy.bool_)
    masks[:, 0, 0] = True
    masks[:, 3, 6] = True
    return {
        "frame_index": index,
        "outputs": {
            "out_obj_ids": numpy.asarray(ids, dtype=numpy.int64),
            "out_probs": numpy.asarray(probabilities, dtype=numpy.float64),
            "out_boxes_xywh": numpy.tile([0.25, 0.25, 0.5, 0.5], (len(ids), 1)),
            "out_binary_masks": masks[:, :, ::2],
        },
    }


def sampled(tmp_path):
    return SampledFrameSet((SampledFrame(0, 30, 3.0, tmp_path / "0.jpg"),))


def parse(response, tmp_path, writer=None):
    return sam31._parse_frame_response(
        response, sampled(tmp_path), allowed_indices=range(1),
        expected_dimensions=(4, 4), mask_writer=writer,
    )


def test_strided_masks_roundtrip_and_sorted_metadata_keep_source_mapping(tmp_path):
    budget = sam31._ArtifactBudget(100_000, 10)
    path = tmp_path / "masks.npz"
    writer = sam31._MaskArchiveWriter(path, budget)
    _, detections, dimensions = parse(arrays(), tmp_path, writer)
    writer.finish(detections, sampled(tmp_path))
    assert dimensions == (4, 4)
    assert [item.object_id for item in detections] == [2, 9]
    assert [item.mask_index for item in detections] == [0, 1]
    assert all(item.area_fraction == 0.125 for item in detections)
    assert all(item.center_xy == (0.5, 0.5) for item in detections)
    with numpy.load(path, allow_pickle=False) as archive:
        assert archive["object_ids"].tolist() == [2, 9]
        assert archive["frame_indices"].tolist() == [30, 30]
        assert archive["local_frame_indices"].tolist() == [0, 0]
        assert archive["sampled_frame_indices"].tolist() == [30]
        assert archive["mask_indices"].tolist() == [0, 1]
        mask = archive["masks/00000000"]
        assert mask.dtype == numpy.bool_
        assert numpy.argwhere(mask).tolist() == [[0, 0], [3, 3]]
    assert budget.actual_bytes == path.stat().st_size


def test_removed_and_empty_masks_do_not_create_tracks_or_mask_members(tmp_path):
    response = arrays(ids=(1, 2, 3), probabilities=(-10000.0, 0.8, 0.9))
    response["outputs"]["out_binary_masks"][1] = False
    writer = sam31._MaskArchiveWriter(tmp_path / "masks.npz", sam31._ArtifactBudget(100_000, 10))
    _, detections, _ = parse(response, tmp_path, writer)
    writer.finish(detections, sampled(tmp_path))
    assert [item.object_id for item in detections] == [3]
    with zipfile.ZipFile(tmp_path / "masks.npz") as archive:
        assert [name for name in archive.namelist() if name.startswith("masks/")] == ["masks/00000000.npy"]


def test_reverse_propagation_mask_indices_follow_sorted_source_frames(tmp_path):
    class FrameMaskPredictor(Predictor):
        def response(self, index):
            response = super().response(index)
            mask = response["outputs"]["out_binary_masks"]
            mask[:] = False
            mask[0, index, index] = True
            return response

    tmp_path.chmod(0o700)
    artifact = provider(FrameMaskPredictor()).analyze(
        request(spatial_prompts=[spatial()]), tmp_path,
    )
    assert [item.frame_index for item in artifact.tracks[0].observations] == [20, 30, 40]
    with numpy.load(tmp_path / "masks/cup.npz", allow_pickle=False) as archive:
        assert archive["frame_indices"].tolist() == [20, 30, 40]
        assert archive["mask_indices"].tolist() == [2, 0, 1]
        for source, member in [(20, "masks/00000002"), (30, "masks/00000000"), (40, "masks/00000001")]:
            assert numpy.argwhere(archive[member]).tolist() == [[(source - 20) // 10] * 2]


@pytest.mark.parametrize("field,value", [
    ("out_obj_ids", numpy.array([2, 2], dtype=numpy.int64)),
    ("out_obj_ids", numpy.array([-1, 2], dtype=numpy.int64)),
    ("out_probs", numpy.array([numpy.nan, 0.9])),
    ("out_probs", numpy.array([1.1, 0.9])),
    ("out_probs", numpy.array([1, 1], dtype=numpy.int64)),
    ("out_boxes_xywh", numpy.array([[0.5, 0.5, 0.8, 0.8], [0.0, 0.0, 1.0, 1.0]])),
    ("out_binary_masks", numpy.ones((2, 4, 4), dtype=numpy.uint8)),
    ("out_binary_masks", numpy.ones((2, 3, 4), dtype=numpy.bool_)),
])
def test_malformed_predictor_arrays_are_rejected_before_archiving(tmp_path, field, value):
    response = arrays()
    response["outputs"][field] = value
    with pytest.raises((TypeError, ValueError)):
        parse(response, tmp_path)


def provider(predictor, **options):
    return sam31.Sam31EvidenceProvider(
        predictor=predictor, torch_module=SimpleNamespace(),
        materialize_frames=materialize, initial_sampler=lambda *args: (20, 40),
        overlay_renderer=lambda **kwargs: kwargs["destination"].write_bytes(b"overlay"),
        **options,
    )


def test_resource_failure_closes_session_stream_and_removes_partial_outputs(tmp_path):
    class StreamingPredictor(Predictor):
        stream_closed = False

        def handle_stream_request(self, payload):
            try:
                yield self.response(1)
                yield self.response(2)
                yield self.response(0)
            finally:
                self.stream_closed = True

    predictor = StreamingPredictor()
    tmp_path.chmod(0o700)
    with pytest.raises(CvProviderError):
        provider(predictor, max_artifact_bytes=1100).analyze(request(), tmp_path)
    assert predictor.stream_closed
    assert predictor.calls[-1]["type"] == "close_session"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("indices", [(1, 0, 2), (1, 2), (1, 2, 2), (1, 2, 0, 0)])
def test_incomplete_duplicate_or_reordered_stream_cannot_publish(tmp_path, indices):
    class InvalidPredictor(Predictor):
        def handle_stream_request(self, payload):
            return iter(self.response(index) for index in indices)

    predictor = InvalidPredictor()
    tmp_path.chmod(0o700)
    with pytest.raises(CvProviderError):
        provider(predictor).analyze(request(), tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert predictor.calls[-1]["type"] == "close_session"


def test_targeted_oom_retry_recomputes_same_source_mapping(tmp_path):
    class OutOfMemoryError(Exception):
        pass

    class OomPredictor(Predictor):
        streams = 0

        def handle_stream_request(self, payload):
            self.streams += 1
            if self.streams == 1:
                def failed_stream():
                    yield self.response(1)
                    raise OutOfMemoryError()
                return failed_stream()
            return super().handle_stream_request(payload)

    predictor = OomPredictor()
    materialized = []

    def frames(video, timeline, indices, destination):
        materialized.append(indices)
        return materialize(video, timeline, indices, destination)

    instance = sam31.Sam31EvidenceProvider(
        predictor=predictor,
        torch_module=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OutOfMemoryError)),
        materialize_frames=frames,
        initial_sampler=lambda *args: (20, 40),
        overlay_renderer=lambda **kwargs: kwargs["destination"].write_bytes(b"overlay"),
    )
    tmp_path.chmod(0o700)
    targeted = request(spatial_prompts=[spatial()])
    with pytest.raises(CvOutOfMemoryError):
        instance.analyze(targeted, tmp_path)
    assert list(tmp_path.iterdir()) == []
    artifact = instance.analyze(targeted, tmp_path)
    assert materialized == [(20, 30, 40), (20, 30, 40)]
    assert [point.frame_index for point in artifact.processed_timeline.frames] == [20, 30, 40]
    assert [item.frame_index for item in artifact.tracks[0].observations] == [20, 30, 40]
    assert sum(call["type"] == "close_session" for call in predictor.calls) == 2


@pytest.mark.parametrize("change", ["missing_entity", "missing_spatial", "multiple_entities", "multiple_spatial"])
def test_provider_rejects_unsupported_untargeted_or_multi_entity_request(tmp_path, change):
    targeted = request(spatial_prompts=[spatial()])
    updates = {
        "missing_entity": {"entities": ()},
        "missing_spatial": {"spatial_prompts": ()},
        "multiple_entities": {"entities": targeted.entities * 2},
        "multiple_spatial": {"spatial_prompts": targeted.spatial_prompts * 2},
    }
    tmp_path.chmod(0o700)
    with pytest.raises(CvProviderError):
        provider(Predictor()).analyze(targeted.model_copy(update=updates[change]), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_one_target_can_return_multiple_candidate_masks_and_reference_overlays(tmp_path):
    class CandidatesPredictor(Predictor):
        def response(self, index):
            return arrays(index)

    tmp_path.chmod(0o700)
    artifact = provider(CandidatesPredictor()).analyze(request(mode="probe"), tmp_path)
    assert [track.track_id for track in artifact.tracks] == ["cup_2", "cup_9"]
    assert [record.track_id for record in artifact.overlay_records] == ["cup_2", "cup_9"]
    assert all(track.observations[0].frame_index == 30 for track in artifact.tracks)
    with numpy.load(tmp_path / "masks/cup.npz", allow_pickle=False) as archive:
        assert archive["object_ids"].tolist() == [2, 9]
        assert archive["frame_indices"].tolist() == [30, 30]


def test_targeted_zero_detection_retains_available_evidence_and_empty_archive(tmp_path):
    class EmptyPredictor(Predictor):
        def response(self, index):
            return arrays(index, ids=(), probabilities=())

    tmp_path.chmod(0o700)
    artifact = provider(EmptyPredictor()).analyze(request(mode="probe"), tmp_path)
    assert artifact.status.value == "available"
    assert artifact.tracks == ()
    assert artifact.overlay_records == ()
    with numpy.load(tmp_path / "masks/cup.npz", allow_pickle=False) as archive:
        assert archive["object_ids"].tolist() == []
        assert archive["sampled_frame_indices"].tolist() == [30]


@pytest.mark.parametrize("fail_body", [False, True])
def test_stream_cleanup_failure_removes_partial_outputs_without_losing_body_error(tmp_path, fail_body):
    class ClosingStream:
        def __iter__(self):
            if fail_body:
                raise RuntimeError("stream failed")
            return iter(arrays(index, ids=(1,), probabilities=(0.9,)) for index in (1, 2, 0))

        def close(self):
            raise KeyboardInterrupt("close failed")

    class ClosingPredictor(Predictor):
        def handle_stream_request(self, payload):
            return ClosingStream()

    predictor = ClosingPredictor()
    tmp_path.chmod(0o700)
    with pytest.raises(CvProviderError):
        provider(predictor).analyze(request(), tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert predictor.calls[-1]["type"] == "close_session"


def git(repository, *arguments):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def source_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    (repository / "sam3" / "assets").mkdir(parents=True)
    (repository / "sam3" / "__init__.py").write_text("")
    (repository / "sam3" / "model_builder.py").write_text("value = 1\n")
    (repository / sam31._BPE_REPOSITORY_PATH).write_bytes(b"vocabulary")
    git(repository, "init")
    git(repository, "add", "sam3")
    git(repository, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
    monkeypatch.setattr(sam31, "_PINNED_REPOSITORY_REVISION", git(repository, "rev-parse", "HEAD"))
    return repository


def test_loader_uses_local_assets_without_copying_and_restores_imports(tmp_path, monkeypatch):
    repository = source_repository(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    predictor = Predictor()
    builder_arguments = {}

    def build(**kwargs):
        builder_arguments.update(kwargs)
        return predictor

    original_import = sam31.importlib.import_module
    monkeypatch.setattr(sam31.importlib, "import_module", lambda name: (
        SimpleNamespace() if name == "torch" else original_import(name)
    ))
    instance = sam31.Sam31EvidenceProvider.load(repository, checkpoint, predictor_factory=build)
    try:
        assert builder_arguments["checkpoint_path"] == str(checkpoint)
        assert builder_arguments["bpe_path"] == str(repository / sam31._BPE_REPOSITORY_PATH)
        assert checkpoint.read_bytes() == b"checkpoint"
    finally:
        instance.close()


@pytest.mark.parametrize("change", ["dirty", "revision", "tokenizer", "empty_checkpoint"])
def test_loader_rejects_unpinned_or_invalid_local_assets(tmp_path, monkeypatch, change):
    repository = source_repository(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    options = {}
    if change == "dirty":
        (repository / "sam3" / "model_builder.py").write_text("value = 999\n")
    elif change == "revision":
        monkeypatch.setattr(sam31, "_PINNED_REPOSITORY_REVISION", "0" * 40)
    elif change == "tokenizer":
        tokenizer = tmp_path / "wrong-bpe.gz"
        tokenizer.write_bytes(b"wrong vocabulary")
        options["bpe_path"] = tokenizer
    else:
        checkpoint.write_bytes(b"")
    with pytest.raises(CvProviderError, match="Unable to load local SAM3.1 runtime"):
        sam31.Sam31EvidenceProvider.load(repository, checkpoint, predictor_factory=lambda **kwargs: Predictor(), **options)


def test_loader_rejects_preloaded_foreign_sam_package(tmp_path, monkeypatch):
    import sys
    repository = source_repository(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setitem(sys.modules, "sam3", SimpleNamespace(__file__=str(tmp_path / "foreign.py")))
    with pytest.raises(CvProviderError):
        sam31.Sam31EvidenceProvider.load(repository, checkpoint, predictor_factory=lambda **kwargs: Predictor())


def test_default_builder_import_is_local_and_close_removes_owned_modules(tmp_path, monkeypatch):
    import sys
    repository = source_repository(tmp_path, monkeypatch)
    (repository / ".gitignore").write_text("__pycache__/\n")
    (repository / "sam3" / "model_builder.py").write_text(
        "from types import SimpleNamespace\n"
        "def build_sam3_multiplex_video_predictor(**kwargs):\n"
        "    return SimpleNamespace(handle_request=lambda p: {}, "
        "handle_stream_request=lambda p: iter(()), assets=kwargs)\n"
    )
    git(repository, "add", ".")
    git(repository, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "builder")
    monkeypatch.setattr(sam31, "_PINNED_REPOSITORY_REVISION", git(repository, "rev-parse", "HEAD"))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    original_import = sam31.importlib.import_module
    monkeypatch.setattr(sam31.importlib, "import_module", lambda name: (
        SimpleNamespace() if name == "torch" else original_import(name)
    ))
    original_path = list(sys.path)
    instance = sam31.Sam31EvidenceProvider.load(repository, checkpoint)
    assert sys.path == original_path
    assert Path(sys.modules["sam3.model_builder"].__file__).is_relative_to(repository)
    instance.close()
    instance.close()
    assert "sam3.model_builder" not in sys.modules
    assert "sam3" not in sys.modules
    sam31._verify_repository_revision(repository)


def test_load_failure_shuts_down_allocated_predictor(tmp_path, monkeypatch):
    repository = source_repository(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    predictor = Predictor()
    shutdown = []
    predictor.shutdown = lambda: shutdown.append(True)
    original_import = sam31.importlib.import_module
    monkeypatch.setattr(sam31.importlib, "import_module", lambda name: (
        SimpleNamespace() if name == "torch" else original_import(name)
    ))
    with pytest.raises(CvProviderError):
        sam31.Sam31EvidenceProvider.load(
            repository, checkpoint, predictor_factory=lambda **kwargs: predictor,
            max_artifact_bytes=0,
        )
    assert shutdown == [True]
