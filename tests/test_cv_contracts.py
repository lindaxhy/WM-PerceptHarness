import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from percept_harness.cv import contracts
from percept_harness.cv.artifacts import CvArtifactError, CvArtifactStore, cv_cache_key
from percept_harness.cv.contracts import (
    ArtifactFile, CvEvidenceArtifact, CvEvidenceRequest, CvTrack, EvidenceStatus, FrameTimeline,
    OverlayRecord, SpatialPrompt, TrackObservation,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"overlay"


def request_data(**overrides):
    data = dict(
        schema_version="cv_request_v1", provider="sam31", model_identity="sam3.1",
        video_path="/videos/clip.mp4", video_identity="/videos/clip.mp4:1000:42",
        duration_seconds=2.0, frame_count=5, checkpoint_identity="/ckpt/sam3.1.pt:9:7",
        timeline={"frames": [{"frame_index": i, "timestamp_seconds": i * 0.25} for i in range(5)]},
        entities=[{"entity_id": "target", "canonical_label": "red cup"}],
        sampling={"max_fps": 2.0},
        spatial_prompts=[{"entity_id": "target", "frame_index": 2, "bbox_xyxy": [0.1, 0.2, 0.5, 0.6]}],
    )
    return {**data, **overrides}


def observation(frame_index, **overrides):
    return {"frame_index": frame_index, "timestamp_seconds": frame_index * 0.25,
            "bbox_xyxy": [0.1, 0.2, 0.5, 0.6], "mask_ref": "masks/target.npz", "visible": True,
            "confidence": 0.9, "area_fraction": 0.16, "center_xy": [0.3, 0.4], **overrides}


def artifact_data(**overrides):
    data = dict(
        schema_version="cv_evidence_v1", status="available", provider="sam31",
        model_identity="sam3.1", video_identity="/videos/clip.mp4:1000:42",
        checkpoint_identity="/ckpt/sam3.1.pt:9:7",
        processed_timeline={"frames": [{"frame_index": i, "timestamp_seconds": i * 0.25} for i in (1, 2, 3)]},
        entities=[{"entity_id": "target", "canonical_label": "red cup"}],
        tracks=[{"track_id": "target_0", "entity_id": "target",
                 "observations": [observation(1), observation(2)]}],
        files=[{"path": "masks/target.npz", "size_bytes": 5},
               {"path": "overlays/target-0-00000002.png", "size_bytes": len(PNG)}],
        overlay_records=[{"path": "overlays/target-0-00000002.png", "track_id": "target_0", "frame_index": 2}],
    )
    return {**data, **overrides}


def test_cv_cache_key_golden():
    request = CvEvidenceRequest.model_validate(request_data())
    assert cv_cache_key(request) == "c078178be72941e83811a8b1bd3d3fb40f75c6fb8cd3de1601b4e27bd65a8131"
    assert cv_cache_key(request.model_copy(update={"video_path": Path("/elsewhere.mp4")})) == cv_cache_key(request)
    probe = CvEvidenceRequest.model_validate(request_data(
        mode="probe", timeline={"frames": [{"frame_index": 2, "timestamp_seconds": 0.5}]}))
    assert cv_cache_key(probe) == "73d79bbf6b58889dd9ca9724a840601e462ab4254f28df9fff98862107f4156e"
    assert CvEvidenceRequest.model_validate_json(request.model_dump_json()) == request


def test_json_schemas_golden():
    from percept_harness.pipelines import event_schemas
    models = [CvEvidenceRequest, CvEvidenceArtifact] + [
        getattr(event_schemas, name) for name in sorted(dir(event_schemas))
        if isinstance(getattr(event_schemas, name), type)
        and issubclass(getattr(event_schemas, name), contracts.StrictModel)]
    encoded = json.dumps([m.model_json_schema() for m in models], sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == "aa9adcd4c3c7e07e318e78deb589abdbbf9e3c2108957560f5c0fedc108b9cfc"


@pytest.mark.parametrize("model, data", [
    (SpatialPrompt, {"entity_id": "target", "frame_index": 0, "bbox_xyxy": [0.5, 0.2, 0.5, 0.6]}),
    (SpatialPrompt, {"entity_id": "target", "frame_index": 0, "bbox_xyxy": [0.1, 0.6, 0.5, 0.2]}),
    (SpatialPrompt, {"entity_id": "target", "frame_index": 0, "bbox_xyxy": [0.1, 0.2, 1.5, 0.6]}),
    (SpatialPrompt, {"entity_id": "Target", "frame_index": 0, "bbox_xyxy": [0.1, 0.2, 0.5, 0.6]}),
    (TrackObservation, observation(1, bbox_xyxy=[0.5, 0.2, 0.1, 0.6])),
    (TrackObservation, observation(1, bbox_xyxy=[0.1, 0.6, 0.5, 0.6])),
    (TrackObservation, observation(1, mask_ref="overlays/a.png")),
    (TrackObservation, observation(1, mask_ref="/masks/a.npz")),
    (TrackObservation, observation(1, mask_ref="masks/../a.npz")),
    (TrackObservation, observation(1, mask_ref="masks\\a.npz")),
    (TrackObservation, observation(1, mask_ref="a" * 513)),
    (TrackObservation, observation(1, confidence=1.5)),
    (FrameTimeline, {"frames": []}),
    (FrameTimeline, {"frames": [{"frame_index": 1, "timestamp_seconds": 0.0},
                                {"frame_index": 1, "timestamp_seconds": 0.1}]}),
    (FrameTimeline, {"frames": [{"frame_index": 0, "timestamp_seconds": 0.1},
                                {"frame_index": 1, "timestamp_seconds": 0.1}]}),
    (CvTrack, {"track_id": "t", "entity_id": "target", "observations": [observation(2), observation(1)]}),
    (CvTrack, {"track_id": "t", "entity_id": "target",
               "observations": [observation(1), observation(2, timestamp_seconds=0.25)]}),
    (CvTrack, {"track_id": "t", "entity_id": "target", "observations": []}),
    (CvTrack, {"track_id": "t", "entity_id": "target", "observations": [observation(1)], "status": "disabled"}),
    (ArtifactFile, {"path": "", "size_bytes": 1}),
    (ArtifactFile, {"path": "a//b", "size_bytes": 1}),
    (ArtifactFile, {"path": "./a", "size_bytes": 1}),
    (ArtifactFile, {"path": "a/" * 256 + "b", "size_bytes": 1}),
    (ArtifactFile, {"path": "a", "size_bytes": -1}),
    (OverlayRecord, {"path": "overlays/a/b.png", "track_id": "t", "frame_index": 0}),
    (OverlayRecord, {"path": "masks/a.png", "track_id": "t", "frame_index": 0}),
    (OverlayRecord, {"path": "overlays/a.jpg", "track_id": "t", "frame_index": 0}),
    (CvEvidenceRequest, request_data(spatial_prompts=[
        {"entity_id": "other", "frame_index": 2, "bbox_xyxy": [0.1, 0.2, 0.5, 0.6]}])),
    (CvEvidenceRequest, request_data(spatial_prompts=[
        {"entity_id": "target", "frame_index": 9, "bbox_xyxy": [0.1, 0.2, 0.5, 0.6]}])),
    (CvEvidenceRequest, request_data(frame_count=4)),
    (CvEvidenceRequest, request_data(duration_seconds=0.75)),
    (CvEvidenceRequest, request_data(entities=[{"entity_id": "target", "canonical_label": "a"}] * 2)),
    (CvEvidenceRequest, request_data(provider="sam2")),
    (CvEvidenceRequest, request_data(extra=1)),
    (CvEvidenceArtifact, artifact_data(processed_timeline=None)),
    (CvEvidenceArtifact, artifact_data(status="disabled")),
    (CvEvidenceArtifact, artifact_data(tracks=[{"track_id": "target_0", "entity_id": "other",
                                                "observations": [observation(1)]}])),
    (CvEvidenceArtifact, artifact_data(tracks=[{"track_id": "target_0", "entity_id": "target",
                                                "observations": [observation(4)]}])),
    (CvEvidenceArtifact, artifact_data(files=[{"path": "overlays/target-0-00000002.png", "size_bytes": 1}])),
    (CvEvidenceArtifact, artifact_data(files=[{"path": "masks/target.npz", "size_bytes": 5}] * 2)),
    (CvEvidenceArtifact, artifact_data(overlay_records=[])),
    (CvEvidenceArtifact, artifact_data(overlay_records=[
        {"path": "overlays/target-0-00000002.png", "track_id": "target_1", "frame_index": 2}])),
    (CvEvidenceArtifact, artifact_data(overlay_records=[
        {"path": "overlays/target-0-00000002.png", "track_id": "target_0", "frame_index": 3}])),
])
def test_contract_rejections(model, data):
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_contract_limits(monkeypatch):
    assert CvEvidenceArtifact.model_validate(artifact_data()).status is EvidenceStatus.AVAILABLE
    assert ArtifactFile(path="a/" * 255 + "bc", size_bytes=0).path.endswith("bc")
    disabled = artifact_data(status="disabled", processed_timeline=None, tracks=[], files=[], overlay_records=[])
    assert CvEvidenceArtifact.model_validate(disabled).status is EvidenceStatus.DISABLED
    many = artifact_data(tracks=[{"track_id": f"target_{i}", "entity_id": "target",
                                  "observations": [observation(2)]} for i in range(3)])
    CvEvidenceArtifact.model_validate(many)
    monkeypatch.setattr(contracts, "MAX_OBSERVATIONS", 2)
    with pytest.raises(ValidationError, match="too many total observations"):
        CvEvidenceArtifact.model_validate(many)
    tracks = tuple(CvTrack.model_validate(track) for track in many["tracks"])
    with pytest.raises(ValidationError, match="too many total observations"):
        CvEvidenceArtifact.model_validate({**many, "tracks": tracks})
    with pytest.raises(ValidationError, match="too many total observations"):
        CvEvidenceArtifact.model_validate_json(json.dumps(many))
    with pytest.raises(ValidationError, match="at most 1 items"):
        CvEvidenceRequest.model_validate(request_data(spatial_prompts=[{}, {}]))


def _publish(root, request, artifact, files):
    store = CvArtifactStore(root)
    with store.staging(cv_cache_key(request)) as staging:
        for path, content in files.items():
            (staging / path).parent.mkdir(exist_ok=True)
            (staging / path).write_bytes(content)
        return store, store.publish(request, staging, artifact)


def test_artifact_store_round_trip_and_manifest_golden(tmp_path):
    request = CvEvidenceRequest.model_validate(request_data())
    artifact = CvEvidenceArtifact.model_validate(artifact_data())
    files = {"masks/target.npz": b"masks", "overlays/target-0-00000002.png": PNG}
    root = tmp_path / "cache"
    store, handle = _publish(root, request, artifact, files)
    key = cv_cache_key(request)
    assert handle.key == key
    entry = root / key[:2] / key
    manifest = (entry / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == "c4a3bec523adf27fdc1d3b6e5f0589e31045cda7083d862f756ab00808dc5c5b"
    assert manifest.startswith(b'{"artifact":{"checkpoint_identity":"/ckpt/sam3.1.pt:9:7","entities":')
    assert sorted(p.relative_to(entry).as_posix() for p in entry.rglob("*") if p.is_file()) == [
        "manifest.json", *sorted(files)]
    assert store.lookup(key) == handle
    assert store.load(handle) == artifact
    assert store.read_overlays(handle, ("overlays/target-0-00000002.png",)) == {
        "overlays/target-0-00000002.png": PNG}
    for paths in (("masks/target.npz",), ["overlays/target-0-00000002.png"]):
        with pytest.raises(CvArtifactError):
            store.read_overlays(handle, paths)
    # Republishing an identical request keeps the existing entry.
    assert _publish(root, request, artifact, files)[1] == handle
    # Only sizes and overlay signatures are checked, not content.
    (entry / "masks" / "target.npz").write_bytes(b"MASKS")
    assert store.read_overlays(handle, ()) == {}
    (entry / "masks" / "target.npz").write_bytes(b"tampered")
    for read in (store.load, lambda h: store.read_overlays(h, ())):
        with pytest.raises(CvArtifactError):
            read(handle)
    assert store.lookup(key) is None and not entry.exists()
    store.close()
    with pytest.raises(CvArtifactError):
        store.lookup(key)


def test_artifact_store_rejects_mismatches(tmp_path):
    request = CvEvidenceRequest.model_validate(request_data())
    artifact = CvEvidenceArtifact.model_validate(artifact_data())
    files = {"masks/target.npz": b"masks", "overlays/target-0-00000002.png": PNG}
    other = artifact.model_copy(update={"model_identity": "sam3"})
    for bad_artifact, bad_files in ((other, files), (artifact, {**files, "extra.bin": b""}),
                                    (artifact, {**files, "masks/target.npz": b"mask"}),
                                    (artifact, {**files, "overlays/target-0-00000002.png": b"x" * len(PNG)})):
        with pytest.raises(CvArtifactError, match="unable to publish"):
            _publish(tmp_path / "cache", request, bad_artifact, bad_files)
    store = CvArtifactStore(tmp_path / "cache")
    with pytest.raises(CvArtifactError):
        store.lookup("A" * 64)
    with pytest.raises(CvArtifactError, match="unable to publish"):
        store.publish(request, tmp_path, artifact)
    assert store.lookup(cv_cache_key(request)) is None
