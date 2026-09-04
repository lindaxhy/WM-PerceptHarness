from __future__ import annotations

from dataclasses import FrozenInstanceError
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import weakref

import pytest

from las_repro.cv.artifacts import CvArtifactError, CvArtifactStore, cv_cache_key
from las_repro.cv.contracts import (
    ArtifactFile,
    CvEvidenceArtifact,
    CvEvidenceRequest,
    CvTrack,
    EntityPrompt,
    EntityRole,
    EvidenceStatus,
    EvidenceThresholds,
    FrameTimeline,
    FrameTimestamp,
    SamplingPolicy,
    TrackObservation,
)


@pytest.fixture
def cv_request() -> CvEvidenceRequest:
    return CvEvidenceRequest(
        schema_version="cv_request_v1",
        provider="fake",
        model_identity="fake-sam31-v1",
        video_path=Path("videos/demo.mp4"),
        video_sha256="a" * 64,
        duration_seconds=3.0,
        frame_count=90,
        checkpoint_sha256="b" * 64,
        timeline=FrameTimeline(
            frames=(
                FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
                FrameTimestamp(frame_index=2, timestamp_seconds=0.1),
            )
        ),
        entities=(
            EntityPrompt(
                entity_id="right_hand",
                canonical_label="right hand",
                aliases=("hand",),
                role=EntityRole.ACTOR,
            ),
            EntityPrompt(
                entity_id="cup",
                canonical_label="cup",
                aliases=("mug",),
                role=EntityRole.MANIPULATED_OBJECT,
            ),
        ),
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


def artifact_for(request: CvEvidenceRequest, payload: bytes) -> CvEvidenceArtifact:
    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status=EvidenceStatus.AVAILABLE,
        provider=request.provider,
        model_identity=request.model_identity,
        video_sha256=request.video_sha256,
        checkpoint_sha256=request.checkpoint_sha256,
        entities=request.entities,
        tracks=(
            CvTrack(
                track_id="right_hand_1",
                entity_id="right_hand",
                observations=(
                    TrackObservation(
                        frame_index=0,
                        timestamp_seconds=0.0,
                        bbox_xyxy=(0.1, 0.1, 0.2, 0.2),
                        mask_ref="masks/0.npz",
                        visible=True,
                        confidence=0.9,
                        area_fraction=0.01,
                        center_xy=(0.15, 0.15),
                    ),
                ),
            ),
        ),
        files=(
            ArtifactFile(
                path="masks/0.npz",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            ),
        ),
    )


def write_artifact_file(staging: Path, payload: bytes) -> None:
    masks = staging / "masks"
    masks.mkdir(mode=0o700)
    (masks / "0.npz").write_bytes(payload)


def published_artifact(store, request, payload=b"mask", **artifact_updates):
    artifact = artifact_for(request, payload).model_copy(update=artifact_updates)
    key = cv_cache_key(request)
    with store.staging(key) as staging:
        write_artifact_file(staging, payload)
        handle = store.publish(request, staging, artifact)
    return artifact, handle


def write_raw_entry(root: Path, key: str, manifest: bytes) -> Path:
    prefix = root / key[:2]
    prefix.mkdir(mode=0o700, exist_ok=True)
    entry = prefix / key
    entry.mkdir(mode=0o700)
    (entry / "manifest.json").write_bytes(manifest)
    return entry


def canonical_manifest_for(
    request: CvEvidenceRequest, artifact: CvEvidenceArtifact
) -> bytes:
    cache_identity = {
        "schema_version": request.schema_version,
        "video_sha256": request.video_sha256,
        "provider": request.provider,
        "model_identity": request.model_identity,
        "checkpoint_sha256": request.checkpoint_sha256,
        "entities": [entity.model_dump(mode="json") for entity in request.entities],
        "sampling": request.sampling.model_dump(mode="json"),
        "thresholds": request.thresholds.model_dump(mode="json"),
    }
    return json.dumps(
        {
            "cache_identity": cache_identity,
            "artifact": artifact.model_dump(mode="json"),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def test_cache_key_has_stable_canonical_digest(cv_request):
    """Changing canonical JSON construction must change the published identity."""
    assert cv_cache_key(cv_request) == (
        "ed87e42882fdb25ca415530ae04d52b22f489e33516c65b134c4f6f98775af15"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "cv_request_v2"},
        {"video_sha256": "1" * 64},
        {"provider": "sam31"},
        {"model_identity": "sam3.1-production"},
        {"checkpoint_sha256": "2" * 64},
    ],
)
def test_cache_key_changes_for_scalar_evidence_inputs(cv_request, change):
    """Omitting any scalar evidence identity must incorrectly reuse an artifact."""
    assert cv_cache_key(cv_request.model_copy(update=change)) != cv_cache_key(cv_request)


def test_cache_key_changes_for_ordered_normalized_prompts(cv_request):
    """Ignoring prompt content or order must conflate different segmentation requests."""
    reordered = cv_request.model_copy(update={"entities": tuple(reversed(cv_request.entities))})
    changed_alias = cv_request.entities[0].model_copy(update={"aliases": ("palm",)})
    changed = cv_request.model_copy(
        update={"entities": (changed_alias, cv_request.entities[1])}
    )

    assert cv_cache_key(reordered) != cv_cache_key(cv_request)
    assert cv_cache_key(changed) != cv_cache_key(cv_request)


def test_cache_key_changes_for_sampling_and_thresholds(cv_request):
    """Dropping tunable evidence parameters must reuse semantically stale evidence."""
    sampling = cv_request.sampling.model_copy(update={"scan_fps": 7.0})
    thresholds = cv_request.thresholds.model_copy(update={"min_confidence": 0.75})

    assert cv_cache_key(cv_request.model_copy(update={"sampling": sampling})) != cv_cache_key(
        cv_request
    )
    assert cv_cache_key(
        cv_request.model_copy(update={"thresholds": thresholds})
    ) != cv_cache_key(cv_request)


def test_cache_key_excludes_mutable_source_location_and_downstream_inputs(cv_request):
    """Moving identical video bytes or changing Qwen text must not invalidate CV evidence."""
    downstream_only = cv_request.model_copy(
        update={
            "video_path": Path("/another/mutable/location.mp4"),
            "duration_seconds": 30.0,
            "frame_count": 300,
            "timeline": FrameTimeline(
                frames=(FrameTimestamp(frame_index=10, timestamp_seconds=4.0),)
            ),
            "qwen_prompt": "a downstream prompt that is not part of the contract",
        }
    )

    assert "qwen_prompt" not in CvEvidenceRequest.model_fields
    assert cv_cache_key(downstream_only) == cv_cache_key(cv_request)


def test_store_publishes_canonical_manifest_and_loads_validated_artifact(
    tmp_path, cv_request
):
    """Skipping publication validation must expose incomplete or changed evidence."""
    payload = b"\x00NPZ-mask-binary\xff"
    artifact = artifact_for(cv_request, payload)
    key = cv_cache_key(cv_request)
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)

    with store.staging(key) as staging:
        assert staging.parent == root / key[:2]
        assert staging.stat().st_mode & 0o077 == 0
        write_artifact_file(staging, payload)
        handle = store.publish(cv_request, staging, artifact)

    entry = root / key[:2] / key
    manifest_bytes = (entry / "manifest.json").read_bytes()
    expected_manifest = canonical_manifest_for(cv_request, artifact)
    assert manifest_bytes == expected_manifest
    assert payload not in manifest_bytes
    assert (entry / "masks" / "0.npz").read_bytes() == payload
    assert store.lookup(key) == handle
    assert store.load(handle) == artifact
    with pytest.raises(FrozenInstanceError):
        handle.key = "0" * 64  # type: ignore[misc]


def test_publish_commit_guard_failure_prevents_atomic_installation(
    tmp_path, cv_request
):
    """A failed generation guard must leave no installed cache entry."""
    payload = b"mask"
    artifact = artifact_for(cv_request, payload)
    key = cv_cache_key(cv_request)
    store = CvArtifactStore(tmp_path / "cv-cache")
    guard_entered = False

    class LeaseLost(RuntimeError):
        pass

    class RejectCommit:
        def __enter__(self):
            nonlocal guard_entered
            guard_entered = True
            raise LeaseLost("generation changed")

        def __exit__(self, *exc_info):
            return None

    with store.staging(key) as staging:
        write_artifact_file(staging, payload)
        with pytest.raises(LeaseLost, match="generation changed"):
            store.publish(
                cv_request,
                staging,
                artifact,
                commit_guard=RejectCommit,
            )

    assert guard_entered is True
    assert store.lookup(key) is None


def test_staging_is_removed_when_producer_fails(tmp_path, cv_request):
    """Leaking failed staging directories must leave unbounded partial artifacts."""
    store = CvArtifactStore(tmp_path / "cv-cache")
    key = cv_cache_key(cv_request)
    staging_path = None

    with pytest.raises(RuntimeError, match="producer failed"):
        with store.staging(key) as staging:
            staging_path = staging
            raise RuntimeError("producer failed")

    assert staging_path is not None
    assert not staging_path.exists()


def test_artifact_json_contains_references_but_never_mask_bytes(cv_request):
    """Embedding masks in the contract must leak large binary data into task JSON."""
    payload = b"\x00NPZ-mask-binary\xff"

    dumped = artifact_for(cv_request, payload).model_dump(mode="json")

    encoded = json.dumps(dumped, sort_keys=True).encode("utf-8")
    assert dumped["files"] == [
        {
            "path": "masks/0.npz",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    ]
    assert payload not in encoded


@pytest.mark.parametrize("malicious_path", ["../outside.npz", "/tmp/outside.npz"])
def test_publish_rejects_traversing_artifact_references(
    tmp_path, cv_request, malicious_path
):
    """Trusting a constructed manifest path must allow reads outside staging."""
    payload = b"mask"
    store = CvArtifactStore(tmp_path / "cv-cache")
    malicious_file = ArtifactFile.model_construct(
        path=malicious_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    artifact = artifact_for(cv_request, payload).model_copy(
        update={"files": (malicious_file,)}
    )

    with store.staging(cv_cache_key(cv_request)) as staging:
        write_artifact_file(staging, payload)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact") as caught:
            store.publish(cv_request, staging, artifact)

    assert malicious_path not in str(caught.value)


def test_publish_rejects_symlink_escape(tmp_path, cv_request):
    """Following a staged symlink must hash and publish attacker-selected bytes."""
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"mask")
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, b"mask")

    with store.staging(cv_cache_key(cv_request)) as staging:
        (staging / "masks").mkdir(mode=0o700)
        (staging / "masks" / "0.npz").symlink_to(outside)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)

    assert outside.read_bytes() == b"mask"


def test_publish_revalidates_constructed_artifact_contract(tmp_path, cv_request):
    """Trusting a constructed model must publish references normal validation forbids."""
    payload = b"mask"
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)
    observation_data = artifact.tracks[0].observations[0].model_dump(mode="python")
    bad_observation = TrackObservation.model_construct(
        **{**observation_data, "mask_ref": "../outside.npz"}
    )
    bad_track = artifact.tracks[0].model_copy(
        update={"observations": (bad_observation,)}
    )
    constructed = artifact.model_copy(
        update={"tracks": (bad_track,)}
    )

    with store.staging(cv_cache_key(cv_request)) as staging:
        write_artifact_file(staging, payload)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, constructed)

    assert store.lookup(cv_cache_key(cv_request)) is None


def test_publish_rejects_file_inode_swap(tmp_path, cv_request, monkeypatch):
    """Hashing an old descriptor after its path is replaced must publish unverified bytes."""
    payload = b"original-mask"
    replacement = tmp_path / "replacement.npz"
    replacement.write_bytes(b"replacement-mask")
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)
    original_open = Path.open
    swapped = False

    def swap_after_open(path, *args, **kwargs):
        nonlocal swapped
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.name == "0.npz" and mode == "rb" and not swapped:
            swapped = True
            os.replace(replacement, path)
        return stream

    with store.staging(cv_cache_key(cv_request)) as staging:
        write_artifact_file(staging, payload)
        monkeypatch.setattr(Path, "open", swap_after_open)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)

    assert swapped


def test_publish_revalidates_files_after_fsync(tmp_path, cv_request, monkeypatch):
    """Replacing a chunk after its first digest pass must publish unchecked bytes."""
    payload = b"original-mask"
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)
    original_fsync_tree = store._fsync_tree
    swapped = False

    def swap_after_fsync(staging):
        nonlocal swapped
        original_fsync_tree(staging)
        replacement = tmp_path / "post-fsync-replacement.npz"
        replacement.write_bytes(b"replacement-mask")
        os.replace(replacement, staging / "masks" / "0.npz")
        swapped = True

    monkeypatch.setattr(store, "_fsync_tree", swap_after_fsync)

    with store.staging(cv_cache_key(cv_request)) as staging:
        write_artifact_file(staging, payload)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)

    assert swapped
    assert store.lookup(cv_cache_key(cv_request)) is None


def test_publish_rejects_mutation_after_final_staging_validation(
    tmp_path, cv_request, monkeypatch
):
    """Mutating a chunk after final validation must not return a valid handle."""
    payload = b"original-mask"
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)
    original_validate_entry = store._validate_entry
    mutated = False

    def mutate_after_validation(entry, expected_key):
        nonlocal mutated
        result = original_validate_entry(entry, expected_key)
        if entry.name.startswith((".staging-", ".publish-")) and not mutated:
            (entry / "masks" / "0.npz").write_bytes(b"post-validation-mutation")
            mutated = True
        return result

    monkeypatch.setattr(store, "_validate_entry", mutate_after_validation)

    with store.staging(cv_cache_key(cv_request)) as staging:
        write_artifact_file(staging, payload)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)

    assert mutated
    assert store.lookup(cv_cache_key(cv_request)) is None


def test_publish_rejects_hard_linked_artifact_file(tmp_path, cv_request):
    """Publishing a shared inode must let an external pathname mutate cached bytes."""
    payload = b"mask"
    outside = tmp_path / "outside.npz"
    outside.write_bytes(payload)
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)

    with store.staging(cv_cache_key(cv_request)) as staging:
        (staging / "masks").mkdir(mode=0o700)
        os.link(outside, staging / "masks" / "0.npz")
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)

    assert outside.read_bytes() == payload


def test_retained_staging_descriptor_cannot_mutate_published_file(
    tmp_path, cv_request, monkeypatch
):
    """Renaming provider-owned inodes must preserve a retained writable alias."""
    payload = b"original-mask"
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)
    original_validate_entry = store._validate_entry
    retained = None
    mutated = False

    def mutate_source_after_destination_validation(entry, expected_key):
        nonlocal mutated
        result = original_validate_entry(entry, expected_key)
        if entry.name == expected_key and not mutated:
            assert retained is not None
            retained.seek(0)
            retained.write(b"changed!-mask")
            retained.flush()
            os.fsync(retained.fileno())
            mutated = True
        return result

    monkeypatch.setattr(store, "_validate_entry", mutate_source_after_destination_validation)
    key = cv_cache_key(cv_request)
    with store.staging(key) as staging:
        write_artifact_file(staging, payload)
        retained = (staging / "masks" / "0.npz").open("r+b")
        try:
            handle = store.publish(cv_request, staging, artifact)
        finally:
            retained.close()

    assert mutated
    assert store.load(handle) == artifact
    assert (tmp_path / "cv-cache" / key[:2] / key / "masks" / "0.npz").read_bytes() == payload


@pytest.mark.parametrize(
    ("file_update", "write_chunk"),
    [
        ({"sha256": "0" * 64}, True),
        ({"size_bytes": 99}, True),
        ({}, False),
    ],
)
def test_publish_rejects_wrong_digest_size_or_missing_chunk(
    tmp_path, cv_request, file_update, write_chunk
):
    """Accepting mismatched file metadata must make manifest validation meaningless."""
    payload = b"mask"
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)
    artifact = artifact.model_copy(
        update={"files": (artifact.files[0].model_copy(update=file_update),)}
    )

    with store.staging(cv_cache_key(cv_request)) as staging:
        if write_chunk:
            write_artifact_file(staging, payload)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)


def test_publish_enforces_file_count_and_total_byte_limits(tmp_path, cv_request):
    """Skipping configured limits must allow an artifact to exhaust local storage reads."""
    payload = b"four"
    key = cv_cache_key(cv_request)
    too_many = CvArtifactStore(tmp_path / "count-cache", max_files=1)
    base = artifact_for(cv_request, payload)
    second = ArtifactFile(path="masks/1.npz", sha256="0" * 64, size_bytes=0)
    two_files = base.model_copy(update={"files": (*base.files, second)})
    with too_many.staging(key) as staging:
        write_artifact_file(staging, payload)
        (staging / "masks" / "1.npz").write_bytes(b"")
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            too_many.publish(cv_request, staging, two_files)

    too_large = CvArtifactStore(tmp_path / "bytes-cache", max_bytes=3)
    with too_large.staging(key) as staging:
        write_artifact_file(staging, payload)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            too_large.publish(cv_request, staging, base)


def test_publish_bounds_traversal_of_undeclared_directories(tmp_path, cv_request):
    """Ignoring directory count must permit unbounded traversal before rejection."""
    payload = b"mask"
    store = CvArtifactStore(tmp_path / "cv-cache", max_files=1)
    artifact = artifact_for(cv_request, payload)

    with store.staging(cv_cache_key(cv_request)) as staging:
        write_artifact_file(staging, payload)
        for index in range(65):
            (staging / f"undeclared-{index}").mkdir(mode=0o700)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)


@pytest.mark.parametrize("mode", [0o770, 0o707])
def test_store_rejects_group_or_world_writable_cache_root(tmp_path, mode):
    """Accepting a writable root must let another principal replace validated entries."""
    root = tmp_path / "unsafe-cache"
    root.mkdir(mode=0o700)
    root.chmod(mode)

    with pytest.raises(CvArtifactError, match="unsafe CV artifact directory"):
        CvArtifactStore(root)


@pytest.mark.parametrize("case", ["partial", "oversized", "duplicate"])
def test_lookup_quarantines_partial_oversized_or_duplicate_manifest(
    tmp_path, cv_request, case
):
    """Returning corrupt cache entries must let partial evidence reach consumers."""
    if case == "partial":
        manifest = b'{"cache_identity":'
    elif case == "oversized":
        manifest = b"x" * 65
    else:
        artifact = artifact_for(cv_request, b"mask")
        manifest_payload = json.loads(canonical_manifest_for(cv_request, artifact))
        duplicate = dict(manifest_payload["artifact"]["files"][0])
        manifest_payload["artifact"]["files"].append(duplicate)
        manifest = json.dumps(
            manifest_payload, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    root = tmp_path / "cv-cache"
    max_manifest_bytes = 64 if case == "oversized" else 4096
    store = CvArtifactStore(root, max_manifest_bytes=max_manifest_bytes)
    key = cv_cache_key(cv_request)
    entry = write_raw_entry(root, key, manifest)

    assert store.lookup(key) is None
    assert not entry.exists()
    names = [path.name for path in (root / "quarantine").iterdir()]
    assert len(names) == 1
    assert re.fullmatch(rf"{key}-[0-9]+-[0-9a-f]+", names[0])
    assert "masks" not in names[0]


def test_lookup_rejects_manifest_inode_swap(tmp_path, cv_request, monkeypatch):
    """Reading a replacement manifest after stat must accept an unverified inode."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    empty_artifact = artifact_for(cv_request, b"mask").model_copy(
        update={"tracks": (), "files": (), "warnings": ("original",)}
    )
    original_manifest = canonical_manifest_for(cv_request, empty_artifact)
    entry = write_raw_entry(root, key, original_manifest)
    replacement = tmp_path / "replacement-manifest.json"
    replacement_artifact = empty_artifact.model_copy(update={"warnings": ("replacement",)})
    replacement.write_bytes(canonical_manifest_for(cv_request, replacement_artifact))
    original_open = os.open
    swapped = False

    def swap_after_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path).name == "manifest.json" and not swapped:
            swapped = True
            destination_descriptor = kwargs.get("dir_fd")
            if destination_descriptor is None:
                os.replace(replacement, path)
            else:
                os.replace(replacement, path, dst_dir_fd=destination_descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", swap_after_open)

    assert store.lookup(key) is None
    assert swapped
    assert not entry.exists()


def test_lookup_sanitizes_quarantine_failure(tmp_path, cv_request, monkeypatch):
    """Leaking a quarantine OS error must expose attacker-influenced local details."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    write_raw_entry(root, key, b"partial")
    original_replace = os.replace

    def fail_quarantine(source, destination, *args, **kwargs):
        if (
            Path(destination).parent.name == "quarantine"
            or kwargs.get("src_dir_fd") != kwargs.get("dst_dir_fd")
        ):
            raise OSError("private-attacker-detail")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_quarantine)

    with pytest.raises(CvArtifactError) as caught:
        store.lookup(key)

    assert "private-attacker-detail" not in str(caught.value)
    assert str(root) not in str(caught.value)


def test_lookup_sanitizes_hostile_lock_file(tmp_path, cv_request):
    """Passing a lock-open error through must disclose its local cache pathname."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    prefix = root / key[:2]
    prefix.mkdir(mode=0o700)
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"")
    (prefix / f".{key}.lock").symlink_to(outside)

    with pytest.raises(CvArtifactError) as caught:
        store.lookup(key)

    assert str(root) not in str(caught.value)


def test_load_revalidates_file_digest_and_quarantines_corruption(tmp_path, cv_request):
    """Trusting a prior handle must allow post-lookup mask tampering."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    _, handle = published_artifact(store, cv_request)
    mask = root / handle.key[:2] / handle.key / "masks" / "0.npz"
    mask.write_bytes(b"tampered")

    with pytest.raises(CvArtifactError, match="unable to load CV artifact"):
        store.load(handle)

    assert store.lookup(handle.key) is None
    assert len(tuple((root / "quarantine").iterdir())) == 1


def test_valid_existing_destination_wins_later_publication(tmp_path, cv_request):
    """Replacing a valid same-key entry must make concurrent results order-dependent."""
    store = CvArtifactStore(tmp_path / "cv-cache")
    first_artifact, first_handle = published_artifact(
        store, cv_request, warnings=("first",)
    )

    _, second_handle = published_artifact(store, cv_request, warnings=("second",))

    assert second_handle == first_handle
    assert store.load(second_handle) == first_artifact


def test_existing_destination_must_match_request_cache_identity(tmp_path, cv_request):
    """A valid artifact under another request key must not win publication."""
    payload = b"mask"
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    sam_request = cv_request.model_copy(
        update={
            "provider": "sam31",
            "model_identity": "sam3.1-production",
            "checkpoint_sha256": "c" * 64,
        }
    )
    key = cv_cache_key(sam_request)
    wrong_artifact = artifact_for(cv_request, payload)
    wrong_manifest = canonical_manifest_for(cv_request, wrong_artifact)
    wrong_entry = write_raw_entry(root, key, wrong_manifest)
    (wrong_entry / "masks").mkdir(mode=0o700)
    (wrong_entry / "masks" / "0.npz").write_bytes(payload)
    expected = artifact_for(sam_request, payload)

    with store.staging(key) as staging:
        write_artifact_file(staging, payload)
        handle = store.publish(sam_request, staging, expected)

    assert store.load(handle) == expected
    assert len(tuple((root / "quarantine").iterdir())) == 1


def test_corrupt_destination_is_quarantined_before_republication(tmp_path, cv_request):
    """Overwriting a corrupt entry in place must erase forensic isolation."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    _, first_handle = published_artifact(store, cv_request)
    entry = root / first_handle.key[:2] / first_handle.key
    original_inode = entry.stat().st_ino
    (entry / "manifest.json").write_bytes(b"partial")

    replacement, replacement_handle = published_artifact(
        store, cv_request, warnings=("replacement",)
    )

    assert store.load(replacement_handle) == replacement
    assert entry.stat().st_ino != original_inode
    assert len(tuple((root / "quarantine").iterdir())) == 1


def test_interrupted_atomic_replace_never_exposes_partial_entry(
    tmp_path, cv_request, monkeypatch
):
    """A failed rename must not leave a manifest-visible partial destination."""
    store = CvArtifactStore(tmp_path / "cv-cache")
    key = cv_cache_key(cv_request)
    artifact = artifact_for(cv_request, b"mask")
    real_replace = os.replace

    def interrupt_replace(source, destination, *args, **kwargs):
        if Path(source).name.startswith((".staging-", ".publish-")):
            raise OSError("injected interruption")
        return real_replace(source, destination, *args, **kwargs)

    with store.staging(key) as staging:
        write_artifact_file(staging, b"mask")
        monkeypatch.setattr(os, "replace", interrupt_replace)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact") as caught:
            store.publish(cv_request, staging, artifact)

    assert "injected interruption" not in str(caught.value)
    assert store.lookup(key) is None


def test_concurrent_publish_returns_one_valid_winner(
    tmp_path, cv_request, monkeypatch
):
    """Two unchecked replacements must return handles for different final manifests."""
    root = tmp_path / "cv-cache"
    stores = (CvArtifactStore(root), CvArtifactStore(root))
    artifacts = (
        artifact_for(cv_request, b"mask").model_copy(update={"warnings": ("one",)}),
        artifact_for(cv_request, b"mask").model_copy(update={"warnings": ("two",)}),
    )
    barrier = threading.Barrier(2)
    original_fsync_tree = CvArtifactStore._fsync_tree

    def synchronize_before_publication(self, stage):
        original_fsync_tree(self, stage)
        barrier.wait(timeout=5)

    monkeypatch.setattr(CvArtifactStore, "_fsync_tree", synchronize_before_publication)
    handles = []
    failures = []

    def publish(index):
        try:
            _, handle = published_artifact(
                stores[index], cv_request, warnings=artifacts[index].warnings
            )
            handles.append(handle)
        except BaseException as error:  # surfaced below from the test threads
            failures.append(error)

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    assert len(handles) == 2
    assert handles[0] == handles[1] == stores[0].lookup(cv_cache_key(cv_request))
    assert stores[0].load(handles[0]) in artifacts


class LimitedScandir:
    """Fail if a caller asks the real scandir iterator for too many entries."""

    def __init__(self, iterator, limit, consumed, on_end=None):
        self._iterator = iterator
        self._limit = limit
        self._consumed = consumed
        self._on_end = on_end

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._consumed[0] >= self._limit:
            raise AssertionError("scandir consumed beyond the configured tree bound")
        try:
            entry = next(self._iterator)
        except StopIteration:
            if self._on_end is not None:
                callback, self._on_end = self._on_end, None
                callback()
            raise
        self._consumed[0] += 1
        return entry

    def close(self):
        self._iterator.close()


def same_directory_endpoint(value, expected: Path) -> bool:
    if isinstance(value, int):
        opened = os.fstat(value)
        target = expected.stat()
        return (opened.st_dev, opened.st_ino) == (target.st_dev, target.st_ino)
    return Path(value) == expected


def test_high_fanout_scan_stops_at_bound_before_materializing_directory(
    tmp_path, cv_request, monkeypatch
):
    """Materializing all directory entries must defeat the configured scan bound."""
    payload = b"mask"
    store = CvArtifactStore(tmp_path / "cv-cache", max_files=1)
    artifact = artifact_for(cv_request, payload)
    key = cv_cache_key(cv_request)
    real_scandir = os.scandir
    consumed = [0]
    wrapped = [False]

    with store.staging(key) as staging:
        write_artifact_file(staging, payload)
        for index in range(256):
            (staging / f"fanout-{index:03}").mkdir(mode=0o700)

        def limited_scandir(path):
            iterator = real_scandir(path)
            if same_directory_endpoint(path, staging) and not wrapped[0]:
                wrapped[0] = True
                return LimitedScandir(iterator, 65, consumed)
            return iterator

        monkeypatch.setattr(os, "scandir", limited_scandir)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)

    assert consumed[0] == 65


def test_fifo_manifest_lookup_returns_promptly_without_writer(tmp_path, cv_request):
    """Opening a FIFO manifest in blocking mode must hang cache lookup."""
    root = tmp_path / "cv-cache"
    CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    prefix = root / key[:2]
    prefix.mkdir(mode=0o700)
    entry = prefix / key
    entry.mkdir(mode=0o700)
    os.mkfifo(entry / "manifest.json", mode=0o600)
    script = (
        "from pathlib import Path; "
        "from las_repro.cv.artifacts import CvArtifactStore; "
        "print(CvArtifactStore(Path(__import__('sys').argv[1])).lookup(__import__('sys').argv[2]))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), key],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "None"


def test_publish_rechecks_membership_after_initial_walk(
    tmp_path, cv_request, monkeypatch
):
    """Adding an undeclared file after membership capture must not be published."""
    payload = b"mask"
    store = CvArtifactStore(tmp_path / "cv-cache")
    artifact = artifact_for(cv_request, payload)
    key = cv_cache_key(cv_request)
    real_scandir = os.scandir
    mutated = False

    with store.staging(key) as staging:
        write_artifact_file(staging, payload)

        def add_file_after_walk():
            nonlocal mutated
            (staging / "unexpected.npz").write_bytes(b"unexpected")
            mutated = True

        def mutating_scandir(path):
            iterator = real_scandir(path)
            if same_directory_endpoint(path, staging) and not mutated:
                return LimitedScandir(iterator, 100, [0], add_file_after_walk)
            return iterator

        monkeypatch.setattr(os, "scandir", mutating_scandir)
        with pytest.raises(CvArtifactError, match="unable to publish CV artifact"):
            store.publish(cv_request, staging, artifact)

    assert mutated
    assert store.lookup(key) is None


def two_file_artifact(
    request: CvEvidenceRequest, first: bytes, second: bytes
) -> CvEvidenceArtifact:
    base = artifact_for(request, first)
    second_file = ArtifactFile(
        path="masks/1.npz",
        sha256=hashlib.sha256(second).hexdigest(),
        size_bytes=len(second),
    )
    return base.model_copy(update={"files": (*base.files, second_file)})


def test_lookup_rechecks_already_hashed_file_metadata(
    tmp_path, cv_request, monkeypatch
):
    """Mutating an already-hashed inode must not produce a validated handle."""
    first = b"first-mask"
    second = b"second-mask"
    artifact = two_file_artifact(cv_request, first, second)
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    entry = write_raw_entry(root, key, canonical_manifest_for(cv_request, artifact))
    (entry / "masks").mkdir(mode=0o700)
    first_path = entry / "masks" / "0.npz"
    first_path.write_bytes(first)
    (entry / "masks" / "1.npz").write_bytes(second)
    expected_digest = hashlib.sha256(first).hexdigest()
    real_sha256 = hashlib.sha256
    mutated = False

    class MutatingHash:
        def __init__(self, value=b""):
            self._hash = real_sha256(value)

        def update(self, value):
            self._hash.update(value)

        def hexdigest(self):
            nonlocal mutated
            result = self._hash.hexdigest()
            if result == expected_digest and not mutated:
                first_path.write_bytes(b"evil!-mask")
                mutated = True
            return result

    monkeypatch.setattr(hashlib, "sha256", MutatingHash)

    assert store.lookup(key) is None
    assert mutated
    assert not entry.exists()


def test_store_rejects_symlinked_or_writable_ancestry(tmp_path):
    """Checking only the terminal root must trust replaceable ancestor paths."""
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    writable_parent = tmp_path / "writable-parent"
    writable_parent.mkdir(mode=0o700)
    writable_parent.chmod(0o777)

    with pytest.raises(CvArtifactError, match="unsafe CV artifact directory"):
        CvArtifactStore(linked_parent / "cache")
    with pytest.raises(CvArtifactError, match="unsafe CV artifact directory"):
        CvArtifactStore(writable_parent / "cache")


def test_store_detects_ancestor_symlink_substitution_after_initialization(
    tmp_path, cv_request
):
    """Reaching the same root inode through a substituted ancestor must be rejected."""
    parent = tmp_path / "cache-parent"
    parent.mkdir(mode=0o700)
    root = parent / "cache"
    store = CvArtifactStore(root)
    displaced = tmp_path / "displaced-parent"
    parent.rename(displaced)
    parent.symlink_to(displaced, target_is_directory=True)

    with pytest.raises(CvArtifactError, match="unsafe CV artifact directory"):
        store.lookup(cv_cache_key(cv_request))


def test_lookup_does_not_follow_ancestor_substitution_after_root_check(
    tmp_path, cv_request, monkeypatch
):
    """A swap immediately after the root check must not redirect the lookup."""
    parent = tmp_path / "cache-parent"
    parent.mkdir(mode=0o700)
    root = parent / "cache"
    store = CvArtifactStore(root)
    _, handle = published_artifact(store, cv_request)
    replacement_parent = tmp_path / "replacement-parent"
    replacement_parent.mkdir(mode=0o700)
    shutil.copytree(root, replacement_parent / "cache")
    displaced = tmp_path / "displaced-parent"
    original_open_prefix = store._open_prefix
    swapped = False

    def substitute_before_prefix_open(key, *, create):
        nonlocal swapped
        if not swapped:
            parent.rename(displaced)
            replacement_parent.rename(parent)
            swapped = True
        return original_open_prefix(key, create=create)

    monkeypatch.setattr(store, "_open_prefix", substitute_before_prefix_open)

    with pytest.raises(CvArtifactError, match="unsafe CV artifact directory"):
        store.lookup(handle.key)

    assert swapped


def test_new_prefix_creation_fsyncs_cache_root(tmp_path, cv_request, monkeypatch):
    """Omitting the parent fsync must lose a newly created prefix after a crash."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    root_identity = (root.stat().st_dev, root.stat().st_ino)
    real_fsync = os.fsync
    fsynced = []

    def record_fsync(descriptor):
        status = os.fstat(descriptor)
        fsynced.append((status.st_dev, status.st_ino))
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    with store.staging(cv_cache_key(cv_request)):
        pass

    assert root_identity in fsynced


def test_new_quarantine_creation_fsyncs_cache_root(tmp_path, cv_request, monkeypatch):
    """Omitting the parent fsync must lose a new quarantine directory after a crash."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    write_raw_entry(root, key, b"partial")
    root_identity = (root.stat().st_dev, root.stat().st_ino)
    real_fsync = os.fsync
    fsynced = []

    def record_fsync(descriptor):
        status = os.fstat(descriptor)
        fsynced.append((status.st_dev, status.st_ino))
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    assert store.lookup(key) is None

    assert root_identity in fsynced


def test_lookup_treats_disappearing_entry_as_sanitized_miss(
    tmp_path, cv_request, monkeypatch
):
    """Disappearance between existence and identity checks must leak its pathname."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    entry = write_raw_entry(root, key, b"partial")
    displaced = tmp_path / "disappeared-entry"
    original_identity = store._entry_identity
    removed = False

    def disappear_before_identity(requested_key):
        nonlocal removed
        if requested_key == key and not removed:
            entry.rename(displaced)
            removed = True
        return original_identity(requested_key)

    monkeypatch.setattr(store, "_entry_identity", disappear_before_identity)

    assert store.lookup(key) is None
    assert removed


def test_lookup_quarantines_deep_json_without_recursion_error(tmp_path, cv_request):
    """Unbounded JSON nesting must escape as an unsanitized RecursionError."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    entry = write_raw_entry(root, key, b"[" * 10_000 + b"]" * 10_000)

    assert store.lookup(key) is None
    assert not entry.exists()
    assert len(tuple((root / "quarantine").iterdir())) == 1


def test_concurrent_prefix_observer_fsyncs_root_before_using_winner(
    tmp_path, cv_request, monkeypatch
):
    """An EEXIST loser must not use a newly observed prefix without its own fsync."""
    root = tmp_path / "cv-cache"
    stores = (CvArtifactStore(root), CvArtifactStore(root))
    key = cv_cache_key(cv_request)
    barrier = threading.Barrier(2)
    real_mkdir = os.mkdir
    real_fsync = os.fsync
    mkdir_threads = set()
    fsync_threads = set()
    failures = []

    def race_prefix_mkdir(path, mode=0o777, *, dir_fd=None):
        if path == key[:2] and same_directory_endpoint(dir_fd, root):
            mkdir_threads.add(threading.get_ident())
            barrier.wait(timeout=5)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    def record_root_fsync(descriptor):
        if same_directory_endpoint(descriptor, root):
            fsync_threads.add(threading.get_ident())
        return real_fsync(descriptor)

    def create_staging(store):
        try:
            with store.staging(key):
                pass
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(os, "mkdir", race_prefix_mkdir)
    monkeypatch.setattr(os, "fsync", record_root_fsync)
    threads = [threading.Thread(target=create_staging, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert len(mkdir_threads) == 2
    assert mkdir_threads <= fsync_threads


def test_lookup_returns_miss_when_entry_disappears_after_identity(
    tmp_path, cv_request, monkeypatch
):
    """A vanished identified entry must remain a cache miss, not an error."""
    root = tmp_path / "cv-cache"
    store = CvArtifactStore(root)
    key = cv_cache_key(cv_request)
    entry = write_raw_entry(root, key, b"partial")
    displaced = tmp_path / "disappeared-after-identity"
    original_validate = store._validate_entry
    removed = False

    def disappear_before_validation(path, expected_key):
        nonlocal removed
        if path == entry and not removed:
            entry.rename(displaced)
            removed = True
        return original_validate(path, expected_key)

    monkeypatch.setattr(store, "_validate_entry", disappear_before_validation)

    assert store.lookup(key) is None
    assert removed


def test_store_context_manager_closes_root_descriptor_without_double_close(tmp_path):
    """Closing twice must not close a new resource that reuses the old fd number."""
    root = tmp_path / "cv-cache"
    with CvArtifactStore(root) as store:
        root_descriptor = store._root_descriptor
        os.fstat(root_descriptor)

    with pytest.raises(OSError):
        os.fstat(root_descriptor)

    source_descriptor = os.open(root, os.O_RDONLY)
    try:
        os.dup2(source_descriptor, root_descriptor)
        store.close()
        os.fstat(root_descriptor)
    finally:
        os.close(root_descriptor)
        if source_descriptor != root_descriptor:
            os.close(source_descriptor)


def test_store_close_waits_for_active_staging_before_releasing_root(
    tmp_path, cv_request
):
    """Close requested inside an active operation must defer descriptor release."""
    store = CvArtifactStore(tmp_path / "cv-cache")
    root_descriptor = store._root_descriptor

    with store.staging(cv_cache_key(cv_request)):
        store.close()
        os.fstat(root_descriptor)

    with pytest.raises(OSError):
        os.fstat(root_descriptor)
    with pytest.raises(CvArtifactError):
        store.lookup(cv_cache_key(cv_request))


def test_store_close_defers_root_release_until_context_exit(tmp_path):
    """The context-manager lease must keep the descriptor alive through its body."""
    store = CvArtifactStore(tmp_path / "cv-cache")

    with store:
        root_descriptor = store._root_descriptor
        store.close()
        os.fstat(root_descriptor)

    with pytest.raises(OSError):
        os.fstat(root_descriptor)


def test_store_finalizer_releases_unclosed_root_descriptor(tmp_path):
    """Dropping an unclosed store must not leak its trusted root descriptor."""
    store = CvArtifactStore(tmp_path / "cv-cache")
    root_descriptor = store._root_descriptor
    reference = weakref.ref(store)

    del store
    gc.collect()

    assert reference() is None
    with pytest.raises(OSError):
        os.fstat(root_descriptor)


def test_rejected_ancestry_closes_newly_opened_child_descriptor(
    tmp_path, monkeypatch
):
    """Rejecting an unsafe child must close both parent and rejected child fds."""
    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    opened_descriptors = set()
    original_open = CvArtifactStore._open_directory_descriptor

    def track_open(path, *, dir_fd=None):
        descriptor = original_open(path, dir_fd=dir_fd)
        opened_descriptors.add(descriptor)
        return descriptor

    monkeypatch.setattr(
        CvArtifactStore, "_open_directory_descriptor", staticmethod(track_open)
    )

    with pytest.raises(CvArtifactError, match="unsafe CV artifact directory"):
        CvArtifactStore(unsafe_parent / "cache")

    assert opened_descriptors
    for descriptor in opened_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_directory_open_closes_descriptor_when_fstat_fails(tmp_path, monkeypatch):
    """An error validating a just-opened directory must not leak its descriptor."""
    opened_descriptors = []
    real_open = os.open
    real_fstat = os.fstat

    def track_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened_descriptors.append(descriptor)
        return descriptor

    def fail_fstat(descriptor):
        raise OSError("injected fstat failure")

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "fstat", fail_fstat)

    with pytest.raises(OSError, match="injected fstat failure"):
        CvArtifactStore._open_directory_descriptor(tmp_path)

    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        real_fstat(opened_descriptors[0])


def test_concurrent_cache_root_creators_reopen_validate_and_fsync_winner(
    tmp_path, monkeypatch
):
    """A root-creation EEXIST loser must validate and fsync the winning entry."""
    root = tmp_path / "cv-cache"
    barrier = threading.Barrier(2)
    real_mkdir = os.mkdir
    real_fsync = os.fsync
    mkdir_threads = set()
    fsync_threads = set()
    stores = []
    failures = []

    def race_root_mkdir(path, mode=0o777, *, dir_fd=None):
        if path == root.name and same_directory_endpoint(dir_fd, tmp_path):
            mkdir_threads.add(threading.get_ident())
            barrier.wait(timeout=5)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    def record_parent_fsync(descriptor):
        if same_directory_endpoint(descriptor, tmp_path):
            fsync_threads.add(threading.get_ident())
        return real_fsync(descriptor)

    def construct_store():
        try:
            stores.append(CvArtifactStore(root))
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(os, "mkdir", race_root_mkdir)
    monkeypatch.setattr(os, "fsync", record_parent_fsync)
    threads = [threading.Thread(target=construct_store) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert len(stores) == 2
    assert len(mkdir_threads) == 2
    assert mkdir_threads <= fsync_threads


def test_concurrent_first_quarantines_reopen_and_fsync_winner(
    tmp_path, cv_request, monkeypatch
):
    """Different corrupt keys racing first quarantine must both finish durably."""
    root = tmp_path / "cv-cache"
    stores = (CvArtifactStore(root), CvArtifactStore(root))
    other_request = cv_request.model_copy(update={"checkpoint_sha256": "c" * 64})
    keys = (cv_cache_key(cv_request), cv_cache_key(other_request))
    for key in keys:
        write_raw_entry(root, key, b"partial")
    barrier = threading.Barrier(2)
    real_mkdir = os.mkdir
    real_fsync = os.fsync
    mkdir_threads = set()
    fsync_threads = set()
    results = []
    failures = []

    def race_quarantine_mkdir(path, mode=0o777, *, dir_fd=None):
        if path == "quarantine" and same_directory_endpoint(dir_fd, root):
            mkdir_threads.add(threading.get_ident())
            barrier.wait(timeout=5)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    def record_root_fsync(descriptor):
        if same_directory_endpoint(descriptor, root):
            fsync_threads.add(threading.get_ident())
        return real_fsync(descriptor)

    def lookup_corrupt(store, key):
        try:
            results.append(store.lookup(key))
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(os, "mkdir", race_quarantine_mkdir)
    monkeypatch.setattr(os, "fsync", record_root_fsync)
    threads = [
        threading.Thread(target=lookup_corrupt, args=(store, key))
        for store, key in zip(stores, keys, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert results == [None, None]
    assert len(tuple((root / "quarantine").iterdir())) == 2
    assert len(mkdir_threads) == 2
    assert mkdir_threads <= fsync_threads
