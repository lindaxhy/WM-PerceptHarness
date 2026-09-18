from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import percept_harness.cv as cv_public
import percept_harness.cv.contracts as cv_contracts
from percept_harness.cv.contracts import (
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
    OverlayRecord,
    SamplingPolicy,
    TrackObservation,
)


SHA256 = "a" * 64


class BombList(list):
    """A sized hostile container that must be rejected before iteration."""

    def __iter__(self):
        raise RuntimeError("hostile list was iterated")


class BombStr(str):
    """A hostile string subclass whose enum operations must never run."""

    def __hash__(self) -> int:
        raise RuntimeError("hostile string was hashed")

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile string was compared")


class BombInt(int):
    """A hostile integer subclass whose enum operations must never run."""

    def __hash__(self) -> int:
        raise RuntimeError("hostile integer was hashed")

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile integer was compared")


def valid_timeline() -> FrameTimeline:
    return FrameTimeline(
        frames=(
            FrameTimestamp(frame_index=0, timestamp_seconds=0.0),
            FrameTimestamp(frame_index=2, timestamp_seconds=0.1),
        )
    )


def valid_request() -> CvEvidenceRequest:
    return CvEvidenceRequest(
        schema_version="cv_request_v1",
        provider="fake",
        model_identity="fake-sam31-v1",
        video_path=Path("videos/demo.mp4"),
        video_sha256=SHA256,
        duration_seconds=3.0,
        frame_count=90,
        checkpoint_sha256="b" * 64,
        timeline=valid_timeline(),
        entities=(
            EntityPrompt(
                entity_id="right_hand",
                canonical_label="right hand",
                aliases=("hand",),
                role=EntityRole.ACTOR,
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


def valid_artifact() -> CvEvidenceArtifact:
    entity = EntityPrompt(
        entity_id="cup",
        canonical_label="cup",
        aliases=(),
        role=EntityRole.MANIPULATED_OBJECT,
    )
    observation = TrackObservation(
        frame_index=0,
        timestamp_seconds=0.0,
        bbox_xyxy=(0.1, 0.1, 0.2, 0.2),
        mask_ref="masks/0.npz",
        visible=True,
        confidence=0.9,
        area_fraction=0.01,
        center_xy=(0.15, 0.15),
    )
    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status=EvidenceStatus.AVAILABLE,
        provider="fake",
        model_identity="fake-sam31-v1",
        video_sha256=SHA256,
        checkpoint_sha256="b" * 64,
        processed_timeline=FrameTimeline(
            frames=(FrameTimestamp(frame_index=0, timestamp_seconds=0.0),)
        ),
        entities=(entity,),
        tracks=(CvTrack(track_id="cup_1", entity_id="cup", observations=(observation,)),),
        files=(ArtifactFile(path="masks/0.npz", sha256="c" * 64, size_bytes=4),),
        overlay_records=(),
    )


@pytest.mark.parametrize("hostile", [BombStr("available"), BombInt(1)])
def test_contract_enum_preflight_rejects_scalar_subclasses(hostile: object) -> None:
    """Every contract enum boundary rejects subclasses before enum lookup."""
    entity_payload = valid_request().entities[0].model_dump(mode="json")
    entity_payload["role"] = hostile
    track_payload = valid_artifact().tracks[0].model_dump(mode="json")
    track_payload["status"] = hostile
    artifact_payload = valid_artifact().model_dump(mode="json")
    artifact_payload["status"] = hostile

    for model, payload in (
        (EntityPrompt, entity_payload),
        (CvTrack, track_payload),
        (CvEvidenceArtifact, artifact_payload),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(payload)

    assert EntityPrompt.model_validate(
        {
            **valid_request().entities[0].model_dump(mode="json"),
            "role": EntityRole.ACTOR,
        }
    ).role is EntityRole.ACTOR
    assert CvTrack.model_validate(
        {
            **valid_artifact().tracks[0].model_dump(mode="json"),
            "status": "available",
        }
    ).status is EvidenceStatus.AVAILABLE


def test_track_observation_rejects_invalid_geometry_and_time():
    """Removing finite/bounded geometry guards must admit unusable evidence."""
    with pytest.raises(ValidationError):
        TrackObservation(
            frame_index=3,
            timestamp_seconds=float("nan"),
            bbox_xyxy=(0.8, 0.1, 0.2, 0.9),
            mask_ref="masks/3.npz",
            visible=True,
            confidence=1.1,
            area_fraction=0.2,
            center_xy=(0.5, 0.5),
        )


def test_contract_models_reject_extra_fields_and_are_frozen():
    """Opening or mutating a boundary model must fail before a job is persisted."""
    request = valid_request()
    with pytest.raises(ValidationError):
        CvEvidenceRequest.model_validate({**request.model_dump(), "typo": True})
    with pytest.raises(ValidationError):
        request.model_identity = "another-model"  # type: ignore[misc]


def test_artifact_extra_key_uses_precise_pydantic_location() -> None:
    """Artifact structural preflight must leave ordinary extras to core."""
    for extra in ("ordinary", BombList(["do-not-touch"])):
        payload = valid_artifact().model_dump(mode="json")
        payload["typo"] = extra

        with pytest.raises(ValidationError) as caught:
            CvEvidenceArtifact.model_validate(payload)

        assert any(
            error["loc"] == ("typo",) and error["type"] == "extra_forbidden"
            for error in caught.value.errors()
        )


def test_contract_models_preserve_strict_json_dict_and_text_roundtrips():
    """Shallow tuple freezing must retain both supported JSON entry paths."""
    request = valid_request()
    artifact = valid_artifact()

    assert type(request).model_validate(
        request.model_dump(mode="json"), strict=True
    ) == request
    assert type(request).model_validate_json(
        request.model_dump_json(), strict=True
    ) == request
    assert type(artifact).model_validate(
        artifact.model_dump(mode="json"), strict=True
    ) == artifact
    assert type(artifact).model_validate_json(
        artifact.model_dump_json(), strict=True
    ) == artifact


def test_contract_rejects_hostile_extra_container_before_iteration():
    """Forbidden values must not be recursively inspected before extra checking."""
    payload = valid_request().model_dump(mode="json")
    payload["forbidden"] = BombList(["do-not-touch"])

    with pytest.raises(ValidationError):
        CvEvidenceRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["entities", "tracks"])
def test_artifact_rejects_hostile_top_level_sequence_before_iteration(field):
    """Artifact-specific preflight must not outrun the shallow base validator."""
    payload = valid_artifact().model_dump(mode="json")
    payload[field] = BombList(payload[field])

    with pytest.raises(ValidationError):
        CvEvidenceArtifact.model_validate(payload)


@pytest.mark.parametrize("entity_id", ["Right_Hand", "right-hand", "right hand", "2hand"])
def test_entity_prompt_rejects_non_snake_case_ids(entity_id):
    """Relaxing object IDs would break deterministic references across artifacts."""
    with pytest.raises(ValidationError):
        EntityPrompt(
            entity_id=entity_id,
            canonical_label="right hand",
            aliases=(),
            role=EntityRole.ACTOR,
        )


def test_timeline_requires_strictly_increasing_indices_and_timestamps():
    """Allowing repeated/out-of-order timestamps would corrupt frame alignment."""
    with pytest.raises(ValidationError):
        FrameTimeline(
            frames=(
                FrameTimestamp(frame_index=1, timestamp_seconds=0.0),
                FrameTimestamp(frame_index=1, timestamp_seconds=0.1),
            )
        )
    with pytest.raises(ValidationError):
        FrameTimeline(
            frames=(
                FrameTimestamp(frame_index=1, timestamp_seconds=0.2),
                FrameTimestamp(frame_index=2, timestamp_seconds=0.2),
            )
        )


@pytest.mark.parametrize("path", ["/absolute/mask.npz", "masks/../mask.npz", "masks\\mask.npz"])
def test_artifact_file_rejects_non_relative_posix_paths(path):
    """Accepting an escaping artifact path would permit cache-root traversal."""
    with pytest.raises(ValidationError):
        ArtifactFile(path=path, sha256=SHA256, size_bytes=1)


def test_artifact_requires_unique_track_ids_and_closed_entity_references():
    """Removing manifest cross-reference checks would create ambiguous evidence."""
    observation = TrackObservation(
        frame_index=0,
        timestamp_seconds=0.0,
        bbox_xyxy=(0.1, 0.1, 0.2, 0.2),
        mask_ref="masks/0.npz",
        visible=True,
        confidence=0.9,
        area_fraction=0.01,
        center_xy=(0.15, 0.15),
    )
    known = EntityPrompt(
        entity_id="cup",
        canonical_label="cup",
        aliases=(),
        role=EntityRole.MANIPULATED_OBJECT,
    )
    duplicate_tracks = (
        CvTrack(track_id="cup_1", entity_id="cup", observations=(observation,)),
        CvTrack(track_id="cup_1", entity_id="cup", observations=(observation,)),
    )
    with pytest.raises(ValidationError):
        CvEvidenceArtifact(
            schema_version="cv_evidence_v1",
            status=EvidenceStatus.AVAILABLE,
            provider="fake",
            model_identity="fake-sam31-v1",
            video_sha256=SHA256,
            checkpoint_sha256="b" * 64,
            processed_timeline=FrameTimeline(
                frames=(FrameTimestamp(frame_index=0, timestamp_seconds=0.0),)
            ),
            entities=(known,),
            tracks=duplicate_tracks,
            files=(ArtifactFile(path="masks/0.npz", sha256="c" * 64, size_bytes=4),),
            overlay_records=(),
        )
    with pytest.raises(ValidationError):
        CvEvidenceArtifact(
            schema_version="cv_evidence_v1",
            status=EvidenceStatus.AVAILABLE,
            provider="fake",
            model_identity="fake-sam31-v1",
            video_sha256=SHA256,
            checkpoint_sha256="b" * 64,
            processed_timeline=FrameTimeline(
                frames=(FrameTimestamp(frame_index=0, timestamp_seconds=0.0),)
            ),
            entities=(known,),
            tracks=(CvTrack(track_id="other_1", entity_id="other", observations=(observation,)),),
            files=(ArtifactFile(path="masks/0.npz", sha256="c" * 64, size_bytes=4),),
            overlay_records=(),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_confidence_rejects_non_finite_values(value):
    """Dropping finite confidence validation would make evidence ordering undefined."""
    with pytest.raises(ValidationError):
        TrackObservation(
            frame_index=0,
            timestamp_seconds=0.0,
            bbox_xyxy=(0.1, 0.1, 0.2, 0.2),
            mask_ref="masks/0.npz",
            visible=True,
            confidence=value,
            area_fraction=0.01,
            center_xy=(0.15, 0.15),
        )


def test_artifact_requires_exact_schema_version():
    """Accepting another manifest version would defeat provider-independent parsing."""
    with pytest.raises(ValidationError):
        CvEvidenceArtifact.model_validate(
            {**valid_artifact().model_dump(), "schema_version": "cv_evidence_v2"}
        )


def test_available_artifact_requires_processed_clock_and_closes_observations():
    """A source timeline cannot stand in for the frames the provider processed."""
    artifact = valid_artifact()
    payload = artifact.model_dump(mode="json")
    payload["processed_timeline"] = {
        "frames": [{"frame_index": 0, "timestamp_seconds": 0.0}]
    }
    payload["overlay_records"] = []

    validated = CvEvidenceArtifact.model_validate(payload)
    assert validated.processed_timeline == FrameTimeline(
        frames=(FrameTimestamp(frame_index=0, timestamp_seconds=0.0),)
    )

    del payload["processed_timeline"]
    with pytest.raises(ValidationError):
        CvEvidenceArtifact.model_validate(payload)

    payload["processed_timeline"] = {
        "frames": [{"frame_index": 0, "timestamp_seconds": 0.125}]
    }
    with pytest.raises(ValidationError, match="processed timeline"):
        CvEvidenceArtifact.model_validate(payload)


def test_overlay_record_requires_real_file_track_and_visible_observation():
    """Filename-shaped files must not manufacture track/frame provenance."""
    overlay_type = getattr(cv_contracts, "OverlayRecord")
    artifact = valid_artifact()
    payload = artifact.model_dump(mode="json")
    payload["files"].append(
        {"path": "overlays/opaque.png", "sha256": "d" * 64, "size_bytes": 8}
    )
    payload["processed_timeline"] = {
        "frames": [{"frame_index": 0, "timestamp_seconds": 0.0}]
    }
    payload["overlay_records"] = [
        overlay_type(path="overlays/opaque.png", track_id="cup_1", frame_index=0)
    ]

    validated = CvEvidenceArtifact.model_validate(payload)
    assert validated.overlay_records[0].track_id == "cup_1"

    for update in (
        {"track_id": "missing_1"},
        {"frame_index": 1},
        {"path": "overlays/unlisted.png"},
    ):
        forged = dict(payload)
        forged["overlay_records"] = [
            {**payload["overlay_records"][0].model_dump(mode="json"), **update}
        ]
        with pytest.raises(ValidationError):
            CvEvidenceArtifact.model_validate(forged)


def test_mask_and_overlay_namespaces_are_disjoint_and_overlay_is_one_file_deep():
    """Rendered prompt PNGs cannot also be interpreted as raw mask payloads."""
    with pytest.raises(ValidationError, match="mask_ref"):
        TrackObservation(
            frame_index=0,
            timestamp_seconds=0.0,
            bbox_xyxy=(0.1, 0.1, 0.2, 0.2),
            mask_ref="overlays/opaque.png",
            visible=True,
            confidence=0.9,
            area_fraction=0.01,
            center_xy=(0.15, 0.15),
        )

    with pytest.raises(ValidationError, match=r"overlays/\*\.png"):
        OverlayRecord(
            path="overlays/nested/opaque.png",
            track_id="cup_1",
            frame_index=0,
        )


@pytest.mark.parametrize("status", [EvidenceStatus.DISABLED, EvidenceStatus.UNAVAILABLE])
def test_nonavailable_artifacts_reject_all_processed_payload(status):
    """Both degradation states must carry no visual evidence or artifact files."""
    payload = valid_artifact().model_dump(mode="json")
    payload["status"] = status.value

    with pytest.raises(ValidationError, match="cannot contain processed evidence"):
        CvEvidenceArtifact.model_validate(payload)

    payload.update(
        processed_timeline=None,
        tracks=[],
        files=[],
        overlay_records=[],
    )
    assert CvEvidenceArtifact.model_validate(payload).status is status


def test_track_status_and_observation_coverage_are_consistent():
    """Available means observed, while degraded tracks cannot retain observations."""
    observation = valid_artifact().tracks[0].observations[0]
    with pytest.raises(ValidationError, match="available track"):
        CvTrack(
            track_id="cup_1",
            entity_id="cup",
            observations=(),
            status=EvidenceStatus.AVAILABLE,
        )
    with pytest.raises(ValidationError, match="nonavailable track"):
        CvTrack(
            track_id="cup_1",
            entity_id="cup",
            observations=(observation,),
            status=EvidenceStatus.UNAVAILABLE,
        )


def test_threshold_contract_rejects_integer_json_scalars():
    """Threshold types must not vary with Pydantic's integer-to-float coercion."""
    with pytest.raises(ValidationError):
        EvidenceThresholds.model_validate(
            {
                "min_confidence": 0,
                "min_area_fraction": 0.01,
                "occlusion_visibility_drop": 0.5,
            }
        )


def test_integer_contracts_reject_values_outside_the_canonical_bound():
    """Finite integer widths bound canonical serialization and comparisons."""
    with pytest.raises(ValidationError):
        FrameTimestamp(frame_index=2**63, timestamp_seconds=0.0)


def test_overlay_record_is_exported_as_a_public_cv_contract() -> None:
    """Consumers should not need to import a private implementation module."""
    assert cv_public.OverlayRecord is cv_contracts.OverlayRecord
    assert "OverlayRecord" in cv_public.__all__


def test_public_contracts_enforce_nested_resource_bounds_directly():
    """Direct model validation must reject attacker-sized text containers."""
    with pytest.raises(ValidationError):
        EntityPrompt(
            entity_id="cup",
            canonical_label="cup",
            aliases=("alias",) * 257,
            role=EntityRole.OTHER,
        )

    artifact = valid_artifact().model_dump(mode="json")
    artifact["processed_timeline"] = {
        "frames": [{"frame_index": 0, "timestamp_seconds": 0.0}]
    }
    artifact["overlay_records"] = []
    artifact["warnings"] = ["x" * 257]
    with pytest.raises(ValidationError):
        CvEvidenceArtifact.model_validate(artifact)

    valid = valid_artifact()
    observation_block = valid.tracks[0].observations * 10_000
    oversized_tracks = tuple(
        CvTrack.model_construct(
            track_id=f"cup_{ordinal}",
            entity_id="cup",
            observations=observation_block,
            status=EvidenceStatus.AVAILABLE,
        )
        for ordinal in range(1, 8)
    )
    oversized_payload = valid.model_dump(mode="python")
    oversized_payload["tracks"] = oversized_tracks
    with pytest.raises(ValidationError, match="total observations"):
        CvEvidenceArtifact.model_validate(oversized_payload)


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64])
def test_contracts_reject_noncanonical_sha256_digests(digest):
    """Relaxing the digest shape would make cache identity ambiguous."""
    with pytest.raises(ValidationError):
        CvEvidenceRequest.model_validate(
            {**valid_request().model_dump(), "video_sha256": digest}
        )
