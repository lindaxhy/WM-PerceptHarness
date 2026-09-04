from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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


SHA256 = "a" * 64


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
        entities=(entity,),
        tracks=(CvTrack(track_id="cup_1", entity_id="cup", observations=(observation,)),),
        files=(ArtifactFile(path="masks/0.npz", sha256="c" * 64, size_bytes=4),),
    )


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
            entities=(known,),
            tracks=duplicate_tracks,
            files=(ArtifactFile(path="masks/0.npz", sha256="c" * 64, size_bytes=4),),
        )
    with pytest.raises(ValidationError):
        CvEvidenceArtifact(
            schema_version="cv_evidence_v1",
            status=EvidenceStatus.AVAILABLE,
            provider="fake",
            model_identity="fake-sam31-v1",
            video_sha256=SHA256,
            checkpoint_sha256="b" * 64,
            entities=(known,),
            tracks=(CvTrack(track_id="other_1", entity_id="other", observations=(observation,)),),
            files=(ArtifactFile(path="masks/0.npz", sha256="c" * 64, size_bytes=4),),
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


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64])
def test_contracts_reject_noncanonical_sha256_digests(digest):
    """Relaxing the digest shape would make cache identity ambiguous."""
    with pytest.raises(ValidationError):
        CvEvidenceRequest.model_validate(
            {**valid_request().model_dump(), "video_sha256": digest}
        )
