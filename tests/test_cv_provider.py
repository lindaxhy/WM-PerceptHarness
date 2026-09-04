from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from las_repro.cv.base import CvEvidenceProvider, FakeCvEvidenceProvider
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
                FrameTimestamp(frame_index=5, timestamp_seconds=0.25),
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


def test_fake_provider_is_deterministic_and_writes_valid_artifact(
    cv_request: CvEvidenceRequest, tmp_path: Path
) -> None:
    """Nondeterministic tracks or files would make Fake-backed tests irreproducible."""
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"

    first_provider = FakeCvEvidenceProvider()
    second_provider = FakeCvEvidenceProvider()
    first = first_provider.analyze(cv_request, first_dir)
    second = second_provider.analyze(cv_request, second_dir)

    assert isinstance(first_provider, CvEvidenceProvider)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.status is EvidenceStatus.AVAILABLE
    assert {track.entity_id for track in first.tracks} == {"right_hand", "cup"}
    assert any(
        previous.visible and not current.visible
        for track in first.tracks
        for previous, current in zip(track.observations, track.observations[1:])
    )
    referenced = {
        observation.mask_ref
        for track in first.tracks
        for observation in track.observations
        if observation.mask_ref is not None
    }
    assert referenced == {artifact_file.path for artifact_file in first.files}
    for artifact_file in first.files:
        first_bytes = (first_dir / artifact_file.path).read_bytes()
        second_bytes = (second_dir / artifact_file.path).read_bytes()
        assert first_bytes == second_bytes
        assert len(first_bytes) == artifact_file.size_bytes
        assert hashlib.sha256(first_bytes).hexdigest() == artifact_file.sha256
    assert first_provider.request_metrics() == {
        "processed_frames": 3,
        "entity_prompts": 2,
        "track_count": 2,
        "peak_allocated_bytes": 0,
    }
