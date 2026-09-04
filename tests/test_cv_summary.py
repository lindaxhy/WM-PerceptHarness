from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import warnings

import pytest
from pydantic import ValidationError

from las_repro.cv.contracts import (
    ArtifactFile,
    CvEvidenceArtifact,
    CvTrack,
    EntityPrompt,
    EntityRole,
    EvidenceStatus,
    EvidenceThresholds,
    FrameTimeline,
    FrameTimestamp,
    TrackObservation,
)
import las_repro.cv.summary as summary_module
from las_repro.cv.summary import (
    CvEvidenceSummary,
    build_occlusion_candidates,
    summarize_cv_evidence,
)


SHA256 = "a" * 64


def _entity(
    entity_id: str,
    *,
    role: EntityRole = EntityRole.OTHER,
    aliases: tuple[str, ...] = (),
) -> EntityPrompt:
    return EntityPrompt(
        entity_id=entity_id,
        canonical_label=entity_id.replace("_", " "),
        aliases=aliases,
        role=role,
    )


def _observation(
    frame_index: int,
    *,
    timestamp_seconds: float | None = None,
    bbox_xyxy: tuple[float, float, float, float] = (0.2, 0.2, 0.4, 0.4),
    visible: bool = True,
    confidence: float = 0.9,
    area_fraction: float | None = None,
    mask_ref: str | None = None,
) -> TrackObservation:
    left, top, right, bottom = bbox_xyxy
    return TrackObservation(
        frame_index=frame_index,
        timestamp_seconds=(
            frame_index / 10.0
            if timestamp_seconds is None
            else timestamp_seconds
        ),
        bbox_xyxy=bbox_xyxy,
        mask_ref=mask_ref,
        visible=visible,
        confidence=confidence,
        area_fraction=(
            (right - left) * (bottom - top)
            if area_fraction is None
            else area_fraction
        ),
        center_xy=((left + right) / 2.0, (top + bottom) / 2.0),
    )


def _track(
    track_id: str,
    entity_id: str,
    observations: tuple[TrackObservation, ...],
) -> CvTrack:
    return CvTrack(
        track_id=track_id,
        entity_id=entity_id,
        observations=observations,
    )


def _artifact(
    tracks: tuple[CvTrack, ...],
    *,
    entities: tuple[EntityPrompt, ...] | None = None,
    files: tuple[ArtifactFile, ...] = (),
    warnings: tuple[str, ...] = (),
) -> CvEvidenceArtifact:
    if entities is None:
        entity_ids = sorted({track.entity_id for track in tracks})
        entities = tuple(_entity(entity_id) for entity_id in entity_ids)
    return CvEvidenceArtifact(
        schema_version="cv_evidence_v1",
        status=EvidenceStatus.AVAILABLE,
        provider="fake",
        model_identity="fake-sam31-v1",
        video_sha256=SHA256,
        checkpoint_sha256="b" * 64,
        entities=entities,
        tracks=tracks,
        files=files,
        warnings=warnings,
    )


def _unsafe_artifact_copy(
    artifact: CvEvidenceArtifact, **updates: object
) -> CvEvidenceArtifact:
    values = {
        field_name: getattr(artifact, field_name)
        for field_name in type(artifact).model_fields
    }
    values.update(updates)
    return CvEvidenceArtifact.model_construct(**values)


def _timeline(*frame_indices: int) -> FrameTimeline:
    return FrameTimeline(
        frames=tuple(
            FrameTimestamp(
                frame_index=frame_index,
                timestamp_seconds=frame_index / 10.0,
            )
            for frame_index in frame_indices
        )
    )


def _thresholds(
    *,
    min_confidence: float = 0.5,
    min_area_fraction: float = 0.01,
    visibility_drop: float = 0.5,
) -> EvidenceThresholds:
    return EvidenceThresholds(
        min_confidence=min_confidence,
        min_area_fraction=min_area_fraction,
        occlusion_visibility_drop=visibility_drop,
    )


def test_summary_contains_no_masks_reads_no_artifacts_and_is_size_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing mask references through, or reading them, would breach the boundary."""
    mask_file = ArtifactFile(
        path="masks/cup.npz", sha256="c" * 64, size_bytes=10_000
    )
    overlay_file = ArtifactFile(
        path="overlays/cup-1-00000000.png",
        sha256="d" * 64,
        size_bytes=100,
    )
    artifact = _artifact(
        (
            _track(
                "cup_1",
                "cup",
                (
                    _observation(0, mask_ref=mask_file.path),
                    _observation(1, mask_ref=mask_file.path),
                ),
            ),
        ),
        files=(mask_file, overlay_file),
    )

    def refuse_file_read(*_: object, **__: object) -> bytes:
        raise AssertionError("summary must not read artifact payloads")

    monkeypatch.setattr(Path, "read_bytes", refuse_file_read)
    summary = summarize_cv_evidence(
        artifact,
        max_tracks=64,
        max_observations_per_track=64,
    )

    encoded = summary.model_dump_json()
    prompt_encoded = json.dumps(
        summary.prompt_record(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "mask" not in encoded.casefold()
    assert "mask" not in prompt_encoded.casefold()
    assert len(encoded) <= 200_000
    assert len(prompt_encoded) <= 200_000
    assert summary.overlay_refs == ("overlays/cup-1-00000000.png",)
    assert set(summary.prompt_record()) == {
        "schema_version",
        "status",
        "entities",
        "tracks",
        "relations",
        "overlay_refs",
        "warnings",
    }


def test_observation_reduction_retains_landmarks_and_state_change_boundaries() -> None:
    """Dropping a mandatory landmark would hide a real visibility or quality change."""
    observations = (
        _observation(0, area_fraction=0.30, confidence=0.8),
        _observation(1, area_fraction=0.05, confidence=0.8),
        _observation(2, area_fraction=0.80, confidence=0.8),
        _observation(3, area_fraction=0.30, confidence=0.1),
        _observation(4, area_fraction=0.30, confidence=0.8, visible=False),
        _observation(5, area_fraction=0.30, confidence=0.8, visible=False),
        _observation(6, area_fraction=0.30, confidence=0.8, visible=True),
        _observation(7, area_fraction=0.40, confidence=0.8),
        _observation(8, area_fraction=0.40, confidence=0.8),
        _observation(9, area_fraction=0.40, confidence=0.8),
    )

    summary = summarize_cv_evidence(
        _artifact((_track("target_1", "target", observations),)),
        timeline=_timeline(*range(10)),
        max_observations_per_track=8,
    )

    assert [item.frame_index for item in summary.tracks[0].observations] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        9,
    ]
    assert any(
        warning.startswith("OBSERVATIONS_TRUNCATED:")
        for warning in summary.warnings
    )


def test_cap_smaller_than_mandatory_landmarks_uses_declared_stable_priority() -> None:
    """Silently exceeding a small cap would make the prompt bound untrustworthy."""
    observations = (
        _observation(0, area_fraction=0.30),
        _observation(1, area_fraction=0.05),
        _observation(2, area_fraction=0.80),
        _observation(3, confidence=0.1),
        _observation(4, visible=False),
        _observation(5, visible=True),
        _observation(6),
    )

    summary = summarize_cv_evidence(
        _artifact((_track("target_1", "target", observations),)),
        timeline=_timeline(*range(7)),
        max_observations_per_track=3,
    )

    # Priority is first, last, then chronological visibility-change boundaries;
    # extrema follow only if capacity remains.
    assert [item.frame_index for item in summary.tracks[0].observations] == [0, 3, 6]
    assert any(
        warning.startswith("MANDATORY_LANDMARKS_TRUNCATED:")
        and "priority=first,last,state_changes,min_area,max_area,lowest_confidence"
        in warning
        for warning in summary.warnings
    )


def test_remaining_observation_slots_are_filled_by_stable_temporal_sampling() -> None:
    """Filling from one end would discard the temporal shape of a long track."""
    observations = tuple(
        _observation(index, area_fraction=0.1, confidence=0.9)
        for index in range(20)
    )
    summary = summarize_cv_evidence(
        _artifact((_track("target_1", "target", observations),)),
        timeline=_timeline(*range(20)),
        max_observations_per_track=6,
    )

    retained = [item.frame_index for item in summary.tracks[0].observations]
    assert retained[0] == 0
    assert retained[-1] == 19
    assert len(retained) == 6
    assert len(set(retained)) == 6
    assert any(3 <= frame <= 6 for frame in retained)
    assert any(8 <= frame <= 11 for frame in retained)
    assert any(13 <= frame <= 16 for frame in retained)


def test_summary_is_stable_across_equivalent_entity_track_and_file_ordering() -> None:
    """Depending on provider container order would change prompts and candidate IDs."""
    first_entity = _entity("alpha", aliases=("first", "a"))
    second_entity = _entity("beta", aliases=("b", "second"))
    first_track = _track(
        "alpha_1", "alpha", (_observation(0), _observation(2))
    )
    second_track = _track(
        "beta_1",
        "beta",
        (
            _observation(0, bbox_xyxy=(0.3, 0.3, 0.5, 0.5)),
            _observation(2, bbox_xyxy=(0.3, 0.3, 0.5, 0.5)),
        ),
    )
    files = (
        ArtifactFile(
            path="overlays/beta-1-00000002.png", sha256="c" * 64, size_bytes=1
        ),
        ArtifactFile(
            path="overlays/alpha-1-00000000.png", sha256="d" * 64, size_bytes=1
        ),
    )
    first = _artifact(
        (second_track, first_track),
        entities=(second_entity, first_entity),
        files=files,
    )
    second = _artifact(
        (first_track, second_track),
        entities=(
            first_entity.model_copy(
                update={"aliases": tuple(reversed(first_entity.aliases))}
            ),
            second_entity.model_copy(
                update={"aliases": tuple(reversed(second_entity.aliases))}
            ),
        ),
        files=tuple(reversed(files)),
    )

    left = summarize_cv_evidence(first, timeline=_timeline(0, 1, 2))
    right = summarize_cv_evidence(second, timeline=_timeline(0, 1, 2))

    assert left.model_dump_json() == right.model_dump_json()


def test_casefold_duplicate_aliases_have_an_order_independent_representative() -> None:
    """Keeping the first case variant would make equivalent alias sets unstable."""
    track = _track("target_1", "target", (_observation(0),))
    left = summarize_cv_evidence(
        _artifact(
            (track,),
            entities=(_entity("target", aliases=("Cup", "cup")),),
        )
    )
    right = summarize_cv_evidence(
        _artifact(
            (track,),
            entities=(_entity("target", aliases=("cup", "Cup")),),
        )
    )

    assert left.model_dump_json() == right.model_dump_json()
    assert left.entities[0].aliases == ("Cup",)


def test_relations_use_only_same_frame_geometry_and_are_capped() -> None:
    """Comparing arbitrary frames would fabricate overlap that was never observed."""
    alpha = _track(
        "alpha_1",
        "alpha",
        (
            _observation(0, bbox_xyxy=(0.1, 0.1, 0.5, 0.5)),
            _observation(2, bbox_xyxy=(0.7, 0.7, 0.9, 0.9)),
        ),
    )
    beta = _track(
        "beta_1",
        "beta",
        (
            _observation(0, bbox_xyxy=(0.2, 0.2, 0.6, 0.6)),
            _observation(1, bbox_xyxy=(0.7, 0.7, 0.9, 0.9)),
        ),
    )
    summary = summarize_cv_evidence(
        _artifact((beta, alpha)),
        timeline=_timeline(0, 1, 2),
        max_relations=1,
    )

    assert len(summary.relations) == 1
    relation = summary.relations[0]
    assert relation.frame_index == 0
    assert relation.timestamp_seconds == 0.0
    assert relation.subject_track_id == "alpha_1"
    assert relation.object_track_id == "beta_1"
    for value in (
        relation.bbox_iou,
        relation.subject_bbox_covered_fraction,
        relation.object_bbox_covered_fraction,
        relation.center_distance_fraction,
        relation.area_similarity,
    ):
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert any(
        warning.startswith("RELATIONS_TRUNCATED:")
        for warning in summary.warnings
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_tracks", 0),
        ("max_tracks", True),
        ("max_observations_per_track", -1),
        ("max_relations", 0),
        ("max_overlays", 0),
        ("max_overlays", 25),
        ("max_prompt_chars", 0),
        ("max_prompt_chars", 200_001),
    ],
)
def test_summary_limits_fail_closed(name: str, value: object) -> None:
    """Accepting invalid caps would permit unbounded or ambiguous prompt records."""
    artifact = _artifact(
        (_track("target_1", "target", (_observation(0),)),)
    )

    with pytest.raises((TypeError, ValueError)):
        summarize_cv_evidence(artifact, **{name: value})


@pytest.mark.parametrize(
    "replacement",
    [
        {"timestamp_seconds": float("nan")},
        {"timestamp_seconds": float("inf")},
        {"bbox_xyxy": (0.1, 0.1, float("inf"), 0.5)},
        {"bbox_xyxy": (0.8, 0.1, 0.2, 0.5)},
        {"center_xy": (float("nan"), 0.3)},
        {"area_fraction": float("inf")},
        {"confidence": float("nan")},
    ],
)
def test_summary_revalidates_unsafe_observation_geometry(
    replacement: dict[str, object],
) -> None:
    """Trusting model_construct would let non-finite geometry poison ordering."""
    valid = _observation(0)
    bad = TrackObservation.model_construct(
        **{**valid.model_dump(mode="python"), **replacement}
    )
    track = CvTrack.model_construct(
        track_id="target_1",
        entity_id="target",
        observations=(bad,),
        status=EvidenceStatus.AVAILABLE,
    )
    artifact = _unsafe_artifact_copy(
        _artifact((_track("target_1", "target", (valid,)),)),
        tracks=(track,),
    )

    with pytest.raises(ValidationError):
        summarize_cv_evidence(artifact)


@pytest.mark.parametrize(
    "observations",
    [
        (_observation(1), _observation(1)),
        (_observation(2), _observation(1)),
        (
            _observation(1, timestamp_seconds=0.2),
            _observation(2, timestamp_seconds=0.1),
        ),
    ],
)
def test_summary_revalidates_unsafe_observation_order_and_duplicates(
    observations: tuple[TrackObservation, ...],
) -> None:
    """Accepting repeated or reversed frames would make lifecycle gaps unstable."""
    bad_track = CvTrack.model_construct(
        track_id="target_1",
        entity_id="target",
        observations=observations,
        status=EvidenceStatus.AVAILABLE,
    )
    valid_artifact = _artifact(
        (_track("target_1", "target", (_observation(0),)),)
    )
    constructed = _unsafe_artifact_copy(
        valid_artifact,
        tracks=(bad_track,),
    )

    with pytest.raises(ValidationError):
        summarize_cv_evidence(constructed)


def test_unsafe_values_are_rejected_without_leaking_serializer_warnings() -> None:
    """Pydantic warning reprs must not echo malformed provider values or paths."""
    artifact = _artifact(
        (_track("target_1", "target", (_observation(0),)),)
    )
    constructed = _unsafe_artifact_copy(
        artifact,
        tracks=({"secret_path": "/private/provider/checkpoint"},),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError):
            summarize_cv_evidence(constructed)

    assert caught == []


def test_summary_rejects_conflicting_global_frame_times_and_timeline_mismatch() -> None:
    """One frame cannot carry two PTS values or be remapped to a uniform clock."""
    first = _track(
        "alpha_1", "alpha", (_observation(0, timestamp_seconds=0.0),)
    )
    conflicting = _track(
        "beta_1", "beta", (_observation(0, timestamp_seconds=0.01),)
    )

    with pytest.raises(ValueError, match="frame timeline"):
        summarize_cv_evidence(_artifact((first, conflicting)))

    aligned_artifact = _artifact((first,))
    mismatched = FrameTimeline(
        frames=(FrameTimestamp(frame_index=0, timestamp_seconds=0.01),)
    )
    with pytest.raises(ValueError, match="frame timeline"):
        summarize_cv_evidence(aligned_artifact, timeline=mismatched)


def test_summary_revalidates_unsafe_overlay_paths_and_allowlists_png_overlays() -> None:
    """An unsafe or non-overlay file must never become a prompt-visible reference."""
    artifact = _artifact(
        (_track("target_1", "target", (_observation(0),)),),
        files=(
            ArtifactFile(
                path="overlays/target-1-00000000.png",
                sha256="c" * 64,
                size_bytes=1,
            ),
            ArtifactFile(path="notes/context.json", sha256="d" * 64, size_bytes=1),
            ArtifactFile(path="masks/preview.png", sha256="e" * 64, size_bytes=1),
        ),
    )
    summary = summarize_cv_evidence(artifact)
    assert summary.overlay_refs == ("overlays/target-1-00000000.png",)

    malicious_file = ArtifactFile.model_construct(
        path="../outside.png", sha256="f" * 64, size_bytes=1
    )
    constructed = _unsafe_artifact_copy(
        artifact,
        files=(malicious_file,),
    )
    with pytest.raises(ValidationError):
        summarize_cv_evidence(constructed)


def test_huge_tracks_aliases_observations_and_overlays_remain_bounded() -> None:
    """Caller caps and prompt budgeting must survive adversarially large manifests."""
    entities = tuple(
        _entity(
            f"entity_{entity_index}",
            aliases=tuple(
                f"alias-{entity_index}-{alias_index}-" + ("x" * 2_000)
                for alias_index in range(80)
            ),
        )
        for entity_index in range(30)
    )
    tracks = tuple(
        _track(
            f"entity_{entity_index}_1",
            f"entity_{entity_index}",
            tuple(
                _observation(
                    frame_index,
                    bbox_xyxy=(
                        0.01 * (entity_index % 10),
                        0.01 * (entity_index % 10),
                        0.2 + 0.01 * (entity_index % 10),
                        0.2 + 0.01 * (entity_index % 10),
                    ),
                )
                for frame_index in range(40)
            ),
        )
        for entity_index in range(30)
    )
    files = tuple(
        ArtifactFile(
            path=f"overlays/entity-{index:05d}.png",
            sha256=f"{index % 16:x}" * 64,
            size_bytes=1,
        )
        for index in range(500)
    )

    summary = summarize_cv_evidence(
        _artifact(tracks, entities=entities, files=files),
        timeline=_timeline(*range(40)),
        max_tracks=12,
        max_observations_per_track=12,
        max_relations=20,
        max_overlays=7,
        max_prompt_chars=10_000,
    )

    assert len(summary.tracks) <= 12
    assert all(len(track.observations) <= 12 for track in summary.tracks)
    assert len(summary.relations) <= 20
    assert len(summary.overlay_refs) <= 7
    assert len(summary.model_dump_json()) <= 10_000
    assert len(
        json.dumps(
            summary.prompt_record(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    ) <= 10_000
    warning_codes = {warning.split(":", 1)[0] for warning in summary.warnings}
    assert {
        "ALIASES_TRUNCATED",
        "TRACKS_TRUNCATED",
        "OBSERVATIONS_TRUNCATED",
        "RELATIONS_TRUNCATED",
        "OVERLAYS_TRUNCATED",
        "PROMPT_DATA_TRUNCATED",
    } <= warning_codes


def test_summary_and_prompt_records_are_deeply_immutable_and_fresh() -> None:
    """Mutating trusted prompt data must not mutate the frozen evidence model."""
    summary = summarize_cv_evidence(
        _artifact(
            (
                _track(
                    "target_1",
                    "target",
                    (_observation(0), _observation(1)),
                ),
            ),
            entities=(_entity("target", aliases=("item",)),),
        )
    )
    first = summary.prompt_record()
    second = summary.prompt_record()

    first["entities"][0]["aliases"].append("injected")
    first["tracks"][0]["observations"][0]["bbox_xyxy"][0] = 0.99
    first["warnings"].append("injected")

    assert second == summary.prompt_record()
    assert "injected" not in summary.model_dump_json()
    with pytest.raises(ValidationError):
        summary.status = EvidenceStatus.UNAVAILABLE  # type: ignore[misc]


def _candidate_artifact() -> tuple[CvEvidenceArtifact, FrameTimeline]:
    frames = tuple(range(5))
    central = (0.30, 0.30, 0.50, 0.50)
    overlapping = (0.35, 0.25, 0.65, 0.55)
    separate = (0.70, 0.70, 0.90, 0.90)
    far_left = (0.00, 0.70, 0.10, 0.90)
    edge = (0.86, 0.30, 1.00, 0.50)
    entities = (
        _entity("behind_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("edge_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("flicker_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("partial_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("permanent_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("no_occluder_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("stable_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("never_visible_target", role=EntityRole.MANIPULATED_OBJECT),
        _entity("board", role=EntityRole.OCCLUDER),
        _entity("distant_board", role=EntityRole.OCCLUDER),
    )
    tracks = (
        _track(
            "behind_target_1",
            "behind_target",
            tuple(_observation(index, bbox_xyxy=central) for index in (0, 1, 3, 4)),
        ),
        _track(
            "edge_target_1",
            "edge_target",
            tuple(_observation(index, bbox_xyxy=edge) for index in (0, 1)),
        ),
        _track(
            "flicker_target_1",
            "flicker_target",
            tuple(
                _observation(
                    index,
                    bbox_xyxy=central,
                    confidence=0.1 if index == 2 else 0.9,
                )
                for index in frames
            ),
        ),
        _track(
            "partial_target_1",
            "partial_target",
            tuple(
                _observation(
                    index,
                    bbox_xyxy=central,
                    area_fraction=0.04 if index == 2 else 0.16,
                )
                for index in frames
            ),
        ),
        _track(
            "permanent_target_1",
            "permanent_target",
            tuple(_observation(index, bbox_xyxy=central) for index in (0, 1)),
        ),
        _track(
            "no_occluder_target_1",
            "no_occluder_target",
            tuple(_observation(index, bbox_xyxy=separate) for index in (0, 1, 3, 4)),
        ),
        _track(
            "stable_target_1",
            "stable_target",
            tuple(_observation(index, bbox_xyxy=central) for index in frames),
        ),
        _track(
            "never_visible_target_1",
            "never_visible_target",
            tuple(
                _observation(index, bbox_xyxy=central, visible=False)
                for index in frames
            ),
        ),
        _track(
            "board_1",
            "board",
            tuple(_observation(index, bbox_xyxy=overlapping) for index in frames),
        ),
        _track(
            "distant_board_1",
            "distant_board",
            tuple(_observation(index, bbox_xyxy=far_left) for index in frames),
        ),
    )
    overlays = tuple(
        ArtifactFile(
            path=f"overlays/{entity_id}-1-{frame_index:08d}.png",
            sha256=f"{ordinal % 16:x}" * 64,
            size_bytes=1,
        )
        for ordinal, (entity_id, frame_index) in enumerate(
            (
                ("behind_target", 1),
                ("behind_target", 3),
                ("board", 1),
                ("board", 3),
                ("unrelated", 4),
            ),
            start=1,
        )
    )
    return _artifact(tracks, entities=entities, files=overlays), _timeline(*frames)


def test_candidate_generation_preserves_evidence_and_counter_signals() -> None:
    """Collapsing geometry and counter-signals would turn proposals into claims."""
    artifact, timeline = _candidate_artifact()
    summary = summarize_cv_evidence(
        artifact,
        timeline=timeline,
        max_tracks=64,
        max_observations_per_track=64,
        max_relations=1_024,
    )

    candidates = build_occlusion_candidates(summary, _thresholds())
    by_target = {candidate.target_entity_id: candidate for candidate in candidates}

    assert set(by_target) == {
        "behind_target",
        "edge_target",
        "flicker_target",
        "partial_target",
        "permanent_target",
        "no_occluder_target",
    }
    assert "board" in by_target["behind_target"].possible_occluder_entity_ids
    assert by_target["behind_target"].last_visible_frame == 1
    assert by_target["behind_target"].first_revisible_frame == 3
    assert by_target["behind_target"].allowed_start_times == (0.1, 0.2)
    assert by_target["behind_target"].allowed_end_times == (0.2, 0.3)
    assert by_target["behind_target"].overlay_refs == (
        "overlays/behind_target-1-00000001.png",
        "overlays/behind_target-1-00000003.png",
        "overlays/board-1-00000001.png",
        "overlays/board-1-00000003.png",
    )
    assert by_target["edge_target"].edge_departure is True
    assert by_target["edge_target"].first_revisible_frame is None
    assert by_target["flicker_target"].low_confidence is True
    assert by_target["partial_target"].low_confidence is False
    assert by_target["permanent_target"].first_revisible_frame is None
    assert by_target["permanent_target"].edge_departure is False
    assert by_target["no_occluder_target"].possible_occluder_entity_ids == ()
    assert "classification" not in type(by_target["behind_target"]).model_fields
    assert "stable_target" not in by_target
    assert "never_visible_target" not in by_target


def test_absence_does_not_invent_an_occluder_or_positive_classification() -> None:
    """A lifecycle gap alone must remain an unresolved candidate, never occlusion."""
    target = _track(
        "target_1", "target", (_observation(0), _observation(2))
    )
    clock = _track(
        "clock_1",
        "clock",
        tuple(
            _observation(
                frame,
                bbox_xyxy=(0.7, 0.7, 0.9, 0.9),
            )
            for frame in range(3)
        ),
    )
    summary = summarize_cv_evidence(
        _artifact((target, clock)), timeline=_timeline(0, 1, 2)
    )

    [candidate] = [
        item
        for item in build_occlusion_candidates(summary, _thresholds())
        if item.target_entity_id == "target"
    ]
    assert candidate.possible_occluder_entity_ids == ()
    assert set(candidate.prompt_record()) == {
        "candidate_id",
        "target_entity_id",
        "possible_occluder_entity_ids",
        "allowed_start_times",
        "allowed_end_times",
        "last_visible_frame",
        "first_revisible_frame",
        "edge_departure",
        "low_confidence",
        "overlay_refs",
    }
    assert "classification" not in candidate.prompt_record()


def test_explicit_missing_interval_retains_low_confidence_counter_signal() -> None:
    """A low-confidence miss must stay visible to the later adjudicator."""
    target = _track(
        "target_1",
        "target",
        (
            _observation(0, confidence=0.9),
            _observation(1, visible=False, confidence=0.1),
            _observation(2, confidence=0.9),
        ),
    )
    summary = summarize_cv_evidence(
        _artifact((target,)), timeline=_timeline(0, 1, 2)
    )

    [candidate] = build_occlusion_candidates(summary, _thresholds())
    assert candidate.low_confidence is True


def test_any_temporally_overlapping_entity_can_remain_a_possible_occluder() -> None:
    """Entity role alone must not semantically discard aligned overlap evidence."""
    target = _track(
        "target_1", "target", (_observation(0), _observation(2))
    )
    other = _track(
        "other_object_1",
        "other_object",
        tuple(
            _observation(frame, bbox_xyxy=(0.25, 0.25, 0.45, 0.45))
            for frame in range(3)
        ),
    )
    summary = summarize_cv_evidence(
        _artifact(
            (target, other),
            entities=(
                _entity("target", role=EntityRole.MANIPULATED_OBJECT),
                _entity("other_object", role=EntityRole.MANIPULATED_OBJECT),
            ),
        ),
        timeline=_timeline(0, 1, 2),
    )

    target_candidate = next(
        item
        for item in build_occlusion_candidates(summary, _thresholds())
        if item.target_entity_id == "target"
    )
    assert target_candidate.possible_occluder_entity_ids == ("other_object",)


def test_possible_occluders_require_temporally_aligned_bbox_evidence() -> None:
    """A box seen only at another frame cannot support an occluder relationship."""
    target = _track(
        "target_1",
        "target",
        (
            _observation(0, bbox_xyxy=(0.2, 0.2, 0.4, 0.4)),
            _observation(2, bbox_xyxy=(0.2, 0.2, 0.4, 0.4)),
        ),
    )
    other = _track(
        "board_1",
        "board",
        (
            _observation(1, bbox_xyxy=(0.2, 0.2, 0.4, 0.4)),
            _observation(2, bbox_xyxy=(0.7, 0.7, 0.9, 0.9)),
        ),
    )
    summary = summarize_cv_evidence(
        _artifact((target, other)), timeline=_timeline(0, 1, 2)
    )

    target_candidate = next(
        item
        for item in build_occlusion_candidates(summary, _thresholds())
        if item.target_entity_id == "target"
    )
    assert target_candidate.possible_occluder_entity_ids == ()


def test_candidate_ids_and_order_survive_hash_prefix_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hash-prefix collision must not duplicate IDs or expose input ordering."""
    artifact, timeline = _candidate_artifact()
    summary = summarize_cv_evidence(artifact, timeline=timeline)
    monkeypatch.setattr(
        summary_module,
        "_candidate_hash_prefix",
        lambda _: "deadbeefcafe",
    )

    first = build_occlusion_candidates(summary, _thresholds())
    reconstructed = CvEvidenceSummary.model_validate(
        {
            **summary.model_dump(mode="python"),
            "tracks": tuple(reversed(summary.tracks)),
            "relations": tuple(reversed(summary.relations)),
        },
        strict=True,
    )
    second = build_occlusion_candidates(reconstructed, _thresholds())

    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]
    assert [item.candidate_id for item in first] == [
        f"occ_deadbeefcafe_{ordinal:04d}"
        for ordinal in range(1, len(first) + 1)
    ]
    assert list(first) == sorted(
        first,
        key=lambda item: (
            item.target_entity_id,
            item.last_visible_frame,
            item.first_revisible_frame
            if item.first_revisible_frame is not None
            else sys.maxsize,
            item.candidate_id,
        ),
    )


def test_candidate_prompt_record_is_fresh_and_model_is_frozen() -> None:
    """A later renderer must not mutate a validated candidate through prompt data."""
    artifact, timeline = _candidate_artifact()
    candidate = build_occlusion_candidates(
        summarize_cv_evidence(artifact, timeline=timeline), _thresholds()
    )[0]
    first = candidate.prompt_record()
    second = candidate.prompt_record()

    first["possible_occluder_entity_ids"].append("injected")
    first["allowed_start_times"].append(99.0)
    first["overlay_refs"].append("../outside.png")

    assert second == candidate.prompt_record()
    assert "injected" not in candidate.model_dump_json()
    assert "outside" not in candidate.model_dump_json()
    with pytest.raises(ValidationError):
        candidate.edge_departure = not candidate.edge_departure  # type: ignore[misc]


def test_candidate_builder_revalidates_unsafe_thresholds() -> None:
    """NaN thresholds would make candidate classification order-dependent."""
    artifact, timeline = _candidate_artifact()
    summary = summarize_cv_evidence(artifact, timeline=timeline)
    unsafe = EvidenceThresholds.model_construct(
        min_confidence=float("nan"),
        min_area_fraction=0.01,
        occlusion_visibility_drop=0.5,
    )

    with pytest.raises(ValidationError):
        build_occlusion_candidates(summary, unsafe)


def test_candidate_builder_rejects_unsafe_cross_track_frame_time_conflicts() -> None:
    """A constructed summary must not align boxes that disagree on one frame's PTS."""
    summary = summarize_cv_evidence(
        _artifact(
            (
                _track("alpha_1", "alpha", (_observation(0),)),
                _track("beta_1", "beta", (_observation(0),)),
            )
        )
    )
    bad_observation = summary.tracks[1].observations[0].model_copy(
        update={"timestamp_seconds": 0.01}
    )
    bad_track = summary.tracks[1].model_copy(
        update={"observations": (bad_observation,)}
    )
    constructed = summary.model_copy(
        update={"tracks": (summary.tracks[0], bad_track), "relations": ()}
    )

    with pytest.raises((ValidationError, ValueError), match="frame timeline"):
        build_occlusion_candidates(constructed, _thresholds())


def test_summary_module_imports_no_gpu_or_sam_runtime() -> None:
    """Importing prompt summarization must remain safe in the core environment."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import las_repro.cv.summary; "
                "assert 'torch' not in sys.modules; "
                "assert not any(name == 'sam3' or name.startswith('sam3.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
