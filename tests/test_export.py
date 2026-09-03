"""Contract tests for export of completed embodied action results."""

from __future__ import annotations

import copy
import errno
import math
import stat
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

import las_repro.export as export_module
from las_repro.export import (
    ActionCaption,
    ActionCaptionExportError,
    iter_action_captions,
    write_action_captions_jsonl,
)


@pytest.fixture
def embodied_result() -> dict[str, object]:
    """A completed, locally validated 0805 action result."""
    return {
        "task_description": "move the red container",
        "segments": [
            {
                "action_index": 0,
                "segment_index": 0,
                "start": 0.0,
                "end": 0.41,
                "description": "right hand reaches the red container",
                "event_type": "approach",
                "start_boundary_id": "a0-start",
                "end_boundary_id": "a0-middle",
                "actor": "right_gripper",
                "actor_state": "reaching",
                "skill": "reach",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.9,
            },
            {
                "action_index": 0,
                "segment_index": 1,
                "start": 0.41,
                "end": 1.0,
                "description": "right hand grasps the red container",
                "event_type": "grasp_secured",
                "start_boundary_id": "a0-middle",
                "end_boundary_id": "a0-end",
                "actor": "right_gripper",
                "actor_state": "grasping",
                "skill": "grasp",
                "target": "red container",
                "visual_motion_state": "active",
                "confidence": 0.95,
            },
        ],
    }


def test_export_uses_half_open_frame_spans_stable_ids_and_current_schema(
    embodied_result: dict[str, object],
) -> None:
    """Changing ordering, IDs, frame arithmetic, or field ownership must fail."""
    rows = list(iter_action_captions("video_0001", embodied_result, source_fps=30.0))

    assert [row.caption_id for row in rows] == [
        "video_0001_cap_0000",
        "video_0001_cap_0001",
    ]
    assert [(row.start_frame, row.end_frame, row.duration_frames) for row in rows] == [
        (0, 12, 12),
        (12, 30, 18),
    ]
    assert rows[0].annotation_stage == "boundary_fine_segments_0805"
    assert rows[0].caption == "right hand reaches the red container"
    assert rows[0].caption_char_count == 36
    assert rows[0].source_segment_index == 0
    assert rows[0].parent_source_segment_index == 0
    assert rows[0].needs_refinement is False
    assert rows[0].over_caption_char_limit is False
    assert (
        rows[0].source_description
        == "actor=right_gripper | actor_state=reaching | skill=reach | "
        "target=red container | visual_motion_state=active | "
        "action=right hand reaches the red container | confidence=0.9"
    )
    assert rows[0].schema_parse_ok is True
    assert rows[0].parse_errors == ()
    assert rows[0].needs_review is False


def test_export_accepts_canonical_enum_normalization_warning_without_mutation(
    embodied_result: dict[str, object],
) -> None:
    """Dropping the canonical pipeline warning must not block normalized exports."""
    normalized = copy.deepcopy(embodied_result)
    segments = normalized["segments"]
    assert isinstance(segments, list)
    segments[0]["actor_state"] = "unknown"
    segments[0]["skill"] = "unknown"
    segments[1]["actor_state"] = "unknown"
    normalized["warnings"] = [
        {
            "code": "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN",
            "fields": ["actor_state", "skill"],
            "count": 3,
        }
    ]
    original = copy.deepcopy(normalized)

    rows = list(iter_action_captions("video_0001", normalized, source_fps=30.0))

    assert [(row.actor_state.value, row.skill.value) for row in rows] == [
        ("unknown", "unknown"),
        ("unknown", "grasp"),
    ]
    assert normalized == original


def test_export_rejects_noncanonical_warning_code_type_without_leaking_content(
    embodied_result: dict[str, object],
) -> None:
    """A string subclass must not smuggle behavior through the warning schema."""

    class CodeSubclass(str):
        pass

    normalized = copy.deepcopy(embodied_result)
    segments = normalized["segments"]
    assert isinstance(segments, list)
    segments[0]["actor_state"] = "unknown"
    normalized["warnings"] = [
        {
            "code": CodeSubclass("ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN"),
            "fields": ["actor_state"],
            "count": 1,
        }
    ]
    original = copy.deepcopy(normalized)

    with pytest.raises(ActionCaptionExportError) as error:
        list(iter_action_captions("video_0001", normalized, source_fps=30.0))

    assert str(error.value) == "completed embodied result is invalid"
    assert normalized == original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update({"private_payload": "DO NOT LEAK"}),
        lambda result: result.update({"warnings": []}),
        lambda result: result.update(
            {"warnings": [result["warnings"][0], result["warnings"][0]]}
        ),
        lambda result: result.update({"warnings": {"code": "DO NOT LEAK"}}),
        lambda result: result["warnings"][0].update({"private_payload": "DO NOT LEAK"}),
        lambda result: result["warnings"][0].update({"code": "DO NOT LEAK"}),
        lambda result: result["warnings"][0].update({"fields": []}),
        lambda result: result["warnings"][0].update(
            {"fields": ["actor_state", "actor_state"]}
        ),
        lambda result: result["warnings"][0].update(
            {"fields": ["skill", "actor_state"]}
        ),
        lambda result: result["warnings"][0].update({"fields": ["target"]}),
        lambda result: result["warnings"][0].update({"count": True}),
        lambda result: result["warnings"][0].update({"count": 0}),
        lambda result: result["warnings"][0].update({"count": 1}),
        lambda result: result["warnings"][0].update({"count": 5}),
        lambda result: result["warnings"][0].update({"count": 4}),
        lambda result: result["warnings"][0].update(
            {"fields": ["actor"], "count": 1}
        ),
    ],
    ids=[
        "unknown-result-key",
        "empty-warning-list",
        "multiple-warnings",
        "warning-not-list",
        "extra-warning-key",
        "unknown-code",
        "empty-fields",
        "duplicate-fields",
        "noncanonical-field-order",
        "unsupported-field",
        "boolean-count",
        "zero-count",
        "count-below-field-count",
        "count-above-segment-capacity",
        "count-above-observed-unknowns",
        "field-without-unknown",
    ],
)
def test_export_rejects_malformed_normalization_warning_without_mutation_or_leak(
    embodied_result: dict[str, object],
    mutation: object,
) -> None:
    """Only the closed pipeline warning envelope may cross the export boundary."""
    normalized = copy.deepcopy(embodied_result)
    segments = normalized["segments"]
    assert isinstance(segments, list)
    segments[0]["actor_state"] = "unknown"
    segments[0]["skill"] = "unknown"
    segments[1]["actor_state"] = "unknown"
    normalized["warnings"] = [
        {
            "code": "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN",
            "fields": ["actor_state", "skill"],
            "count": 3,
        }
    ]
    assert callable(mutation)
    mutation(normalized)
    original = copy.deepcopy(normalized)

    with pytest.raises(ActionCaptionExportError) as error:
        list(iter_action_captions("video_0001", normalized, source_fps=30.0))

    assert str(error.value) == "completed embodied result is invalid"
    assert "DO NOT LEAK" not in str(error.value)
    assert normalized == original


@pytest.mark.parametrize("source_fps", [0.0, -1.0, math.inf, math.nan, True, "30"])
def test_export_rejects_invalid_source_fps(
    embodied_result: dict[str, object], source_fps: object
) -> None:
    """Allowing a non-finite or coercive rate corrupts every frame span."""
    with pytest.raises(ActionCaptionExportError, match="source_fps is invalid"):
        list(iter_action_captions("video_0001", embodied_result, source_fps=source_fps))  # type: ignore[arg-type]


def test_export_rejects_incomplete_nonfinite_or_frame_collapsed_results_without_mutation(
    embodied_result: dict[str, object],
) -> None:
    """A malformed completed result must never become a plausible training row."""
    broken = copy.deepcopy(embodied_result)
    segments = broken["segments"]
    assert isinstance(segments, list)
    segments[0].pop("target")
    segments[1]["confidence"] = math.nan
    original_broken = copy.deepcopy(broken)

    with pytest.raises(ActionCaptionExportError, match="completed embodied result is invalid"):
        list(iter_action_captions("video_0001", broken, source_fps=30.0))
    assert broken == original_broken

    with pytest.raises(ActionCaptionExportError, match="frame span collapsed"):
        list(iter_action_captions("video_0001", embodied_result, source_fps=1.0))


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda result: result["segments"][0].update({"end": 1.01}),
            "fine segment duration",
        ),
        (
            lambda result: result["segments"][0].update({"description": "x" * 61}),
            "caption limit",
        ),
        (
            lambda result: result["segments"][0].update(
                {"description": "right Hand reaches container"}
            ),
            "caption style",
        ),
        (
            lambda result: result["segments"][0].update({"start": 5.0, "end": 5.41}),
            "complete-result start",
        ),
        (
            lambda result: result["segments"][1].update({"segment_index": 0}),
            "source-index uniqueness",
        ),
    ],
)
def test_export_rejects_completed_results_that_cannot_truthfully_be_clean(
    embodied_result: dict[str, object],
    mutate: object,
    reason: str,
) -> None:
    """Final flags must only be emitted after all available 0805 checks pass."""
    broken = copy.deepcopy(embodied_result)
    assert callable(mutate), reason
    mutate(broken)
    original_broken = copy.deepcopy(broken)

    with pytest.raises(ActionCaptionExportError, match="completed embodied result is invalid"):
        list(iter_action_captions("video_0001", broken, source_fps=30.0))
    assert broken == original_broken


def test_export_rejects_rounded_frame_overlap_at_extreme_fps(
    embodied_result: dict[str, object],
) -> None:
    """Second-level epsilon must not silently create overlapping source frames."""
    overlapping = copy.deepcopy(embodied_result)
    segments = overlapping["segments"]
    assert isinstance(segments, list)
    segments[0].update({"end": 1.0})
    segments[1].update({"start": 0.9999999995, "end": 1.5})

    with pytest.raises(ActionCaptionExportError, match="completed embodied result is invalid"):
        list(iter_action_captions("video_0001", overlapping, source_fps=1e12))


def test_export_rejects_epsilon_sized_initial_gap_at_extreme_fps(
    embodied_result: dict[str, object],
) -> None:
    """A completed video cannot omit even frames hidden by seconds epsilon."""
    gapped = copy.deepcopy(embodied_result)
    segments = gapped["segments"]
    assert isinstance(segments, list)
    segments[0].update({"start": 5e-10})

    with pytest.raises(ActionCaptionExportError, match="completed embodied result is invalid"):
        list(iter_action_captions("video_0001", gapped, source_fps=1e12))


def test_export_contains_hostile_mapping_and_frame_overflow_with_stable_errors(
    embodied_result: dict[str, object],
) -> None:
    """Untrusted mapping protocols and extreme arithmetic must not leak content."""

    class HostileResult(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("HOSTILE PAYLOAD SENTINEL")

        def __iter__(self):
            raise RuntimeError("HOSTILE PAYLOAD SENTINEL")

        def __len__(self) -> int:
            return 2

    with pytest.raises(ActionCaptionExportError) as hostile_error:
        list(iter_action_captions("video_0001", HostileResult(), source_fps=30.0))
    assert str(hostile_error.value) == "completed embodied result is invalid"

    overflowing = copy.deepcopy(embodied_result)
    segments = overflowing["segments"]
    assert isinstance(segments, list)
    segments[0].update({"end": 1.0})
    segments[1].update({"start": 1.0, "end": 2.0})
    with pytest.raises(ActionCaptionExportError) as overflow_error:
        list(iter_action_captions("video_0001", overflowing, source_fps=float.fromhex("0x1.fffffffffffffp+1023")))
    assert str(overflow_error.value) == "completed embodied result frame span is invalid"


def test_export_rejects_result_mapping_that_changes_after_key_validation(
    embodied_result: dict[str, object],
) -> None:
    """A stateful mapping must not add malformed warnings after its key check."""

    class StatefulResult(Mapping[str, object]):
        def __init__(self) -> None:
            self._values = copy.deepcopy(embodied_result)
            self._values["warnings"] = [{"private_payload": "DO NOT LEAK"}]
            self._iterations = 0

        def __getitem__(self, key: str) -> object:
            return self._values[key]

        def __iter__(self):
            self._iterations += 1
            yield from ("task_description", "segments")
            if self._iterations > 1:
                yield "warnings"

        def __len__(self) -> int:
            return 2

    with pytest.raises(ActionCaptionExportError) as error:
        list(iter_action_captions("video_0001", StatefulResult(), source_fps=30.0))

    assert str(error.value) == "completed embodied result is invalid"
    assert "DO NOT LEAK" not in str(error.value)


def test_export_rejects_integer_subclass_that_bypasses_warning_count_bounds(
    embodied_result: dict[str, object],
) -> None:
    """Count comparisons must not execute attacker-controlled integer methods."""

    class DeceptiveCount(int):
        def __le__(self, other: object) -> bool:
            return other != 0

        def __gt__(self, other: object) -> bool:
            return False

    normalized = copy.deepcopy(embodied_result)
    segments = normalized["segments"]
    assert isinstance(segments, list)
    segments[0]["actor_state"] = "unknown"
    normalized["warnings"] = [
        {
            "code": "ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN",
            "fields": ["actor_state"],
            "count": DeceptiveCount(1000),
        }
    ]

    with pytest.raises(ActionCaptionExportError) as error:
        list(iter_action_captions("video_0001", normalized, source_fps=30.0))

    assert str(error.value) == "completed embodied result is invalid"


def test_export_preserves_ordered_sparse_source_indexes(
    embodied_result: dict[str, object],
) -> None:
    """Caption IDs use stable output order, not an assumed dense model index."""
    sparse = copy.deepcopy(embodied_result)
    segments = sparse["segments"]
    assert isinstance(segments, list)
    segments[0]["segment_index"] = 4
    segments[1]["segment_index"] = 9

    rows = list(iter_action_captions("video_0001", sparse, source_fps=30.0))
    assert [(row.caption_id, row.source_segment_index) for row in rows] == [
        ("video_0001_cap_0000", 4),
        ("video_0001_cap_0001", 9),
    ]


def test_jsonl_writer_is_utf8_deterministic_atomic_and_cleans_up_iterable_failures(
    tmp_path: Path, embodied_result: dict[str, object]
) -> None:
    """A failed stream must preserve the old file and leave no partial replacement."""
    destination = tmp_path / "action_captions.jsonl"
    destination.write_text("old\n", encoding="utf-8")
    rows = tuple(iter_action_captions("视频", embodied_result, source_fps=30.0))

    write_action_captions_jsonl(destination, rows)
    first_write = destination.read_bytes()
    assert "视频".encode() in first_write
    assert first_write.endswith(b"\n")
    assert b"NaN" not in first_write

    write_action_captions_jsonl(destination, rows)
    assert destination.read_bytes() == first_write

    def failing_rows():
        yield rows[0]
        raise RuntimeError("producer failure")

    with pytest.raises(RuntimeError, match="producer failure"):
        write_action_captions_jsonl(destination, failing_rows())
    assert destination.read_bytes() == first_write
    assert list(tmp_path.glob(".action_captions.jsonl.*.tmp")) == []


def test_writer_revalidates_dataclass_values_and_rejects_subclasses(
    tmp_path: Path, embodied_result: dict[str, object]
) -> None:
    """Frozen annotations alone cannot guarantee a valid serialized training row."""
    [row] = list(iter_action_captions("video_0001", embodied_result, source_fps=30.0))[:1]
    destination = tmp_path / "action_captions.jsonl"
    destination.write_text("old\n", encoding="utf-8")

    with pytest.raises(ActionCaptionExportError, match="action caption is invalid"):
        replace(
            row,
            start_sec="bad",  # type: ignore[arg-type]
            duration_frames=-99,
            schema_parse_ok="yes",  # type: ignore[arg-type]
            parse_errors="error",  # type: ignore[arg-type]
        )
    assert destination.read_text(encoding="utf-8") == "old\n"

    class CaptionSubclass(ActionCaption):
        pass

    with pytest.raises(ActionCaptionExportError, match="action caption is invalid"):
        CaptionSubclass(**row.to_dict())

    spoofed = object.__new__(ActionCaption)
    with pytest.raises(ActionCaptionExportError, match="action caption is invalid"):
        write_action_captions_jsonl(destination, [spoofed])

    uppercase_caption = "right Hand moves object"
    uppercase_source = row.source_description.replace(
        "action=right hand reaches the red container",
        f"action={uppercase_caption}",
    )
    with pytest.raises(ActionCaptionExportError, match="action caption is invalid"):
        replace(
            row,
            caption=uppercase_caption,
            caption_char_count=len(uppercase_caption),
            source_description=uppercase_source,
        )


@pytest.mark.parametrize("failure", ["serialization", "file_fsync", "replace"])
def test_writer_pre_replace_failures_preserve_destination_and_cleanup_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embodied_result: dict[str, object],
    failure: str,
) -> None:
    """Every pre-replace write path must leave a previous export recoverable."""
    destination = tmp_path / "action_captions.jsonl"
    destination.write_text("old\n", encoding="utf-8")
    rows = tuple(iter_action_captions("video_0001", embodied_result, source_fps=30.0))
    if failure == "serialization":
        monkeypatch.setattr(export_module, "_jsonl_line", lambda _: (_ for _ in ()).throw(ValueError("injected")))
    elif failure == "file_fsync":
        monkeypatch.setattr(export_module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))
    else:
        monkeypatch.setattr(export_module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError(errno.EIO, "injected")))

    with pytest.raises(Exception):
        write_action_captions_jsonl(destination, rows)
    assert destination.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".action_captions.jsonl.*.tmp")) == []


@pytest.mark.parametrize("control_exception", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_writer_cleans_temp_and_reraises_control_exceptions_unchanged(
    tmp_path: Path,
    embodied_result: dict[str, object],
    control_exception: type[BaseException],
) -> None:
    """Cancellation must not leave a partial export beside the preserved target."""
    destination = tmp_path / "action_captions.jsonl"
    destination.write_text("old\n", encoding="utf-8")
    [row] = list(iter_action_captions("video_0001", embodied_result, source_fps=30.0))[:1]

    def cancelled_rows():
        yield row
        raise control_exception()

    with pytest.raises(control_exception):
        write_action_captions_jsonl(destination, cancelled_rows())
    assert destination.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".action_captions.jsonl.*.tmp")) == []


def test_writer_mode_directory_sync_errors_and_destination_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embodied_result: dict[str, object],
) -> None:
    """Directory sync failures are visible after replacement and symlinks are replaced."""
    rows = tuple(iter_action_captions("video_0001", embodied_result, source_fps=30.0))
    destination = tmp_path / "action_captions.jsonl"
    target = tmp_path / "target.jsonl"
    target.write_text("target-old\n", encoding="utf-8")
    destination.symlink_to(target)

    write_action_captions_jsonl(destination, rows)
    assert not destination.is_symlink()
    assert target.read_text(encoding="utf-8") == "target-old\n"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600

    calls = 0
    original_fsync = export_module.os.fsync

    def failing_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "injected")
        original_fsync(descriptor)

    monkeypatch.setattr(export_module.os, "fsync", failing_directory_fsync)
    with pytest.raises(ActionCaptionExportError, match="durability is uncertain"):
        write_action_captions_jsonl(destination, rows)
    assert destination.read_bytes().endswith(b"\n")
    assert list(tmp_path.glob(".action_captions.jsonl.*.tmp")) == []


def test_writer_ignores_only_documented_unsupported_directory_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embodied_result: dict[str, object],
) -> None:
    """A filesystem that declares directory fsync unsupported remains usable."""
    rows = tuple(iter_action_captions("video_0001", embodied_result, source_fps=30.0))
    destination = tmp_path / "action_captions.jsonl"
    calls = 0
    original_fsync = export_module.os.fsync

    def unsupported_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.ENOTSUP, "unsupported")
        original_fsync(descriptor)

    monkeypatch.setattr(export_module.os, "fsync", unsupported_directory_fsync)
    write_action_captions_jsonl(destination, rows)
    assert destination.read_bytes().endswith(b"\n")
