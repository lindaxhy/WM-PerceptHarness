from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from percept_harness.video_metrics import videophy_joint as joint
import build_reference_manifest as references  # noqa: E402
from validate_metric_manifest import load_manifest, validate_manifest  # noqa: E402


def _pc(value=4, path="/data/a.mp4"):
    return [{"videopath": path, "pc": value}]


def _sa(value=4, path="/data/a.mp4"):
    return [{"videopath": path, "sa": value}]


@pytest.mark.parametrize("value", [None, True, False, "", "NaN", float("nan"), float("inf"), "-inf", 3.5, "4.5", 0, 6])
def test_joint_rejects_non_integer_or_invalid_ratings(value):
    with pytest.raises(ValueError):
        joint.aggregate(_pc(value), _sa())


@pytest.mark.parametrize("rows", [None, "", {}, 5, [], [None], [{}], [{"videopath": "", "pc": 4}], [{"videopath": None, "pc": 4}]])
def test_joint_rejects_empty_or_malformed_rows(rows):
    with pytest.raises(ValueError):
        joint.aggregate(rows, _sa())


def test_joint_requires_identical_unique_video_sets():
    with pytest.raises(ValueError, match="Duplicate"):
        joint.aggregate(_pc() + _pc(path="/data/sub/../a.mp4"), _sa())
    with pytest.raises(ValueError, match="differ"):
        joint.aggregate(_pc(), _sa(path="/data/b.mp4"))


def test_joint_does_not_collapse_case_sensitive_linux_names():
    # This tests lexical keys, so it also runs without those files on Windows.
    assert joint._key("/data/A.mp4", None) != joint._key("/data/a.mp4", None)
    result = joint.aggregate(_pc(5, "/data/A.mp4") + _pc(3), _sa(4, "/data/A.mp4") + _sa())
    assert len(result["rows"]) == 2
    assert result["joint_rate"] == 0.5


def test_json_scores_preserve_types_and_reject_boolean(tmp_path):
    source = tmp_path / "scores.json"
    source.write_text(json.dumps(_pc(True)), encoding="utf-8")
    rows = joint._read_rows(source)
    assert rows[0]["pc"] is True
    with pytest.raises(ValueError):
        joint.aggregate(rows, _sa())


@pytest.mark.parametrize("body", ["videopath,pc,pc\na.mp4,4,5\n", "videopath,pc\na.mp4\n", "videopath,pc\na.mp4,4,extra\n"])
def test_csv_rejects_duplicate_columns_and_bad_row_width(tmp_path, body):
    source = tmp_path / "scores.csv"
    source.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        joint._read_rows(source)


def _frame_dirs(tmp_path):
    generated, reference = tmp_path / "generated", tmp_path / "reference"
    generated.mkdir()
    reference.mkdir()
    (generated / "000.png").write_bytes(b"generated")
    (reference / "000.png").write_bytes(b"real reference")
    return generated, reference


def test_reference_manifest_does_not_claim_alignment(tmp_path):
    generated, reference = _frame_dirs(tmp_path)
    result = references.build_pairs(generated, reference)
    assert result["alignment_verified"] is False
    assert "do not establish" in result["warning"]
    assert result["counts"]["matched"] == 1


def test_reference_refuses_self_reference_and_nested_roots(tmp_path):
    generated, reference = _frame_dirs(tmp_path)
    with pytest.raises(ValueError, match="different"):
        references.build_pairs(generated, generated)
    nested = generated / "reference"
    nested.mkdir()
    with pytest.raises(ValueError, match="contain"):
        references.build_pairs(generated, nested)
    # Model an alias/hardlink without relying on Windows symlink privileges.
    original_samefile = Path.samefile

    def samefile(left, right):
        if left.name == "000.png" and right.name == "000.png":
            return True
        return original_samefile(left, right)

    with patch.object(Path, "samefile", samefile), pytest.raises(ValueError, match="same file"):
        references.build_pairs(generated, reference)


def test_reference_rejects_empty_or_unmatched_directories(tmp_path):
    generated, reference = tmp_path / "g", tmp_path / "r"
    generated.mkdir()
    reference.mkdir()
    with pytest.raises(ValueError, match="at least one"):
        references.build_pairs(generated, reference)
    (generated / "a.png").write_bytes(b"g")
    (reference / "b.png").write_bytes(b"r")
    with pytest.raises(ValueError, match="No matching"):
        references.build_pairs(generated, reference)


def test_reference_complete_flag_checks_unmatched_files(tmp_path):
    generated, reference = _frame_dirs(tmp_path)
    (generated / "001.png").write_bytes(b"extra")
    output = tmp_path / "manifest.json"
    assert references.main(["--generated-frames", str(generated), "--reference-frames", str(reference), "--output", str(output), "--require-complete"]) == 1
    assert not output.exists()


@pytest.mark.parametrize("payload", [[], {}, None, [None], ["record"]])
def test_manifest_rejects_empty_or_wrong_top_level(payload):
    with pytest.raises(ValueError):
        validate_manifest(payload)


@pytest.mark.parametrize("patch_fields", [{"video_id": " "}, {"video": None}, {"dimensions": None}, {"dimensions": "mechanics"}, {"dimensions": [""]}, {"dimensions": [False]}, {"dimensions": ["mechanics_typo"]}, {"dimensions": ["mechanics", "mechanics"]}, {"prompt": 123}, {"reference_frames": []}])
def test_manifest_rejects_field_types_even_when_falsy(patch_fields):
    item = {"video_id": "a", "video": "/data/a.mp4"}
    item.update(patch_fields)
    with pytest.raises(ValueError):
        validate_manifest([item])


def test_manifest_rejects_duplicate_ids_and_resolved_paths(tmp_path):
    with pytest.raises(ValueError, match="duplicate video_id"):
        validate_manifest([{"video_id": "a", "video": "a.mp4"}, {"video_id": " a ", "video": "b.mp4"}])
    with pytest.raises(ValueError, match="duplicate video path"):
        validate_manifest([{"video_id": "a", "video": "a.mp4"}, {"video_id": "b", "video": "sub/../a.mp4"}], root=tmp_path)


@pytest.mark.parametrize("auxiliary", [None, "questions.json", {}, [], [""], [123], ["one action only"]])
def test_vbench2_rejects_invalid_order_auxiliary(auxiliary):
    item = {"video_id": "a", "video": "/data/a.mp4", "prompt": "Sit then sweep.", "dimensions": ["motion_order_understanding"], "auxiliary_info": auxiliary}
    with pytest.raises(ValueError, match="auxiliary_info"):
        validate_manifest([item])


def test_vbench2_accepts_official_auxiliary_list_and_reports_scope():
    item = {"video_id": "a", "video": "/data/a.mp4", "prompt": "Sit then sweep.", "dimensions": ["motion_order_understanding"], "auxiliary_info": ["sitting", "get up and sweeping"]}
    result = validate_manifest([item])
    assert result["dimensions_checked"] == ["motion_order_understanding"]
    assert "Preparation schema only" in result["validation_scope"]


def test_pc_does_not_require_prompt_but_sa_does():
    item = {"video_id": "a", "video": "/data/a.mp4"}
    validate_manifest([item], dimensions={"video_phy_pc"})
    with pytest.raises(ValueError, match="prompt"):
        validate_manifest([item], dimensions={"video_phy_sa"})


def test_manifest_checks_relative_files_and_reference_directory(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"video")
    (tmp_path / "gt").mkdir()
    item = {"video_id": "a", "video": "a.mp4", "reference_frames": "gt", "dimensions": ["psnr"]}
    validate_manifest([item], root=tmp_path, check_files=True)
    item["reference_frames"] = "a.mp4"
    with pytest.raises(ValueError, match="generated video"):
        validate_manifest([item], root=tmp_path, check_files=True)


def test_manifest_load_rejects_non_object_rows(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('[null]', encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_manifest(path)


def test_writers_exclusively_create_output_and_preserve_existing(tmp_path):
    pc, sa, output = tmp_path / "pc.json", tmp_path / "sa.json", tmp_path / "joint.json"
    pc.write_text(json.dumps(_pc()), encoding="utf-8")
    sa.write_text(json.dumps(_sa()), encoding="utf-8")
    calls = []
    original_open = Path.open

    def track_open(path, mode="r", *args, **kwargs):
        if path == output:
            calls.append(mode)
        return original_open(path, mode, *args, **kwargs)

    with patch.object(Path, "open", track_open):
        assert joint.main(["--pc", str(pc), "--sa", str(sa), "--output", str(output)]) == 0
    assert "x" in calls
    original = output.read_bytes()
    assert joint.main(["--pc", str(pc), "--sa", str(sa), "--output", str(output)]) == 1
    assert output.read_bytes() == original
    generated, reference = _frame_dirs(tmp_path)
    ref_output = tmp_path / "refs.json"
    output = ref_output
    calls.clear()
    with patch.object(Path, "open", track_open):
        assert references.main(["--generated-frames", str(generated), "--reference-frames", str(reference), "--output", str(output)]) == 0
    assert "x" in calls
    original = output.read_bytes()
    assert references.main(["--generated-frames", str(generated), "--reference-frames", str(reference), "--output", str(output)]) == 1
    assert output.read_bytes() == original
