from __future__ import annotations

import json
import stat
from copy import deepcopy
from pathlib import Path

import pytest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _manifest(sample_id: str = "demo_0001", duration: float = 2.0) -> dict[str, object]:
    return {
        "schema_version": "comparison_viewer_manifest_v1",
        "reference_set_id": "test-reference",
        "samples": [
            {
                "sample_id": sample_id,
                "duration_seconds": duration,
                "media_path": f"evaluation/viewer/media/{sample_id}.mp4",
                "las_path": f"evaluation/references/{sample_id}.json",
                "local_path": f"evaluation/viewer/data/local/{sample_id}.json",
                "caveat": "",
            }
        ],
    }


def _local_result() -> dict[str, object]:
    return {
        "task_description": "describe visible actions",
        "warnings": [{"code": "PRIVATE_WARNING"}],
        "task_id": "private-task",
        "request": {"video_url": "https://private.invalid/video.mp4"},
        "segments": [
            {
                "segment_index": 0,
                "action_index": 0,
                "start": 0.0,
                "end": 1.0,
                "description": "right hand reaches for block",
                "actor": "right_hand",
                "actor_state": "reaching",
                "skill": "reach",
                "target": "block",
                "visual_motion_state": "active",
                "event_type": "pre_contact",
                "confidence": 0.9,
                "start_boundary_id": "private-start",
                "end_boundary_id": "private-end",
            }
        ],
        "grouped_semantic_events": [
            {
                "event_index": 0,
                "start": 0.0,
                "end": 1.0,
                "description": "right hand reaches for block",
                "actor": "right_hand",
                "action": "reach",
                "target": "block",
                "confidence": 0.9,
                "source_segment_indices": [0],
            }
        ],
        "semantic_events": [
            {
                "event_index": 0,
                "start": 0.0,
                "end": 1.0,
                "event_type": "reach",
                "actor": "right_hand",
                "target_object_id": "block",
                "description": "right hand reaches for block",
                "confidence": 0.9,
            }
        ],
        "objects": [
            {"object_id": "block", "name": "block", "description": "red block"}
        ],
        "initial_state": [
            {
                "object_id": "block",
                "state": "on table",
                "visual_evidence": "block is visible on table",
                "confidence": 0.8,
            }
        ],
        "final_state": [
            {
                "object_id": "block",
                "state": "held",
                "visual_evidence": "hand encloses block",
                "confidence": 0.8,
            }
        ],
        "outcome": {"status": "success", "description": "block held", "confidence": 0.8},
    }


def test_export_projects_only_display_fields(tmp_path: Path) -> None:
    from scripts.build_comparison_viewer_data import main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    source_path = input_dir / "demo_0001.json"
    source = _local_result()
    _write_json(source_path, source)
    _write_json(manifest_path, _manifest())

    assert (
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )

    projected = json.loads((output_dir / "demo_0001.json").read_text(encoding="utf-8"))
    assert projected == {
        "duration_seconds": 2.0,
        "final_state": source["final_state"],
        "fine_segments": [
            {
                key: source["segments"][0][key]
                for key in (
                    "segment_index",
                    "action_index",
                    "start",
                    "end",
                    "description",
                    "actor",
                    "actor_state",
                    "skill",
                    "target",
                    "visual_motion_state",
                    "event_type",
                    "confidence",
                )
            }
        ],
        "grouped_events": source["grouped_semantic_events"],
        "initial_state": source["initial_state"],
        "objects": source["objects"],
        "outcome": source["outcome"],
        "sample_id": "demo_0001",
        "scene_events": source["semantic_events"],
        "schema_version": "comparison_viewer_local_v1",
        "source_result_sha256": __import__("hashlib").sha256(source_path.read_bytes()).hexdigest(),
    }
    serialized = (output_dir / "demo_0001.json").read_text(encoding="utf-8")
    for prohibited in ("warnings", "task_id", "request", "video_url", "boundary_id"):
        assert prohibited not in serialized
    assert stat.S_IMODE((output_dir / "demo_0001.json").stat().st_mode) == 0o600


def test_export_is_byte_identical_on_an_unchanged_second_run(tmp_path: Path) -> None:
    from scripts.build_comparison_viewer_data import main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    _write_json(input_dir / "demo_0001.json", _local_result())
    _write_json(manifest_path, _manifest())
    arguments = [
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
    ]

    assert main(arguments) == 0
    first = (output_dir / "demo_0001.json").read_bytes()
    assert main(arguments) == 0

    assert (output_dir / "demo_0001.json").read_bytes() == first


@pytest.mark.parametrize("case", ["unexpected", "missing"])
def test_export_rejects_a_wrong_input_sample_set(tmp_path: Path, case: str) -> None:
    from scripts.build_comparison_viewer_data import ViewerDataError, main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    input_dir.mkdir()
    if case == "unexpected":
        _write_json(input_dir / "demo_0001.json", _local_result())
        _write_json(input_dir / "extra.json", _local_result())
    _write_json(manifest_path, _manifest())

    with pytest.raises(ViewerDataError, match="INPUT_SAMPLE_SET_INVALID"):
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(manifest_path),
            ]
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("start", "end"),
    [(-0.1, 1.0), (0.0, 2.1), (1.0, 1.0), (1.1, 1.0), (float("nan"), 1.0)],
)
def test_export_rejects_invalid_intervals(tmp_path: Path, start: float, end: float) -> None:
    from scripts.build_comparison_viewer_data import ViewerDataError, main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    source = _local_result()
    source["segments"][0]["start"] = start
    source["segments"][0]["end"] = end
    _write_json(input_dir / "demo_0001.json", source)
    _write_json(manifest_path, _manifest())

    with pytest.raises(ViewerDataError, match="INVALID_INTERVAL"):
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(manifest_path),
            ]
        )

    assert not output_dir.exists()


def test_export_validates_every_sample_before_publishing(tmp_path: Path) -> None:
    from scripts.build_comparison_viewer_data import ViewerDataError, main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    valid = _local_result()
    invalid = deepcopy(valid)
    invalid["semantic_events"] = {}
    _write_json(input_dir / "demo_0001.json", valid)
    _write_json(input_dir / "demo_0002.json", invalid)
    manifest = _manifest()
    manifest["samples"].append(
        {
            "sample_id": "demo_0002",
            "duration_seconds": 2.0,
            "media_path": "evaluation/viewer/media/demo_0002.mp4",
            "las_path": "evaluation/references/demo_0002.json",
            "local_path": "evaluation/viewer/data/local/demo_0002.json",
            "caveat": "",
        }
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(ViewerDataError, match="EXPECTED_ARRAY"):
        main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(manifest_path),
            ]
        )

    assert not output_dir.exists()


def test_repository_viewer_data_is_complete() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest_path = repository / "evaluation/viewer/data/demo-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads(
        (
            repository
            / "evaluation/references/las_official_english_2026-09-04/comparison_with_postfix_local.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        "full_0001": 10.933333333333334,
        "full_0002": 14.7,
        "full_0024": 4.566666666666666,
        "full_0021": 7.3,
        "full_0004": 8.8,
    }

    assert manifest["schema_version"] == "comparison_viewer_manifest_v1"
    assert manifest["reference_set_id"] == "las_official_english_2026-09-04"
    assert [item["sample_id"] for item in manifest["samples"]] == list(expected)
    assert {item["sample_id"]: item["duration_seconds"] for item in manifest["samples"]} == expected

    result_hashes = comparison["local_artifact"]["result_sha256"]
    for item in manifest["samples"]:
        assert not Path(item["las_path"]).is_absolute()
        assert not Path(item["local_path"]).is_absolute()
        las_path = repository / item["las_path"]
        local_path = repository / item["local_path"]
        assert las_path.is_file()
        local = json.loads(local_path.read_text(encoding="utf-8"))
        assert local["schema_version"] == "comparison_viewer_local_v1"
        assert local["sample_id"] == item["sample_id"]
        assert local["duration_seconds"] == item["duration_seconds"]
        assert local["source_result_sha256"] == result_hashes[item["sample_id"]]
