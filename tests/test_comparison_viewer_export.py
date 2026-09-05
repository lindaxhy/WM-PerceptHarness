from __future__ import annotations

import base64
import hashlib
import json
import shutil
import stat
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def hybrid_export_case(tmp_path, monkeypatch):
    """Synthetic SAM-shaped artifacts; only source decoding/reference loading are doubles."""
    from test_cv_artifacts import cv_request, overlay_artifact_for, write_overlay_file
    from test_hybrid_result import source_segments
    from test_las_alignment import metadata_fixture

    from las_repro.cv.artifacts import CvArtifactStore, cv_cache_key
    from las_repro.cv.summary import build_cv_prompt_bundle, summarize_cv_evidence
    from las_repro.evaluation import las_alignment as las
    from las_repro.pipelines.hybrid_result import build_hybrid_result
    from las_repro.pipelines.scene_semantics import unavailable_scene_semantics

    root = tmp_path / "repo"
    root.mkdir()
    results, media, artifacts = root / "input", root / "media", root / "artifacts"
    results.mkdir()
    media.mkdir()
    metadata = metadata_fixture()
    model = "doubao-seed-2-1-pro-260628"
    metadata.update(model_identity=model, provider="ark", samples=[])
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a3ioAAAAASUVORK5CYII="
    )
    references, entries, handles = {}, {}, {}
    sample_ids = ("full_0001", "full_0002", "full_0004", "full_0021", "full_0024")
    timeline = cv_request.__wrapped__().timeline
    monkeypatch.setattr(
        "las_repro.cv.timeline.probe_frame_timeline", lambda _: timeline
    )
    for sid in sample_ids:
        video = f"synthetic source {sid}".encode()
        (media / f"{sid}.mp4").write_bytes(video)
        request = cv_request.__wrapped__().model_copy(
            update={
                "provider": "sam31",
                "model_identity": "sam3.1-test-contract",
                "duration_seconds": 1.0,
                "frame_count": 3,
                "video_sha256": hashlib.sha256(video).hexdigest(),
            }
        )
        artifact = overlay_artifact_for(request, png)
        with (
            CvArtifactStore(artifacts) as store,
            store.staging(cv_cache_key(request)) as staging,
        ):
            write_overlay_file(staging, png)
            handle = store.publish(request, staging, artifact)
        handles[sid] = handle
        summary = summarize_cv_evidence(artifact, timeline=timeline)
        bundle = build_cv_prompt_bundle(summary, request.thresholds)
        result = build_hybrid_result(
            task_description="move cup",
            segments=source_segments(),
            scene=unavailable_scene_semantics(),
            scene_status="unavailable",
            cv_evidence={
                "status": "available",
                "artifact_key": handle.key,
                "manifest_sha256": handle.manifest_sha256,
                "cache_hit": False,
            },
            warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}],
            performance={
                "stages": [],
                "total_seconds": 1.0,
                "repair_count": 0,
                "degradation_count": 1,
            },
            evidence_summary=summary,
            occlusion={
                "status": "available",
                "events": [],
                "decisions": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "classification": "unknown",
                        "target_entity_id": candidate.target_entity_id,
                        "occluder_entity_id": "unknown",
                        "visual_evidence": "Insufficient visual evidence",
                        "confidence": 0.2,
                        "events": [],
                    }
                    for candidate in bundle.candidates
                ],
            },
        )
        raw = las.canonical_json(result).encode()
        (results / f"{sid}.json").write_bytes(raw)
        entries[sid] = {
            "source_video": {"sha256": request.video_sha256, "duration_seconds": 1.0},
            "reference_file_sha256": "e" * 64,
        }
        references[sid] = las.Annotation(sid)
        metadata["samples"].append(
            {
                "sample_id": sid,
                "result_sha256": hashlib.sha256(raw).hexdigest(),
                "source_video_sha256": request.video_sha256,
                "model_identity": model,
                "configuration_sha256": "pending",
                "status": "COMPLETED",
                "wall_seconds": 1.0,
                "stage_seconds": None,
            }
        )
    metadata["configuration"]["cv"] = {
        "sampling": request.sampling.model_dump(mode="json"),
        "thresholds": request.thresholds.model_dump(mode="json"),
        "summary_limits": {
            "max_tracks": 64,
            "max_observations_per_track": 64,
            "max_relations": 512,
            "max_overlays": 24,
            "max_prompt_chars": 200000,
        },
        "bundle_limits": {"max_candidates": 256, "max_prompt_chars": 200000},
    }
    metadata["configuration_sha256"] = las.digest(metadata["configuration"])
    for row in metadata["samples"]:
        row["configuration_sha256"] = metadata["configuration_sha256"]
    metadata_path = root / "metadata.json"
    _write_json(metadata_path, metadata)
    monkeypatch.setattr(las, "load_references", lambda _: (references, entries))
    return (
        {
            "repository_root": root,
            "input_dir": results,
            "output_dir": root / "evaluation/viewer/data/hybrid",
            "metadata_path": metadata_path,
            "media_dir": media,
            "artifact_root": artifacts,
            "variant_id": "doubao_sam31",
            "reference_manifest": Path("synthetic-reference.json"),
            "mapping_path": Path("evaluation/config/las_alignment_mapping_v1.json"),
        },
        handles,
        png,
    )


def test_hybrid_export_publishes_five_verified_jsons_and_only_referenced_overlays(
    hybrid_export_case,
):
    from scripts.build_comparison_viewer_data import export_hybrid_dataset

    kwargs, _, png = hybrid_export_case
    manifest = export_hybrid_dataset(**kwargs)
    output = kwargs["output_dir"]
    assert len(manifest["samples"]) == 5
    assert len(list(output.glob("full_*.json"))) == 5
    overlay_path = output / "overlays" / f"{hashlib.sha256(png).hexdigest()}.png"
    assert overlay_path.read_bytes() == png
    assert not list(output.rglob("*.npz"))
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for entry in manifest["samples"]:
        path = kwargs["repository_root"] / entry["variant"]["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["variant"]["sha256"]
        data = json.loads(raw)
        assert (
            data["provenance"]["overlays"][0]["sha256"]
            == hashlib.sha256(png).hexdigest()
        )
        assert (
            data["provenance"]["source_result_sha256"]
            == entry["variant"]["source_result_sha256"]
        )
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    before = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }
    assert export_hybrid_dataset(**kwargs) == manifest
    assert before == {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "failure",
    [
        "missing_sample",
        "source_changed",
        "result_changed",
        "artifact_changed",
        "foreign_track",
        "wrong_model",
        "bad_review",
    ],
)
def test_hybrid_export_validation_failure_preserves_old_tree(
    hybrid_export_case, failure
):
    from scripts.build_comparison_viewer_data import export_hybrid_dataset

    kwargs, handles, _ = hybrid_export_case
    output = kwargs["output_dir"]
    output.mkdir(parents=True)
    (output / "old.txt").write_text("old complete tree")
    source = kwargs["input_dir"] / "full_0001.json"
    if failure == "missing_sample":
        source.unlink()
    elif failure == "source_changed":
        (kwargs["media_dir"] / "full_0001.mp4").write_bytes(b"wrong")
    elif failure == "result_changed":
        source.write_bytes(source.read_bytes() + b" ")
    elif failure == "artifact_changed":
        handle = handles["full_0001"]
        (
            kwargs["artifact_root"]
            / handle.key[:2]
            / handle.key
            / "overlays/opaque.png"
        ).write_bytes(b"changed")
    elif failure == "foreign_track":
        data = json.loads(source.read_bytes())
        data["annotation_branches"]["action_events"][0]["source_track_ids"] = [
            "foreign"
        ]
        _write_json(source, data)
        metadata = json.loads(kwargs["metadata_path"].read_bytes())
        metadata["samples"][0]["result_sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        _write_json(kwargs["metadata_path"], metadata)
    elif failure == "wrong_model":
        kwargs["variant_id"] = "qwen_sam31"
    else:
        review = kwargs["repository_root"] / "bad-review.json"
        _write_json(review, {"schema_version": "las_review_set_v1", "samples": []})
        kwargs["review_path"] = review
    with pytest.raises(ValueError):
        export_hybrid_dataset(**kwargs)
    assert (output / "old.txt").read_text() == "old complete tree"
    assert sorted(p.name for p in output.iterdir()) == ["old.txt"]


def test_hybrid_export_publication_failure_rolls_back(hybrid_export_case, monkeypatch):
    from scripts import build_comparison_viewer_data as exporter

    kwargs, _, _ = hybrid_export_case
    output = kwargs["output_dir"]
    output.mkdir(parents=True)
    (output / "old.txt").write_text("old complete tree")
    original_replace = exporter.os.replace

    def fail_staging_publication(source, target):
        if Path(target) == output and Path(source).name.startswith(".hybrid-staging-"):
            raise OSError("simulated publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(exporter.os, "replace", fail_staging_publication)
    with pytest.raises(ValueError):
        exporter.export_hybrid_dataset(**kwargs)
    assert (output / "old.txt").read_text() == "old complete tree"


def test_hybrid_export_rejects_output_symlink_and_broad_target(hybrid_export_case):
    from scripts.build_comparison_viewer_data import export_hybrid_dataset

    kwargs, _, _ = hybrid_export_case
    output = kwargs["output_dir"]
    output.parent.mkdir(parents=True)
    outside = kwargs["repository_root"] / "keep"
    outside.mkdir()
    (outside / "sentinel").write_text("keep")
    output.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        export_hybrid_dataset(**kwargs)
    assert (outside / "sentinel").read_text() == "keep"
    kwargs["output_dir"] = kwargs["repository_root"]
    with pytest.raises(ValueError):
        export_hybrid_dataset(**kwargs)


@pytest.fixture
def hybrid_cli_case(hybrid_export_case, monkeypatch):
    from las_repro.evaluation import las_alignment as las
    from scripts import build_comparison_viewer_data as exporter

    kwargs, _, _ = hybrid_export_case
    root = kwargs["repository_root"]
    monkeypatch.setattr(exporter, "REPOSITORY_ROOT", root)
    qwen = root / "qwen-input"
    local = root / "evaluation/viewer/data/local"
    manifest = {
        "schema_version": "comparison_viewer_manifest_v1",
        "reference_set_id": "las_official_english_2026-09-04",
        "samples": [],
    }
    frozen = {}
    for sid in las.FROZEN_QWEN:
        _write_json(qwen / f"{sid}.json", _local_result())
        frozen[sid] = hashlib.sha256((qwen / f"{sid}.json").read_bytes()).hexdigest()
        manifest["samples"].extend(_manifest(sid, duration=1.0)["samples"])
    monkeypatch.setattr(las, "FROZEN_QWEN", frozen)
    manifest_path = root / "demo-manifest.json"
    _write_json(manifest_path, manifest)
    local.mkdir(parents=True)
    (local / "sentinel").write_text("frozen viewer data")
    args = [
        "--input-dir",
        str(qwen),
        "--output-dir",
        str(local),
        "--manifest",
        str(manifest_path),
        "--hybrid-input-dir",
        str(kwargs["input_dir"]),
        "--hybrid-output-dir",
        str(kwargs["output_dir"]),
        "--hybrid-metadata",
        str(kwargs["metadata_path"]),
        "--media-dir",
        str(kwargs["media_dir"]),
        "--artifact-root",
        str(kwargs["artifact_root"]),
        "--reference-manifest",
        str(kwargs["reference_manifest"]),
        "--mapping",
        str(kwargs["mapping_path"]),
    ]
    return exporter, args, kwargs, local, manifest_path, manifest


def test_hybrid_cli_exports_new_variant_without_rewriting_frozen_local(hybrid_cli_case):
    exporter, args, kwargs, local, manifest_path, manifest = hybrid_cli_case
    assert exporter.main(args) == 0
    assert sorted(path.name for path in local.iterdir()) == ["sentinel"]
    assert json.loads(manifest_path.read_bytes()) == manifest
    assert len(list(kwargs["output_dir"].glob("full_*.json"))) == 5


@pytest.mark.parametrize(
    ("option", "relationship"),
    [
        (option, relationship)
        for option in ("--input-dir", "--output-dir")
        for relationship in ("equal", "ancestor", "descendant")
    ]
    + [("--manifest", "ancestor")],
)
def test_hybrid_cli_rejects_overlap_with_legacy_paths_and_preserves_bytes(
    hybrid_cli_case, option, relationship
):
    exporter, args, kwargs, _, _, _ = hybrid_cli_case
    target = kwargs["output_dir"]
    index = args.index(option) + 1
    original = Path(args[index])
    if option == "--manifest":
        protected = target / "demo-manifest.json"
        protected.parent.mkdir(parents=True)
        shutil.copyfile(original, protected)
    else:
        protected = {
            "equal": target,
            "ancestor": target / "frozen",
            "descendant": target.parent,
        }[relationship]
        shutil.copytree(original, protected, dirs_exist_ok=True)
    args[index] = str(protected)
    target.mkdir(parents=True, exist_ok=True)
    (target / "sentinel").write_bytes(b"previous hybrid output")
    before = {
        path: path.read_bytes() for path in target.parent.rglob("*") if path.is_file()
    }
    with pytest.raises(exporter.ViewerDataError):
        exporter.main(args)
    assert {path: path.read_bytes() for path in before} == before
    assert not (target / "variant-manifest.json").exists()


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
        "outcome": {
            "status": "success",
            "description": "block held",
            "confidence": 0.8,
        },
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
        "source_result_sha256": __import__("hashlib")
        .sha256(source_path.read_bytes())
        .hexdigest(),
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
def test_export_rejects_invalid_intervals(
    tmp_path: Path, start: float, end: float
) -> None:
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


@pytest.mark.parametrize(
    ("collection", "field", "value", "error"),
    [
        ("segments", "description", {"task_id": "private-task"}, "INVALID_TEXT"),
        ("segments", "segment_index", True, "INVALID_INDEX"),
        (
            "grouped_semantic_events",
            "source_segment_indices",
            [0, "private-task"],
            "INVALID_INDEX_LIST",
        ),
        ("objects", "name", ["block"], "INVALID_TEXT"),
        ("initial_state", "confidence", 1.1, "INVALID_CONFIDENCE"),
    ],
)
def test_export_rejects_invalid_display_field_values(
    tmp_path: Path,
    collection: str,
    field: str,
    value: object,
    error: str,
) -> None:
    from scripts.build_comparison_viewer_data import ViewerDataError, main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    source = _local_result()
    source[collection][0][field] = value
    _write_json(input_dir / "demo_0001.json", source)
    _write_json(manifest_path, _manifest())

    with pytest.raises(ViewerDataError, match=error):
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


def test_export_rejects_nested_metadata_in_outcome(tmp_path: Path) -> None:
    from scripts.build_comparison_viewer_data import ViewerDataError, main

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.json"
    source = _local_result()
    source["outcome"]["description"] = {
        "request": {"video_url": "https://private.invalid/video.mp4"}
    }
    _write_json(input_dir / "demo_0001.json", source)
    _write_json(manifest_path, _manifest())

    with pytest.raises(ViewerDataError, match="INVALID_TEXT"):
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

    assert manifest["schema_version"] == "comparison_viewer_manifest_v2"
    assert manifest["reference_set_id"] == "las_official_english_2026-09-04"
    assert [item["sample_id"] for item in manifest["samples"]] == list(expected)
    assert {
        item["sample_id"]: item["duration_seconds"] for item in manifest["samples"]
    } == expected

    result_hashes = comparison["local_artifact"]["result_sha256"]
    for item in manifest["samples"]:
        assert not Path(item["las_path"]).is_absolute()
        las_path = repository / item["las_path"]
        assert las_path.is_file()
        assert hashlib.sha256(las_path.read_bytes()).hexdigest() == item["las_sha256"]
        variants = {row["id"]: row for row in item["local_variants"]}
        assert len(item["local_variants"]) == len(variants) == 3
        assert set(variants) == {"qwen_only", "doubao_only", "doubao_sam31"}
        for variant in variants.values():
            path = Path(variant["path"])
            assert not path.is_absolute() and ".." not in path.parts
            assert hashlib.sha256((repository / path).read_bytes()).hexdigest() == variant["sha256"]
        local_path = repository / variants["qwen_only"]["path"]
        local = json.loads(local_path.read_text(encoding="utf-8"))
        assert local["schema_version"] == "comparison_viewer_local_v1"
        assert local["sample_id"] == item["sample_id"]
        assert local["duration_seconds"] == item["duration_seconds"]
        assert local["source_result_sha256"] == result_hashes[item["sample_id"]]
        for variant_id, metadata_name in (
            ("doubao_only", "doubao"), ("doubao_sam31", "hybrid")
        ):
            metadata = json.loads((repository / (
                "evaluation/results/sam31_2026-09-04/attempt2-cold/"
                f"{metadata_name}-metadata.json"
            )).read_bytes())
            source = next(row for row in metadata["samples"] if row["sample_id"] == item["sample_id"])
            variant = variants[variant_id]
            projection = json.loads((repository / variant["path"]).read_bytes())
            assert projection["sample"]["sample_id"] == item["sample_id"]
            assert variant["source_result_sha256"] == source["result_sha256"]
            assert projection["provenance"]["source_result_sha256"] == source["result_sha256"]
            assert projection["provenance"]["source_video_sha256"] == item["source_video_sha256"]
            assert variant["model_identity"] == "doubao-seed-2-1-pro-260628"
