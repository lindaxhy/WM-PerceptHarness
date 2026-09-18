"""Build deterministic, display-only local annotation files for the demo viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

FINE_FIELDS = (
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
GROUPED_FIELDS = (
    "event_index",
    "start",
    "end",
    "description",
    "actor",
    "action",
    "target",
    "confidence",
    "source_segment_indices",
)
SCENE_FIELDS = (
    "event_index",
    "start",
    "end",
    "event_type",
    "actor",
    "target_object_id",
    "description",
    "confidence",
)
OBJECT_FIELDS = ("object_id", "name", "description")
STATE_FIELDS = ("object_id", "state", "visual_evidence", "confidence")
OUTCOME_FIELDS = ("status", "description", "confidence")
FINE_TEXT_FIELDS = (
    "description",
    "actor",
    "actor_state",
    "skill",
    "target",
    "visual_motion_state",
    "event_type",
)
GROUPED_TEXT_FIELDS = ("description", "actor", "action", "target")
SCENE_TEXT_FIELDS = ("event_type", "actor", "target_object_id", "description")
OBJECT_TEXT_FIELDS = ("object_id", "name", "description")
STATE_TEXT_FIELDS = ("object_id", "state", "visual_evidence")
OUTCOME_TEXT_FIELDS = ("status", "description")


class ViewerDataError(ValueError):
    """An input cannot be safely projected into viewer data."""


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ViewerDataError(code)
    return value


def _array(value: object, code: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ViewerDataError(code)
    return value


def _finite_number(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ViewerDataError(code)
    number = float(value)
    if not math.isfinite(number):
        raise ViewerDataError(code)
    return number


def _text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ViewerDataError(code)
    return value


def _index(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ViewerDataError(code)
    return value


def _confidence(value: object, code: str) -> float:
    number = _finite_number(value, code)
    if not 0.0 <= number <= 1.0:
        raise ViewerDataError(code)
    return number


def _index_list(value: object, code: str) -> list[int]:
    if not isinstance(value, list):
        raise ViewerDataError(code)
    for item in value:
        _index(item, code)
    return value


def _validate_interval(item: Mapping[str, object], duration: float) -> None:
    start = _finite_number(item.get("start"), "INVALID_INTERVAL")
    end = _finite_number(item.get("end"), "INVALID_INTERVAL")
    if not 0.0 <= start < end <= duration:
        raise ViewerDataError("INVALID_INTERVAL")


def _required_fields(
    item: Mapping[str, object], fields: Sequence[str], code: str
) -> Mapping[str, object]:
    if not set(fields) <= set(item):
        raise ViewerDataError(code)
    return item


def _project_fields(
    value: Mapping[str, object], fields: Sequence[str]
) -> dict[str, object]:
    return {field: value[field] for field in fields}


def _project_record(
    item: Mapping[str, object],
    fields: Sequence[str],
    *,
    collection: str,
    missing_code: str,
    text_fields: Sequence[str] = (),
    index_fields: Sequence[str] = (),
    index_list_fields: Sequence[str] = (),
    confidence_field: str | None = None,
    duration: float | None = None,
) -> dict[str, object]:
    _required_fields(item, fields, missing_code)
    if duration is not None:
        _validate_interval(item, duration)
    for field in text_fields:
        _text(item[field], f"INVALID_TEXT:{collection}.{field}")
    for field in index_fields:
        _index(item[field], f"INVALID_INDEX:{collection}.{field}")
    for field in index_list_fields:
        _index_list(item[field], f"INVALID_INDEX_LIST:{collection}.{field}")
    if confidence_field is not None:
        _confidence(
            item[confidence_field],
            f"INVALID_CONFIDENCE:{collection}.{confidence_field}",
        )
    return _project_fields(item, fields)


def project_local_result(
    sample_id: str,
    duration: float,
    source: Mapping[str, object],
    *,
    source_sha256: str,
) -> dict[str, object]:
    duration = _finite_number(duration, "INVALID_DURATION")
    if duration <= 0.0:
        raise ViewerDataError("INVALID_DURATION")
    fine = _array(source.get("segments"), "EXPECTED_ARRAY:segments")
    grouped = _array(
        source.get("grouped_semantic_events"),
        "EXPECTED_ARRAY:grouped_semantic_events",
    )
    scene = _array(source.get("semantic_events"), "EXPECTED_ARRAY:semantic_events")
    objects = _array(source.get("objects"), "EXPECTED_ARRAY:objects")
    initial = _array(source.get("initial_state"), "EXPECTED_ARRAY:initial_state")
    final = _array(source.get("final_state"), "EXPECTED_ARRAY:final_state")
    outcome = _mapping(source.get("outcome"), "EXPECTED_OBJECT:outcome")
    return {
        "schema_version": "comparison_viewer_local_v1",
        "sample_id": sample_id,
        "duration_seconds": duration,
        "source_result_sha256": source_sha256,
        "fine_segments": [
            _project_record(
                item,
                FINE_FIELDS,
                collection="segments",
                missing_code="MISSING_FINE_FIELD",
                text_fields=FINE_TEXT_FIELDS,
                index_fields=("segment_index", "action_index"),
                confidence_field="confidence",
                duration=duration,
            )
            for item in fine
        ],
        "grouped_events": [
            _project_record(
                item,
                GROUPED_FIELDS,
                collection="grouped_semantic_events",
                missing_code="MISSING_GROUPED_FIELD",
                text_fields=GROUPED_TEXT_FIELDS,
                index_fields=("event_index",),
                index_list_fields=("source_segment_indices",),
                confidence_field="confidence",
                duration=duration,
            )
            for item in grouped
        ],
        "scene_events": [
            _project_record(
                item,
                SCENE_FIELDS,
                collection="semantic_events",
                missing_code="MISSING_SCENE_FIELD",
                text_fields=SCENE_TEXT_FIELDS,
                index_fields=("event_index",),
                confidence_field="confidence",
                duration=duration,
            )
            for item in scene
        ],
        "objects": [
            _project_record(
                item,
                OBJECT_FIELDS,
                collection="objects",
                missing_code="MISSING_OBJECT_FIELD",
                text_fields=OBJECT_TEXT_FIELDS,
            )
            for item in objects
        ],
        "initial_state": [
            _project_record(
                item,
                STATE_FIELDS,
                collection="initial_state",
                missing_code="MISSING_STATE_FIELD",
                text_fields=STATE_TEXT_FIELDS,
                confidence_field="confidence",
            )
            for item in initial
        ],
        "final_state": [
            _project_record(
                item,
                STATE_FIELDS,
                collection="final_state",
                missing_code="MISSING_STATE_FIELD",
                text_fields=STATE_TEXT_FIELDS,
                confidence_field="confidence",
            )
            for item in final
        ],
        "outcome": _project_record(
            outcome,
            OUTCOME_FIELDS,
            collection="outcome",
            missing_code="MISSING_OUTCOME_FIELD",
            text_fields=OUTCOME_TEXT_FIELDS,
            confidence_field="confidence",
        ),
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hybrid_output_target(
    output: Path, repository: Path, protected: Sequence[Path]
) -> Path:
    target = Path(os.path.abspath(output))
    repository = repository.resolve()
    relative = target.relative_to(repository)
    if len(relative.parts) < 4 or relative.parts[:3] != (
        "evaluation",
        "viewer",
        "data",
    ):
        raise ValueError("hybrid output must name a viewer dataset directory")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in relative.parts):
        raise ValueError("unsafe hybrid output name")
    current = target
    while current != repository:
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ValueError("unsafe hybrid output directory")
        if current.exists() and (
            current.stat().st_uid != os.geteuid() or current.stat().st_mode & 0o022
        ):
            raise ValueError("hybrid output directory is not owner-controlled")
        current = current.parent
    for path in protected:
        protected_path = path.resolve()
        if (
            protected_path == target
            or protected_path.is_relative_to(target)
            or target.is_relative_to(protected_path)
        ):
            raise ValueError("hybrid output overlaps protected input")
    return target


def _publish_hybrid_tree(target: Path, files: Mapping[str, bytes]) -> None:
    """Stage a complete tree and switch its name; roll back a failed switch."""
    target.parent.mkdir(parents=True, exist_ok=True)
    original = target.stat() if target.exists() else None
    staging = Path(tempfile.mkdtemp(prefix=".hybrid-staging-", dir=target.parent))
    backup: Path | None = None
    try:
        for relative, payload in files.items():
            path = staging / relative
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            status = path.lstat()
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or path.read_bytes() != payload
            ):
                raise ValueError("staged hybrid file verification failed")
        # Do not overwrite a dataset that changed while validation/staging ran.
        if target.is_symlink():
            raise ValueError("hybrid output changed")
        current = target.stat() if target.exists() else None
        if (original is None) != (current is None) or (
            original is not None
            and (original.st_dev, original.st_ino, original.st_mtime_ns)
            != (current.st_dev, current.st_ino, current.st_mtime_ns)
        ):
            raise ValueError("hybrid output changed")
        if original is not None:
            backup = Path(
                tempfile.mkdtemp(prefix=".hybrid-previous-", dir=target.parent)
            )
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if backup is not None:
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        # On rollback failure leave the previous tree recoverable at its private
        # backup name. Never delete the only remaining copy in this error path.


def export_hybrid_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    media_dir: Path,
    artifact_root: Path | None,
    repository_root: Path,
    reference_manifest: Path,
    mapping_path: Path,
    variant_id: str = "doubao_sam31",
    review_path: Path | None = None,
    include_fine_segments: bool = False,
    protected_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Authenticate and publish a complete five-demo variant, not a raw task dump.

    The existing demo manifest is intentionally not changed here. The returned
    digest-bearing variant manifest supports Task 15's reviewed publication.
    """
    from percept_harness.cv.artifacts import CvArtifactHandle, CvArtifactStore
    from percept_harness.evaluation import las_alignment as las
    from percept_harness.evaluation.viewer_projection import project_hybrid_viewer_data

    try:
        labels = {
            "doubao_sam31": "Doubao + SAM3.1",
            "qwen_sam31": "Qwen + SAM3.1",
            "doubao_only": "Doubao-only",
        }
        if variant_id not in labels:
            raise ValueError("unsupported hybrid viewer variant")
        protected = [
            input_dir,
            media_dir,
            metadata_path,
            reference_manifest,
            mapping_path,
            repository_root / "evaluation/viewer/data/local",
            *protected_paths,
        ]
        if artifact_root is not None:
            protected.append(artifact_root)
        if review_path is not None:
            protected.append(review_path)
        target = _hybrid_output_target(output_dir, repository_root, protected)
        prefix = target.relative_to(repository_root.resolve()).as_posix()
        references, entries = las.load_references(reference_manifest)
        if set(references) != set(las.FROZEN_QWEN):
            raise ValueError("exact frozen five samples required")
        metadata = las.strict_json(metadata_path.read_bytes())
        model = metadata["model_identity"]
        if (
            variant_id.startswith("doubao") and model != "doubao-seed-2-1-pro-260628"
        ) or (variant_id == "qwen_sam31" and not model.casefold().startswith("qwen")):
            raise ValueError("variant model mismatch")
        if (metadata["configuration"]["cv"] is None) != (variant_id == "doubao_only"):
            raise ValueError("variant CV configuration mismatch")
        reviews = None
        if review_path is not None:
            review_set = las.strict_json(review_path.read_bytes())
            if (
                type(review_set) is not dict
                or set(review_set) != {"schema_version", "samples"}
                or review_set["schema_version"] != "las_review_set_v1"
            ):
                raise ValueError("invalid viewer review set")
            rows = review_set["samples"]
            if type(rows) is not list:
                raise ValueError("invalid review samples")
            reviews = {row["sample_id"]: row for row in rows}
            if len(reviews) != len(rows) or set(reviews) != set(references):
                raise ValueError("incomplete viewer review set")
        # Reuse the full Task 12 boundary: real SAM policy, source hashes/PTS,
        # configuration-bound summary/candidates, result bytes and human review.
        las.evaluate_run(
            input_dir,
            metadata,
            references,
            entries,
            las.load_mapping(mapping_path),
            role="doubao" if variant_id == "doubao_only" else "hybrid",
            media_dir=media_dir,
            artifact_root=artifact_root,
            reviews=reviews,
        )
        files: dict[str, bytes] = {}
        samples = []
        for row in sorted(metadata["samples"], key=lambda entry: entry["sample_id"]):
            sid = row["sample_id"]
            raw = las.read_verified(input_dir / f"{sid}.json", row["result_sha256"])
            branches = raw["annotation_branches"]
            scene = branches["scene_facts"]
            consumers = (
                branches["action_events"]
                + branches["occlusion"]["events"]
                + scene["events"]
                + scene["locations"]
                + scene["relations"]
            )
            needed = {
                key for event in consumers for key in event["source_keyframe_ids"]
            }
            overlays = []
            cv = raw["cv_evidence"]
            if cv["status"] == "available":
                handle = CvArtifactHandle(cv["artifact_key"], cv["manifest_sha256"])
                with CvArtifactStore(artifact_root) as store:
                    artifact = store.load(handle)
                    registered = {
                        Path(record.path).stem: record
                        for record in artifact.overlay_records
                    }
                    if not needed <= set(registered):
                        raise ValueError("missing viewer overlays")
                    requested = tuple(registered[key].path for key in sorted(needed))
                    contents = store.read_overlays(handle, requested)
                descriptions = {file.path: file for file in artifact.files}
                clock = {
                    frame.frame_index: frame.timestamp_seconds
                    for frame in artifact.processed_timeline.frames
                }
                for key in sorted(needed):
                    record = registered[key]
                    descriptor = descriptions[record.path]
                    payload = contents[record.path]
                    relative = f"overlays/{descriptor.sha256}.png"
                    # Match the preview's bounded decode header, in addition to
                    # the store's descriptor/PNG signature authentication.
                    if (
                        len(payload) < 24
                        or payload[8:16] != b"\x00\x00\x00\rIHDR"
                        or not 0 < int.from_bytes(payload[16:20], "big") <= 4096
                        or not 0 < int.from_bytes(payload[20:24], "big") <= 4096
                    ):
                        raise ValueError("invalid or oversized viewer PNG")
                    if relative in files and files[relative] != payload:
                        raise ValueError("overlay digest collision")
                    files[relative] = payload
                    overlays.append(
                        {
                            "keyframe_id": key,
                            "path": f"{prefix}/{relative}",
                            "sha256": descriptor.sha256,
                            "size_bytes": descriptor.size_bytes,
                            "track_id": record.track_id,
                            "frame_index": record.frame_index,
                            "timestamp_seconds": clock[record.frame_index],
                        }
                    )
            projected = project_hybrid_viewer_data(
                sid,
                entries[sid]["source_video"]["duration_seconds"],
                raw,
                source_sha256=row["source_video_sha256"],
                source_result_sha256=row["result_sha256"],
                model_identity=model,
                review=(reviews or {}).get(sid),
                include_fine_segments=include_fine_segments,
                overlay_references=overlays,
            )
            payload = las.canonical_json(projected).encode()
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError("viewer JSON exceeds reader budget")
            files[f"{sid}.json"] = payload
            samples.append(
                {
                    "sample_id": sid,
                    "duration_seconds": entries[sid]["source_video"][
                        "duration_seconds"
                    ],
                    "source_video_sha256": row["source_video_sha256"],
                    "las_sha256": entries[sid]["reference_file_sha256"],
                    "variant": {
                        "id": variant_id,
                        "label": labels[variant_id],
                        "path": f"{prefix}/{sid}.json",
                        "format": "comparison_viewer_hybrid_v1",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "source_result_sha256": row["result_sha256"],
                        "model_identity": model,
                    },
                }
            )
        manifest = {
            "schema_version": "comparison_viewer_variant_set_v1",
            "samples": samples,
        }
        files["variant-manifest.json"] = las.canonical_json(manifest).encode()
        # Recheck directory containment after the potentially lengthy validation.
        _hybrid_output_target(output_dir, repository_root, protected)
        _publish_hybrid_tree(target, files)
        return manifest
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        raise ViewerDataError("HYBRID_EXPORT_FAILED") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hybrid-input-dir", type=Path)
    parser.add_argument("--hybrid-output-dir", type=Path)
    parser.add_argument("--hybrid-metadata", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument(
        "--hybrid-variant",
        choices=("doubao_sam31", "qwen_sam31", "doubao_only"),
        default="doubao_sam31",
    )
    parser.add_argument("--include-fine-segments", action="store_true")
    parser.add_argument(
        "--media-dir", type=Path, default=REPOSITORY_ROOT / "evaluation/viewer/media"
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=REPOSITORY_ROOT
        / "evaluation/references/las_official_english_2026-09-04/manifest.json",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/config/las_alignment_mapping_v1.json",
    )
    arguments = parser.parse_args(argv)
    hybrid_options = (
        arguments.hybrid_input_dir,
        arguments.hybrid_output_dir,
        arguments.hybrid_metadata,
    )
    hybrid_requested = any(value is not None for value in hybrid_options)
    if (
        hybrid_requested and not all(value is not None for value in hybrid_options)
    ) or (
        not hybrid_requested
        and (
            arguments.artifact_root
            or arguments.review
            or arguments.include_fine_segments
        )
    ):
        parser.error(
            "hybrid export requires input directory, output directory and metadata together"
        )

    manifest = _mapping(_load_json(arguments.manifest), "INVALID_MANIFEST")
    samples = _array(manifest.get("samples"), "INVALID_MANIFEST_SAMPLES")
    sample_ids = [sample.get("sample_id") for sample in samples]
    if (
        not samples
        or any(
            not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids
        )
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ViewerDataError("INVALID_MANIFEST_SAMPLES")
    expected = set(sample_ids)
    actual = {
        path.stem for path in arguments.input_dir.glob("*.json") if path.is_file()
    }
    if actual != expected:
        raise ViewerDataError("INPUT_SAMPLE_SET_INVALID")
    if hybrid_requested:
        from percept_harness.evaluation import las_alignment as las

        references, entries = las.load_references(arguments.reference_manifest)
        if (
            expected != set(references)
            or expected != set(las.FROZEN_QWEN)
            or manifest.get("reference_set_id") != "las_official_english_2026-09-04"
        ):
            raise ViewerDataError("FROZEN_VIEWER_SAMPLE_SET_REQUIRED")
        for sample in samples:
            if (
                sample["duration_seconds"]
                != entries[sample["sample_id"]]["source_video"]["duration_seconds"]
            ):
                raise ViewerDataError("VIEWER_DURATION_MISMATCH")

    outputs: dict[str, bytes] = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        source_path = arguments.input_dir / f"{sample_id}.json"
        raw = source_path.read_bytes()
        if (
            hybrid_requested
            and hashlib.sha256(raw).hexdigest() != las.FROZEN_QWEN[sample_id]
        ):
            raise ViewerDataError("FROZEN_QWEN_DIGEST_MISMATCH")
        source = _mapping(json.loads(raw), "INVALID_LOCAL_RESULT")
        projected = project_local_result(
            sample_id,
            sample["duration_seconds"],
            source,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )
        outputs[sample_id] = _canonical_json(projected)
    if hybrid_requested:
        # The old Qwen output and demo manifest remain untouched. Publish only
        # the named new variant; Task 15 switches the viewer manifest afterwards.
        if arguments.hybrid_output_dir.resolve() == arguments.output_dir.resolve():
            raise ViewerDataError("HYBRID_OUTPUT_OVERLAPS_LOCAL")
        export_hybrid_dataset(
            input_dir=arguments.hybrid_input_dir,
            output_dir=arguments.hybrid_output_dir,
            metadata_path=arguments.hybrid_metadata,
            media_dir=arguments.media_dir,
            artifact_root=arguments.artifact_root,
            repository_root=REPOSITORY_ROOT,
            reference_manifest=arguments.reference_manifest,
            mapping_path=arguments.mapping,
            variant_id=arguments.hybrid_variant,
            review_path=arguments.review,
            include_fine_segments=arguments.include_fine_segments,
            protected_paths=(
                arguments.input_dir,
                arguments.output_dir,
                arguments.manifest,
            ),
        )
        return 0
    for sample_id, payload in outputs.items():
        _atomic_write(arguments.output_dir / f"{sample_id}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
