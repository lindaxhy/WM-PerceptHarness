from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import tomllib
from pathlib import Path

from las_repro.config import Settings


def test_repository_has_no_asr_module_dependency_configuration_or_legacy_test():
    """Reintroducing any removed ASR surface must fail the no-audio policy gate."""
    repository_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    optional_dependencies = project["project"]["optional-dependencies"]
    dependency_groups = [
        project["project"]["dependencies"],
        *optional_dependencies.values(),
    ]
    dependency_names = {
        re.sub(r"[-_.]+", "-", match.group(1).lower())
        for group in dependency_groups
        for requirement in group
        if (match := re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)) is not None
    }

    assert importlib.util.find_spec("las_repro.asr") is None
    assert all(extra.lower() != "asr" for extra in optional_dependencies)
    assert "faster-whisper" not in dependency_names
    assert {
        "asr_backend",
        "asr_model_path",
        "asr_device",
        "asr_compute_type",
        "asr_required",
    }.isdisjoint(Settings.model_fields)
    assert not (repository_root / "tests" / "test_asr.py").exists()


def _tracked_text_files(repository_root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        capture_output=True,
        check=True,
    )
    return [
        repository_root / Path(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw and (repository_root / Path(raw.decode("utf-8"))).is_file()
    ]


def test_repository_contains_no_committed_access_token_values() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    token = re.compile(
        rb"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
        rb"github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})"
    )

    offenders = [
        path.relative_to(repository_root).as_posix()
        for path in _tracked_text_files(repository_root)
        if token.search(path.read_bytes()) is not None
    ]

    assert offenders == []


def test_cv_gpu_packages_are_imported_only_by_the_lazy_sam31_adapter() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    cv_root = repository_root / "src" / "las_repro" / "cv"
    gpu_packages = {"cv2", "numpy", "sam3", "torch"}
    offenders: list[str] = []

    for path in sorted(cv_root.glob("*.py")):
        if path.name == "sam31.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        if imported & gpu_packages:
            offenders.append(path.relative_to(repository_root).as_posix())

    assert offenders == []


def test_env_example_documents_disabled_local_cv_defaults() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    values = {
        key: value.strip("'\"")
        for raw_line in (repository_root / ".env.example").read_text(
            encoding="utf-8"
        ).splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
        for key, value in [raw_line.split("=", 1)]
    }

    assert values["LAS_GPU_DEVICES"] == "0,1,2"
    assert values["LAS_CV_DEVICE"] == "3"
    assert values["LAS_CV_PROVIDER"] == "disabled"
    assert values["LAS_CV_MODEL_ALIAS"] == "sam3.1"
    assert values["LAS_CV_ENTITY_LIMIT"] == "16"
    assert values["LAS_CV_SHORT_VIDEO_SECONDS"] == "30.0"
    assert values["LAS_CV_SCAN_FPS"] == "8.0"
    assert values["LAS_CV_MAX_FPS"] == "30.0"
    assert values["LAS_CV_REFINEMENT_RADIUS_SECONDS"] == "1.0"
    assert values["LAS_CV_MIN_CONFIDENCE"] == "0.5"
    assert values["LAS_CV_MIN_AREA_FRACTION"] == "0.01"
    assert values["LAS_CV_OCCLUSION_VISIBILITY_DROP"] == "0.5"
    assert values["LAS_CV_OVERLAP_THRESHOLD"] == "0.1"
    assert values["LAS_CV_EXECUTION_CHUNK_FRAMES"] == "8"
    assert values["LAS_CV_TIMEOUT_SECONDS"] == "300.0"
    assert values["LAS_CV_COMPILE_MODEL"] == "false"
    assert values["LAS_CV_CACHE_MAX_BYTES"] == str(8 * 1024 * 1024 * 1024)
    assert values["LAS_CV_CACHE_MAX_FILES"] == "10000"
    assert re.fullmatch(r"[0-9a-f]{64}", values["LAS_CV_CHECKPOINT_SHA256"])
    for field in (
        "LAS_CV_REPOSITORY_PATH",
        "LAS_CV_CHECKPOINT_PATH",
        "LAS_CV_BPE_PATH",
        "LAS_CV_CACHE_ROOT",
    ):
        assert values[field]
        assert not Path(values[field]).is_absolute()
