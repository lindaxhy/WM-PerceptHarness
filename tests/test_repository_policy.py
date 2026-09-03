from __future__ import annotations

import importlib.util
import re
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
