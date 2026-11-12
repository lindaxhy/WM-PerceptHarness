"""Check portable source delivery without importing GPU/model dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "metric_packages"
ENTRIES = {
    "01_semantic_adherence": ["modified/sapc/run_sa.py"],
    "02_physical_commonsense": ["modified/sapc/run_pc.py"],
    "06_psnr": ["modified/run_psnr.py"],
    "07_dynamic_degree": ["run_dynamic_degree.py"],
    "08_motion_smoothness": ["run_motion_smoothness.py"],
}


def test_selected_package_directories():
    assert {p.name for p in PACKAGES.iterdir() if p.is_dir()} == set(ENTRIES)


@pytest.mark.parametrize("name", ENTRIES)
def test_package_manifest_covers_exact_source_bytes(name):
    root = PACKAGES / name
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = set()
    for row in manifest["files"]:
        relative = row["path"]
        assert relative not in expected
        expected.add(relative)
        path = (root / relative).resolve()
        assert path.is_relative_to(root.resolve())
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], relative
        assert len(data) == row["bytes"], relative
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
        and not p.name.endswith(".pyc") and p.name != "MANIFEST.json"
    }
    assert actual == expected


@pytest.mark.parametrize("name", ENTRIES)
def test_upstream_manifests_match_vendored_sources(name):
    root = PACKAGES / name
    for filename in ("official_sources.json", "OFFICIAL_MANIFEST.json", "SOURCES.json"):
        manifest_path = root / filename
        if not manifest_path.exists():
            continue
        for row in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]:
            data = (root / row["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
            if "git_blob_sha1" in row:
                blob = b"blob " + str(len(data)).encode() + b"\0" + data
                assert hashlib.sha1(blob).hexdigest() == row["git_blob_sha1"]


@pytest.mark.parametrize("name,entry", [(n, e) for n, es in ENTRIES.items() for e in es])
def test_standalone_help_without_site_packages(name, entry, tmp_path):
    # -S excludes site-packages: help must not need torch, models or NumPy.
    result = subprocess.run(
        [sys.executable, "-S", str(PACKAGES / name / entry), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
