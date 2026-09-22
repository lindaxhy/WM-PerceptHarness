from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_official_external import SPECS, preflight, run_official  # noqa: E402


EXTERNAL = {
    "instruction_following", "action_binding", "object_interactions", "motion_binding",
    "motion_order_understanding", "motion_rationality", "mechanics", "thermotics", "material",
    "phygen_eval_pca", "videophy_pc", "videophy_sa", "psnr", "ssim", "lpips", "fid", "fvd",
}


def _checkout(tmp_path: Path, revision: str | None) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    return source


def _runner_factory(revision: str):
    calls = []

    def runner(command, **kwargs):
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=revision + "\n", stderr="")

    return runner, calls


def test_specs_cover_all_external_caller_metrics():
    assert EXTERNAL == set(SPECS)
    assert all(spec.command_hint and spec.source_url for spec in SPECS.values())
    assert SPECS["fvd"].revision == "4700efb9afa54286b0e04473ba80a13e8461e25f"


def test_check_only_verifies_pinned_revision_and_preserves_command(tmp_path):
    source = _checkout(tmp_path, "unused")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    runner, calls = _runner_factory(SPECS["action_binding"].revision)
    command = ["python", "official.py", "--read-prompt-file", str(manifest), "--flag", "value"]
    result = run_official(metric="action_binding", source_dir=source, checkpoint=None,
                          manifest=manifest, generated=None, reference=None,
                          command=command, check_only=True, runner=runner)
    assert result["status"] == "preflight_passed"
    assert result["command"] == command
    assert calls[0][0][:4] == ["git", "-C", str(source), "rev-parse"]
    assert len(calls) == 1


def test_wrong_revision_fails_before_official_command(tmp_path):
    source = _checkout(tmp_path, "unused")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    runner, calls = _runner_factory("wrong-revision")
    with pytest.raises(ValueError, match="revision mismatch"):
        run_official(metric="object_interactions", source_dir=source, checkpoint=None,
                     manifest=manifest, generated=None, reference=None,
                     command=["python", "official.py"], check_only=True, runner=runner)
    assert len(calls) == 1


def test_iqa_requires_real_generated_and_reference_directories(tmp_path):
    source = _checkout(tmp_path, "unused")
    generated = tmp_path / "generated"
    reference = tmp_path / "reference"
    generated.mkdir()
    reference.mkdir()
    runner, _ = _runner_factory(SPECS["psnr"].revision)
    result = preflight(metric="psnr", source_dir=source, checkpoint=None, manifest=None,
                       generated=generated, reference=reference, runner=runner)
    assert result["metric"] == "psnr"
    assert result["generated"] == str(generated.resolve())


def test_run_forwards_command_and_writes_log(tmp_path):
    source = _checkout(tmp_path, None)
    generated = tmp_path / "generated"
    reference = tmp_path / "reference"
    generated.mkdir()
    reference.mkdir()
    output = tmp_path / "result.json"
    calls = []

    def runner(command, **kwargs):
        calls.append((list(command), kwargs))
        stdout = ""
        if list(command)[:4] == ["git", "-C", str(source), "rev-parse"]:
            stdout = SPECS["fvd"].revision + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    command = ["python", "official_fvd.py", "--real", str(reference), "--generated", str(generated)]
    result = run_official(metric="fvd", source_dir=source, checkpoint=None,
                          manifest=None, generated=generated, reference=reference,
                          command=command, output=output, runner=runner)
    assert result["status"] == "complete"
    assert result["command"] == command
    assert calls[1][0] == command
    assert calls[1][1]["cwd"] == str(source)
    assert Path(result["log"]).is_file()


def test_missing_pinned_source_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="requires --source-dir"):
        preflight(metric="action_binding", source_dir=None, checkpoint=None,
                  manifest=manifest, generated=None, reference=None,
                  runner=lambda *args, **kwargs: None)


def test_missing_required_checkpoint_is_rejected(tmp_path):
    source = _checkout(tmp_path, None)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="requires --checkpoint"):
        preflight(metric="videophy_pc", source_dir=source, checkpoint=None,
                  manifest=manifest, generated=None, reference=None,
                  runner=lambda *args, **kwargs: None)
