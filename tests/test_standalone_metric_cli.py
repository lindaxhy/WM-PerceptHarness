"""Dispatch and argument forwarding for the standalone ``percept score`` commands.

These tests never load a metric model: they check command registration, help,
argument forwarding and exit-code propagation only.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from percept_harness import cli
from percept_harness.video_metrics import (
    dynamic_degree,
    motion_smoothness,
    physical_commonsense,
    psnr,
    semantic_adherence,
    standalone,
)

MODULES = {
    "semantic-adherence": semantic_adherence,
    "physical-commonsense": physical_commonsense,
    "psnr": psnr,
    "dynamic-degree": dynamic_degree,
    "motion-smoothness": motion_smoothness,
}

REQUIRED = {
    "semantic-adherence": ["--input-csv", "in.csv", "--checkpoint", "ckpt", "--output", "out"],
    "physical-commonsense": ["--input-csv", "in.csv", "--checkpoint", "ckpt", "--output", "out"],
    "psnr": ["--manifest", "manifest.jsonl", "--output-dir", "out"],
    "dynamic-degree": ["--mode", "official", "--video", "v.mp4", "--cache-dir", "cache",
                       "--gpu", "0", "--output", "out.json"],
    "motion-smoothness": ["--mode", "cpu-staging", "--video", "v.mp4", "--cache-dir", "cache",
                          "--gpu", "0", "--output", "out.json"],
}

RUNNERS = {
    "semantic-adherence": "01_semantic_adherence/modified/sapc/run_sa.py",
    "physical-commonsense": "02_physical_commonsense/modified/sapc/run_pc.py",
    "psnr": "06_psnr/modified/run_psnr.py",
    "dynamic-degree": "07_dynamic_degree/run_dynamic_degree.py",
    "motion-smoothness": "08_motion_smoothness/run_motion_smoothness.py",
}


@pytest.mark.parametrize("name", MODULES)
def test_score_registry_points_at_standalone_module(name):
    assert cli._SCORE_MODULES[name] == f".video_metrics.{name.replace('-', '_')}"


@pytest.mark.parametrize("name", MODULES)
def test_help_names_the_harness_command(name, capsys):
    with pytest.raises(SystemExit) as raised:
        MODULES[name].main(["--help"])
    assert raised.value.code == 0
    assert f"usage: percept score {name}" in capsys.readouterr().out


def test_score_dispatch_reaches_the_module(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["score", "psnr", "--help"])
    assert raised.value.code == 0
    assert "usage: percept score psnr" in capsys.readouterr().out


@pytest.mark.parametrize("name", MODULES)
def test_dry_run_prints_the_packaged_runner(name, capsys):
    assert MODULES[name].main([*REQUIRED[name], "--dry-run"]) == 0
    error = capsys.readouterr().err.replace("\\", "/")
    assert RUNNERS[name] in error
    assert "Running: " in error


@pytest.mark.parametrize("name", MODULES)
def test_runner_exit_code_is_forwarded(name, tmp_path):
    module = MODULES[name]
    script = tmp_path / "metric_packages" / module.PACKAGE / module.RUNNER
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(7)\n", encoding="utf-8")
    assert module.main([*REQUIRED[name], "--metric-packages-root", str(tmp_path),
                        "--python", sys.executable]) == 7


def test_missing_runner_is_reported(tmp_path, capsys):
    assert psnr.main([*REQUIRED["psnr"], "--metric-packages-root", str(tmp_path),
                      "--python", sys.executable]) == 1
    assert "Missing standalone runner" in capsys.readouterr().err


def test_root_accepts_checkout_or_metric_packages(tmp_path):
    (tmp_path / "metric_packages").mkdir()
    assert standalone.resolve_root(tmp_path) == (tmp_path / "metric_packages").resolve()


def test_psnr_forwards_limit_and_workers(tmp_path, monkeypatch):
    output = tmp_path / "arguments.json"
    script = tmp_path / "metric_packages" / psnr.PACKAGE / psnr.RUNNER
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, os, sys\n"
        "with open(os.environ['ARGS_OUT'], 'w', encoding='utf-8') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARGS_OUT", str(output))
    arguments = [*REQUIRED["psnr"], "--limit", "2", "--workers", "5"]
    assert psnr.main([*arguments, "--metric-packages-root", str(tmp_path),
                      "--python", sys.executable]) == 0
    forwarded = json.loads(Path(output).read_text(encoding="utf-8"))
    assert forwarded == ["--manifest", "manifest.jsonl", "--output-dir", "out",
                         "--workers", "5", "--limit", "2"]
