"""End-to-end tests for the synchronous runner and the percept CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from las_repro.cli import main as cli_main
from las_repro.config import Settings
from las_repro.models.fake import FakeVideoModel
from las_repro.runner import (
    SUPPORTED_TEMPLATES,
    SyncRunner,
    collect_videos,
    run_batch,
)


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("videos")
    path = directory / "demo.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
            "-an", "-y", str(path),
        ],
        check=True,
    )
    return path


def _runner(tmp_path: Path) -> SyncRunner:
    settings = Settings(work_root=tmp_path / "work")
    return SyncRunner(
        FakeVideoModel(),
        settings,
        model_alias="qwen3-vl-8b-instruct",
    )


@pytest.mark.parametrize("template", SUPPORTED_TEMPLATES)
def test_each_template_completes_with_fake_backend(
    tmp_path: Path, silent_video: Path, template: str
) -> None:
    outcome = _runner(tmp_path).evaluate(silent_video, template)
    assert outcome.status == "completed", outcome.error
    assert outcome.data
    assert outcome.elapsed_seconds is not None


def test_unknown_template_fails_without_raising(
    tmp_path: Path, silent_video: Path
) -> None:
    outcome = _runner(tmp_path).evaluate(silent_video, "not_a_template")
    assert outcome.status == "failed"
    assert "unsupported template" in (outcome.error or "")


def test_missing_video_fails_without_raising(tmp_path: Path) -> None:
    outcome = _runner(tmp_path).evaluate(
        tmp_path / "absent.mp4", "general_video_captioning"
    )
    assert outcome.status == "failed"


def test_task_work_dir_is_cleaned_after_each_video(
    tmp_path: Path, silent_video: Path
) -> None:
    runner = _runner(tmp_path)
    outcome = runner.evaluate(silent_video, "general_video_captioning")
    assert outcome.status == "completed"
    work_root = tmp_path / "work"
    assert list(work_root.iterdir()) == []


def test_collect_videos_expands_directories_and_deduplicates(
    tmp_path: Path, silent_video: Path
) -> None:
    (tmp_path / "media").mkdir()
    copy = tmp_path / "media" / "copy.mp4"
    copy.write_bytes(silent_video.read_bytes())
    (tmp_path / "media" / "notes.txt").write_text("not a video")

    videos = collect_videos([tmp_path / "media", copy, silent_video])
    assert videos == [copy.resolve(), silent_video.resolve()]


def test_collect_videos_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_videos([tmp_path / "absent"])


def test_run_batch_writes_results_and_resumes(
    tmp_path: Path, silent_video: Path
) -> None:
    runner = _runner(tmp_path)
    output = tmp_path / "out"
    logs: list[str] = []

    first = run_batch(
        runner, [silent_video], "general_video_captioning", output, log=logs.append
    )
    assert [outcome.status for outcome in first] == ["completed"]

    record = json.loads((output / f"{silent_video.stem}.json").read_text())
    assert record["status"] == "completed"
    assert record["template"] == "general_video_captioning"
    assert record["data"]

    aggregate = [
        json.loads(line)
        for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert len(aggregate) == 1 and aggregate[0]["status"] == "completed"

    second = run_batch(
        runner, [silent_video], "general_video_captioning", output, log=logs.append
    )
    assert [outcome.status for outcome in second] == ["skipped"]
    assert any("skip" in line for line in logs)


def test_failed_video_is_not_skipped_on_resume(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    output = tmp_path / "out"
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a real video")

    first = run_batch(runner, [broken], "general_video_captioning", output, log=lambda _ : None)
    assert [outcome.status for outcome in first] == ["failed"]
    second = run_batch(runner, [broken], "general_video_captioning", output, log=lambda _: None)
    assert [outcome.status for outcome in second] == ["failed"]


def test_cli_eval_end_to_end_with_fake_backend(
    tmp_path: Path,
    silent_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LAS_WORK_ROOT", str(tmp_path / "work"))
    output = tmp_path / "out"
    code = cli_main([
        "eval",
        "--videos", str(silent_video),
        "--template", "embodied_action_captioning",
        "--backend", "fake",
        "--output", str(output),
        "--prompt-context", "visible interacted object: red container",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "1 completed" in captured.out
    record = json.loads((output / f"{silent_video.stem}.json").read_text())
    assert record["status"] == "completed"


def test_cli_eval_reports_missing_videos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli_main([
        "eval",
        "--videos", str(tmp_path / "absent"),
        "--template", "general_video_captioning",
        "--backend", "fake",
        "--output", str(tmp_path / "out"),
    ])
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_cli_eval_doubao_requires_api_key(
    tmp_path: Path,
    silent_video: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("LAS_ARK_API_KEY", raising=False)
    code = cli_main([
        "eval",
        "--videos", str(silent_video),
        "--template", "general_video_captioning",
        "--backend", "doubao",
        "--output", str(tmp_path / "out"),
    ])
    assert code == 2
    assert "LAS_ARK_API_KEY" in capsys.readouterr().err


def test_cli_eval_exit_code_is_nonzero_when_any_video_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAS_WORK_ROOT", str(tmp_path / "work"))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a real video")
    code = cli_main([
        "eval",
        "--videos", str(broken),
        "--template", "general_video_captioning",
        "--backend", "fake",
        "--output", str(tmp_path / "out"),
    ])
    assert code == 1
