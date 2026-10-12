"""Persistent GPU slots, manifest safety, resume, and request throttling."""
import json
from pathlib import Path

import pytest

from test_runner import silent_video
from percept_harness.batch import read_manifest, assignment, RequestGate


def test_manifest_rejects_duplicate_jobs_and_output_escape(tmp_path, silent_video):
    p = tmp_path / "manifest.jsonl"
    for ids in [["../outside"], ["same", "same"]]:
        p.write_text("".join(json.dumps({"job_id": i, "video_path": str(silent_video)}) + "\n" for i in ids))
        with pytest.raises(ValueError):
            read_manifest(p)


def test_request_gate_ramps_and_backs_off_without_waiting_in_logger():
    gate = RequestGate(9)
    for _ in range(8):
        with gate:
            pass
    assert gate.limit == 4
    gate.throttled()
    assert gate.limit == 2
    assert gate.snapshot()["rate_limits"] == 1


def test_two_gpu_batch_keeps_conservative_cv_resume_and_reuses_artifacts(tmp_path, silent_video, monkeypatch):
    from percept_harness.cli import main
    p = tmp_path / "manifest.jsonl"
    p.write_text("".join(json.dumps({"job_id": f"model/sample_{i}", "video_path": str(silent_video)}) + "\n" for i in range(3)))
    monkeypatch.setenv("PERCEPT_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("PERCEPT_CV_CACHE_ROOT", str(tmp_path / "cv-cache"))
    root = tmp_path / "out"
    args = ["annotate", "--manifest", str(p), "--template", "embodied_action_captioning",
            "--backend", "fake", "--cv", "fake", "--cv-devices", "0,1", "--output", str(root)]
    assert main(args) == 0
    status = json.loads((root / "shard-00.status.json").read_text())
    assert status["counts"] == {"pending": 0, "running": 0, "completed": 3, "failed": 0}
    for value in status["jobs"].values():
        result = json.loads(Path(value["result_file"]).read_text())
        assert result["data"]["cv_evidence"]["status"] == "available"
    assert main(args) == 0
    status = json.loads((root / "shard-00.status.json").read_text())
    assert not any(value["resumed"] for value in status["jobs"].values())
    assert all(json.loads(Path(value["result_file"]).read_text())["data"]["cv_evidence"]["cache_hit"] for value in status["jobs"].values())


def test_shards_partition_every_job_once():
    jobs = [f"dataset/model/sample_{i}" for i in range(100)]
    shards = [{j for j in jobs if assignment(j, 4) == i} for i in range(4)]
    assert set.union(*shards) == set(jobs)
    assert sum(map(len, shards)) == len(jobs)
