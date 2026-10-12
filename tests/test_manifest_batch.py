"""Persistent GPU slots, manifest safety, resume, and request throttling."""
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize('free_gib', [1, 20])
def test_two_gpu_batch_keeps_cv_evidence_and_respects_cache_pressure(tmp_path, silent_video, monkeypatch, free_gib):
    from percept_harness.cli import main
    # Exercise both cache policies regardless of the test host's free disk space.
    monkeypatch.setattr('percept_harness.batch.shutil.disk_usage',
                        lambda path: SimpleNamespace(free=free_gib * 1024**3))
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
    cache_hits = [json.loads(Path(value["result_file"]).read_text())["data"]["cv_evidence"]["cache_hit"]
                  for value in status["jobs"].values()]
    if free_gib >= 12:
        assert all(cache_hits)
    else:
        assert not all(cache_hits)
        assert not list((tmp_path / "cv-cache").rglob("manifest.json"))
    from percept_harness.batch import _verify_cv_archive
    for value in status["jobs"].values():
        evidence = json.loads(Path(value["result_file"]).read_text())["data"]["cv_evidence"]
        _verify_cv_archive(Path(value["cv_archive"]), evidence["manifest_sha256"])


def test_shards_partition_every_job_once():
    jobs = [f"dataset/model/sample_{i}" for i in range(100)]
    shards = [{j for j in jobs if assignment(j, 4) == i} for i in range(4)]
    assert set.union(*shards) == set(jobs)
    assert sum(map(len, shards)) == len(jobs)
