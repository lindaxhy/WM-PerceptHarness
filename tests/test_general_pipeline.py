"""End-to-end behavior for deterministic general video captioning."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from las_repro.config import Settings
from las_repro.domain import InferenceStatus, TaskStatus
from las_repro.media import MediaResolver, VideoMetadata
from las_repro.models.base import ModelRequest
from las_repro.models.fake import FakeVideoModel
from las_repro.pipelines.base import PipelineRegistry
from las_repro.pipelines.general import GeneralCaptionPipeline
from las_repro.store import SQLiteTaskStore
from las_repro.workers import Coordinator, GPUWorker, JobWaitTimeout, wait_for_jobs


class _RecordingModel:
    def __init__(self, wrapped: Any) -> None:
        self.wrapped = wrapped
        self.calls: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        self.calls.append(request)
        return self.wrapped.generate(request)


class _DistinctSegmentModel(FakeVideoModel):
    """Keep planned overlapping spans visible instead of deduplicating them."""

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        if request.stage == "general_segment":
            return {
                "segments": [_event(
                    request.span.start,
                    request.span.end,
                    f"visual event at {request.span.start:.3f}",
                    scene=[f"scene {request.span.start:.3f}"],
                    subjects=["visible person"],
                    actions=["moves"],
                    visible_text=["label"],
                    uncertainty=["identity uncertain"],
                )]
            }
        return super().generate(request)


class _QueuedSegmentModel(FakeVideoModel):
    def __init__(self, segment_results: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.segment_results = list(segment_results)

    def generate(self, request: ModelRequest) -> dict[str, Any]:
        if request.stage == "general_segment":
            return self.segment_results.pop(0)
        return super().generate(request)


class _PipelineHarness:
    def __init__(
        self,
        tmp_path: Path,
        model: Any,
        *,
        duration: float = 2.0,
        segment_seconds: float = 1.2,
        overlap_seconds: float = 0.2,
        wait_timeout: float = 0.75,
        timeout_stage: str | None = None,
        summary_on_fallback_worker: bool = False,
        model_name: str = "qwen3-vl-8b-instruct",
    ) -> None:
        self.allowed = tmp_path / "allowed"
        self.allowed.mkdir()
        self.video_path = self.allowed / "video.mp4"
        self.video_path.write_bytes(b"deterministic visual fixture")
        self.store = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
        self.store.initialize()
        self.settings = Settings(
            database_path=self.store.database_path,
            work_root=tmp_path / "work",
            allowed_media_roots=(self.allowed,),
            segment_seconds=segment_seconds,
            segment_overlap_seconds=overlap_seconds,
            lease_seconds=5,
        )
        self.model = _RecordingModel(model)
        self.model_name = model_name
        self.worker = GPUWorker(
            self.store,
            self.model,
            "gpu-0",
            "cuda:0",
            model_name=model_name,
            lease_seconds=5.0,
        )
        self.fallback_worker = GPUWorker(
            self.store,
            self.model,
            "gpu-1",
            "cuda:1",
            model_name=model_name,
            lease_seconds=5.0,
        )
        self.probe_calls: list[Path] = []
        self.wait_timeouts: list[float] = []

        def probe(path: Path) -> VideoMetadata:
            self.probe_calls.append(path)
            return VideoMetadata(duration=duration, width=320, height=180, fps=10.0)

        self.probe = probe

        def wait_jobs(
            store: SQLiteTaskStore,
            task_id: str,
            job_ids: list[str] | tuple[str, ...],
            timeout: float,
        ) -> list[dict[str, Any]]:
            self.wait_timeouts.append(timeout)
            requested_jobs = [store.get_inference_job(job_id) for job_id in job_ids]
            if requested_jobs and all(
                job is not None and job.stage == timeout_stage for job in requested_jobs
            ):
                raise JobWaitTimeout("private path and token=must-not-survive")
            if (
                summary_on_fallback_worker
                and requested_jobs
                and all(job is not None and job.stage == "general_summary" for job in requested_jobs)
            ):
                fallback_at = requested_jobs[0].affinity_fallback_at
                assert fallback_at is not None
                assert self.fallback_worker.run_once(now=fallback_at - 0.001) is False
                assert self.fallback_worker.run_once(now=fallback_at) is True
                return wait_for_jobs(store, task_id, job_ids, 0.0)
            while self.worker.run_once():
                pass
            return wait_for_jobs(
                store,
                task_id,
                job_ids,
                0.0,
                monotonic=lambda: 0.0,
                sleep=lambda _: pytest.fail("terminal jobs must not sleep"),
            )

        self.wait_jobs = wait_jobs

        self.pipeline = GeneralCaptionPipeline(
            probe=probe,
            wait_jobs=wait_jobs,
            wait_timeout=wait_timeout,
        )

    def run(self, **payload_overrides: Any) -> Any:
        payload = {
            "video_url": str(self.video_path),
            "task_template": "general_video_captioning",
            "model_name": self.model_name,
        }
        payload.update(payload_overrides)
        task = self.store.create_task(payload)
        registry = PipelineRegistry()
        registry.register("general_video_captioning", lambda: self.pipeline)
        coordinator = Coordinator(
            self.store,
            MediaResolver(self.settings),
            self.settings,
            registry,
            worker_id="coordinator-0",
            cleanup_on_terminal=False,
        )

        assert coordinator.run_once() is True
        completed = self.store.get_task(task.task_id)
        assert completed is not None
        return completed


def _event(
    start: float,
    end: float,
    description: str,
    *,
    scene: list[str] | None = None,
    subjects: list[str] | None = None,
    actions: list[str] | None = None,
    visible_text: list[str] | None = None,
    uncertainty: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "start_time": start,
        "end_time": end,
        "scene": scene or ["indoor room"],
        "subjects": subjects or ["visible person"],
        "actions": actions or ["moves"],
        "visible_text": visible_text or [],
        "uncertainty": uncertainty or [],
        "description": description,
        "warnings": warnings or [],
    }


def test_general_pipeline_probes_once_and_fans_out_clip_aware_visual_jobs(
    tmp_path: Path,
) -> None:
    """Ignoring persisted FPS/clip tuning would send the wrong visual evidence."""
    harness = _PipelineHarness(
        tmp_path,
        _DistinctSegmentModel(),
        duration=4.0,
        segment_seconds=1.5,
        overlap_seconds=0.5,
    )

    completed = harness.run(
        fps=4.25,
        start=1.0,
        end=3.0,
        query="describe visible actions",
        media_resolution="high",
        reasoning_effort="low",
        clip_context="medium",
    )

    assert completed.status is TaskStatus.COMPLETED
    assert completed.result is not None
    assert completed.result["summary"]
    assert [item["start_time"] for item in completed.result["timeline"]] == [1.0, 2.0]
    assert completed.result["metadata"] == {
        "model_name": "qwen3-vl-8b-instruct",
        "duration": 4.0,
        "width": 320,
        "height": 180,
        "source_fps": 10.0,
        "sampling_fps": 4.25,
        "clip_start": 1.0,
        "clip_end": 3.0,
        "segment_seconds": 1.5,
        "segment_overlap_seconds": 0.5,
        "segment_count": 2,
    }
    assert harness.probe_calls == [harness.video_path.resolve()]
    assert harness.wait_timeouts == [0.75, 0.75]

    segment_calls = [call for call in harness.model.calls if call.stage == "general_segment"]
    assert [(call.span.start, call.span.end) for call in segment_calls] == [
        (1.0, 2.5),
        (2.0, 3.0),
    ]
    assert all(call.fps == 4.25 for call in harness.model.calls)
    assert all(call.video_session_id == completed.task_id for call in harness.model.calls)
    assert all(call.media_resolution == "high" for call in harness.model.calls)
    assert all(call.reasoning_effort == "low" for call in harness.model.calls)
    assert all(call.clip_context == "medium" for call in harness.model.calls)
    prompt = json.loads(segment_calls[0].prompt)
    assert prompt["evidence"] == "visual_frames_only"
    assert prompt["query"] == "describe visible actions"
    assert prompt["tuning"] == {
        "clip_context": "medium",
        "media_resolution": "high",
        "reasoning_effort": "low",
    }
    summary_prompt = json.loads(
        next(call.prompt for call in harness.model.calls if call.stage == "general_summary")
    )
    assert summary_prompt["tuning"] == prompt["tuning"]
    assert set(prompt["output_schema"]["$defs"]["GeneralTimelineEvent"]["required"]) == {
        "start_time",
        "end_time",
        "scene",
        "subjects",
        "actions",
        "visible_text",
        "uncertainty",
        "description",
        "warnings",
    }
    assert all(
        {
            "scene",
            "subjects",
            "actions",
            "visible_text",
            "uncertainty",
        }
        <= event.keys()
        for event in completed.result["timeline"]
    )
    assert summary_prompt["timeline"] == completed.result["timeline"]
    segment_payloads = [
        job.payload
        for job in harness.store.list_inference_jobs(completed.task_id)
        if job.stage == "general_segment"
    ]
    summary_payloads = [
        job.payload
        for job in harness.store.list_inference_jobs(completed.task_id)
        if job.stage == "general_summary"
    ]
    assert all(
        {
            key: payload[key]
            for key in ("media_resolution", "reasoning_effort", "clip_context")
        }
        == {
            "media_resolution": "high",
            "reasoning_effort": "low",
            "clip_context": "medium",
        }
        for payload in segment_payloads + summary_payloads
    )
    assert all(set(payload["schema_context"]) == {"span"} for payload in segment_payloads)
    assert all(
        set(payload["schema_context"]) == {"span", "expected_timeline"}
        for payload in summary_payloads
    )
    assert all("audio" not in json.dumps(payload).casefold() for payload in segment_payloads)


def test_general_result_reports_the_alias_bound_to_every_inference_request(
    tmp_path: Path,
) -> None:
    """Result metadata must identify the model queue that actually ran."""
    harness = _PipelineHarness(
        tmp_path,
        _DistinctSegmentModel(),
        model_name="model-b",
    )

    completed = harness.run()

    assert completed.status is TaskStatus.COMPLETED
    assert completed.result["metadata"]["model_name"] == "model-b"
    assert {
        job.model_name
        for job in harness.store.list_inference_jobs(completed.task_id)
    } == {"model-b"}
    assert {call.model_name for call in harness.model.calls} == {"model-b"}


def test_overlap_merge_normalizes_description_but_keeps_nonoverlapping_repeats(
    tmp_path: Path,
) -> None:
    """Description equality alone must never collapse separate repeated actions."""
    model = _QueuedSegmentModel(
        [
            {
                "warnings": ["scene warning"],
                "segments": [
                    _event(
                        0.2,
                        1.1,
                        "  Person   MOVES ",
                        scene=["kitchen"],
                        subjects=["person"],
                        actions=["walks"],
                        visible_text=["EXIT"],
                        uncertainty=["face unclear"],
                        warnings=["blurred"],
                    )
                ],
            },
            {
                "segments": [
                    _event(
                        1.0,
                        1.4,
                        "person moves",
                        scene=["hallway"],
                        subjects=["person", "cart"],
                        actions=["pushes cart"],
                        visible_text=["EXIT", "A12"],
                        uncertainty=["cart partly hidden"],
                        warnings=["occluded", "blurred"],
                    ),
                    _event(
                        1.4,
                        1.8,
                        "PERSON MOVES",
                        scene=["hallway"],
                        warnings=["late repeat"],
                    ),
                ]
            },
        ]
    )
    harness = _PipelineHarness(tmp_path, model)

    completed = harness.run()

    assert completed.status is TaskStatus.COMPLETED
    assert completed.result is not None
    assert completed.result["timeline"] == [
        {
            "start_time": 0.2,
            "end_time": 1.4,
            "scene": ["hallway", "kitchen"],
            "subjects": ["cart", "person"],
            "actions": ["pushes cart", "walks"],
            "visible_text": ["A12", "EXIT"],
            "uncertainty": ["cart partly hidden", "face unclear"],
            "description": "Person MOVES",
            "warnings": ["blurred", "occluded"],
        },
        {
            "start_time": 1.4,
            "end_time": 1.8,
            "scene": ["hallway"],
            "subjects": ["visible person"],
            "actions": ["moves"],
            "visible_text": [],
            "uncertainty": [],
            "description": "PERSON MOVES",
            "warnings": ["late repeat"],
        },
    ]
    assert completed.result["warnings"] == [
        "blurred",
        "late repeat",
        "occluded",
        "scene warning",
    ]


def test_invalid_segment_timestamp_fails_top_level_with_sanitized_ordinal(
    tmp_path: Path,
) -> None:
    """An out-of-span event must not reach a completed public timeline."""
    model = _QueuedSegmentModel(
        [
            {
                "segments": [
                    _event(-0.1, 0.5, "token=must-not-survive")
                ]
            },
            {"segments": []},
        ]
    )
    harness = _PipelineHarness(tmp_path, model)

    failed = harness.run()

    assert failed.status is TaskStatus.FAILED
    assert failed.error == "general segment 0 failed: event timestamps are outside its visual span"
    assert "must-not-survive" not in (failed.error or "")
    [first_job, _] = harness.store.list_inference_jobs(failed.task_id)
    assert first_job.result == {
        "_schema_validation": {
            "schema_name": "general_segment",
            "status": "invalid",
            "issue_codes": ["GENERAL_SEGMENT_TIMESTAMP_OUT_OF_SPAN"],
        }
    }
    assert "must-not-survive" not in json.dumps(first_job.result)


def test_failed_segment_job_fails_top_level_without_model_diagnostics(
    tmp_path: Path,
) -> None:
    """A worker exception must preserve the ordinal but not its sensitive text."""
    model = FakeVideoModel(
        failure_script={
            "general_segment": [
                RuntimeError("authorization=must-not-survive /private/customer/path")
            ]
        }
    )
    harness = _PipelineHarness(tmp_path, model, duration=1.0)

    failed = harness.run()

    assert failed.status is TaskStatus.FAILED
    assert failed.error == "general segment 0 failed: local inference job failed"
    assert "must-not-survive" not in (failed.error or "")
    [job] = harness.store.list_inference_jobs(failed.task_id)
    assert job.error == "model inference failed"


def test_summary_validation_gets_exactly_one_affinitized_repair(
    tmp_path: Path,
) -> None:
    """A malformed first summary should trigger one same-session correction."""
    model = FakeVideoModel(
        failure_script={
            "general_summary": [
                {"summary": "", "timeline": []},
                {
                    "summary": "repaired summary",
                    "timeline": [_event(
                        0.0,
                        2.0,
                        "deterministic visual event",
                        scene=["deterministic indoor scene"],
                        subjects=["deterministic visible subject"],
                        actions=["deterministic visible action"],
                    )],
                    "warnings": [],
                },
            ]
        }
    )
    harness = _PipelineHarness(tmp_path, model)

    completed = harness.run()

    assert completed.status is TaskStatus.COMPLETED
    assert completed.result is not None
    assert completed.result["summary"] == "repaired summary"
    assert "summary_generation_failed" not in completed.result["warnings"]
    assert [call.stage for call in harness.model.calls] == [
        "general_segment",
        "general_segment",
        "general_summary",
        "general_summary",
    ]
    repair_prompt = json.loads(harness.model.calls[-1].prompt)
    assert repair_prompt["repair"]["issues"] == ["summary_non_blank"]
    summary_jobs = [
        job
        for job in harness.store.list_inference_jobs(completed.task_id)
        if job.stage == "general_summary"
    ]
    assert [job.ordinal for job in summary_jobs] == [0, 1]
    assert summary_jobs[0].result == {
        "_schema_validation": {
            "schema_name": "general_summary",
            "status": "invalid",
            "issue_codes": ["GENERAL_SUMMARY_SCHEMA_INVALID"],
        }
    }
    assert all(job.affinity_worker_id == "gpu-0" for job in summary_jobs)
    segment_jobs = [
        job
        for job in harness.store.list_inference_jobs(completed.task_id)
        if job.stage == "general_segment"
    ]
    assert all(
        job.affinity_fallback_at
        == job.created_at + 0.25
        for job in summary_jobs
    )
    assert all(job.completed_by == "gpu-0" for job in summary_jobs)


def test_summary_affinity_has_time_for_a_second_worker_before_wait_timeout(
    tmp_path: Path,
) -> None:
    """Affinity fallback must open strictly before the bounded summary deadline."""
    harness = _PipelineHarness(
        tmp_path,
        FakeVideoModel(),
        summary_on_fallback_worker=True,
    )

    completed = harness.run()

    assert completed.status is TaskStatus.COMPLETED
    [summary_job] = [
        job
        for job in harness.store.list_inference_jobs(completed.task_id)
        if job.stage == "general_summary"
    ]
    assert summary_job.completed_by == "gpu-1"
    assert summary_job.affinity_fallback_at == summary_job.created_at + 0.25


def test_summary_job_spec_is_identical_when_pipeline_reenters_after_crash(
    tmp_path: Path,
) -> None:
    """Recovery must adopt the exact durable summary job instead of conflicting."""
    harness = _PipelineHarness(tmp_path, FakeVideoModel())
    task = harness.store.create_task(
        {
            "video_url": str(harness.video_path),
            "task_template": "general_video_captioning",
            "model_name": "qwen3-vl-8b-instruct",
        }
    )
    class SimulatedCoordinatorCrash(BaseException):
        pass

    def crash_after_summary_creation(
        store: SQLiteTaskStore,
        task_id: str,
        job_ids: list[str] | tuple[str, ...],
        timeout: float,
    ) -> list[dict[str, Any]]:
        jobs = [store.get_inference_job(job_id) for job_id in job_ids]
        if jobs and all(job is not None and job.stage == "general_summary" for job in jobs):
            raise SimulatedCoordinatorCrash
        return harness.wait_jobs(store, task_id, job_ids, timeout)

    crashing = GeneralCaptionPipeline(
        probe=harness.probe,
        wait_jobs=crash_after_summary_creation,
        wait_timeout=0.75,
    )
    crashing_registry = PipelineRegistry()
    crashing_registry.register("general_video_captioning", lambda: crashing)
    first_coordinator = Coordinator(
        harness.store,
        MediaResolver(harness.settings),
        harness.settings,
        crashing_registry,
        worker_id="coordinator-first",
        cleanup_on_terminal=False,
    )
    with pytest.raises(SimulatedCoordinatorCrash):
        first_coordinator.run_once(now=100.0)

    [before] = [
        job
        for job in harness.store.list_inference_jobs(task.task_id)
        if job.stage == "general_summary"
    ]
    assert before.status is InferenceStatus.PENDING
    interrupted = harness.store.get_task(task.task_id)
    assert interrupted is not None
    assert interrupted.status is TaskStatus.RUNNING
    assert interrupted.worker_id is None

    recovered_registry = PipelineRegistry()
    recovered_registry.register("general_video_captioning", lambda: harness.pipeline)
    recovered_coordinator = Coordinator(
        harness.store,
        MediaResolver(harness.settings),
        harness.settings,
        recovered_registry,
        worker_id="coordinator-recovered",
        cleanup_on_terminal=False,
    )
    assert recovered_coordinator.run_once(now=106.0) is True

    [after] = [
        job
        for job in harness.store.list_inference_jobs(task.task_id)
        if job.stage == "general_summary"
    ]
    completed = harness.store.get_task(task.task_id)
    assert completed is not None
    assert completed.status is TaskStatus.COMPLETED
    assert completed.attempt == 2
    assert completed.result is not None and completed.result["summary"]
    assert after.job_id == before.job_id
    assert after.payload == before.payload
    assert after.affinity_worker_id == before.affinity_worker_id
    assert after.affinity_fallback_at == before.affinity_fallback_at
    assert after.status is InferenceStatus.COMPLETED


def test_two_summary_failures_return_deterministic_timeline_fallback(
    tmp_path: Path,
) -> None:
    """Summary failure must not discard already validated visual events."""
    model = _DistinctSegmentModel(
        failure_script={
            "general_summary": [
                RuntimeError("api_key=must-not-survive"),
                {"summary": [], "timeline": []},
            ]
        }
    )
    harness = _PipelineHarness(tmp_path, model)

    completed = harness.run()

    assert completed.status is TaskStatus.COMPLETED
    assert completed.result is not None
    assert completed.result["summary"] == (
        "Visual timeline contains 2 events from 0.000 to 2.000 seconds."
    )
    assert len(completed.result["timeline"]) == 2
    assert completed.result["warnings"] == ["summary_generation_failed"]
    assert [call.stage for call in harness.model.calls].count("general_summary") == 2
    stored = json.dumps(
        [job.error for job in harness.store.list_inference_jobs(completed.task_id)]
    )
    assert "must-not-survive" not in stored


def test_summary_wait_timeout_falls_back_without_enqueuing_concurrent_repair(
    tmp_path: Path,
) -> None:
    """A nonterminal timed-out job must not race a newly enqueued repair."""
    harness = _PipelineHarness(
        tmp_path,
        _DistinctSegmentModel(),
        timeout_stage="general_summary",
    )

    completed = harness.run()

    assert completed.status is TaskStatus.COMPLETED
    assert completed.result is not None
    assert completed.result["warnings"] == ["summary_generation_failed"]
    assert [
        job.ordinal
        for job in harness.store.list_inference_jobs(completed.task_id)
        if job.stage == "general_summary"
    ] == [0]
    assert harness.wait_timeouts == [0.75, 0.75]
    [summary_job] = [
        job
        for job in harness.store.list_inference_jobs(completed.task_id)
        if job.stage == "general_summary"
    ]
    assert summary_job.status is InferenceStatus.FAILED
    assert summary_job.error == "parent task is terminal"
    assert harness.worker.run_once() is False


def test_segment_wait_timeout_fails_with_a_stable_public_reason(tmp_path: Path) -> None:
    """Coordinator persistence must not expose waiter diagnostics on timeout."""
    harness = _PipelineHarness(
        tmp_path,
        FakeVideoModel(),
        timeout_stage="general_segment",
    )

    failed = harness.run()

    assert failed.status is TaskStatus.FAILED
    assert failed.error == "general segment inference timed out"
    assert "must-not-survive" not in (failed.error or "")
    assert all(
        job.status is InferenceStatus.FAILED
        and job.error == "parent task is terminal"
        for job in harness.store.list_inference_jobs(failed.task_id)
    )
    assert harness.worker.run_once() is False


@pytest.mark.parametrize(
    "clip",
    [
        {"start": 1.0},
        {"start": -0.1, "end": 1.0},
        {"start": 0.5, "end": 0.5},
        {"start": 0.0, "end": 2.1},
    ],
)
def test_pipeline_defensively_rejects_invalid_persisted_clip_bounds(
    tmp_path: Path,
    clip: Mapping[str, float],
) -> None:
    """Corrupt legacy rows must not create out-of-video model requests."""
    harness = _PipelineHarness(tmp_path, FakeVideoModel(), duration=2.0)

    failed = harness.run(**clip)

    assert failed.status is TaskStatus.FAILED
    assert failed.error == "general caption request has invalid clip bounds"
    assert harness.model.calls == []


def test_empty_span_plan_fails_without_requesting_a_summary(tmp_path: Path) -> None:
    """No visual spans means there is no evidence from which to summarize."""
    harness = _PipelineHarness(tmp_path, FakeVideoModel())
    harness.pipeline = GeneralCaptionPipeline(
        probe=harness.probe,
        planner=lambda duration, maximum, overlap: [],
        wait_jobs=harness.wait_jobs,
        wait_timeout=0.75,
    )

    failed = harness.run()

    assert failed.status is TaskStatus.FAILED
    assert failed.error == "general segment planner returned no visual spans"
    assert harness.model.calls == []
    assert harness.store.list_inference_jobs(failed.task_id) == []


@pytest.mark.parametrize(
    "hostile_event",
    [
        {
            "start_time": 0.0,
            "end_time": 1.0,
            "description": "missing designed evidence fields",
            "warnings": [],
        },
        {
            **_event(0.0, 1.0, "otherwise valid"),
            "authorization": "must-not-survive",
        },
        {
            **_event(0.0, 1.0, "otherwise valid"),
            "subjects": ["person", 7],
        },
    ],
)
def test_segment_schema_rejects_missing_extra_and_wrong_typed_evidence(
    tmp_path: Path,
    hostile_event: dict[str, Any],
) -> None:
    """Loose model objects must not be truncated into apparently valid evidence."""
    model = _QueuedSegmentModel(
        [
            {"segments": [hostile_event]},
            {"segments": []},
        ]
    )
    harness = _PipelineHarness(tmp_path, model)

    failed = harness.run()

    assert failed.status is TaskStatus.FAILED
    assert failed.error == "general segment 0 failed: result schema is invalid"
    assert "must-not-survive" not in (failed.error or "")
