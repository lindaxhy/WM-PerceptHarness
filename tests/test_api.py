from __future__ import annotations

import copy
import hashlib
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from las_repro.api import create_app
from las_repro.config import Settings
from las_repro.export import iter_action_captions
from las_repro.pipelines.hybrid_result import (
    build_hybrid_result,
    validate_hybrid_result,
)
from las_repro.pipelines.scene_semantics import unavailable_scene_semantics
from las_repro.store import SQLiteTaskStore


@pytest.fixture
def store(tmp_path):
    value = SQLiteTaskStore(tmp_path / "tasks.sqlite3")
    value.initialize()
    return value


@pytest.fixture
def client(store):
    settings = Settings(
        database_path=store.database_path,
        api_key_sha256=hashlib.sha256(b"local-test-key").hexdigest(),
    )
    return TestClient(create_app(settings, store))


@pytest.fixture
def auth_header():
    return {"Authorization": "Bearer local-test-key"}


def test_submit_never_persists_ark_secret(client, store, auth_header):
    response = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
                "ark_api_key": "secret-ark-value",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["task_status"] == "PENDING"
    assert body["metadata"]["warnings"] == ["ark_api_key was ignored; inference is fully local"]
    task = store.get_task(body["metadata"]["task_id"])
    assert task is not None
    assert "secret-ark-value" not in json.dumps(task.payload)


def test_submit_accepts_configured_ark_alias_and_preserves_local_alias(store, auth_header):
    settings = Settings(database_path=store.database_path,
        api_key_sha256=hashlib.sha256(b"local-test-key").hexdigest(),
        ark_model_registry={"doubao-pro": "doubao-seed-2-1-pro-260628"})
    client = TestClient(create_app(settings, store))
    for alias in ("doubao-pro", "qwen3-vl-8b-instruct"):
        response = client.post("/api/v1/submit", headers=auth_header, json={
            "operator_id": "las_video_understanding", "operator_version": "v1",
            "data": {"video_url": "/allowed/demo.mp4", "task_template": "general_video_captioning", "model_name": alias}})
        assert response.status_code == 200


def test_overlapping_local_and_ark_aliases_are_rejected():
    with pytest.raises(Exception):
        Settings(ark_model_registry={"qwen3-vl-8b-instruct": "remote-id"})


def test_application_exposes_only_the_two_post_api_routes(client):
    routes = {(route.path, frozenset(route.methods or ())) for route in client.app.routes}

    assert routes == {
        ("/api/v1/submit", frozenset({"POST"})),
        ("/api/v1/poll", frozenset({"POST"})),
    }


def test_validation_error_never_echoes_malformed_nested_cloud_secret(client, auth_header):
    response = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
                "ark_api_key": {"raw_secret": "secret-ark-value"},
            },
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}
    assert "secret-ark-value" not in response.text


@pytest.mark.parametrize("unknown_alias", ["private-model-alias", "/etc/passwd"])
def test_submit_rejects_unregistered_model_alias_without_persisting_or_echoing_it(
    client, store, auth_header, unknown_alias
):
    """An unknown local-looking alias must never become durable queued work."""
    response = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
                "model_name": unknown_alias,
            },
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}
    assert unknown_alias not in response.text
    assert store.claim_task("coordinator", lease_seconds=1.0) is None


def test_poll_distinguishes_unknown_task(client, auth_header):
    response = client.post(
        "/api/v1/poll",
        headers=auth_header,
        json={
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "task_id": str(uuid.uuid4()),
        },
    )

    assert response.status_code == 404
    assert response.json()["metadata"]["business_code"] == "TASK_NOT_FOUND"


@pytest.mark.parametrize(
    "operator_id", ["las_long_video_understand", "las_video_understanding"]
)
def test_submit_and_poll_preserve_each_supported_operator(
    client, store, auth_header, operator_id
):
    """Hard-coding one operator at submit or poll must break this round trip."""
    submitted = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": operator_id,
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
            },
        },
    )

    assert submitted.status_code == 200
    task_id = submitted.json()["metadata"]["task_id"]
    task = store.get_task(task_id)
    assert task is not None
    assert task.operator_id == operator_id

    polled = client.post(
        "/api/v1/poll",
        headers=auth_header,
        json={
            "operator_id": operator_id,
            "operator_version": "v1",
            "task_id": task_id,
        },
    )

    assert polled.status_code == 200
    assert polled.json()["metadata"]["task_status"] == "PENDING"


def test_query_only_submit_persists_effective_local_template(client, store, auth_header):
    """Omitting query-mode defaulting must leave the local worker without a pipeline."""
    response = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": {
                "video_url": "https://example.test/demo.mp4",
                "query": "Describe the visible actions in order.",
            },
        },
    )

    assert response.status_code == 200
    task = store.get_task(response.json()["metadata"]["task_id"])
    assert task is not None
    assert task.payload["task_template"] == "general_video_captioning"
    assert task.payload["query"] == "Describe the visible actions in order."


@pytest.mark.parametrize(
    "data",
    [
        {"video_url": "https://example.test/demo.mp4"},
        {"video_url": "https://example.test/demo.mp4", "query": " \t "},
    ],
)
def test_submit_without_template_or_query_returns_sanitized_422(
    client, auth_header, data
):
    """Validation details must not expose rejected request data."""
    response = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": data,
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}
    assert "example.test" not in response.text


def test_submit_persists_documented_tuning_and_discards_all_cloud_fields(
    client, store, auth_header
):
    """Dropping a local option or persisting a cloud option must break this boundary."""
    cloud_values = {
        "ark_api_key": "secret-ark-value",
        "ark_endpoint_id": "secret-endpoint-value",
        "use_responses_api": False,
        "previous_response_ids": ["secret-response-value"],
        "expire_in": 60,
    }
    response = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": {
                "video_url": "https://example.test/demo.mp4",
                "task_template": "general_video_captioning",
                "query": "Describe the selected interval.",
                "fps": 5.0,
                "media_resolution": "high",
                "reasoning_effort": "medium",
                "clip_context": "low",
                "start": 1.25,
                "end": 3.5,
                **cloud_values,
            },
        },
    )

    assert response.status_code == 200
    assert len(response.json()["metadata"]["warnings"]) == 5
    task = store.get_task(response.json()["metadata"]["task_id"])
    assert task is not None
    assert task.payload == {
        "video_url": "https://example.test/demo.mp4",
        "task_template": "general_video_captioning",
        "query": "Describe the selected interval.",
        "model_name": "qwen3-vl-8b-instruct",
        "fps": 5.0,
        "media_resolution": "high",
        "reasoning_effort": "medium",
        "clip_context": "low",
        "start": 1.25,
        "end": 3.5,
    }
    persisted = json.dumps(task.payload)
    assert all(field not in persisted for field in cloud_values)
    assert all(str(value) not in persisted for value in cloud_values.values())


def test_submit_warns_for_explicit_null_cloud_fields_without_persisting_them(
    client, store, auth_header
):
    """Treating null as omitted must not suppress compatibility warnings."""
    cloud_fields = {
        "ark_api_key": None,
        "ark_endpoint_id": None,
        "use_responses_api": None,
        "previous_response_ids": None,
        "expire_in": None,
    }
    response = client.post(
        "/api/v1/submit",
        headers=auth_header,
        json={
            "operator_id": "las_video_understanding",
            "operator_version": "v1",
            "data": {
                "video_url": "/allowed/demo.mp4",
                "task_template": "general_video_captioning",
                **cloud_fields,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["warnings"] == [
        "ark_api_key was ignored; inference is fully local",
        "ark_endpoint_id was ignored; inference is fully local",
        "use_responses_api was ignored; local VideoSession caching is used",
        "previous_response_ids was ignored; local VideoSession caching is used",
        "expire_in was ignored; local task retention is configured by the service",
    ]
    task = store.get_task(response.json()["metadata"]["task_id"])
    assert task is not None
    assert all(field not in task.payload for field in cloud_fields)


def test_poll_returns_completed_result_at_top_level(client, store, auth_header):
    task = store.create_task({"video_url": "/allowed/demo.mp4"})
    claimed = store.claim_task("coordinator", lease_seconds=10, now=1.0)
    assert claimed is not None
    store.complete_task(
        task.task_id,
        {"summary": "done"},
        worker_id="coordinator",
        attempt=claimed.attempt,
        now=2.0,
    )

    response = client.post(
        "/api/v1/poll",
        headers=auth_header,
        json={
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "task_id": task.task_id,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "metadata": {
            "task_id": task.task_id,
            "task_status": "COMPLETED",
            "business_code": "0",
            "error_msg": "",
            "warnings": [],
            "progress": None,
        },
        "data": {"summary": "done"},
    }


def _complete_result(store, result):
    task = store.create_task({"video_url": "/allowed/demo.mp4"})
    claimed = store.claim_task("coordinator", lease_seconds=10, now=1.0)
    assert claimed is not None
    store.complete_task(
        task.task_id,
        result,
        worker_id="coordinator",
        attempt=claimed.attempt,
        now=2.0,
    )
    return task.task_id


def _poll(client, auth_header, task_id):
    return client.post(
        "/api/v1/poll",
        headers=auth_header,
        json={
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "task_id": task_id,
        },
    )


def _hybrid_result(*, available):
    segments = [
        {
            "action_index": 0,
            "segment_index": 0,
            "start": 0.0,
            "end": 1.0,
            "event_type": "approach",
            "start_boundary_id": "a0-start",
            "end_boundary_id": "a0-end",
            "actor": "right_gripper",
            "actor_state": "reaching",
            "skill": "reach",
            "target": "cup",
            "visual_motion_state": "active",
            "description": "right hand reaches for the cup",
            "confidence": 0.8,
        }
    ]
    stage = {
        "stage": "pass_a",
        "model_stage": "embodied_pass_a",
        "queue_seconds": 0.1,
        "wall_seconds": 0.3,
        "inference_seconds": 0.2,
        "attempt_count": 1,
        "worker_id": "cpu-0",
        "cache_hit": False,
        "provider_metrics": {"input_tokens": 12, "output_tokens": 7},
    }
    performance = {
        "stages": [stage],
        "total_seconds": 0.3,
        "repair_count": 0,
        "degradation_count": 0 if available else 1,
    }
    if not available:
        stage["provider_metrics"].pop("output_tokens")
        return build_hybrid_result(
            task_description="move cup",
            segments=segments,
            scene=unavailable_scene_semantics(),
            scene_status="unavailable",
            cv_evidence={"status": "disabled"},
            warnings=[{"code": "SCENE_SEMANTICS_UNAVAILABLE"}],
            performance=performance,
        )

    scene = unavailable_scene_semantics()
    scene.update(
        objects=[{"object_id": "cup", "name": "cup", "description": "a cup"}],
        semantic_events=[
            {
                "event_index": 0,
                "start": 0.0,
                "end": 1.0,
                "event_type": "move",
                "actor": "right_hand",
                "target_object_id": "cup",
                "description": "cup moves",
                "confidence": 0.8,
            }
        ],
        locations=[
            {
                "object_id": "cup",
                "location": "center",
                "start": 0.0,
                "end": 1.0,
                "visual_evidence": "cup visible in center",
                "confidence": 0.8,
                "branch": "scene",
                "model_stage": "scene_semantics",
                "evidence_mode": "hybrid",
                "source_track_ids": ["cup_1"],
                "source_keyframe_ids": ["frame-000"],
                "source_segment_indices": [0],
                "repair_history": ["initial"],
                "review_status": "not_required",
            }
        ],
        relations=[
            {
                "subject_object_id": "cup",
                "object_object_id": "cup",
                "relation": "unknown",
                "start": 0.0,
                "end": 1.0,
                "visual_evidence": "cup position is unchanged",
                "confidence": 0.8,
                "branch": "scene",
                "model_stage": "scene_semantics",
                "evidence_mode": "hybrid",
                "source_track_ids": ["cup_1"],
                "source_keyframe_ids": ["frame-000"],
                "source_segment_indices": [0],
                "repair_history": ["initial"],
                "review_status": "not_required",
            }
        ],
    )
    occlusion = {
        "status": "available",
        "decisions": [
            {
                "candidate_id": "occ_aaaaaaaaaaaa_0000",
                "classification": "occlusion",
                "target_entity_id": "cup",
                "occluder_entity_id": "hand",
                "events": [{"event_type": "occluded", "start": 0.2, "end": 0.4}],
                "visual_evidence": "cup is briefly hidden",
                "confidence": 0.8,
            }
        ],
        "events": [
            {
                "event_index": 0,
                "start": 0.2,
                "end": 0.4,
                "event_type": "occluded",
                "target_entity_id": "cup",
                "occluder_entity_id": "hand",
                "description": "cup is briefly hidden",
                "confidence": 0.8,
                "source_candidate_id": "occ_aaaaaaaaaaaa_0000",
                "branch": "occlusion",
                "model_stage": "occlusion_semantics",
                "evidence_mode": "hybrid",
                "source_track_ids": ["cup_1"],
                "source_keyframe_ids": ["frame-000"],
                "source_segment_indices": [0],
                "repair_history": ["initial"],
                "review_status": "unreviewed",
            }
        ],
    }
    result = build_hybrid_result(
        task_description="move cup",
        segments=segments,
        scene=scene,
        scene_status="available",
        cv_evidence={
            "status": "available",
            "artifact_key": "a" * 64,
            "manifest_sha256": "b" * 64,
            "cache_hit": False,
        },
        warnings=[],
        performance=performance,
        occlusion=occlusion,
    )
    for event in (
        result["annotation_branches"]["action_events"]
        + result["annotation_branches"]["scene_facts"]["events"]
    ):
        event.update(
            source_track_ids=["cup_1"],
            source_keyframe_ids=["frame-000"],
            evidence_mode="hybrid",
        )
    return result


@pytest.mark.parametrize("available", [False, True], ids=["disabled-cv", "available-cv"])
def test_poll_preserves_validated_hybrid_provenance_and_usage(
    client, store, auth_header, available
):
    original = _hybrid_result(available=available)
    validate_hybrid_result(original)
    stored_snapshot = copy.deepcopy(original)
    task_id = _complete_result(store, original)

    response = _poll(client, auth_header, task_id)

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"] == {
        "task_id": task_id,
        "task_status": "COMPLETED",
        "business_code": "0",
        "error_msg": "",
        "warnings": [],
        "progress": None,
    }
    validate_hybrid_result(body["data"])
    assert body["data"] == stored_snapshot
    assert store.get_task(task_id).result == stored_snapshot
    assert len(list(iter_action_captions("poll", body["data"], source_fps=10.0))) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result["annotation_branches"]["action_events"][0].update(
            source_keyframe_ids="frame-000"
        ),
        lambda result: result["performance"]["stages"][0]["provider_metrics"].update(
            input_tokens="12"
        ),
    ],
)
def test_poll_does_not_restore_fields_from_invalid_hybrid_results(
    client, store, auth_header, mutation
):
    original = _hybrid_result(available=True)
    mutation(original)
    with pytest.raises((ValueError, TypeError, KeyError)):
        validate_hybrid_result(original)
    task_id = _complete_result(store, original)

    data = _poll(client, auth_header, task_id).json()["data"]

    assert data["annotation_branches"]["action_events"][0]["source_keyframe_ids"] == "***"
    assert data["performance"]["stages"][0]["provider_metrics"]["input_tokens"] == "***"


def test_poll_does_not_restore_sensitive_names_at_unapproved_paths(
    client, store, auth_header
):
    original = _hybrid_result(available=True)
    original["unapproved"] = {
        "source_keyframe_ids": ["frame-000"],
        "artifact_key": "harmless-placeholder",
        "input_tokens": 12,
        "output_tokens": 7,
        "credential_password": "placeholder-value",
    }
    task_id = _complete_result(store, original)

    data = _poll(client, auth_header, task_id).json()["data"]

    assert data["unapproved"] == {
        "source_keyframe_ids": "***",
        "artifact_key": "***",
        "input_tokens": "***",
        "output_tokens": "***",
        "credential_password": "***",
    }


def test_poll_returns_stable_sanitized_failure(client, store, auth_header):
    task = store.create_task({"video_url": "/allowed/demo.mp4"})
    claimed = store.claim_task("coordinator", lease_seconds=10, now=1.0)
    assert claimed is not None
    store.fail_task(
        task.task_id,
        "backend crashed: AKIAIOSFODNN7EXAMPLE",
        worker_id="coordinator",
        attempt=claimed.attempt,
        now=2.0,
    )

    response = client.post(
        "/api/v1/poll",
        headers=auth_header,
        json={
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "task_id": task.task_id,
        },
    )

    assert response.status_code == 200
    assert "AKIAIOSFODNN7EXAMPLE" not in response.text
    assert response.json()["metadata"] == {
        "task_id": task.task_id,
        "task_status": "FAILED",
        "business_code": "TASK_FAILED",
        "error_msg": "Task execution failed",
        "warnings": [],
        "progress": None,
    }


@pytest.mark.parametrize(
    ("stored_operator_id", "stored_operator_version", "code"),
    [
        ("las_video_understanding", "v1", "OPERATOR_MISMATCH"),
        ("las_long_video_understand", "v0", "OPERATOR_VERSION_MISMATCH"),
    ],
)
def test_poll_distinguishes_task_operator_and_version_mismatches(
    client, store, auth_header, stored_operator_id, stored_operator_version, code
):
    task = store.create_task(
        {"video_url": "/allowed/demo.mp4"},
        operator_id=stored_operator_id,
        operator_version=stored_operator_version,
    )

    response = client.post(
        "/api/v1/poll",
        headers=auth_header,
        json={
            "operator_id": "las_long_video_understand",
            "operator_version": "v1",
            "task_id": task.task_id,
        },
    )

    assert response.status_code == 409
    assert response.json()["metadata"]["business_code"] == code
