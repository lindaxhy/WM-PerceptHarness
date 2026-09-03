from __future__ import annotations

import hashlib
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from las_repro.api import create_app
from las_repro.config import Settings
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
