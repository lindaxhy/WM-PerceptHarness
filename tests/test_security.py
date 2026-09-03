from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from las_repro.api import create_app
from las_repro.config import Settings
from las_repro.security import redact
from las_repro.store import SQLiteTaskStore


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_path=tmp_path / "tasks.sqlite3",
        api_key_sha256=hashlib.sha256(b"local-test-key").hexdigest(),
    )


@pytest.fixture
def client(settings):
    store = SQLiteTaskStore(settings.database_path)
    store.initialize()
    return TestClient(create_app(settings, store))


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Basic local-test-key"}, {"Authorization": "Bearer wrong-key"}],
)
def test_protected_routes_reject_missing_malformed_and_invalid_bearer_tokens(client, headers):
    response = client.post("/api/v1/submit", headers=headers, json={})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "Unauthorized"}


def test_bearer_token_allows_request_validation_to_run(client):
    response = client.post(
        "/api/v1/submit",
        headers={"Authorization": "Bearer local-test-key"},
        json={},
    )

    assert response.status_code == 422


def test_redact_recursively_masks_sensitive_keys_without_mutating_input():
    value = {
        "apiKey": "top-secret",
        "nested": [{"AUTHORIZATION": "Bearer token"}, {"safe": "value"}],
        "details": {"password_hint": "also-secret"},
    }

    redacted = redact(value)

    assert redacted == {
        "apiKey": "***",
        "nested": [{"AUTHORIZATION": "***"}, {"safe": "value"}],
        "details": {"password_hint": "***"},
    }
    assert value["apiKey"] == "top-secret"
    assert value["nested"][0]["AUTHORIZATION"] == "Bearer token"
    assert value["details"]["password_hint"] == "also-secret"
