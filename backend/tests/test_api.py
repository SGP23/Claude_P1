"""
Integration tests for the FastAPI backend.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Ensure backend imports resolve
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient

    # Import the fastapi_app (not the socketio-wrapped 'app')
    os.chdir(os.path.join(PROJECT_ROOT, "backend"))
    from server import fastapi_app

    return TestClient(fastapi_app)


class TestRootEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "SignLens API"
        assert data["status"] == "running"
        assert "endpoints" in data

    def test_root_contains_new_endpoints(self, client):
        response = client.get("/")
        endpoints = response.json()["endpoints"]
        assert "/suggest-words" in endpoints
        assert "/complete-word" in endpoints
        assert "/groups" in endpoints


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "device" in data


class TestModelStatus:
    def test_model_status(self, client):
        response = client.get("/model-status")
        assert response.status_code == 200
        data = response.json()
        assert "loaded" in data
        assert "classes" in data
        assert "device" in data
        assert "temporal_smoothing" in data


class TestDatasetInfo:
    def test_dataset_info(self, client):
        response = client.get("/dataset-info")
        assert response.status_code == 200
        data = response.json()
        assert "class_names" in data
        assert "num_classes" in data


class TestWordSuggestions:
    def test_suggest_words(self, client):
        response = client.post("/suggest-words", json={"sentence": "HE"})
        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) == 4

    def test_suggest_words_empty(self, client):
        response = client.post("/suggest-words", json={"sentence": ""})
        assert response.status_code == 200
        data = response.json()
        assert all(s == "" for s in data["suggestions"])

    def test_complete_word(self, client):
        response = client.post(
            "/complete-word",
            json={"sentence": "I AM H", "suggestion": "HELLO"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentence"] == "I AM HELLO"


class TestGroups:
    def test_groups(self, client):
        response = client.get("/groups")
        assert response.status_code == 200
        data = response.json()
        assert "groups" in data
        assert "excluded" in data
        assert "J" in data["excluded"]
        assert "Z" in data["excluded"]


class TestLogs:
    def test_logs(self, client):
        response = client.get("/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert isinstance(data["logs"], list)


class TestConfidenceSettings:
    def test_confidence_settings(self, client):
        response = client.get("/confidence-settings")
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "always_predict"
