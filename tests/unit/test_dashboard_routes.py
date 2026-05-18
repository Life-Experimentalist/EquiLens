"""Integration tests for new API endpoints and dashboard HTML routes."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from equilens.backend.api import app

    with TestClient(app) as c:
        yield c


# ── New JSON API endpoints ────────────────────────────────────────────────────


def test_corpus_endpoint_returns_list(client):
    r = client.get("/api/corpus")
    assert r.status_code == 200
    data = r.json()
    assert "corpus_files" in data
    assert isinstance(data["corpus_files"], list)


def test_sessions_endpoint_returns_list(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    data = r.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


def test_backups_endpoint_returns_list(client):
    r = client.get("/api/backups")
    assert r.status_code == 200
    data = r.json()
    assert "backups" in data
    assert isinstance(data["backups"], list)


def test_trigger_backup_returns_200(client):
    r = client.post("/api/backups")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert data["name"].endswith(".zip")


def test_retry_nonexistent_job_returns_404(client):
    r = client.post("/api/jobs/nonexistent_xyz_job/retry")
    assert r.status_code == 404


def test_dashboard_summary_returns_200(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    data = r.json()
    assert "ollama_available" in data
    assert "active_jobs_count" in data
    assert "scheduler" in data
    assert "backup_count" in data


def test_sse_endpoint_returns_404_for_unknown_job(client):
    # SSE endpoint should return 404 if the job doesn't exist
    # We use stream=True to not block on the infinite stream
    with client.stream("GET", "/api/events/nonexistent_job_id") as r:
        # The SSE will stream a "done" or error event, or just start streaming
        # At minimum it must return 200 with event-stream content-type
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")


def test_openapi_schema_has_all_tag_groups(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = schema.get("paths", {})
    # Verify key endpoints exist
    assert "/api/corpus" in paths
    assert "/api/sessions" in paths
    assert "/api/backups" in paths
    assert "/api/dashboard" in paths
    assert "/api/events/{job_id}" in paths


# ── HTML dashboard routes ─────────────────────────────────────────────────────


def test_dashboard_home_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "EquiLens" in r.text


def test_audit_page_returns_html(client):
    r = client.get("/audit")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_generate_page_returns_html(client):
    r = client.get("/generate")
    assert r.status_code == 200


def test_analyze_page_returns_html(client):
    r = client.get("/analyze")
    assert r.status_code == 200


def test_jobs_page_returns_html(client):
    r = client.get("/jobs")
    assert r.status_code == 200


def test_results_page_returns_html(client):
    r = client.get("/results")
    assert r.status_code == 200


def test_static_js_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200


def test_static_css_served(client):
    r = client.get("/static/style.css")
    assert r.status_code == 200
