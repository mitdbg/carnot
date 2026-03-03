"""Tier 4 — Smoke tests for the docker-compose deployment.

These tests verify that the backend, database, and frontend are up and
responding correctly after ``docker compose up``.  They do not require
Auth0 credentials (the endpoints tested here are unauthenticated).
"""

from __future__ import annotations

import requests


class TestHealthSmoke:
    """Smoke tests that confirm the docker-compose stack is alive."""

    def test_backend_health(self, backend_root_url: str):
        """``GET /health`` returns 200 with ``{"status": "healthy"}``.

        This endpoint is unauthenticated and confirms both the backend
        process and its database connection (Alembic migrations ran).
        """
        resp = requests.get(f"{backend_root_url}/health", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"

    def test_root_endpoint(self, backend_root_url: str):
        """``GET /`` returns the API identity message."""
        resp = requests.get(f"{backend_root_url}/", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["message"] == "Carnot Web API"
        assert body["status"] == "running"

    def test_config_endpoint(self, backend_url: str):
        """``GET /api/config/`` returns local-mode directory paths.

        In ``LOCAL_ENV=true`` mode the backend serves paths under
        ``/carnot/`` inside the container.
        """
        resp = requests.get(f"{backend_url}/config/", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["base_dir"] == "/carnot/"
        assert body["data_dir"] == "/carnot/data/"
        assert body["shared_data_dir"] == "/carnot/shared/"

    def test_frontend_reachable(self, docker_compose_up):
        """``GET http://localhost:80`` returns 200 (nginx serves the React app).

        We don't inspect the HTML content — just confirm the frontend
        container is up and nginx is serving something.
        """
        resp = requests.get("http://localhost:80", timeout=10)
        assert resp.status_code == 200
