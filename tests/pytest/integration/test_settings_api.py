"""Integration tests for the ``/api/settings`` routes.

Validates user settings management against the live Docker Compose stack:
empty initial state, saving API keys, and masked retrieval.

All tests require ``RUN_INTEGRATION_TESTS=1`` and valid Auth0 credentials.
"""

from __future__ import annotations

import os

import requests


class TestSettingsAPI:
    """CRUD operations on the ``/api/settings`` endpoints.

    Representation invariant:
        Tests run against a fresh database.  The first ``GET`` for a new
        user returns an empty dict; subsequent ``POST``/``GET`` pairs
        validate encryption round-tripping.

    Abstraction function:
        Represents the contract between the frontend settings panel and
        the FastAPI settings routes, exercised through
        nginx → uvicorn → PostgreSQL with encrypted storage.
    """

    def test_get_empty_settings(
        self,
        backend_url: str,
        auth_headers: dict,
    ) -> None:
        """``GET /api/settings/`` for a user with no saved keys returns
        an empty dict (``{}``), or masked keys when ``RUN_TESTS_WITH_LLM=1``
        has provisioned keys via a session-scoped fixture.

        Returns:
            200 with ``{}`` (no keys) or masked values (keys provisioned).
        """
        resp = requests.get(
            f"{backend_url}/settings/",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        llm_enabled = os.environ.get("RUN_TESTS_WITH_LLM", "0") == "1"
        if llm_enabled:
            # When LLM tests are enabled a session-scoped fixture may
            # have already saved real API keys, so we accept masked
            # values (e.g. "...SCgA") as well as empty.
            for key, value in body.items():
                assert value == "" or value is None or value.startswith("..."), (
                    f"Expected empty or masked value for {key}, got: {value!r}"
                )
        else:
            # Without LLM tests, no keys should have been saved yet
            if body:
                for value in body.values():
                    assert value == "" or value is None, (
                        f"Expected empty settings, got non-empty value: {value!r}"
                    )

    def test_save_api_keys(
        self,
        backend_url: str,
        auth_headers: dict,
    ) -> None:
        """``POST /api/settings/keys`` saves API key values.

        Returns:
            200 with ``{"status": "success"}``.
        """
        resp = requests.post(
            f"{backend_url}/settings/keys",
            headers=auth_headers,
            json={
                "OPENAI_API_KEY": "sk-test-integration-key-12345678",
                "ANTHROPIC_API_KEY": "",
                "GEMINI_API_KEY": "",
                "TOGETHER_API_KEY": "",
            },
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "success"

    def test_get_masked_settings(
        self,
        backend_url: str,
        auth_headers: dict,
    ) -> None:
        """After saving keys, ``GET /api/settings/`` returns masked
        versions (e.g., ``...5678``).

        Requires:
            ``test_save_api_keys`` has run in this session.

        Returns:
            200 with ``OPENAI_API_KEY`` showing masked last-4 characters.
        """
        # First ensure a key is saved
        requests.post(
            f"{backend_url}/settings/keys",
            headers=auth_headers,
            json={
                "OPENAI_API_KEY": "sk-test-integration-key-12345678",
                "ANTHROPIC_API_KEY": "",
                "GEMINI_API_KEY": "",
                "TOGETHER_API_KEY": "",
            },
            timeout=10,
        )

        resp = requests.get(
            f"{backend_url}/settings/",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        openai_key = body.get("OPENAI_API_KEY", "")
        assert openai_key, "Expected a masked OPENAI_API_KEY, got empty"
        # Masked format: "...5678" (last 4 chars visible)
        assert openai_key.startswith("..."), (
            f"Expected masked key starting with '...', got {openai_key!r}"
        )
        assert openai_key.endswith("5678"), (
            f"Expected masked key ending with '5678', got {openai_key!r}"
        )
