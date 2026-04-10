"""Integration tests for the ``/api/query`` routes.

Validates the end-to-end query lifecycle against the live Docker Compose
stack: plan generation and query execution with a real LLM.

These tests require **three** environment gates:
    - ``RUN_INTEGRATION_TESTS=1`` — Docker Compose stack is available.
    - ``RUN_TESTS_WITH_LLM=1`` — live LLM calls are permitted.
    - ``OPENAI_API_KEY`` — a valid OpenAI key in the environment.

Run with::

    RUN_INTEGRATION_TESTS=1 RUN_TESTS_WITH_LLM=1 \\
        pytest tests/pytest/integration/test_query_api.py -v
"""

from __future__ import annotations

import contextlib
import json
import os
from uuid import uuid4

import pytest
import requests

# ── Skip unless live LLM is enabled ─────────────────────────────────────

_LLM_ENABLED = os.getenv("RUN_TESTS_WITH_LLM", "").lower() in (
    "1",
    "true",
    "yes",
)

pytestmark = pytest.mark.llm


# ── Helpers ──────────────────────────────────────────────────────────────

# A small test file with unambiguous content so the LLM can answer
# a simple question deterministically.
_TEST_FILENAME = "query_test_animals.txt"
_TEST_CONTENT = (
    "Animal Facts\n"
    "============\n"
    "1. The giraffe is the tallest living terrestrial animal.\n"
    "2. The blue whale is the largest animal ever known to have existed.\n"
    "3. The cheetah is the fastest land animal, reaching speeds up to 70 mph.\n"
    "4. The emperor penguin is the tallest of all penguin species.\n"
    "5. The African elephant is the largest living terrestrial animal by mass.\n"
)


def _upload_test_file(backend_url: str, auth_headers: dict) -> None:
    """Upload the test animals file to the container's data directory."""
    resp = requests.post(
        f"{backend_url}/files/upload",
        headers=auth_headers,
        files={"file": (_TEST_FILENAME, _TEST_CONTENT.encode(), "text/plain")},
        data={"path": "/carnot/data/"},
        timeout=15,
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"


def _find_uploaded_path(backend_url: str, auth_headers: dict) -> str:
    """Return the container-side path of the uploaded test file."""
    resp = requests.get(
        f"{backend_url}/files/browse",
        params={"path": "/carnot/data/"},
        headers=auth_headers,
        timeout=10,
    )
    assert resp.status_code == 200, resp.text
    for item in resp.json()["items"]:
        if item["display_name"] == _TEST_FILENAME:
            return item["path"]
    raise AssertionError(f"{_TEST_FILENAME!r} not found after upload")


def _create_dataset(
    backend_url: str,
    auth_headers: dict,
    file_path: str,
    name: str = "query-integration-animals",
) -> int:
    """Create a dataset from the test file and return its ID."""
    resp = requests.post(
        f"{backend_url}/datasets/",
        headers=auth_headers,
        json={
            "name": name,
            "shared": False,
            "annotation": "Animal facts for query integration test",
            "files": [file_path],
        },
        timeout=15,
    )
    assert resp.status_code == 200, f"Dataset creation failed: {resp.text}"
    return resp.json()["id"]


def _parse_sse_events(response: requests.Response) -> list[dict]:
    """Parse an SSE text/event-stream response into a list of JSON dicts.

    Each ``data: {...}`` line is decoded; keep-alive comments and blank
    lines are skipped.

    Requires:
        - *response* has ``Content-Type: text/event-stream`` (or similar).

    Returns:
        A list of parsed JSON objects from the SSE ``data:`` lines.
    """
    events: list[dict] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith(":"):
            # SSE comment / keep-alive
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(payload))
    return events


# ── Module-scoped fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def query_test_dataset(backend_with_llm_key, auth_headers):
    """Upload a file and create a dataset once per test module.

    This avoids repeating the expensive upload + dataset creation for
    every test.  The dataset is cleaned up after all tests in this
    module complete.

    Returns:
        A tuple ``(dataset_id, file_path)`` for use in query tests.
    """
    url = backend_with_llm_key

    _upload_test_file(url, auth_headers)
    file_path = _find_uploaded_path(url, auth_headers)
    dataset_id = _create_dataset(url, auth_headers, file_path)

    yield dataset_id, file_path

    # Cleanup
    with contextlib.suppress(Exception):
        requests.delete(
            f"{url}/datasets/{dataset_id}",
            headers=auth_headers,
            timeout=10,
        )
    with contextlib.suppress(Exception):
        requests.post(
            f"{url}/files/delete",
            json={"files": [file_path]},
            headers=auth_headers,
            timeout=10,
        )


@pytest.fixture(scope="module")
def query_plan(backend_with_llm_key, auth_headers, query_test_dataset):
    """Generate a logical plan once and reuse it across tests.

    This avoids the expensive (~3 min) data-discovery + planning step
    for every test.  The plan endpoint is called once; the resulting
    plan dict and session_id are shared.

    Returns:
        A tuple ``(plan_dict, session_id, dataset_id)``.
    """
    dataset_id, _ = query_test_dataset
    session_id = str(uuid4())

    resp = requests.post(
        f"{backend_with_llm_key}/query/plan",
        headers=auth_headers,
        json={
            "query": "Which animal is the tallest?",
            "dataset_ids": [dataset_id],
            "session_id": session_id,
        },
        timeout=600,
    )
    assert resp.status_code == 200, (
        f"Plan generation failed ({resp.status_code}): {resp.text[:500]}"
    )
    body = resp.json()
    return body["plan"], session_id, dataset_id


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _LLM_ENABLED, reason="RUN_TESTS_WITH_LLM not set")
class TestQueryAPI:
    """End-to-end query lifecycle through the web API with a real LLM.

    Representation invariant:
        Tests in this class require a running Docker Compose stack **and**
        a valid ``OPENAI_API_KEY``.  The LLM key is provisioned into the
        backend once per session via the ``backend_with_llm_key`` fixture.
        The dataset and plan are created once per module via module-scoped
        fixtures to avoid repeating the expensive data-discovery cycle.

    Abstraction function:
        Represents the contract between the frontend query interface and
        the FastAPI query routes: plan generation (``/query/plan``) and
        query execution (``/query/execute``) with streamed results.
    """

    # ── Validation edge cases ────────────────────────────────────────

    def test_plan_rejects_empty_query(
        self,
        backend_with_llm_key: str,
        auth_headers: dict,
    ) -> None:
        """``POST /api/query/plan`` with an empty query returns 400.

        Returns:
            400 with a detail message about an empty query.
        """
        resp = requests.post(
            f"{backend_with_llm_key}/query/plan",
            headers=auth_headers,
            json={
                "query": "   ",
                "dataset_ids": [1],
                "session_id": str(uuid4()),
            },
            timeout=15,
        )
        assert resp.status_code == 400, resp.text

    def test_plan_rejects_no_datasets(
        self,
        backend_with_llm_key: str,
        auth_headers: dict,
    ) -> None:
        """``POST /api/query/plan`` with no datasets returns 400.

        Returns:
            400 with a detail message about requiring at least one dataset.
        """
        resp = requests.post(
            f"{backend_with_llm_key}/query/plan",
            headers=auth_headers,
            json={
                "query": "Which animal is the tallest?",
                "dataset_ids": [],
                "session_id": str(uuid4()),
            },
            timeout=15,
        )
        assert resp.status_code == 400, resp.text

    # ── Plan generation ──────────────────────────────────────────────

    def test_plan_query(
        self,
        query_plan: tuple[dict, str, int],
    ) -> None:
        """``POST /api/query/plan`` generates a logical plan for a query.

        The plan endpoint is called once via the module-scoped
        ``query_plan`` fixture (which also uploads a file and creates a
        dataset).  This test verifies the plan structure.

        Returns:
            The plan fixture yields a non-empty dict representing the
            logical execution plan, a session ID, and a dataset ID.
        """
        plan, session_id, dataset_id = query_plan
        assert isinstance(plan, dict), f"Expected plan to be a dict, got {type(plan)}"
        assert len(plan) > 0, "Plan dict is empty"
        assert isinstance(session_id, str)
        assert isinstance(dataset_id, int)

    # ── Full execution ───────────────────────────────────────────────

    def test_execute_query(
        self,
        backend_with_llm_key: str,
        auth_headers: dict,
        query_plan: tuple[dict, str, int],
    ) -> None:
        """``POST /api/query/execute`` runs a pre-generated plan and
        streams back results via SSE.

        Reuses the plan from the module-scoped ``query_plan`` fixture to
        avoid a second expensive planning cycle.  Validates that the SSE
        stream contains at least a ``result`` or ``done`` event with no
        errors.

        Returns:
            A ``text/event-stream`` response containing at least one
            ``result`` or ``done`` event.
        """
        plan, session_id, dataset_id = query_plan

        exec_resp = requests.post(
            f"{backend_with_llm_key}/query/execute",
            headers=auth_headers,
            json={
                "query": "Which animal is the tallest?",
                "dataset_ids": [dataset_id],
                "session_id": session_id,
                "plan": plan,
            },
            timeout=600,
            stream=True,
        )
        assert exec_resp.status_code == 200, (
            f"Execute failed ({exec_resp.status_code}): {exec_resp.text[:500]}"
        )

        events = _parse_sse_events(exec_resp)
        event_types = [e.get("type") for e in events]

        # We must have received at least one event
        assert len(events) > 0, "No SSE events received from /query/execute"

        # Check for error events first — their messages are most diagnostic
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 0, (
            f"Query execution produced errors: "
            f"{[e.get('message', '') for e in error_events]}"
        )

        # Verify we got a terminal result or done event
        assert "done" in event_types or "result" in event_types, (
            f"Expected 'done' or 'result' event in SSE stream, got: {event_types}"
        )
