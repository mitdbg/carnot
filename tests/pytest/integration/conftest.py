"""Session-scoped fixtures for Tier 4 Docker Compose integration tests.

Manages the full lifecycle of a local docker-compose deployment:
    setup  → build context prep → compose up → health poll → yield
    teardown → compose down -v → cleanup build context → cleanup temp dirs

Provides:
    - ``docker_compose_up``: session fixture that starts/stops the stack.
    - ``backend_url``: the base API URL for the running backend.
    - ``auth_headers``: a ``{"Authorization": "Bearer <jwt>"}`` dict
      obtained via Auth0 ROPC grant against a dedicated test application.

All tests in this directory are auto-marked with ``@pytest.mark.integration``
and skipped unless ``RUN_INTEGRATION_TESTS=1`` is set.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

# ── Paths ───────────────────────────────────────────────────────────────

# Absolute path to the repo root (tests/pytest/integration/conftest.py → repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"
COMPOSE_DIR = DEPLOY_DIR / "compose"
COMPOSE_FILE = "docker-compose.yaml"
OVERRIDE_FILE = "docker-compose.local.yaml"

# Isolated temp directory for integration test data (not the user's ~/.carnot)
INTEGRATION_BASE_DIR = Path("/tmp/carnot-integration-test/carnot")
INTEGRATION_PG_DATA = Path("/tmp/carnot-integration-test/pg-data/data")

# The local override (docker-compose.local.yaml) hardcodes this path for the
# DB volume.  We must clean it before each run to avoid initdb errors when
# PostgreSQL finds a non-empty data directory from a previous session.
_LOCAL_OVERRIDE_PG_DATA = Path("/tmp/pg-data/data")

# ── Environment gate ────────────────────────────────────────────────────

_INTEGRATION_ENABLED = os.getenv("RUN_INTEGRATION_TESTS", "").lower() in (
    "1",
    "true",
    "yes",
)


def pytest_collection_modifyitems(config, items):
    """Auto-apply ``@pytest.mark.integration`` to every test in this directory
    and skip them unless ``RUN_INTEGRATION_TESTS`` is set."""
    skip_marker = pytest.mark.skip(
        reason="RUN_INTEGRATION_TESTS not set; skipping integration tests",
    )
    integration_marker = pytest.mark.integration
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(integration_marker)
            if not _INTEGRATION_ENABLED:
                item.add_marker(skip_marker)


# ── Auth0 configuration ──────────────────────────────────────────────────

# Auth0 app-level constants for the local Carnot deployment (frontend SPA).
# These are used to configure the backend container's JWT validation.
_AUTH0_DEFAULTS = {
    "VITE_AUTH0_DOMAIN": os.getenv("VITE_AUTH0_DOMAIN"),
    "VITE_AUTH0_CLIENT_ID": os.getenv("VITE_AUTH0_CLIENT_ID"),
    "VITE_AUTH0_AUDIENCE": os.getenv("VITE_AUTH0_AUDIENCE"),
}

# Auth0 ROPC credentials for the dedicated "Carnot Integration Tests"
# application.  Loaded from tests/pytest/.auth (dotenv-style).
_AUTH_FILE = Path(__file__).resolve().parent.parent / ".auth"

def _load_auth_file() -> dict[str, str]:
    """Parse ``tests/pytest/.auth`` into a dict of key-value pairs.

    Requires:
        - The ``.auth`` file exists and contains ``KEY=VALUE`` lines.

    Returns:
        A dict mapping variable names to values.  Lines that are empty
        or start with ``#`` are ignored.

    Raises:
        ``FileNotFoundError`` if the file does not exist.
    """
    result: dict[str, str] = {}
    for line in _AUTH_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key and value:
            result[key.strip()] = value.strip()
    return result


def _get_ropc_token() -> str:
    """Exchange username/password for a JWT via Auth0 ROPC grant.

    Uses the dedicated "Carnot Integration Tests" Auth0 application
    (Regular Web App with ``password`` grant enabled) so integration
    tests authenticate with real JWTs — no auth bypass required.

    The Auth0 domain used for token exchange is the **same tenant
    domain** that the backend validates against (passed via
    ``VITE_AUTH0_DOMAIN``).  This ensures the JWT issuer matches
    ``AUTH0_ISSUER`` in the backend.

    Requires:
        - ``tests/pytest/.auth`` contains ``CLIENT_SECRET``,
          ``USERNAME``, and ``PASSWORD``.
        - ``VITE_AUTH0_DOMAIN`` and ``VITE_AUTH0_AUDIENCE`` are set in
          the environment.
        - The Auth0 app has ``password`` grant enabled and the DB
          connection is enabled for it.

    Returns:
        An opaque JWT access token string.

    Raises:
        ``pytest.skip`` if credentials are missing.
        ``RuntimeError`` if the token exchange fails.
    """
    auth_vars = _load_auth_file()

    client_secret = auth_vars.get("CLIENT_SECRET")
    username = auth_vars.get("USERNAME")
    password = auth_vars.get("PASSWORD")
    if not all([client_secret, username, password]):
        pytest.skip(
            "Auth0 ROPC credentials missing from tests/pytest/.auth "
            "(need CLIENT_SECRET, USERNAME, PASSWORD)"
        )

    # Use the same Auth0 domain that the backend validates JWTs against
    # so the issuer claim matches.
    auth0_domain = _AUTH0_DEFAULTS.get("VITE_AUTH0_DOMAIN") or os.getenv("VITE_AUTH0_DOMAIN")
    audience = _AUTH0_DEFAULTS.get("VITE_AUTH0_AUDIENCE") or os.getenv("VITE_AUTH0_AUDIENCE")
    # The dedicated test app's client ID (Regular Web App with ROPC grant)
    client_id = auth_vars.get("CLIENT_ID", "ptYVVxjeXmqxWph1GzLb3gyWHcmer6Xq")

    if not auth0_domain:
        pytest.skip("VITE_AUTH0_DOMAIN not set — cannot exchange ROPC token")
    if not audience:
        pytest.skip("VITE_AUTH0_AUDIENCE not set — cannot exchange ROPC token")

    token_url = f"https://{auth0_domain}/oauth/token"
    payload = {
        "grant_type": "http://auth0.com/oauth/grant-type/password-realm",
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password,
        "audience": audience,
        "scope": "openid profile email",
        "realm": "Username-Password-Authentication",
    }

    resp = requests.post(token_url, json=payload, timeout=15)
    if resp.status_code != 200:
        body = resp.text[:500]
        raise RuntimeError(
            f"Auth0 ROPC token exchange failed ({resp.status_code}): {body}"
        )
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"Auth0 response missing access_token: {data}")
    return token


# ── Build-context helpers ───────────────────────────────────────────────

# Files/dirs that ``start_local.sh`` copies into the compose build context.
_BUILD_CONTEXT_COPIES = [
    ("app/frontend", "frontend"),
    ("app/backend", "backend"),
    ("src", "src"),
    ("pyproject.toml", "pyproject.toml"),
    ("README.md", "README.md"),
]


def _prepare_build_context() -> None:
    """Copy source files into ``deploy/compose/`` so Dockerfiles can build."""
    for src_rel, dst_name in _BUILD_CONTEXT_COPIES:
        src = REPO_ROOT / src_rel
        dst = COMPOSE_DIR / dst_name
        # Remove stale copies first
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.is_file():
            dst.unlink(missing_ok=True)
        # Copy
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Clean local artifacts that shouldn't be in the Docker build context
    for pattern in [
        "frontend/node_modules",
        "frontend/dist",
        "backend/.venv",
        "backend/__pycache__",
        "backend/src/__pycache__",
    ]:
        p = COMPOSE_DIR / pattern
        if p.is_dir():
            shutil.rmtree(p)


def _cleanup_build_context() -> None:
    """Remove files copied into ``deploy/compose/`` by ``_prepare_build_context``."""
    for _, dst_name in _BUILD_CONTEXT_COPIES:
        dst = COMPOSE_DIR / dst_name
        if dst.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
        elif dst.is_file():
            dst.unlink(missing_ok=True)


def _write_secrets() -> None:
    """Write DB secrets files into ``deploy/compose/secrets/``."""
    secrets_dir = COMPOSE_DIR / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    (secrets_dir / "db_password.txt").write_text("testpassword\n")
    (secrets_dir / "db_user.txt").write_text("testuser\n")
    (secrets_dir / "db_name.txt").write_text("testdb\n")


def _get_compose_env() -> dict[str, str]:
    """Build the environment dict for ``docker compose`` subprocesses.

    Mirrors the exports in ``deploy/start_local.sh`` plus Auth0 app
    constants (defaults or env var overrides) and test auth bypass.
    """
    env = os.environ.copy()
    # Apply Auth0 defaults first (env vars win if already set)
    for key, default in _AUTH0_DEFAULTS.items():
        env.setdefault(key, default)
    env.update(
        {
            "ENV_NAME": "dev",
            "LOCAL_ENV": "true",
            "LOCAL_BASE_DIR": str(INTEGRATION_BASE_DIR),
            "DOCKERHUB_USERNAME": "carnotlocal",
            "SETTINGS_ENCRYPTION_KEY": "12u1STDIIImTyKtTfkqwPDRCK4dCe65xHfXrPjrTeIU=",
            "VITE_API_BASE_URL": "http://localhost:8000/api",
            "BASE_ORIGINS": "http://localhost",
        },
    )
    return env


# ── Health polling ──────────────────────────────────────────────────────

_HEALTH_URL = "http://localhost:8000/health"
_MAX_WAIT_SECONDS = 120
_POLL_INTERVAL_START = 2
_POLL_INTERVAL_MAX = 10


def _wait_for_backend() -> None:
    """Block until the backend ``/health`` endpoint returns 200.

    Uses exponential back-off capped at ``_POLL_INTERVAL_MAX`` seconds.
    Raises ``TimeoutError`` after ``_MAX_WAIT_SECONDS``.
    """
    deadline = time.monotonic() + _MAX_WAIT_SECONDS
    interval = _POLL_INTERVAL_START
    while time.monotonic() < deadline:
        try:
            resp = requests.get(_HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(interval)
        interval = min(interval * 1.5, _POLL_INTERVAL_MAX)
    raise TimeoutError(
        f"Backend did not become healthy within {_MAX_WAIT_SECONDS}s"
    )


# ── Docker Compose commands ─────────────────────────────────────────────

def _compose_cmd(*args: str) -> list[str]:
    """Build a ``docker compose`` command list with base and local override."""
    return [
        "docker", "compose",
        "-f", COMPOSE_FILE,
        "-f", OVERRIDE_FILE,
        *args,
    ]


def _run_compose(
    *args: str,
    env: dict[str, str],
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run a ``docker compose`` command inside ``deploy/compose/``.

    When *capture* is ``False`` the command's stdout/stderr stream to the
    terminal in real-time — necessary for long-running builds so the
    subprocess does not block on pipe buffering.
    """
    kwargs: dict = {
        "cwd": str(COMPOSE_DIR),
        "env": env,
        "check": True,
    }
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    else:
        # Stream to the terminal so Docker build output is visible
        kwargs["stdout"] = None
        kwargs["stderr"] = None
    return subprocess.run(_compose_cmd(*args), **kwargs)


# ── Session fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def docker_compose_up():
    """Start the full docker-compose stack; tear it down after the session.

    Setup:
        1. Create isolated temp directories.
        2. Copy source into build context.
        3. Write DB secrets.
        4. ``docker compose up --build -d``.
        5. Poll ``/health`` until ready.

    Teardown:
        1. ``docker compose down -v`` (removes containers + volumes).
        2. Clean up build context.
        3. Remove temp directories.
    """
    env = _get_compose_env()

    # ── setup ───────────────────────────────────────────────────────────
    INTEGRATION_BASE_DIR.mkdir(parents=True, exist_ok=True)
    INTEGRATION_PG_DATA.mkdir(parents=True, exist_ok=True)

    # Clean stale PostgreSQL data so initdb can reinitialise cleanly.
    # The local override hardcodes /tmp/pg-data/data for the DB volume.
    if _LOCAL_OVERRIDE_PG_DATA.exists():
        shutil.rmtree(_LOCAL_OVERRIDE_PG_DATA, ignore_errors=True)
    _LOCAL_OVERRIDE_PG_DATA.mkdir(parents=True, exist_ok=True)

    _prepare_build_context()
    _write_secrets()

    try:
        _run_compose("up", "--build", "-d", env=env, capture=False)
        _wait_for_backend()
    except Exception:
        # If startup fails, still try to capture logs and tear down
        try:
            logs = subprocess.run(
                _compose_cmd("logs", "--tail=80"),
                cwd=str(COMPOSE_DIR),
                env=env,
                capture_output=True,
                text=True,
            )
            print("=== docker compose logs (last 80 lines) ===")
            print(logs.stdout[-4000:] if logs.stdout else "(empty stdout)")
            print(logs.stderr[-2000:] if logs.stderr else "(empty stderr)")
        except Exception:
            pass
        # Tear down whatever came up
        subprocess.run(
            _compose_cmd("down", "-v"),
            cwd=str(COMPOSE_DIR),
            env=env,
            capture_output=True,
        )
        _cleanup_build_context()
        raise

    yield

    # ── teardown ────────────────────────────────────────────────────────
    subprocess.run(
        _compose_cmd("down", "-v"),
        cwd=str(COMPOSE_DIR),
        env=env,
        capture_output=True,
    )
    _cleanup_build_context()

    # Remove temp dirs (integration base + hardcoded pg-data)
    shutil.rmtree("/tmp/carnot-integration-test", ignore_errors=True)
    shutil.rmtree(str(_LOCAL_OVERRIDE_PG_DATA.parent), ignore_errors=True)


@pytest.fixture(scope="session")
def backend_url(docker_compose_up) -> str:
    """Base URL for the running backend API.

    Returns:
        ``"http://localhost:8000/api"``
    """
    return "http://localhost:8000/api"


@pytest.fixture(scope="session")
def backend_root_url(docker_compose_up) -> str:
    """Root URL for the running backend (no ``/api`` prefix).

    Returns:
        ``"http://localhost:8000"``
    """
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def auth_headers(docker_compose_up) -> dict[str, str]:
    """Auth headers for integration tests using a real Auth0 JWT.

    Obtains a JWT via the Auth0 Resource Owner Password Credential
    (ROPC) grant using the dedicated "Carnot Integration Tests" Auth0
    application.  This ensures the backend validates a real token with
    proper signature, audience, and issuer — no auth bypass needed.

    Falls back to ``AUTH0_TEST_TOKEN`` env var if set (useful for
    manual debugging).

    Returns:
        A dict ``{"Authorization": "Bearer <token>"}`` suitable for passing
        to ``requests.get(..., headers=auth_headers)``.

    Raises:
        ``pytest.skip`` if credentials are missing.
        ``RuntimeError`` if the ROPC token exchange fails.
    """
    manual_token = os.getenv("AUTH0_TEST_TOKEN")
    if manual_token:
        return {"Authorization": f"Bearer {manual_token}"}

    token = _get_ropc_token()
    return {"Authorization": f"Bearer {token}"}


# ── Cleanup helpers for CRUD tests ──────────────────────────────────────


@pytest.fixture()
def created_dataset_ids(backend_url, auth_headers):
    """Track dataset IDs created during a test; delete them on teardown.

    Usage::

        def test_something(backend_url, auth_headers, created_dataset_ids):
            resp = requests.post(f"{backend_url}/datasets/", json=..., headers=auth_headers)
            created_dataset_ids.append(resp.json()["id"])
            # ... assertions ...
            # datasets are deleted automatically after the test
    """
    ids: list[int] = []
    yield ids
    for dataset_id in ids:
        with contextlib.suppress(Exception):
            requests.delete(
                f"{backend_url}/datasets/{dataset_id}",
                headers=auth_headers,
                timeout=10,
            )


@pytest.fixture()
def created_conversation_ids(backend_url, auth_headers):
    """Track conversation IDs created during a test; delete them on teardown."""
    ids: list[int] = []
    yield ids
    for cid in ids:
        with contextlib.suppress(Exception):
            requests.delete(
                f"{backend_url}/conversations/{cid}",
                headers=auth_headers,
                timeout=10,
            )


@pytest.fixture()
def uploaded_file_paths(backend_url, auth_headers):
    """Track file paths uploaded during a test; delete them on teardown."""
    paths: list[str] = []
    yield paths
    if paths:
        with contextlib.suppress(Exception):
            requests.post(
                f"{backend_url}/files/delete",
                json={"files": paths},
                headers=auth_headers,
                timeout=10,
            )


# ── LLM key provisioning ───────────────────────────────────────────────


_LLM_ENABLED = os.getenv("RUN_TESTS_WITH_LLM", "").lower() in (
    "1",
    "true",
    "yes",
)


@pytest.fixture(scope="session")
def llm_api_key() -> str:
    """Return an OpenAI API key from the environment.

    Requires:
        - ``OPENAI_API_KEY`` is set in the shell environment.
        - ``RUN_TESTS_WITH_LLM`` is truthy.

    Returns:
        The OpenAI API key string.

    Raises:
        ``pytest.skip`` if the key or env gate is missing.
    """
    if not _LLM_ENABLED:
        pytest.skip("RUN_TESTS_WITH_LLM not set; skipping live LLM test")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set; cannot run query integration tests")
    return key


@pytest.fixture(scope="session")
def backend_with_llm_key(backend_url, auth_headers, llm_api_key) -> str:
    """Save the OpenAI API key to the backend so query endpoints work.

    This is session-scoped: the key is saved once and reused across all
    query integration tests.

    Returns:
        The ``backend_url`` unchanged (convenience for chaining).

    Raises:
        ``AssertionError`` if the save request fails.
    """
    resp = requests.post(
        f"{backend_url}/settings/keys",
        headers=auth_headers,
        json={
            "OPENAI_API_KEY": llm_api_key,
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "TOGETHER_API_KEY": "",
        },
        timeout=15,
    )
    assert resp.status_code == 200, (
        f"Failed to save LLM key to backend: {resp.status_code} {resp.text}"
    )
    return backend_url
