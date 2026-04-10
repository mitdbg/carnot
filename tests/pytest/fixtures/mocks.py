"""Mock fixtures for dual-mode (mocked / live) LLM testing.

Provides:
    - ``mock_llm_config``: a synthetic LLM config dict that does **not**
      require a real API key.
    - ``mock_litellm``: patches ``litellm.completion`` and
      ``litellm.embedding`` with deterministic fakes so operator tests
      can validate wiring without network calls.

These fixtures are used by **Tier 2** tests (see ``TEST_SUITE.md``).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Environment gate ────────────────────────────────────────────────────────

LIVE_LLM = os.getenv("RUN_TESTS_WITH_LLM", "").lower() in ("1", "true", "yes")

requires_llm = pytest.mark.skipif(
    not LIVE_LLM,
    reason="RUN_TESTS_WITH_LLM not set; skipping live LLM test",
)
"""Decorator that skips a test unless ``RUN_TESTS_WITH_LLM`` is truthy."""


# ── Deterministic fake embedding ────────────────────────────────────────────

_EMBEDDING_DIM = 256


def _deterministic_embedding(text: str) -> list[float]:
    """Return a deterministic unit-length embedding derived from *text*.

    Uses SHA-256 to generate a repeatable float vector.  The result is
    normalised so cosine-similarity comparisons are meaningful in tests.

    Requires:
        - *text* is a non-empty string.

    Returns:
        A list of *_EMBEDDING_DIM* floats in the range [-1, 1].
    """
    digest = hashlib.sha256(text.encode()).digest()
    # Expand to _EMBEDDING_DIM bytes by repeated hashing.
    raw = digest
    while len(raw) < _EMBEDDING_DIM:
        raw += hashlib.sha256(raw).digest()
    values = [(b / 127.5) - 1.0 for b in raw[:_EMBEDDING_DIM]]
    # Normalise to unit length.
    norm = max(sum(v * v for v in values) ** 0.5, 1e-9)
    return [v / norm for v in values]


# ── Fake litellm responses ──────────────────────────────────────────────────

def _make_completion_response(content: str) -> MagicMock:
    """Build a minimal object that mimics ``litellm.completion(...)``.

    The response satisfies the access pattern used by
    ``LiteLLMModel.generate``::

        response.choices[0].message.model_dump(include={"role", "content", "tool_calls"})
        response.usage.prompt_tokens
        response.usage.completion_tokens
        response._hidden_params["response_cost"]
    """
    message = MagicMock()
    message.content = content
    message.role = "assistant"
    message.tool_calls = None
    message.model_dump = lambda include=None: {
        k: v
        for k, v in {"role": "assistant", "content": content, "tool_calls": None}.items()
        if include is None or k in include
    }

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.prompt_tokens_details = None
    usage.completion_tokens_details = None
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response._hidden_params = {"response_cost": 0.001}
    return response


def _make_embedding_response(texts: list[str]) -> MagicMock:
    """Build a minimal object that mimics ``litellm.embedding(...)``."""
    data = []
    for text in texts:
        item = {"embedding": _deterministic_embedding(text)}
        data.append(item)

    usage = MagicMock()
    usage.total_tokens = len(texts) * 10  # deterministic token count

    response = MagicMock()
    response.data = data
    response.usage = usage
    response._hidden_params = {"response_cost": 0.0001 * len(texts)}
    return response


# ── Default canned completion logic ─────────────────────────────────────────

def _default_completion_handler(
    model: str, messages: list[dict], **kwargs: Any
) -> MagicMock:
    """Fallback completion handler.

    Returns a simple affirmative answer.  Override per-test via the
    ``mock_litellm`` fixture's ``set_completion_handler`` helper if you
    need different behaviour.
    """
    return _make_completion_response("yes")


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_config() -> dict:
    """Synthetic LLM config that never contacts a real provider.

    Returns:
        A dict with a placeholder ``OPENAI_API_KEY`` suitable for
        constructing operators in mocked tests.
    """
    return {"OPENAI_API_KEY": "test-key-not-real"}


@pytest.fixture
def mock_litellm(monkeypatch):
    """Patch ``litellm.completion`` and ``litellm.embedding`` globally.

    The patched ``completion`` delegates to a handler that can be swapped
    per-test via ``mock_litellm.set_completion_handler(fn)``.

    The patched ``embedding`` always returns deterministic vectors derived
    from a SHA-256 hash of the input text.

    Returns:
        A namespace object with:
        - ``set_completion_handler(fn)`` — replace the completion handler
          for the current test.
        - ``completion_calls`` — list of ``(model, messages, kwargs)``
          tuples recorded for inspection.
        - ``embedding_calls`` — list of ``(model, input_texts, kwargs)``
          tuples recorded for inspection.
    """

    class _State:
        def __init__(self):
            self.completion_handler = _default_completion_handler
            self.completion_calls: list[tuple] = []
            self.embedding_calls: list[tuple] = []

        def set_completion_handler(self, fn):
            self.completion_handler = fn

    state = _State()

    def fake_completion(model, messages, **kwargs):
        state.completion_calls.append((model, messages, kwargs))
        return state.completion_handler(model, messages, **kwargs)

    def fake_embedding(model, input, **kwargs):  # noqa: A002 — shadows builtin
        texts = input if isinstance(input, list) else [input]
        state.embedding_calls.append((model, texts, kwargs))
        return _make_embedding_response(texts)

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.setattr("litellm.embedding", fake_embedding)

    return state
