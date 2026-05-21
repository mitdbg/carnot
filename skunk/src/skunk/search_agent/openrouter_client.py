"""Minimal `OpenRouter` shim — local stand-in for the unknown `openrouter`
package the teammate imported on their branch (`from openrouter import
OpenRouter`). No such package was committed and the API shape
(`.chat.send(...)` for streamed chat, `.embeddings.generate(...)` for
embeddings) does not match any common PyPI release we could find.

This shim adapts the `openai` SDK (already a project dependency, used by
`skunk.common.LLMClient`) onto that surface so the teammate's
`SearchAgent` and `_make_vector_search` run unmodified.

When we port the agent to `skunk.common.LLMClient` (see ARCHITECTURE.md
"TODO after merge"), this file goes away.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class _ChatNamespace:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def send(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Forward to `openai.chat.completions.create`. The agent only ever
        calls this with `stream=True`; we expose both modes for symmetry.
        Returns the openai stream/response object directly — its
        `.choices[0].delta.content` (stream) / `.choices[0].message.content`
        (non-stream) shape matches what the agent already consumes."""
        return self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )


class _EmbeddingsNamespace:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def generate(self, *, input: str | list[str], model: str, **kwargs: Any) -> Any:
        """Forward to `openai.embeddings.create`. The response shape
        (`.data[N].embedding`) is already what the caller expects."""
        return self._client.embeddings.create(input=input, model=model, **kwargs)


class OpenRouter:
    """Adapter that exposes `.chat.send(...)` and `.embeddings.generate(...)`
    on top of an `openai.OpenAI` client pointed at OpenRouter's base URL."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set — required by skunk.search_agent.OpenRouter shim"
            )
        self._client = OpenAI(api_key=key, base_url=_OPENROUTER_BASE_URL)
        self.chat = _ChatNamespace(self._client)
        self.embeddings = _EmbeddingsNamespace(self._client)
