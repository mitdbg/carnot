"""Gemini LLM client with configurable retry logic.

Use ctx.llm_client.call(...) in subagents; inject a mock LLMClient for tests via
HarnessContext(llm_client=MockLLMClient()).
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
import time
from typing import Any, ClassVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from skunk.config import SkunkConfig


def _is_transient(exc: genai_errors.APIError) -> bool:
    return getattr(exc, "code", None) in (429, 500, 503, 504)


class LLMClient:
    """Thin wrapper around the Gemini API with configurable retry logic.

    A single genai.Client is reused across calls. Transient server errors
    (429, 500, 503, 504) are retried with a fixed sleep; all other errors
    propagate immediately.

    Static system prompts are routed through Gemini's explicit cache. Cache
    state is class-level so it survives across HarnessContext instances within
    one process. If cache creation fails (too small, transient, unsupported
    model), the call falls back to passing system_instruction directly.
    """

    _system_caches: ClassVar[dict[str, str]] = {}     # sha256(system) -> cache_name
    _cache_disabled: ClassVar[set[str]] = set()       # hashes that failed to cache

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if self._config.use_vertex:
                project = os.environ.get("GOOGLE_CLOUD_PROJECT")
                if not project:
                    raise RuntimeError("GOOGLE_CLOUD_PROJECT not set (required for Vertex AI)")
                self._client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                )
            else:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY not set")
                self._client = genai.Client(api_key=api_key)
        return self._client

    def _resolve_cache(self, client: genai.Client, system: str) -> str | None:
        """Return a cached_content name for `system`, creating one lazily.

        Returns None if caching is unavailable for this system text (too small,
        transient failure, model doesn't support it). Caller should then pass
        system_instruction directly.
        """
        key = hashlib.sha256(system.encode()).hexdigest()
        if key in self._cache_disabled:
            return None
        if key in self._system_caches:
            return self._system_caches[key]
        # Gemini explicit caching requires ≥1024 input tokens. Skip the round trip for
        # short system prompts; ~4 chars/token is a safe English-text heuristic.
        if len(system) < 4000:
            self._cache_disabled.add(key)
            return None
        try:
            cache = client.caches.create(
                model=self._config.gemini_model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system,
                    ttl="3600s",
                ),
            )
            self._system_caches[key] = cache.name
            return cache.name
        except Exception as e:
            print(
                f"[LLMClient] cache create failed ({type(e).__name__}: {e}); "
                f"falling back to direct system_instruction",
                file=sys.stderr,
                flush=True,
            )
            self._cache_disabled.add(key)
            return None

    def call(
        self,
        system: str,
        user: str,
        images: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        parts: list[Any] = []
        if images:
            for mime_type, b64_data in images:
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime_type)
                )
        parts.append(types.Part.from_text(text=user))

        cache_name = self._resolve_cache(client, system) if system else None
        if cache_name:
            gen_config = types.GenerateContentConfig(
                cached_content=cache_name,
                max_output_tokens=65535,
                temperature=temperature,
            )
        else:
            gen_config = types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=65535,
                temperature=temperature,
            )

        model = self._config.gemini_model
        max_retries = self._config.gemini_max_retries
        delay = self._config.gemini_retry_delay_s

        for attempt in range(1, max_retries + 2):
            try:
                resp = client.models.generate_content(model=model, contents=parts, config=gen_config)
                return (resp.text or "").strip()
            except genai_errors.APIError as e:
                if not _is_transient(e) or attempt > max_retries:
                    raise
                print(
                    f"[LLMClient] {e.code} from Gemini ({e.status or 'transient'}), "
                    f"sleeping {delay}s then retrying (attempt {attempt}/{max_retries})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

        raise RuntimeError("unreachable: retry loop fell through")
