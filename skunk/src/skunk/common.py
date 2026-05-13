"""Shared runtime: Gemini client with rate limiting/retry, and HarnessContext.

Rate limiting
-------------
A single process-wide rate limiter paces all Gemini calls at
`config.gemini_rpm` requests/minute. The configured rpm is converted to
requests/second (rps = rpm/60); slots refill continuously at that rate and
the bucket caps at one second's worth of capacity, so bursts are bounded to
~1s of requests rather than a full minute. `call()` blocks on `acquire()`
until a slot is free, then issues the API call.

Retry
-----
Any exception from the Gemini SDK is treated as transient and retried with
exponential backoff (`gemini_retry_initial_delay_s`, doubling each attempt,
capped at `gemini_retry_max_delay_s`) up to `gemini_max_retries` extra attempts.
Each failure is logged to stderr. After exhaustion the last exception
propagates.
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from skunk.config import SkunkConfig


class _RateLimiter:
    """Process-wide request rate limiter. Blocks until a request slot is free."""

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0 (got {rate_per_sec})")
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0 (got {capacity})")
        self._rate = rate_per_sec
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self) -> None:
        """Block until 1 request slot is available, then deduct it."""
        with self._cond:
            while True:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._rate
                self._cond.wait(timeout=wait_s)


_RATE_LIMITER_LOCK = threading.Lock()
_RATE_LIMITER: _RateLimiter | None = None
_RATE_LIMITER_RPM: float | None = None


def extract_grounding_urls(api_resp: Any) -> list[str]:
    urls: list[str] = []
    try:
        gm = api_resp.candidates[0].grounding_metadata
        if gm and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if chunk.web and chunk.web.uri:
                    urls.append(chunk.web.uri)
    except (IndexError, AttributeError):
        pass
    return urls


def extract_grounding_titles(api_resp: Any) -> list[str]:
    """Grounding chunk `web.title` values — typically the resolved source domain
    (e.g. "macrotrends.net"), unlike the `uri` which is a Vertex redirect."""
    titles: list[str] = []
    try:
        gm = api_resp.candidates[0].grounding_metadata
        if gm and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if chunk.web and chunk.web.title:
                    titles.append(chunk.web.title)
    except (IndexError, AttributeError):
        pass
    return titles


def _get_rate_limiter(rpm: float) -> _RateLimiter:
    """Process-singleton rate limiter. Reset if rpm changes between calls."""
    global _RATE_LIMITER, _RATE_LIMITER_RPM
    with _RATE_LIMITER_LOCK:
        if _RATE_LIMITER is None or _RATE_LIMITER_RPM != rpm:
            rate = rpm / 60.0
            _RATE_LIMITER = _RateLimiter(rate_per_sec=rate, capacity=max(1.0, rate))
            _RATE_LIMITER_RPM = rpm
        return _RATE_LIMITER


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    grounding_urls: list[str] = field(default_factory=list)
    grounding_titles: list[str] = field(default_factory=list)


class LLMClient:
    """Gemini API client with process-wide rate limiting and retry."""

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

    def call(
        self,
        system: str,
        user: str,
        images: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        thinking_budget: int = 0,
        use_google_search: bool = False,
        ctx: "HarnessContext | None" = None,
    ) -> LLMResponse:
        client = self._get_client()
        parts: list[Any] = []
        if images:
            for mime_type, b64_data in images:
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime_type)
                )
        parts.append(types.Part.from_text(text=user))

        thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
        tools = (
            [types.Tool(google_search=types.GoogleSearch())] if use_google_search else None
        )

        gen_config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=65535,
            temperature=temperature,
            thinking_config=thinking_config,
            tools=tools,
        )

        limiter = _get_rate_limiter(self._config.gemini_rpm)
        max_retries = self._config.gemini_max_retries
        delay = self._config.gemini_retry_initial_delay_s
        max_delay = self._config.gemini_retry_max_delay_s
        model = self._config.gemini_model

        for attempt in range(max_retries + 1):
            limiter.acquire()
            t0 = time.monotonic()
            try:
                api_resp = client.models.generate_content(
                    model=model, contents=parts, config=gen_config
                )
                latency_s = time.monotonic() - t0
                usage = api_resp.usage_metadata
                output_text = (api_resp.text or "").strip()
                if ctx is not None:
                    ctx.emit(
                        "llm", "call",
                        model=model,
                        temperature=temperature,
                        thinking_budget=thinking_budget,
                        latency_s=round(latency_s, 3),
                        input_tokens=getattr(usage, "prompt_token_count", None),
                        output_tokens=getattr(usage, "candidates_token_count", None),
                        total_tokens=getattr(usage, "total_token_count", None),
                        thinking_tokens=getattr(usage, "thoughts_token_count", None),
                        input_text=system + "\n\n---\n\n" + user,
                        output_text=output_text,
                    )
                return LLMResponse(
                    text=output_text,
                    latency_s=latency_s,
                    input_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                    grounding_urls=extract_grounding_urls(api_resp),
                    grounding_titles=extract_grounding_titles(api_resp),
                )
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(
                    f"[LLMClient] attempt {attempt + 1}/{max_retries + 1} failed: "
                    f"{type(e).__name__}: {e}; sleeping {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

        raise RuntimeError("unreachable: retry loop fell through")


@dataclass
class HarnessContext:
    question: str
    verbose: bool = False     # live-print orchestrator + subagent events to stdout
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events
    config: SkunkConfig = field(default_factory=SkunkConfig.from_env)
    llm_client: LLMClient | None = None  # inject a mock for tests; auto-created otherwise

    def __post_init__(self) -> None:
        if self.llm_client is None:
            self.llm_client = LLMClient(self.config)

    def emit(self, source: str, message: str, **fields: Any) -> None:
        """Record a diagnostic event. Subagents call this with their op name as `source`."""
        evt = {"source": source, "message": message, **fields}
        self.events.append(evt)
        if self.verbose:
            extra = ""
            if fields:
                bits = []
                for k, v in fields.items():
                    s = repr(v)
                    if len(s) > 200:
                        s = s[:200] + "..."
                    bits.append(f"{k}={s}")
                extra = " | " + ", ".join(bits)
            print(f"  [{source}] {message}{extra}")
