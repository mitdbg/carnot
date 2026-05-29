"""Shared runtime: LLM client with rate limiting and retry.

All LLM traffic goes through the direct Gemini API (AI Studio) via the
`google-genai` SDK, authenticated by `GEMINI_API_KEY` from `.env`.

Google Search grounding (used by `lookup_external`) is requested by passing
`use_google_search=True` to `LLMClient.call`; AI Studio supplies it via
`Tool(google_search=...)` with the same grounding-metadata shape as the
Vertex endpoint.

Rate limiting
-------------
A single process-wide rate limiter paces all LLM calls at `config.llm_rpm`
requests/minute. The configured rpm is converted to requests/second
(rps = rpm/60); slots refill continuously at that rate and the bucket caps at
one second's worth of capacity, so bursts are bounded to ~1s of requests
rather than a full minute. `call()` blocks on `acquire()` until a slot is free,
then issues the API call.

Retry
-----
Any exception from the SDK is treated as transient and retried with
exponential backoff (`llm_retry_initial_delay_s`, doubling each attempt,
capped at `llm_retry_max_delay_s`) up to `llm_max_retries` extra attempts.
Each failure is logged to stderr. After exhaustion the last exception propagates.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal


# Reasoning-effort knob shared across LLM backends. Maps cleanly to
# the Gemini SDK's `thinking_level` enum.
# Replaces the legacy `thinking_budget: int` API — Gemini 3 treats the
# legacy int as a soft suggestion only, not a cap. "off" is kept for
# explicit no-thinking intent (legacy thinking_budget=0); "minimal" is
# the cheapest Gemini-3 thinking tier (e.g. Gemini 3 Flash).
Effort = Literal["off", "minimal", "low", "medium", "high"]
_EFFORT_VALUES = ("off", "minimal", "low", "medium", "high")

from google import genai
from google.genai import types

from skunk.config import SkunkConfig

if TYPE_CHECKING:
    from skunk.models import HarnessContext


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


def load_env_file(path: Path) -> None:
    """Read a KEY=VALUE .env file and set any unset env vars from it."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n?")


def strip_code_fence(s: str) -> str:
    """Strip a leading/trailing ```lang fence from an LLM response body."""
    s = s.strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s)
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def parse_json_response(text: str) -> Any | None:
    """Strip optional code fences and parse the body as JSON. Returns None
    on parse failure rather than raising — LLM responses are noisy."""
    try:
        return json.loads(strip_code_fence(text))
    except json.JSONDecodeError:
        return None


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


def extract_search_queries(api_resp: Any) -> list[str]:
    """`web_search_queries` from grounding_metadata — populated whenever the
    Google Search tool actually executed, even when Vertex returns no
    `grounding_chunks`. Used as a 'search ran' signal independent of chunk
    surfacing, which Vertex AI populates much less reliably than AI Studio."""
    try:
        gm = api_resp.candidates[0].grounding_metadata
        if gm and gm.web_search_queries:
            return list(gm.web_search_queries)
    except (IndexError, AttributeError):
        pass
    return []


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
    # web_search_queries from Vertex grounding metadata. Populated whenever the
    # Google Search tool actually fired; more reliable than grounding_urls /
    # grounding_titles, which Vertex often omits even on grounded responses.
    search_queries: list[str] = field(default_factory=list)


def _make_genai_client() -> genai.Client:
    """Build a direct-Gemini (AI Studio) genai.Client from `GEMINI_API_KEY`.
    Auth via api-key; no GCP project required."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (required for Gemini API)")
    return genai.Client(api_key=api_key)


class LLMClient:
    """LLM client. All calls go through AI Studio (direct Gemini API);
    `use_google_search=True` attaches the GoogleSearch grounding tool."""

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._gemini_client: genai.Client | None = None

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = _make_genai_client()
        return self._gemini_client

    def call(
        self,
        system: str,
        user: str,
        images: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        use_google_search: bool = False,
        ctx: "HarnessContext | None" = None,
    ) -> LLMResponse:
        if effort not in _EFFORT_VALUES:
            raise ValueError(f"effort must be one of {_EFFORT_VALUES}, got {effort!r}")
        return self._call_gemini(
            system, user, images, temperature, effort, ctx, with_search=use_google_search,
        )

    def _retry_call(self, do_call: "Callable[[], LLMResponse]") -> LLMResponse:
        """Run `do_call` under the rate limiter with exponential-backoff retry.
        `do_call` owns API invocation, timing, response parsing, and the
        ctx.emit on success; this helper only owns rate-limit acquire +
        retry policy."""
        limiter = _get_rate_limiter(self._config.llm_rpm)
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s
        max_delay = self._config.llm_retry_max_delay_s

        for attempt in range(max_retries + 1):
            limiter.acquire()
            try:
                return do_call()
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

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "CLUSTERING",
        dim: int = 768,
        model: str = "gemini-embedding-001",
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Batched embedding via Vertex AI. Build-time only — used by
        `page_index.embed_cluster` for L1-bucket clustering at corpus-build
        time. The retrieval path is embedding-free; do not wire this into
        per-query code.

        `task_type` defaults to "CLUSTERING" per Google's docs. Output is
        L2-unnormalized; callers L2-normalize before cosine. Input is
        chunked at `batch_size` because Gemini's embed endpoint caps
        single requests at 100 contents. No retries / no rate limiter:
        this is a one-shot build-time call, not a sustained workload.
        """
        if not texts:
            return []
        client = self._get_gemini_client()
        cfg = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=dim,
        )
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            resp = client.models.embed_content(
                model=model, contents=chunk, config=cfg,
            )
            vecs = [list(e.values) for e in resp.embeddings]
            if len(vecs) != len(chunk):
                raise RuntimeError(
                    f"embed model {model!r} returned {len(vecs)} vectors "
                    f"for {len(chunk)} inputs; model likely requires batch_size=1"
                )
            out.extend(vecs)
        return out

    @staticmethod
    def _gemini_parts(user: str, images: list[tuple[str, str]] | None) -> list[Any]:
        parts: list[Any] = []
        if images:
            for mime_type, b64_data in images:
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime_type)
                )
        parts.append(types.Part.from_text(text=user))
        return parts

    @staticmethod
    def _effort_to_thinking_config(effort: "Effort") -> "types.ThinkingConfig":
        # Map skunk's Effort onto Gemini's ThinkingLevel enum, which is
        # the only knob that actually caps thinking spend for Gemini 3.
        # The legacy `thinking_budget=int` is accepted but soft-bucketed
        # by Google — not a hard cap.
        if effort == "off":
            return types.ThinkingConfig(thinking_budget=0)
        level_map = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
        }
        return types.ThinkingConfig(thinking_level=level_map[effort])

    @staticmethod
    def _gemini_config(
        system: str,
        temperature: float,
        effort: "Effort",
        *,
        with_search: bool,
    ) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=65535,
            temperature=temperature,
            thinking_config=LLMClient._effort_to_thinking_config(effort),
            tools=[types.Tool(google_search=types.GoogleSearch())] if with_search else None,
        )

    def _call_gemini(
        self,
        system: str,
        user: str,
        images: list[tuple[str, str]] | None,
        temperature: float,
        effort: "Effort",
        ctx: "HarnessContext | None",
        *,
        with_search: bool,
    ) -> LLMResponse:
        """Vertex AI Gemini call. `with_search=True` attaches the Google Search
        grounding tool and surfaces grounding metadata on the response."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(user, images)
        gen_config = self._gemini_config(
            system, temperature, effort, with_search=with_search
        )
        model = self._config.llm_model

        def do() -> LLMResponse:
            t0 = time.monotonic()
            api_resp = client.models.generate_content(
                model=model, contents=parts, config=gen_config,
            )
            latency_s = time.monotonic() - t0
            usage = api_resp.usage_metadata
            output_text = (api_resp.text or "").strip()
            if ctx is not None:
                ctx.emit(
                    "llm", "call",
                    model=model,
                    temperature=temperature,
                    effort=effort,
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
                grounding_urls=extract_grounding_urls(api_resp) if with_search else [],
                grounding_titles=extract_grounding_titles(api_resp) if with_search else [],
                search_queries=extract_search_queries(api_resp) if with_search else [],
            )

        return self._retry_call(do)
