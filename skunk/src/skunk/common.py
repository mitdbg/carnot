"""Shared runtime: LLM clients with rate limiting/retry, and HarnessContext.

Two LLM paths
-------------
- OpenRouter (default): all non-search calls go through the OpenAI-compatible
  OpenRouter API using the `openai` SDK. Configured via OPENROUTER_API_KEY and
  `config.llm_model` (e.g. "google/gemini-2.5-flash").
- Gemini direct (search only): `lookup_external` passes `use_google_search=True`
  to get native Google Search grounding. This path uses the `google-genai` SDK
  directly with GEMINI_API_KEY. All other callers use the OpenRouter path.

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
Any exception from either SDK is treated as transient and retried with
exponential backoff (`gemini_retry_initial_delay_s`, doubling each attempt,
capped at `gemini_retry_max_delay_s`) up to `gemini_max_retries` extra attempts.
Each failure is logged to stderr. After exhaustion the last exception propagates.
"""

from __future__ import annotations

import base64
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types
from openai import OpenAI

from skunk.config import SkunkConfig

if TYPE_CHECKING:
    from skunk.dsl import Plan


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

_EMBED_LIMITER_LOCK = threading.Lock()
_EMBED_LIMITER: _RateLimiter | None = None
_EMBED_LIMITER_RPM: float | None = None


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


def _get_embed_limiter(rpm: float) -> _RateLimiter:
    """Process-singleton rate limiter for the embed endpoint. The embed quota
    sits on a different metric than chat; using a dedicated bucket lets us
    pace embed calls (e.g. 3000 RPM = 50 req/s) without contending with the
    chat limiter."""
    global _EMBED_LIMITER, _EMBED_LIMITER_RPM
    with _EMBED_LIMITER_LOCK:
        if _EMBED_LIMITER is None or _EMBED_LIMITER_RPM != rpm:
            rate = rpm / 60.0
            _EMBED_LIMITER = _RateLimiter(rate_per_sec=rate, capacity=max(1.0, rate))
            _EMBED_LIMITER_RPM = rpm
        return _EMBED_LIMITER


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    grounding_urls: list[str] = field(default_factory=list)
    grounding_titles: list[str] = field(default_factory=list)


class LLMClient:
    """LLM client: OpenRouter for all calls, direct Gemini only for Google Search grounding."""

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._openrouter_client: OpenAI | None = None
        self._gemini_client: genai.Client | None = None

    def _get_openrouter_client(self) -> OpenAI:
        if self._openrouter_client is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._openrouter_client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        return self._openrouter_client

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            if self._config.use_vertex:
                project = os.environ.get("GOOGLE_CLOUD_PROJECT")
                if not project:
                    raise RuntimeError("GOOGLE_CLOUD_PROJECT not set (required for Vertex AI)")
                self._gemini_client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                )
            else:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY not set (required for lookup_external Google Search)")
                self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

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
        if use_google_search:
            return self._call_gemini_search(system, user, images, temperature, thinking_budget, ctx)
        return self._call_openrouter(system, user, images, temperature, thinking_budget, ctx)

    def _call_openrouter(
        self,
        system: str,
        user: str,
        images: list[tuple[str, str]] | None,
        temperature: float,
        thinking_budget: int,
        ctx: "HarnessContext | None",
    ) -> LLMResponse:
        content: list[dict] = []
        if images:
            for mime_type, b64_data in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                })
        content.append({"type": "text", "text": user})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        # OpenRouter unifies reasoning across providers under `reasoning`; the
        # native Gemini `thinking.thinking_budget` field is silently dropped.
        # Map our existing semantics:
        #   0  → explicitly disabled (gemini-3-flash-preview reasons by default)
        #   -1 → max effort (Gemini's "unlimited" sentinel)
        #   >0 → token cap
        if thinking_budget == 0:
            reasoning: dict = {"enabled": False}
        elif thinking_budget < 0:
            reasoning = {"effort": "high"}
        else:
            reasoning = {"max_tokens": thinking_budget}
        extra_body: dict = {"reasoning": reasoning}

        client = self._get_openrouter_client()
        limiter = _get_rate_limiter(self._config.llm_rpm)
        max_retries = self._config.gemini_max_retries
        delay = self._config.gemini_retry_initial_delay_s
        max_delay = self._config.gemini_retry_max_delay_s
        model = self._config.llm_model

        for attempt in range(max_retries + 1):
            limiter.acquire()
            t0 = time.monotonic()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=65535,
                    temperature=temperature,
                    extra_body=extra_body,
                )
                latency_s = time.monotonic() - t0
                output_text = (resp.choices[0].message.content or "").strip()
                usage = resp.usage
                ctd = getattr(usage, "completion_tokens_details", None)
                thinking_tokens = getattr(ctd, "reasoning_tokens", None) if ctd is not None else None
                if ctx is not None:
                    ctx.emit(
                        "llm", "call",
                        model=model,
                        temperature=temperature,
                        thinking_budget=thinking_budget,
                        latency_s=round(latency_s, 3),
                        input_tokens=getattr(usage, "prompt_tokens", None),
                        output_tokens=getattr(usage, "completion_tokens", None),
                        total_tokens=getattr(usage, "total_tokens", None),
                        thinking_tokens=thinking_tokens,
                        input_text=system + "\n\n---\n\n" + user,
                        output_text=output_text,
                    )
                return LLMResponse(
                    text=output_text,
                    latency_s=latency_s,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
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

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str,
        dim: int = 768,
        model: str = "gemini-embedding-001",
        batch_size: int = 100,
        ctx: "HarnessContext | None" = None,
    ) -> list[list[float]]:
        """Embed a list of strings via the direct Gemini endpoint.

        `task_type` must be one of Gemini's documented values (e.g.
        "RETRIEVAL_DOCUMENT" for indexing, "RETRIEVAL_QUERY" for queries).
        Output is L2-unnormalized; callers should normalize before cosine.
        Inputs are split into chunks of `batch_size` (Gemini caps embed
        requests at 100 contents). Each batch goes through the same rate
        limiter and retry policy as `.call()`.
        """
        if not texts:
            return []
        client = self._get_gemini_client()
        cfg = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=dim,
        )
        limiter = _get_embed_limiter(self._config.embed_rpm)
        max_retries = self._config.gemini_max_retries

        # Embed endpoint has its own quota window (60 s buckets); the standard
        # max-delay cap (~1 s) is too short to clear a `retryDelay: 56s` hint
        # from a 429. Use a much higher cap here, and parse the server's hint
        # when present so we sleep just past the cooldown.
        EMBED_MAX_DELAY_S = 90.0

        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            delay = self._config.gemini_retry_initial_delay_s
            for attempt in range(max_retries + 1):
                limiter.acquire()
                t0 = time.monotonic()
                try:
                    resp = client.models.embed_content(
                        model=model, contents=chunk, config=cfg,
                    )
                    latency_s = time.monotonic() - t0
                    vecs = [list(e.values) for e in resp.embeddings]
                    if ctx is not None:
                        ctx.emit(
                            "llm", "embed",
                            model=model, task_type=task_type, dim=dim,
                            batch_size=len(chunk),
                            latency_s=round(latency_s, 3),
                        )
                    out.extend(vecs)
                    break
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    msg = str(e)
                    sleep_s = delay
                    m = re.search(r"retryDelay'?\s*:\s*'?(\d+(?:\.\d+)?)s", msg)
                    if m:
                        sleep_s = float(m.group(1)) + 1.0
                    elif "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                        sleep_s = max(sleep_s, 60.0)
                    sleep_s = min(sleep_s, EMBED_MAX_DELAY_S)
                    print(
                        f"[LLMClient.embed] attempt {attempt + 1}/{max_retries + 1} "
                        f"failed: {type(e).__name__}; sleeping {sleep_s:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    delay = min(delay * 2, EMBED_MAX_DELAY_S)
        return out

    def _call_gemini_search(
        self,
        system: str,
        user: str,
        images: list[tuple[str, str]] | None,
        temperature: float,
        thinking_budget: int,
        ctx: "HarnessContext | None",
    ) -> LLMResponse:
        # TECH DEBT: google-genai is kept alive only for Google Search grounding in
        # lookup_external. When OpenRouter supports a native web-search tool, delete
        # this method and remove the google-genai dependency.
        client = self._get_gemini_client()
        parts: list[Any] = []
        if images:
            for mime_type, b64_data in images:
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=mime_type)
                )
        parts.append(types.Part.from_text(text=user))

        thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)
        gen_config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=65535,
            temperature=temperature,
            thinking_config=thinking_config,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        limiter = _get_rate_limiter(self._config.llm_rpm)
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
    plan: Plan | None = None  # set by the orchestrator at execute() entry so subagents can read top-level constraints; None for tests / before planning

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
