"""Shared runtime: the LLM client plus the cross-cutting types threaded between
operators and the orchestrator (`PageRef`, `AnnotatedValue`, `ExecutionContext`).

All LLM traffic goes through the direct Gemini API (AI Studio) via `google-genai`,
authenticated by `GEMINI_API_KEY`. LLM calls are paced by the process-wide
token-bucket limiter named `"llm"` (see `_RATE_LIMITS` / `get_rate_limiter`); only
transient SDK exceptions (HTTP 429 + 5xx, network timeouts / connection resets —
see `_is_retryable`) are retried with exponential backoff up to `llm_max_retries`
times. The limiter paces traffic; the retry rides out throttling and blips the
limiter can't prevent (provider-side 429, TPM quotas, multi-process fan-out)."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import logging
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, model_validator
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from skunk import trace
from skunk.config import SkunkConfig

if TYPE_CHECKING:
    from skunk.prompted_call import PromptOverride

# Reasoning-effort knob, mapped onto Gemini's `thinking_level` enum. "off" means
# no thinking; "minimal" is the cheapest thinking tier.
Effort = Literal["off", "minimal", "low", "medium", "high"]


@dataclass
class B64Image:
    """A single rendered page image ready to pass to an LLM: MIME type + base64 data."""

    mime: str
    data: str
_EFFORT_VALUES = ("off", "minimal", "low", "medium", "high")

# Process-scoped logger: retries happen with no per-question ctx in scope (see
# `_retry_call`), so they go through stdlib logging rather than `ctx.emit`.
log = logging.getLogger(__name__)


def _is_retryable(e: BaseException) -> bool:
    """True for transient failures worth retrying: HTTP 429 (throttling) and 5xx
    (server-side), plus network-layer timeouts / connection resets from the
    underlying transport (`httpx` async, `requests` sync). Non-429 4xx — bad
    request, auth, context-length overflow — is a permanent error that will never
    succeed, so it raises immediately instead of burning the retry budget."""
    if isinstance(e, genai_errors.APIError):
        code = getattr(e, "code", None)
        return code == 429 or (code is not None and 500 <= code < 600)
    # Transport faults: connection resets, read timeouts, DNS failures, etc.
    return isinstance(
        e,
        (
            httpx.TimeoutException,
            httpx.TransportError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    )


class _RateLimiter:
    """Process-wide request rate limiter. Blocks until a request slot is free."""

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0 (got {rate_per_sec})")
        self._rate = rate_per_sec
        # Burst capacity = ~1s of refill, floored at 1 token so sub-1/s rates still work.
        self._capacity = max(1.0, rate_per_sec)
        self._tokens = self._capacity
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


# Single source of truth for every external rate cap we pace against. Each row is
# `name: (env override, default rpm)`. The rpm is read ONCE, on first use of the
# name, and frozen for the rest of the run — buckets are created on demand and
# never rebuilt (rate limits are immutable for a run). To add a service: add a row
# and call `get_rate_limiter("<name>")` at the call site.
_RATE_LIMITS: dict[str, tuple[str, float]] = {
    # name          (env override,            default rpm)  # rationale
    "llm":          ("SKUNK_LLM_RPM",        1000.0),  # Gemini generation; provider-side quota
    "embed":        ("SKUNK_EMBED_RPM",       600.0),  # Gemini embeddings (search_agent.vector_search)
    "fred":         ("SKUNK_FRED_RPM",         60.0),  # hard 120/min per API key; stay well under —
                                                        # FRED escalates to an extended key-wide ban
                                                        # (persistent 429s) once the cap is tripped.
    "bls":          ("SKUNK_BLS_RPM",          50.0),  # 500/day (registered); smooth worker bursts
    "world_bank":   ("SKUNK_WORLD_BANK_RPM",  120.0),  # no published cap; stay a good citizen
    "tavily":       ("SKUNK_TAVILY_RPM",      100.0),  # ~100/min on the dev tier
}

_LIMITERS_LOCK = threading.Lock()
_LIMITERS: dict[str, _RateLimiter] = {}


def _resolve_rate_per_sec(name: str, rate_per_min: float | None) -> float:
    """Rate (req/sec) for a limiter bucket. A known service in `_RATE_LIMITS`
    reads its env override (once); any other name (e.g. a per-model `llm:<model>`
    bucket) requires an explicit `rate_per_min`."""
    if name in _RATE_LIMITS:
        env_var, default = _RATE_LIMITS[name]
        return float(os.environ.get(env_var, default)) / 60.0
    if rate_per_min is None:
        raise KeyError(f"unknown rate-limit name {name!r} and no rate_per_min given")
    return rate_per_min / 60.0


def get_rate_limiter(name: str, rate_per_min: float | None = None) -> _RateLimiter:
    """Process-wide token-bucket limiter for external service `name`, paced at its
    rpm — read ONCE at first use, then frozen for the run. Known services key off
    `_RATE_LIMITS` (env override); dynamic buckets (per-model `llm:<model>`) pass
    an explicit `rate_per_min`. All threads in this process share the named bucket.

    Scope is per process, NOT per API key: separate processes get independent
    buckets and do not coordinate, so a multi-process deployment would not be
    bounded by the underlying per-key cap. The eval harness is single-process."""
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(name)
        if lim is None:
            lim = _RateLimiter(rate_per_sec=_resolve_rate_per_sec(name, rate_per_min))
            _LIMITERS[name] = lim
        return lim


class _AsyncRateLimiter:
    """Async twin of `_RateLimiter`. Same token bucket, but `acquire` yields to
    the loop (`await asyncio.sleep`) instead of blocking a thread on a `Condition`.

    The instance is a process-wide singleton (see `_ASYNC_LIMITERS`) shared across
    every per-question event loop — and those loops run on *different* worker
    threads — so the token math IS contended. A `threading.Lock` guards the
    refill+deduct; it is held only across that synchronous section, never across
    the `await asyncio.sleep`."""

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0 (got {rate_per_sec})")
        self._rate = rate_per_sec
        self._capacity = max(1.0, rate_per_sec)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Yield until 1 request slot is available, then deduct it."""
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait_s)


# Async limiters mirror the sync `_LIMITERS` registry, keyed by the same names
# (same rpm from `_RATE_LIMITS`). Distinct instances: the sync buckets pace the
# offline build, these pace the request path. Shared across every per-question
# loop (each on its own worker thread), so creation is lock-guarded like the sync
# side.
_ASYNC_LIMITERS_LOCK = threading.Lock()
_ASYNC_LIMITERS: dict[str, _AsyncRateLimiter] = {}


def get_async_rate_limiter(name: str, rate_per_min: float | None = None) -> _AsyncRateLimiter:
    """Process-wide async token-bucket limiter for service `name`, paced at its
    rpm. Request-time analogue of `get_rate_limiter` (same `rate_per_min` rule for
    dynamic per-model buckets); the returned bucket is shared across all
    per-question loop threads (its own lock makes `acquire` thread-safe). See
    `get_rate_limiter` for the scope caveat."""
    with _ASYNC_LIMITERS_LOCK:
        lim = _ASYNC_LIMITERS.get(name)
        if lim is None:
            lim = _AsyncRateLimiter(rate_per_sec=_resolve_rate_per_sec(name, rate_per_min))
            _ASYNC_LIMITERS[name] = lim
        return lim


_MODEL_RPM: dict[str, float] | None = None


def _llm_model_rpm(model: str) -> float:
    """Per-minute request cap for an LLM `model`. Parsed once from `SKUNK_MODEL_RPM`
    ("model=rpm,..."); a model not listed falls back to `SKUNK_LLM_RPM` (default
    1000), so single-model runs are unaffected. Each model gets its own limiter
    bucket (`llm:<model>`), so mixed-model runs pace independently."""
    global _MODEL_RPM
    if _MODEL_RPM is None:
        out: dict[str, float] = {}
        for entry in os.environ.get("SKUNK_MODEL_RPM", "").split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, _, rpm = entry.partition("=")
            out[name.strip()] = float(rpm.strip())
        _MODEL_RPM = out
    return _MODEL_RPM.get(model, float(os.environ.get("SKUNK_LLM_RPM", "1000")))


def _requires_thinking(model: str) -> bool:
    """True for models that mandate thinking mode (reject `thinking_budget=0`).
    Gemini 3.x Pro tiers are thinking-only; Flash/Flash-Lite accept budget=0."""
    m = model.lower()
    return "gemini-3" in m and "pro" in m


# Active (step_idx, op) frame for the current operator call. A ContextVar, not
# thread-local, because within a question the branches run as concurrent asyncio
# tasks on that question's event loop (one loop per worker thread): a
# `threading.local` would let those sibling tasks collide on the shared thread,
# whereas a ContextVar gives each `asyncio.Task` a copy-on-write snapshot, so a
# task that rebinds the frame is isolated from its siblings. Operator steps don't
# nest (the orchestrator opens exactly one per traced call), so this holds a
# single frame, not a stack.
_step_frame: contextvars.ContextVar[tuple[int, str] | None] = contextvars.ContextVar(
    "skunk_step_frame", default=None
)


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


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None


def _make_genai_client() -> genai.Client:
    """Build a direct-Gemini (AI Studio) genai.Client from `GEMINI_API_KEY`.
    Auth via api-key; no GCP project required."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (required for Gemini API)")
    return genai.Client(api_key=api_key)


class LLMClient:
    """LLM client. All calls go through AI Studio (direct Gemini API)."""

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
        images: list[B64Image] | None = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        ctx: "ExecutionContext | None" = None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        return self._call_gemini(
            system, user, images, temperature, effort, ctx, call_site,
            model or self._config.llm_model,
        )

    async def acall(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        ctx: "ExecutionContext | None" = None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """Async twin of `call` for the request path."""
        return await self._acall_gemini(
            system, user, images, temperature, effort, ctx, call_site,
            model or self._config.llm_model,
        )

    def _retry_call(self, do_call: "Callable[[], LLMResponse]", model: str) -> LLMResponse:
        """Run `do_call` under `model`'s rate limiter with exponential-backoff
        retry. `do_call` owns the API invocation, timing, parsing, and success emit."""
        limiter = get_rate_limiter(f"llm:{model}", rate_per_min=_llm_model_rpm(model))
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s

        for attempt in range(max_retries + 1):
            limiter.acquire()
            try:
                return do_call()
            except Exception as e:
                # Log EVERY failure with its message — including the final one
                # before we re-raise — so a fatal error is never silent.
                stop = attempt == max_retries or not _is_retryable(e)
                log.warning(
                    "llm call failed (attempt %d/%d): %s: %s%s",
                    attempt + 1, max_retries + 1, type(e).__name__, e,
                    "" if stop else f"; retrying in {delay:.1f}s",
                )
                if stop:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable: retry loop fell through")

    async def _aretry_call(
        self, do_call: "Callable[[], Awaitable[LLMResponse]]", model: str
    ) -> LLMResponse:
        """Async twin of `_retry_call`: awaits `model`'s async rate limiter and the
        coroutine `do_call`, backing off via `asyncio.sleep` (never blocking the
        event loop). `do_call` owns the API invocation, timing, parsing, and emit."""
        limiter = get_async_rate_limiter(f"llm:{model}", rate_per_min=_llm_model_rpm(model))
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s

        for attempt in range(max_retries + 1):
            await limiter.acquire()
            try:
                return await do_call()
            except Exception as e:
                # Log EVERY failure with its message — including the final one
                # before we re-raise — so a fatal error is never silent.
                stop = attempt == max_retries or not _is_retryable(e)
                log.warning(
                    "llm call failed (attempt %d/%d): %s: %s%s",
                    attempt + 1, max_retries + 1, type(e).__name__, e,
                    "" if stop else f"; retrying in {delay:.1f}s",
                )
                if stop:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
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
        """Batched embedding — build-time / offline corpus-prep only (the query
        path is embedding-free). Output is L2-unnormalized; callers normalize
        before cosine. Chunked at `batch_size` (endpoint caps at 100/request).
        No retries / no rate limiter — one-shot build call."""
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
                model=model, contents=chunk, config=cfg,  # type: ignore[arg-type]
            )
            if not resp.embeddings:
                raise RuntimeError(f"embed model {model!r} returned no embeddings for {len(chunk)} inputs")
            vecs = [list(e.values or ()) for e in resp.embeddings]
            if len(vecs) != len(chunk):
                raise RuntimeError(
                    f"embed model {model!r} returned {len(vecs)} vectors "
                    f"for {len(chunk)} inputs; model likely requires batch_size=1"
                )
            out.extend(vecs)
        return out

    @staticmethod
    def _gemini_parts(user: str, images: list[B64Image] | None) -> list[Any]:
        parts: list[Any] = []
        if images:
            for img in images:
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(img.data), mime_type=img.mime)
                )
        parts.append(types.Part.from_text(text=user))
        return parts

    @staticmethod
    def _effort_to_thinking_config(effort: "Effort", model: str) -> "types.ThinkingConfig":
        # ThinkingLevel is the only knob that hard-caps thinking spend for Gemini 3
        # (the legacy thinking_budget int is soft-bucketed).
        if _requires_thinking(model):
            # Gemini 3.x Pro: thinking is mandatory AND MINIMAL is unsupported, so
            # "off"/"minimal" floor at LOW (the cheapest tier this model accepts).
            level_map = {
                "off": types.ThinkingLevel.LOW,
                "minimal": types.ThinkingLevel.LOW,
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
            }
            return types.ThinkingConfig(thinking_level=level_map[effort])
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
        model: str,
    ) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=65535,
            temperature=temperature,
            thinking_config=LLMClient._effort_to_thinking_config(effort, model),
        )

    @staticmethod
    def _usage_tokens(usage: Any) -> dict:
        """Token counts from a Gemini `usage_metadata` (best-effort — any may be
        None, e.g. when streaming omits usage)."""
        return {
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        }

    def _call_gemini(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: "Effort",
        ctx: "ExecutionContext | None",
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """Single Gemini call. `call_site` attributes the envelope log to the caller."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(user, images)
        model = model or self._config.llm_model
        gen_config = self._gemini_config(system, temperature, effort, model)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            api_resp = client.models.generate_content(
                model=model, contents=parts, config=gen_config,
            )
            latency_s = time.monotonic() - t0
            usage = api_resp.usage_metadata
            output_text = (api_resp.text or "").strip()
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return self._retry_call(do, model)

    async def _acall_gemini(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: "Effort",
        ctx: "ExecutionContext | None",
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """Async twin of `_call_gemini` — uses `client.aio.models.generate_content`."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(user, images)
        model = model or self._config.llm_model
        gen_config = self._gemini_config(system, temperature, effort, model)

        async def do() -> LLMResponse:
            t0 = time.monotonic()
            api_resp = await client.aio.models.generate_content(
                model=model, contents=parts, config=gen_config,
            )
            latency_s = time.monotonic() - t0
            usage = api_resp.usage_metadata
            output_text = (api_resp.text or "").strip()
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return await self._aretry_call(do, model)

    def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: "Callable[[str], bool] | None" = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        ctx: "ExecutionContext | None" = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        """Multi-turn streaming call, accumulating chunks until `should_stop(acc)`
        or the stream ends. `messages` are the {role, content} turns after the
        system message ('assistant' → model role, else user). Lets multi-turn
        agents share this client's rate-limit + retry + logging. `temperature`
        defaults to 0.0 — agent loops are deterministic like every other call site.
        `effort` maps onto Gemini's thinking config exactly like the single-shot
        path (`_gemini_config`), so agent loops are tunable like every other call site."""
        client = self._get_gemini_client()
        model_id = (model or self._config.llm_model).removeprefix("google/")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        gen_config = self._gemini_config(system, temperature, effort, model_id)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            resp_stream = client.models.generate_content_stream(
                model=model_id, contents=contents, config=gen_config,  # type: ignore[arg-type]
            )
            accumulated = ""
            usage = None
            for chunk in resp_stream:
                accumulated += chunk.text or ""
                usage = getattr(chunk, "usage_metadata", None) or usage
                if should_stop is not None and should_stop(accumulated):
                    break
            close = getattr(resp_stream, "close", None)
            if close is not None:
                try:  # noqa: SIM105
                    close()
                except Exception:
                    pass
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return self._retry_call(do, model_id)

    async def astream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: "Callable[[str], bool] | None" = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        ctx: "ExecutionContext | None" = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        """Async twin of `stream` — uses `client.aio.models.generate_content_stream`
        and `async for`, so the event loop runs other tasks between chunks."""
        client = self._get_gemini_client()
        model_id = (model or self._config.llm_model).removeprefix("google/")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        gen_config = self._gemini_config(system, temperature, effort, model_id)

        async def do() -> LLMResponse:
            t0 = time.monotonic()
            resp_stream = await client.aio.models.generate_content_stream(
                model=model_id, contents=contents, config=gen_config,  # type: ignore[arg-type]
            )
            accumulated = ""
            usage = None
            async for chunk in resp_stream:
                accumulated += chunk.text or ""
                usage = getattr(chunk, "usage_metadata", None) or usage
                if should_stop is not None and should_stop(accumulated):
                    break
            aclose = getattr(resp_stream, "aclose", None)
            if aclose is not None:
                try:  # noqa: SIM105
                    await aclose()
                except Exception:
                    pass
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return await self._aretry_call(do, model_id)


# --- Cross-cutting runtime types threaded between operators and the orchestrator ---


@dataclass
class PageRef:
    """Canonical page coordinate."""
    month: str | None = None        # "YYYY-MM"
    page: int | None = None         # 1-based PDF page index (canonical)

    @property
    def year(self) -> int | None:
        return int(self.month[:4]) if self.month else None

    def __post_init__(self) -> None:
        if self.page is not None and self.month is None:
            raise ValueError(
                f"PageRef with page={self.page} requires month for parsed-JSON lookup"
            )

    def __repr__(self) -> str:
        parts = []
        if self.year:
            parts.append(f"year={self.year}")
        if self.month:
            parts.append(f"month={self.month}")
        if self.page is not None:
            parts.append(f"page={self.page}")
        return f"PageRef({', '.join(parts)})"


def page_key_to_pageref(key: str) -> PageRef:
    """Parse a search-agent page key (`"YYYY_MM_pageid"` or `"YYYY-MM-pageid"`)
    into a `PageRef`. Splits on the last separator so the page id is unambiguous."""
    sep = "_" if "_" in key and key.count("_") >= 2 else "-"
    try:
        year_str, month_str, page_str = key.rsplit(sep, 2)
    except ValueError as e:
        raise ValueError(f"page key {key!r} not in YYYY{sep}MM{sep}pageid form") from e
    return PageRef(month=f"{year_str}-{month_str}", page=int(page_str))


VALUE_KIND_VOCAB: frozenset[str] = frozenset({"scalar", "vector", "table"})


class AnnotatedValue(BaseModel):
    """One described, annotated datum: a payload plus minimal metadata.

    `description` uniquely distinguishes this entry from its siblings. `unit` is
    a natural-language label (empty for non-measurements). Payload shape is set
    by `kind` (enforced at construction — cells are always primitive, no nesting):
      - "scalar": int|float|str, or a list of those (lookup_external multi-value)
      - "vector": dict[str, primitive] keyed by `index_name` labels
      - "table":  dict[str, dict[str, primitive]], outer key `row_name`, inner `col_name`

    `.frame` exposes the payload as a uniform `pd.DataFrame` so downstream code
    needn't branch on `kind`.
    """
    model_config = ConfigDict(frozen=True)

    description: str
    value: Any
    unit: str = ""
    kind: Literal["scalar", "vector", "table"] = "scalar"
    index_name: str | None = None
    row_name: str | None = None
    col_name: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> AnnotatedValue:
        # bool is an int subclass in Python — exclude it explicitly.
        def is_prim(c: Any) -> bool:
            return not isinstance(c, bool) and isinstance(c, (int, float, str))

        v = self.value
        if self.kind == "scalar":
            ok = is_prim(v) or (isinstance(v, list) and all(is_prim(c) for c in v))
            if not ok:
                raise ValueError(
                    f"kind=scalar requires int|float|str (or list of those), "
                    f"got {type(v).__name__}"
                )
        elif self.kind == "vector":
            if not (isinstance(v, dict) and all(
                isinstance(k, str) and is_prim(c) for k, c in v.items()
            )):
                raise ValueError("vector value must be flat dict[str, scalar]")
            if not self.index_name:
                raise ValueError("vector entry missing non-empty 'index_name'")
        else:  # table
            if not (isinstance(v, dict) and all(
                isinstance(r, str)
                and isinstance(row, dict)
                and all(isinstance(k, str) and is_prim(c) for k, c in row.items())
                for r, row in v.items()
            )):
                raise ValueError(
                    "table value must be dict[str, dict[str, scalar]] (no nesting)"
                )
            if not self.row_name or not self.col_name:
                raise ValueError("table entry missing non-empty 'row_name'/'col_name'")
        return self

    @property
    def frame(self):
        """Uniform pandas view of the payload: scalar → 1×N column, vector → N×1
        with `index.name == index_name`, table → R×C with named index/columns."""
        v = self.value
        label = (self.description or "value").strip() or "value"
        if self.kind == "scalar":
            rows = list(v) if isinstance(v, list) else [v]
            return pd.DataFrame({label: rows})
        if self.kind == "vector":
            s = pd.Series(v, name=label, dtype=object if not v else None)
            df = s.to_frame()
            df.index.name = self.index_name
            return df
        df = pd.DataFrame.from_dict(v, orient="index")
        df.index.name = self.row_name
        df.columns.name = self.col_name
        return df


_SAMPLE_ROWS = 2
_PAD = "         "  # 9-space continuation indent for an entry's detail lines


def _describe_entry(i: int, e: AnnotatedValue) -> list[str]:
    """Schema view of one `AnnotatedValue`: the meta line, then (for non-scalars) the
    full index labels — load-bearing, they're what generated code keys `.loc[...]` on,
    and aren't otherwise in the prompt — plus a tiny `head` sample. Scalars show in full."""
    head = f"  input_values[{i}]  description: {(e.description or '(no description)')!r}"
    if e.kind == "scalar":
        return [head, f"{_PAD}value={e.value!r}  (kind=scalar, unit={e.unit!r})"]

    df = e.frame
    n_rows, n_cols = df.shape
    if e.kind == "vector":
        meta = f"kind=vector, index_name={e.index_name!r}, unit={e.unit!r}, shape=({n_rows}, {n_cols})"
    else:
        meta = (
            f"kind=table, row_name={e.row_name!r}, col_name={e.col_name!r}, "
            f"unit={e.unit!r}, shape=({n_rows}, {n_cols})"
        )
    lines = [head, f"{_PAD}{meta}"]
    if n_rows == 0:
        lines.append(f"{_PAD}frame: (empty)")
        return lines

    if e.kind == "vector":
        lines.append(f"{_PAD}dtype: {df.dtypes.iloc[0]}")
    else:
        cols = ", ".join(str(c) for c in df.columns)
        dtypes = ", ".join(str(t) for t in df.dtypes)
        lines.append(f"{_PAD}columns: [{cols}]   dtypes: [{dtypes}]")

    labels = [str(x) for x in df.index]
    lines.append(f"{_PAD}index ({len(labels)} labels): [{', '.join(labels)}]")

    with pd.option_context("display.max_columns", None, "display.width", 120):
        rendered = df.head(_SAMPLE_ROWS).to_string()
    lines.append(f"{_PAD}sample (first {min(_SAMPLE_ROWS, n_rows)} of {n_rows} rows):")
    lines.extend(_PAD + ln for ln in rendered.splitlines())
    return lines


def input_values_desc(input_values: list[AnnotatedValue]) -> str:
    """Render `input_values` for the codegen / critique / re-planner prompts as a
    **schema view** — axis labels + dtypes + a small `head` sample, NOT a full cell
    dump. Generated code operates on the frames symbolically (full frames live in the
    exec env), so it needs the labels (to write selections) and a small sample (number
    encoding, sentinels, magnitude for the unit decision), not the interior grid.
    Codegen and critique share this one view, so there's no info asymmetry."""
    lines = [f"input_values ({len(input_values)} entries)"]
    for i, e in enumerate(input_values):
        lines.extend(_describe_entry(i, e))
    return "\n".join(lines)


@dataclass
class ExecutionContext:
    """Per-question execution state, owned by exactly one `Orchestrator` and never
    shared across questions or threads.
    """
    question: str
    uid: str | None = None    # benchmark UID, when run from the eval harness; tags every event
    verbose: bool = False     # also echo orchestrator + operator events to the console
    log_path: str | None = None  # when set, stream this question's events to that file (live, flushed)
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events
    config: SkunkConfig = field(default_factory=SkunkConfig.from_env)
    llm_client: LLMClient | None = None  # auto-created in __post_init__; pass a mock to override
    prompt_overrides: tuple[PromptOverride, ...] = ()  # corpus/few_shot/lesson overrides; operators pick out their own entries by name

    def __post_init__(self) -> None:
        if self.llm_client is None:
            self.llm_client = LLMClient(self.config)
        # The active (step_idx, op) frame lives in the module-level `_step_frame`
        # ContextVar (per asyncio task), not on the instance — see its definition.
        # Per-question log file: opened when log_path is set, written line-per-event
        # and flushed live so a single question's stream lands in its own file (and
        # survives a crash). No lock: this ctx is touched by one thread only (see the
        # class docstring), so the file write needs no synchronization.
        self._logfile = None
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._logfile = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        # Monotonic step-index allocator. Owned here (not by the caller) so the
        # ctx is the single source of step identities. Incremented synchronously
        # inside `step()` before any `await`, so the single-thread/cooperative
        # contract (see class docstring) makes it race-free without a lock.
        self._next_step_idx = 1

    def close(self) -> None:
        """Close the per-question log file, if one was opened."""
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None

    @contextmanager
    def step(self, op: str):
        """Open an observability step (a span) for one operator call, yielding its
        allocated `step_idx`. Every `emit` inside the block is auto-stamped with
        `(step_idx, op)` via the `_step_frame` ContextVar, so callees emit without
        knowing their step. The ctx allocates the index; the caller does not pass
        one in. Steps don't nest, so this rebinds a single frame (set/reset) rather
        than pushing a stack, and the contextmanager guarantees the frame is reset
        on every exit path. Per-task isolation comes from the ContextVar — see
        `_step_frame`."""
        step_idx = self._next_step_idx
        self._next_step_idx += 1
        token = _step_frame.set((step_idx, op))
        try:
            yield step_idx
        finally:
            _step_frame.reset(token)

    def _current_step(self) -> tuple[int | None, str | None]:
        return _step_frame.get() or (None, None)

    def emit(self, message: str, level: str | None = None) -> None:
        """Record a request-scoped diagnostic event onto this question's event stream.

        Convention (see ARCHITECTURE.md "Logging & observability"):
        - There is no separate `source`/`op` argument: the active step's `op`
          (from `_step_frame`, stamped here) identifies who emitted. Out-of-step
          emits carry `op=None` and rely on the message alone.
        - `message` is a single human-readable string. Lead it with a stable
          snake_case event key, then interpolate any variables inline
          (`f"verifier_dropped n_dropped={n} n_parsed={m}"`). There are no
          structured fields — keep large blobs (full prompts, transcripts) out of
          the message; log a count or short `repr` instead.
        - Emit the fact at the layer that owns it, and only there: the
          orchestrator owns operator boundaries (the `"step"` event), so operators
          do NOT emit their own "starting"/"done"; each operator emits only its own
          internal decisions. Use stdlib `logging` (not `emit`) from code that has
          no per-question ctx (build pipelines, offline prep).

        Severity defaults to "warning" when the message contains a `_failed` event
        key or an `error=` field, else "info"; pass `level=` to override. The event
        is always captured to `self.events` (per-question, consumed by the trace
        dump) and the JSONL sink; streamed to this question's `.log` file when one
        is open; and echoed to the console only when `verbose` (rendered via
        `trace.render_line`). The per-question file omits `uid` (implied by the
        filename); the shared console prepends it so interleaved lines stay
        attributable.
        """
        step_idx, op = self._current_step()
        if level is None:
            level = "warning" if ("_failed" in message or "error=" in message) else "info"
        evt = {
            "message": message,
            "step_idx": step_idx,
            "op": op,
            "level": level,
        }
        self.events.append(evt)
        trace.write_jsonl({"uid": self.uid, **evt})
        if self._logfile is not None:
            self._logfile.write(trace.render_line(evt) + "\n")
            self._logfile.flush()
        if self.verbose:
            print(trace.render_line({"uid": self.uid, **evt}))
