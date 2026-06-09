"""The LLM client and its provider-specific scaffolding.

All LLM traffic goes through either the direct Gemini API (AI Studio) via
`google-genai` (authenticated by `GEMINI_API_KEY`, the default) or OpenRouter
(`provider=openrouter`, authenticated by `OPENROUTER_API_KEY`). LLM calls are paced
by the process-wide token-bucket limiters from `skunk.common` (the per-model
`llm:<model>` bucket); only transient SDK exceptions (HTTP 429 + 5xx, network
timeouts / connection resets — see `_is_retryable`) are retried with exponential
backoff up to `llm_max_retries` times. The limiter paces traffic; the retry rides
out throttling and blips the limiter can't prevent (provider-side 429, TPM quotas,
multi-process fan-out)."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
import requests
from google.genai import errors as genai_errors
from google.genai import types

from skunk.common import (
    Effort,
    make_genai_client,
    get_rate_limiter,
)

if TYPE_CHECKING:
    from google import genai
    from openrouter import OpenRouter

    from skunk.common import B64Image, ExecutionContext
    from skunk.config import SkunkConfig

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
    # OpenRouter SDK errors all derive from `OpenRouterError`, which exposes the
    # HTTP `status_code` (lazy import — the SDK is only loaded on the openrouter path).
    try:
        from openrouter.errors import OpenRouterError
    except ImportError:
        OpenRouterError = ()  # type: ignore[assignment]
    if isinstance(e, OpenRouterError):
        code = getattr(e, "status_code", None)
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


_MODEL_RPM: dict[str, float] | None = None

# Built-in per-model request caps (provider account limits). Overridable per model via
# SKUNK_MODEL_RPM ("model=rpm,..."); a model in neither falls back to SKUNK_LLM_RPM.
_DEFAULT_RPM: dict[str, float] = {
    "gemini-3.5-flash": 1000.0,
    "gemini-3.1-flash-lite": 4000.0,
    "gemini-3.1-pro-preview": 150.0,
}


def _llm_model_rpm(model: str) -> float:
    """Per-minute request cap for an LLM `model`. Parsed once from `SKUNK_MODEL_RPM`
    ("model=rpm,..."); a model not listed falls back to its `_DEFAULT_RPM`, then to
    `SKUNK_LLM_RPM` (default 1000). Each model gets its own limiter bucket
    (`llm:<model>`), so mixed-model runs pace independently."""
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
    if model in _MODEL_RPM:
        return _MODEL_RPM[model]
    if model in _DEFAULT_RPM:
        return _DEFAULT_RPM[model]
    return float(os.environ.get("SKUNK_LLM_RPM", "1000"))


# ---------------------------------------------------------------------------
# Tokens-per-minute (TPM) throttle — async, opt-in via SKUNK_MODEL_TPM.
#
# The RPM limiter alone can't bound token throughput: one request can carry tens
# of thousands of tokens, so a request-paced run still blows a TPM quota (the
# full-text page-index filter pushed ~36M tok/min and 429-stormed). This bucket
# meters estimated *input* tokens per call. Mirrors `_RateLimiter.acquire_async`'s
# cross-loop safety (threading.Lock around refill+deduct, sleep outside the lock).
# Off unless `SKUNK_MODEL_TPM` names the model, so it's scoped to experiments.
# ---------------------------------------------------------------------------

class _AsyncTokenBudget:
    """Like `_RateLimiter.acquire_async` but `acquire(amount)` deducts a variable token
    count (the call's estimated input tokens). `capacity` allows a short burst
    and must exceed the largest single request, or `acquire` would cap-clamp it."""

    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0 (got {rate_per_sec})")
        self._rate = rate_per_sec
        self._capacity = max(capacity, rate_per_sec)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

    async def acquire(self, amount: float) -> None:
        amount = max(0.0, min(float(amount), self._capacity))
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                wait_s = (amount - self._tokens) / self._rate
            await asyncio.sleep(wait_s)

    def settle(self, delta: float) -> None:
        """Post-call correction: charge `delta` (actual minus estimated tokens)
        without awaiting. May drive the balance negative so the next `acquire`
        waits longer — this is what makes the throttle track ACTUAL token usage
        even when the pre-call estimate is off. No-op for delta 0."""
        if not delta:
            return
        with self._lock:
            self._refill_locked()
            self._tokens -= delta


_ASYNC_TPM_LOCK = threading.Lock()
_ASYNC_TPM_LIMITERS: dict[str, _AsyncTokenBudget] = {}


def get_async_tpm_limiter(model: str, tpm: float) -> _AsyncTokenBudget:
    """Process-wide TPM bucket for `model`, paced at `tpm` tokens/min. Capacity is
    ~4s of budget so a few large concurrent requests can burst, then throttle."""
    with _ASYNC_TPM_LOCK:
        lim = _ASYNC_TPM_LIMITERS.get(model)
        if lim is None:
            rate_per_sec = tpm / 60.0
            lim = _AsyncTokenBudget(rate_per_sec, capacity=max(rate_per_sec * 4.0, 256_000.0))
            _ASYNC_TPM_LIMITERS[model] = lim
        return lim


_MODEL_TPM: dict[str, float] | None = None

# Built-in per-model token-per-minute caps (provider account limits). Overridable per model
# via SKUNK_MODEL_TPM ("model=tpm,..."); a model in neither is unthrottled (None) and paced
# by RPM alone (e.g. Pro).
_DEFAULT_TPM: dict[str, float] = {
    "gemini-3.5-flash": 4_000_000.0,
    "gemini-3.1-flash-lite": 25_000_000.0,
}


def _llm_model_tpm(model: str) -> float | None:
    """Per-minute *token* cap for `model`, parsed once from `SKUNK_MODEL_TPM`
    ("model=tpm,..."), else its `_DEFAULT_TPM`. Returns None (no throttle) when neither
    sets it — so the TPM bucket is inert for unlisted models (e.g. Pro, paced by RPM)."""
    global _MODEL_TPM
    if _MODEL_TPM is None:
        out: dict[str, float] = {}
        for entry in os.environ.get("SKUNK_MODEL_TPM", "").split(","):
            entry = entry.strip()
            if not entry:
                continue
            name, _, tpm = entry.partition("=")
            out[name.strip()] = float(tpm.strip())
        _MODEL_TPM = out
    if model in _MODEL_TPM:
        return _MODEL_TPM[model]
    return _DEFAULT_TPM.get(model)


def _estimate_prompt_tokens(system: str, user: str) -> float:
    """Cheap pre-call input-token estimate (~4 chars/token) for the TPM bucket.
    Approximate by design — it only paces throughput, it doesn't bill."""
    return (len(system) + len(user)) / 4.0


def _requires_thinking(model: str) -> bool:
    """True for models that mandate thinking mode (reject `thinking_budget=0`).
    Gemini 3.x Pro tiers are thinking-only; Flash/Flash-Lite accept budget=0."""
    m = model.lower()
    return "gemini-3" in m and "pro" in m


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None


def _make_openrouter_client() -> "OpenRouter":
    """Build an OpenRouter client from `OPENROUTER_API_KEY` (for `provider=openrouter`).
    Lazy import: the SDK is only needed when this provider is selected."""
    from openrouter import OpenRouter

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (required for provider=openrouter)")
    return OpenRouter(api_key=api_key)


class LLMClient:
    """LLM client. Generation calls route to either the AI Studio Gemini API
    (`provider=genai`, default) or OpenRouter (`provider=openrouter`); both share
    the same rate-limit / retry / logging scaffolding. Embeddings stay on Gemini."""

    def __init__(self, config: SkunkConfig) -> None:
        self._config = config
        self._gemini_client: genai.Client | None = None
        self._openrouter_client: OpenRouter | None = None

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = make_genai_client()
        return self._gemini_client

    def _get_openrouter_client(self) -> "OpenRouter":
        if self._openrouter_client is None:
            self._openrouter_client = _make_openrouter_client()
        return self._openrouter_client

    def call(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        args = (system, user, images, temperature, effort, ctx, call_site,
                model or self._config.llm_model)
        if self._config.llm_provider == "openrouter":
            return self._call_openrouter(*args)
        return self._call_gemini(*args)

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
        args = (system, user, images, temperature, effort, ctx, call_site,
                model or self._config.llm_model)
        if self._config.llm_provider == "openrouter":
            return await self._acall_openrouter(*args)
        return await self._acall_gemini(*args)

    def _retry_call(self, do_call: Callable[[], LLMResponse], model: str) -> LLMResponse:
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
        self, do_call: Callable[[], Awaitable[LLMResponse]], model: str
    ) -> LLMResponse:
        """Async twin of `_retry_call`: awaits `model`'s async rate limiter and the
        coroutine `do_call`, backing off via `asyncio.sleep` (never blocking the
        event loop). `do_call` owns the API invocation, timing, parsing, and emit."""
        limiter = get_rate_limiter(f"llm:{model}", rate_per_min=_llm_model_rpm(model))
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s

        for attempt in range(max_retries + 1):
            await limiter.acquire_async()
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
    def _effort_to_thinking_config(effort: Effort, model: str) -> "types.ThinkingConfig":
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
        effort: Effort,
        model: str,
    ) -> types.GenerateContentConfig:
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
        effort: Effort,
        ctx: ExecutionContext | None,
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
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
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
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """Async twin of `_call_gemini` — uses `client.aio.models.generate_content`."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(user, images)
        model = model or self._config.llm_model
        gen_config = self._gemini_config(system, temperature, effort, model)

        async def do() -> LLMResponse:
            # TPM throttle (opt-in via SKUNK_MODEL_TPM): meter input tokens so
            # throughput stays under quota. Charge an estimate up front for pacing,
            # then `settle` the actual-vs-estimate delta after the call so the
            # bucket tracks REAL token usage (the char/4 estimate runs ~2x low on
            # dense tabular text). Inside `do` so each retry re-charges. Separate
            # from the RPM limiter.
            tpm = _llm_model_tpm(model)
            tpm_lim, est = None, 0.0
            if tpm:
                est = _estimate_prompt_tokens(system, user)
                tpm_lim = get_async_tpm_limiter(model, tpm)
                await tpm_lim.acquire(est)
            t0 = time.monotonic()
            api_resp = await client.aio.models.generate_content(
                model=model, contents=parts, config=gen_config,
            )
            latency_s = time.monotonic() - t0
            usage = api_resp.usage_metadata
            output_text = (api_resp.text or "").strip()
            toks = self._usage_tokens(usage)
            if tpm_lim is not None:
                tpm_lim.settle((toks["input_tokens"] or 0) - est)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
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
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        """Multi-turn streaming call, accumulating chunks until `should_stop(acc)`
        or the stream ends. `messages` are the {role, content} turns after the
        system message ('assistant' → model role, else user). Lets multi-turn
        agents share this client's rate-limit + retry + logging. `temperature`
        defaults to 0.0 — agent loops are deterministic like every other call site.
        `effort` maps onto Gemini's thinking config exactly like the single-shot
        path (`_gemini_config`), so agent loops are tunable like every other call site."""
        kw = dict(system=system, messages=messages, model=model, should_stop=should_stop,
                  temperature=temperature, effort=effort, ctx=ctx, call_site=call_site)
        if self._config.llm_provider == "openrouter":
            return self._stream_openrouter(**kw) # type: ignore
        return self._stream_gemini(**kw) # type: ignore

    def _stream_gemini(
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
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
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
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        """Async twin of `stream` — uses `client.aio.models.generate_content_stream`
        and `async for`, so the event loop runs other tasks between chunks."""
        kw = dict(system=system, messages=messages, model=model, should_stop=should_stop,
                  temperature=temperature, effort=effort, ctx=ctx, call_site=call_site)
        if self._config.llm_provider == "openrouter":
            return await self._astream_openrouter(**kw) # type: ignore
        return await self._astream_gemini(**kw) # type: ignore

    async def _astream_gemini(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
    ) -> LLMResponse:
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
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return await self._aretry_call(do, model_id)

    # --- OpenRouter generation path (provider="openrouter") ---
    # Mirrors the Gemini helpers above: same do()/_retry_call shape, same ctx.emit
    # envelope and LLMResponse, so the rate-limit/retry/logging scaffolding is shared.

    @staticmethod
    def _effort_to_reasoning(effort: Effort) -> dict:
        """Map our Effort tier onto OpenRouter's `reasoning` arg. "off" maps to the
        lowest tier ("minimal") rather than disabling reasoning outright, since some
        endpoints (e.g. Gemini 3) mandate reasoning and reject `effort=none` with a
        400. The rest pass straight through (OpenRouter's `effort` accepts
        minimal/low/medium/high). Models without reasoning ignore it."""
        return {"effort": "minimal" if effort == "off" else effort}

    @staticmethod
    def _openrouter_content(user: str, images: list[B64Image] | None) -> Any:
        """User-turn content: a plain string, or OpenAI-style content parts when
        images are attached (base64 `data:` URLs), mirroring `_gemini_parts`."""
        if not images:
            return user
        parts: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{img.mime};base64,{img.data}"}}
            for img in images
        ]
        parts.append({"type": "text", "text": user})
        return parts

    @staticmethod
    def _openrouter_text(resp: Any) -> str:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        return getattr(msg, "content", None) or ""

    @staticmethod
    def _usage_tokens_openrouter(usage: Any) -> dict:
        """Token counts from an OpenRouter `ChatUsage` (best-effort — any may be None)."""
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def _call_openrouter(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """Single OpenRouter chat call. `call_site` attributes the envelope log."""
        client = self._get_openrouter_client()
        model = model or self._config.llm_model
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self._openrouter_content(user, images)})
        reasoning = self._effort_to_reasoning(effort)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            resp = client.chat.send(
                model=model, messages=messages, stream=False, # type: ignore
                temperature=temperature, reasoning=reasoning, # type: ignore
            )
            latency_s = time.monotonic() - t0
            output_text = self._openrouter_text(resp).strip()
            toks = self._usage_tokens_openrouter(getattr(resp, "usage", None))
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return self._retry_call(do, model)

    async def _acall_openrouter(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str = "llm",
        model: str | None = None,
    ) -> LLMResponse:
        """Async twin of `_call_openrouter` — uses `client.chat.send_async`."""
        client = self._get_openrouter_client()
        model = model or self._config.llm_model
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self._openrouter_content(user, images)})
        reasoning = self._effort_to_reasoning(effort)

        async def do() -> LLMResponse:
            # TPM throttle (opt-in via SKUNK_MODEL_TPM) — identical to the Gemini path.
            tpm = _llm_model_tpm(model)
            tpm_lim, est = None, 0.0
            if tpm:
                est = _estimate_prompt_tokens(system, user)
                tpm_lim = get_async_tpm_limiter(model, tpm)
                await tpm_lim.acquire(est)
            t0 = time.monotonic()
            resp = await client.chat.send_async(
                model=model, messages=messages, stream=False, # type: ignore
                temperature=temperature, reasoning=reasoning, # type: ignore
            )
            latency_s = time.monotonic() - t0
            output_text = self._openrouter_text(resp).strip()
            toks = self._usage_tokens_openrouter(getattr(resp, "usage", None))
            if tpm_lim is not None:
                tpm_lim.settle((toks["input_tokens"] or 0) - est)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return await self._aretry_call(do, model)

    @staticmethod
    def _openrouter_chat_messages(system: str, messages: list[dict]) -> list[dict]:
        """Multi-turn messages in OpenRouter format: a leading system message (if any)
        then the {role, content} turns ('assistant' → assistant, else user)."""
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(
            {"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
            for m in messages
        )
        return out

    @staticmethod
    def _openrouter_chunk_text(chunk: Any) -> str:
        text = ""
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            text += getattr(delta, "content", None) or ""
        return text

    def _stream_openrouter(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        client = self._get_openrouter_client()
        model_id = model or self._config.llm_model
        or_messages = self._openrouter_chat_messages(system, messages)
        reasoning = self._effort_to_reasoning(effort)

        def do() -> LLMResponse:
            t0 = time.monotonic()
            accumulated = ""
            usage = None
            with client.chat.send(
                model=model_id, messages=or_messages, stream=True, # type: ignore
                temperature=temperature, reasoning=reasoning, # type: ignore
            ) as resp_stream:
                for chunk in resp_stream:
                    accumulated += self._openrouter_chunk_text(chunk)
                    usage = getattr(chunk, "usage", None) or usage
                    if should_stop is not None and should_stop(accumulated):
                        break
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens_openrouter(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return self._retry_call(do, model_id)

    async def _astream_openrouter(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: Callable[[str], bool] | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        ctx: ExecutionContext | None = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        """Async twin of `_stream_openrouter` — uses `client.chat.send_async` + `async for`."""
        client = self._get_openrouter_client()
        model_id = model or self._config.llm_model
        or_messages = self._openrouter_chat_messages(system, messages)
        reasoning = self._effort_to_reasoning(effort)

        async def do() -> LLMResponse:
            t0 = time.monotonic()
            accumulated = ""
            usage = None
            async with await client.chat.send_async(
                model=model_id, messages=or_messages, stream=True, # type: ignore
                temperature=temperature, reasoning=reasoning, # type: ignore
            ) as resp_stream:
                async for chunk in resp_stream:
                    accumulated += self._openrouter_chunk_text(chunk)
                    usage = getattr(chunk, "usage", None) or usage
                    if should_stop is not None and should_stop(accumulated):
                        break
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens_openrouter(usage)
            if ctx is not None:
                ctx.emit(
                    f"call call_site={call_site} model={model_id} temp={temperature} "
                    f"effort={effort} latency_s={round(latency_s, 3)} "
                    f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                    f"think_tok={toks['thinking_tokens']}"
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return await self._aretry_call(do, model_id)
