"""The LLM client and its provider-specific scaffolding.

All LLM traffic goes through OpenRouter (`provider=openrouter`, authenticated by
`OPENROUTER_API_KEY`) or a local vLLM server speaking the OpenAI API
(`provider=vllm`, addressed per model via `config.vllm_base_urls`). Routing is
per call: a model listed in `vllm_base_urls` goes to the vLLM server that serves
it, every other model uses `config.llm_provider` — so one run can keep e.g. its
judge on OpenRouter while the agent runs on a local model. LLM calls are paced
by the process-wide token-bucket limiters from `skunk.common` (the per-model
`llm:<model>` bucket); only transient SDK exceptions (HTTP 429 + 5xx, network
timeouts / connection resets — see `_is_retryable`) are retried with exponential
backoff up to `llm_max_retries` times. The limiter paces traffic; the retry rides
out throttling and blips the limiter can't prevent (provider-side 429, TPM quotas,
multi-process fan-out)."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
import requests

from skunk.common import (
    AsyncTokenBudget,
    Effort,
    get_async_tpm_limiter,
    get_rate_limiter,
)
from skunk.usage import UsageTracker

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openrouter import OpenRouter

    from skunk.common import B64Image, ExecutionContext
    from skunk.config import SystemConfig

def _warn(ctx: "ExecutionContext | None", message: str) -> None:
    """Route a client warning onto the owning question's event stream when a ctx
    is in scope (retry/usage warnings become attributable per question); ctx-less
    callers (offline corpus prep) fall back to plain stderr."""
    if ctx is not None:
        ctx.emit(message, level="warning")
    else:
        print(message, file=sys.stderr)

# Return type of an `attempt` driven by `_retry_call` — an `LLMResponse` for the
# generation paths, a `list[float]` for the embedding path. The retry loop never
# inspects the value, so it is generic over it.
R = TypeVar("R")


class EmptyCompletionError(RuntimeError):
    """A generation call returned HTTP 200 but with no message content. OpenRouter
    relays some upstream-provider failures this way — empty `choices` (or empty
    `content`) and zeroed usage, with no exception raised. For a text-generation call
    (final answer, judge, planner) an empty completion is never a usable result, so we
    treat it as a transient fault: the retry loop re-issues the call, and only if it
    stays empty past the retry budget does it surface as a failed row — far better than
    silently recording an empty answer that then scores 0."""


class EmptyEmbeddingError(EmptyCompletionError):
    """An embedding call returned no vector (empty `data`, or a zero-length embedding).
    Subclasses `EmptyCompletionError` so `_is_retryable` already treats it as transient:
    the same OpenRouter-relays-an-upstream-failure-as-200 mode applies to the embeddings
    endpoint, so re-issue rather than crash a question on a blip."""


def _is_retryable(e: BaseException) -> bool:
    """True for transient failures worth retrying: HTTP 429 (throttling) and 5xx
    (server-side), plus network-layer timeouts / connection resets from the
    underlying transport (`httpx` async, `requests` sync), and an empty-content 200
    (`EmptyCompletionError`). Non-429 4xx — bad request, auth, context-length overflow —
    is a permanent error that will never succeed, so it raises immediately instead of
    burning the retry budget."""
    if isinstance(e, EmptyCompletionError):
        return True
    # OpenRouter SDK errors all derive from `OpenRouterError`, which exposes the
    # HTTP `status_code` (lazy import — the SDK is only loaded on the openrouter path).
    try:
        from openrouter.errors import OpenRouterError
    except ImportError:
        OpenRouterError = ()  # type: ignore[assignment]
    if isinstance(e, OpenRouterError):
        code = getattr(e, "status_code", None)
        return code == 429 or (code is not None and 500 <= code < 600)
    # openai SDK errors (the vllm path): `APIStatusError` carries the HTTP status;
    # `APIConnectionError` (which includes `APITimeoutError`) wraps httpx transport
    # faults — always transient (lazy import, same pattern as OpenRouterError above).
    try:
        from openai import APIConnectionError, APIStatusError
    except ImportError:
        APIConnectionError = APIStatusError = ()  # type: ignore[assignment,misc]
    if isinstance(e, APIStatusError):
        code = getattr(e, "status_code", None)
        return code == 429 or (code is not None and 500 <= code < 600)
    if isinstance(e, APIConnectionError):
        return True
    # Transport faults: connection resets, read timeouts, DNS failures, etc.
    # `TimeoutError` covers both asyncio timeouts on the async SDK path and our own
    # per-request wall-clock cap (asyncio.wait_for in the streaming path raises builtin
    # TimeoutError) — treat a tripped timeout as a transient fault worth retrying, same as a
    # transport-level read timeout.
    retryable: tuple[type[BaseException], ...] = (
        TimeoutError,
        httpx.TimeoutException,
        httpx.TransportError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    )

    return isinstance(e, retryable)


def _error_detail(e: BaseException) -> str:
    """Diagnostic suffix for an LLM error. OpenRouter collapses upstream-provider
    failures into a terse `...ResponseError: Provider returned error`; the actionable
    cause — the HTTP status, the provider's raw error body, and OpenRouter's
    `metadata` (which carries the provider name + the raw upstream message) — lives on
    the exception, not in `str(e)`. The openai SDK (vllm path) likewise buries the
    server's error body on `e.response`. Surface both so an intermittent 400 is
    debuggable straight from the logs. Best-effort: never raises, returns "" when
    nothing to add."""
    try:
        from openrouter.errors import OpenRouterError
    except ImportError:
        OpenRouterError = ()  # type: ignore[assignment]
    if isinstance(e, OpenRouterError):
        parts: list[str] = []
        try:
            code = getattr(e, "status_code", None)
            if code is not None:
                parts.append(f"status={code}")
            err = getattr(getattr(e, "data", None), "error", None)
            meta = getattr(err, "metadata", None)
            if meta:
                parts.append(f"metadata={str(meta)[:1000]}")
            rr = getattr(e, "raw_response", None)
            body = getattr(rr, "text", None) if rr is not None else getattr(e, "body", None)
            if body:
                parts.append(f"body={str(body)[:1000]}")
        except Exception:  # noqa: BLE001 — diagnostics must never mask the original error
            return ""
        return (" | " + " ".join(parts)) if parts else ""
    try:
        from openai import APIStatusError
    except ImportError:
        return ""
    if isinstance(e, APIStatusError):
        parts = []
        try:
            code = getattr(e, "status_code", None)
            if code is not None:
                parts.append(f"status={code}")
            rr = getattr(e, "response", None)
            body = getattr(rr, "text", None) if rr is not None else getattr(e, "body", None)
            if body:
                parts.append(f"body={str(body)[:1000]}")
        except Exception:  # noqa: BLE001 — diagnostics must never mask the original error
            return ""
        return (" | " + " ".join(parts)) if parts else ""
    return ""


# The TPM throttle (AsyncTokenBudget + its process-wide registry) lives in
# `skunk.common`, next to the RPM registry — one home for all pacing state.


def _estimate_prompt_tokens(system: str, user: str) -> float:
    """Cheap pre-call input-token estimate (~4 chars/token) for the TPM bucket.
    Approximate by design — it only paces throughput, it doesn't bill."""
    return (len(system) + len(user)) / 4.0


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    # Cached (prompt-cache hit) input tokens, when the provider reports them.
    # Subset of input_tokens; None when unknown.
    cache_input_tokens: int | None = None
    # Thinking/reasoning tokens (billed at the output rate); None when unreported.
    thinking_tokens: int | None = None


@dataclass(frozen=True)
class CallSpec:
    """One generation request, provider-agnostic — the value threaded from the
    public `call` / `acall` / `astream` methods into the per-provider bodies
    (which used to take it as 8–10 positional args). `model` is always resolved
    by the public method (never empty inside a provider body). Exactly one of
    `user` (single-shot) / `messages` (multi-turn streaming) is meaningful,
    matching the public method it came through."""

    system: str
    model: str
    user: str = ""
    messages: tuple[dict, ...] | None = None
    images: list[B64Image] | None = None
    temperature: float = 0.0
    effort: Effort = "off"
    ctx: ExecutionContext | None = None
    call_site: str = "llm"
    max_output_tokens: int | None = None
    timeout_s: float | None = None
    should_stop: Callable[[str], bool] | None = None
    # Per-call OpenRouter provider order (no fallback). None => use config.llm_provider_order.
    # Lets a single client route one model (e.g. a cheaper semantic-filter judge) to specific
    # providers while other models on the same client stay unpinned. OpenRouter-only: ignored
    # when the model routes to a vLLM backend.
    provider_order: list[str] | None = None


# Raw HTTP body of the most recent OpenRouter response, captured by an httpx response
# hook (below) and read only when a completion comes back empty — so we can log the
# UNFILTERED upstream body (not the SDK's typed object, which could in principle drop a
# field) and settle whether an empty completion is genuinely empty or a parse artifact.
# A ContextVar isolates the value per async task / thread: the hook fires synchronously
# within the same send() call we're about to inspect, so no cross-call bleed.
_RAW_BODY_MAX = 2000
_last_openrouter_raw_body: ContextVar[str | None] = ContextVar(
    "_last_openrouter_raw_body", default=None
)


def _capture_response_body_sync(response: "httpx.Response") -> None:
    """httpx sync `response` event hook: buffer the body and stash it. `read()` caches
    `.content`, so the SDK's later `.json()` still works. Best-effort — never raise into
    the request path."""
    try:
        response.read()
        _last_openrouter_raw_body.set(response.text[:_RAW_BODY_MAX])
    except Exception:  # noqa: BLE001
        pass


async def _capture_response_body_async(response: "httpx.Response") -> None:
    """Async twin of `_capture_response_body_sync` (async clients require async hooks)."""
    try:
        await response.aread()
        _last_openrouter_raw_body.set(response.text[:_RAW_BODY_MAX])
    except Exception:  # noqa: BLE001
        pass


def _raw_body_suffix() -> str:
    """Diagnostic suffix carrying the last captured raw OpenRouter body, for empty-completion
    errors. The body is the ground truth the SDK's typed `ChatResult` is parsed from, so it
    reveals whether `content` was truly null on the wire vs. dropped in parsing."""
    body = _last_openrouter_raw_body.get()
    return f" raw_body={body}" if body else ""


def _make_openrouter_client(api_key: str | None = None) -> OpenRouter:
    """Build an OpenRouter client (for `provider=openrouter`), authenticated by the
    explicit `api_key` else `OPENROUTER_API_KEY`. Lazy import: the SDK is only needed
    when this provider is selected. Inject httpx clients carrying a response hook that
    captures each raw body (read on empty completions only) — with a generous timeout,
    since a bare httpx client defaults to 5s and would abort long generations (the app
    enforces its own per-request wall-clock cap)."""
    import httpx
    from openrouter import OpenRouter

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (required for provider=openrouter)")
    timeout = httpx.Timeout(600.0)
    sync_client = httpx.Client(timeout=timeout, event_hooks={"response": [_capture_response_body_sync]})
    async_client = httpx.AsyncClient(
        timeout=timeout, event_hooks={"response": [_capture_response_body_async]}
    )
    return OpenRouter(api_key=api_key, client=sync_client, async_client=async_client)


# vLLM (openai SDK) clients are cached PROCESS-WIDE (not per LLMClient): the eval builds
# one LLMClient per question and runs many concurrently, so per-instance clients would
# open one connection pool per question per server. The openai clients are thread- and
# async-safe, so one shared (sync, async) pair per server is correct. Keyed by
# (base_url, api_key); the lock only guards the one-time construction.
_VLLM_CLIENTS: dict[tuple[str, str], tuple["OpenAI", "AsyncOpenAI"]] = {}
_VLLM_CLIENTS_LOCK = threading.Lock()


def _make_vllm_clients(base_url: str, api_key: str | None = None) -> tuple["OpenAI", "AsyncOpenAI"]:
    """Sync + async `openai` clients for one vLLM server (`provider=vllm`). vLLM speaks
    the OpenAI API; a server started without `--api-key` accepts any key, so the
    conventional placeholder "EMPTY" is sent when neither the explicit `api_key` nor
    `VLLM_API_KEY` is set. `max_retries=0` — our retry loop owns retries. The generous
    timeout mirrors `_make_openrouter_client` (a bare client would abort long
    generations; the app enforces its own per-request wall-clock cap)."""
    import httpx
    from openai import AsyncOpenAI, OpenAI

    key = api_key or os.environ.get("VLLM_API_KEY") or "EMPTY"
    cache_key = (base_url, key)
    clients = _VLLM_CLIENTS.get(cache_key)
    if clients is None:
        with _VLLM_CLIENTS_LOCK:
            clients = _VLLM_CLIENTS.get(cache_key)  # re-check: another thread may have built it
            if clients is None:
                timeout = httpx.Timeout(600.0)
                clients = (
                    OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0),
                    AsyncOpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=0),
                )
                _VLLM_CLIENTS[cache_key] = clients
    return clients


class _LLMBackend:
    """Per-provider backend: the shared retry/backoff drivers, RPM/TPM pacing, and the
    `_build_response`/`_finish` trace+usage chokepoint, over abstract provider bodies
    (`_gen_call` / `_gen_acall` / `_gen_astream` / `_embed_once`). Backends are built
    only by the `LLMClient` facade, which injects its own `UsageTracker` — every
    backend of a client bills into the one tracker, so a mixed OpenRouter+vLLM run
    still reports one usage/cost total."""

    provider: str = "?"

    def __init__(self, config: SystemConfig, usage: UsageTracker) -> None:
        self._config = config
        self.usage = usage

    # --- entry points (called by the LLMClient facade with a resolved CallSpec) -------

    def call(self, spec: CallSpec) -> LLMResponse:
        return self._retry_call(lambda: self._gen_call(spec), self.provider, spec.model, ctx=spec.ctx)

    async def acall(self, spec: CallSpec) -> LLMResponse:
        return await self._aretry_call(
            lambda: self._gen_acall(spec), self.provider, spec.model, ctx=spec.ctx
        )

    async def astream(self, spec: CallSpec) -> LLMResponse:
        return await self._aretry_call(
            lambda: self._gen_astream(spec), self.provider, spec.model, ctx=spec.ctx
        )

    def embed_query(
        self, text: str, *, model: str, ctx: ExecutionContext | None = None
    ) -> list[float]:
        """Embed a single query string for vector search, routing through the same
        rate-limit / retry / usage-accounting scaffolding as generation so embedding
        spend lands on this client's `usage` tracker (tokens, cost, calls). The provider
        body is `_embed_once`; when `ctx` is given, emits the uniform `call ...` envelope
        (so the embed call's latency shows in the per-question trace like every LLM
        call); usage is recorded regardless of `ctx`."""

        def attempt() -> list[float]:
            t0 = time.monotonic()
            vector, in_tok = self._embed_once(model, text)
            latency_s = time.monotonic() - t0
            self.usage.add_embed(model, in_tok)
            if ctx is not None:
                cost = self.usage.price_embed(model, in_tok)
                ctx.emit(
                    f"call call_site=embed model={model} provider={self.provider} "
                    f"latency_s={round(latency_s, 3)} in_tok={in_tok} dim={len(vector)}",
                    kind="call",
                    data=None if cost is None else {"cost": cost},
                )
            return vector

        # Embeddings share one process-wide "embed" rate bucket across all workers
        # (separate from the per-model generation buckets).
        return self._retry_call(attempt, self.provider, model, bucket="embed", ctx=ctx)

    # --- provider bodies (subclass responsibility) -------------------------------------

    def _gen_call(self, spec: CallSpec) -> LLMResponse:
        raise NotImplementedError(f"provider {self.provider!r} does not implement call")

    async def _gen_acall(self, spec: CallSpec) -> LLMResponse:
        raise NotImplementedError(f"provider {self.provider!r} does not implement acall")

    async def _gen_astream(self, spec: CallSpec) -> LLMResponse:
        raise NotImplementedError(f"provider {self.provider!r} does not implement astream")

    def _embed_once(self, model: str, text: str) -> tuple[list[float], int]:
        """One embedding request (no retry — the retry loop owns that). Returns the
        vector and the provider-reported input-token count (0 when unreported)."""
        raise NotImplementedError(f"provider {self.provider!r} does not implement embed_query")

    # --- shared machinery ---------------------------------------------------------------

    def _retry_call(
        self,
        attempt: Callable[[], R],
        provider: str,
        model: str,
        bucket: str | None = None,
        ctx: "ExecutionContext | None" = None,
    ) -> R:
        """Drive `attempt()` with exponential-backoff retry, paced by a
        rate-limiter bucket. Retries only transient faults (`_is_retryable`: 429 / 5xx /
        transport blips); a non-429 4xx (bad request, auth, context overflow) raises
        immediately. `attempt` owns the API call, timing, parsing, and the success emit;
        `provider`/`model` are for the failure warning (routed onto `ctx`'s event
        stream when given — see `_warn`). `bucket` overrides the limiter (a
        shared process-wide bucket like `"embed"`); when None, the per-model
        `llm:<model>` bucket is used at the configured RPM."""
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s
        limiter = (
            get_rate_limiter(bucket)
            if bucket is not None
            else get_rate_limiter(
                f"llm:{model}",
                rate_per_min=self._config.llm_model_rpm.get(model, self._config.llm_default_rpm),
            )
        )

        for i in range(max_retries + 1):
            limiter.acquire()
            try:
                return attempt()
            except Exception as e:
                # Log EVERY failure with its message — including the final one
                # before we re-raise — so a fatal error is never silent.
                stop = i == max_retries or not _is_retryable(e)
                _warn(
                    ctx,
                    f"llm_call_failed attempt={i + 1}/{max_retries + 1} provider={provider} "
                    f"model={model} error={type(e).__name__}: {e}{_error_detail(e)}"
                    f"{'' if stop else f'; retrying in {delay:.1f}s'}",
                )
                if stop:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable: retry loop fell through")

    async def _aretry_call(
        self,
        attempt: Callable[[], Awaitable[LLMResponse]],
        provider: str,
        model: str,
        ctx: "ExecutionContext | None" = None,
    ) -> LLMResponse:
        """Async twin of `_retry_call`: awaits the model's async rate limiter and the
        coroutine `attempt()`, backing off via `asyncio.sleep` (never blocking the
        event loop). `provider`/`model` are for the failure warning."""
        max_retries = self._config.llm_max_retries
        delay = self._config.llm_retry_initial_delay_s
        limiter = get_rate_limiter(
            f"llm:{model}",
            rate_per_min=self._config.llm_model_rpm.get(model, self._config.llm_default_rpm),
        )

        for i in range(max_retries + 1):
            await limiter.acquire_async()
            try:
                return await attempt()
            except Exception as e:
                stop = i == max_retries or not _is_retryable(e)
                _warn(
                    ctx,
                    f"llm_call_failed attempt={i + 1}/{max_retries + 1} provider={provider} "
                    f"model={model} error={type(e).__name__}: {e}{_error_detail(e)}"
                    f"{'' if stop else f'; retrying in {delay:.1f}s'}",
                )
                if stop:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable: retry loop fell through")

    def _build_response(
        self,
        text: str,
        toks: dict,
        latency_s: float,
        *,
        model: str,
        temperature: float,
        effort: Effort,
        ctx: ExecutionContext | None,
        call_site: str,
    ) -> LLMResponse:
        """Shared tail for every generation path: emit the uniform `call ...` envelope
        (when `ctx` is set), accumulate the call into this client's `usage` tracker, and
        pack the provider-agnostic token dict (`_usage_tokens_chat`) into an
        `LLMResponse`. `toks` may omit any key — the chat dict has no
        `thinking_tokens`, hence the `.get`s. The single chokepoint every call path funnels
        through, so usage accounting lives here once rather than at each public method."""
        if ctx is not None:
            # Exact per-call USD (priced as `cost()` aggregates) → the trace viewer shows
            # per-step and cumulative spend without re-deriving it from a duplicate price table.
            cost = self.usage.price_call(
                model, toks["input_tokens"], toks.get("cache_input_tokens") or 0,
                toks["output_tokens"], toks.get("thinking_tokens") or 0,
            )
            ctx.emit(
                f"call call_site={call_site} model={model} temp={temperature} "
                f"effort={effort} latency_s={round(latency_s, 3)} "
                f"in_tok={toks['input_tokens']} out_tok={toks['output_tokens']} "
                f"think_tok={toks.get('thinking_tokens')}",
                data=None if cost is None else {"cost": cost},
            )
        resp = LLMResponse(
            text=text,
            latency_s=latency_s,
            input_tokens=toks["input_tokens"],
            output_tokens=toks["output_tokens"],
            cache_input_tokens=toks.get("cache_input_tokens"),
            thinking_tokens=toks.get("thinking_tokens"),
        )
        self.usage.add(resp, model)
        return resp

    async def _tpm_acquire(
        self, model: str, system: str, user: str
    ) -> tuple[AsyncTokenBudget | None, float]:
        """Acquire the per-model token budget (`config.llm_model_tpm`, falling back to
        `llm_default_tpm`) before a call, charging an up-front input-token estimate.
        Returns (limiter, estimate) so the caller can `_tpm_settle` the actual-vs-estimate
        delta afterward; (None, 0.0) when the model is unthrottled (paced by RPM alone)."""
        tpm = self._config.llm_model_tpm.get(model, self._config.llm_default_tpm)
        if not tpm:
            return None, 0.0
        est = _estimate_prompt_tokens(system, user)
        lim = get_async_tpm_limiter(model, tpm)
        await lim.acquire(est)
        return lim, est

    @staticmethod
    def _tpm_settle(lim: AsyncTokenBudget | None, est: float, input_tokens: int | None) -> None:
        """Post-call correction: charge the actual-minus-estimated input tokens so the
        bucket tracks REAL usage (the char/4 estimate runs ~2x low on dense tabular text)."""
        if lim is not None:
            lim.settle((input_tokens or 0) - est)

    def _finish(
        self, spec: CallSpec, text: str, toks: dict, latency_s: float, model_id: str
    ) -> LLMResponse:
        """`_build_response` with the spec's envelope fields unpacked — the one tail
        every provider body calls."""
        return self._build_response(
            text, toks, latency_s,
            model=model_id, temperature=spec.temperature, effort=spec.effort,
            ctx=spec.ctx, call_site=spec.call_site,
        )


class _OpenAIChatBackend(_LLMBackend):
    """Shared shaping for the two OpenAI-chat-flavored backends (OpenRouter and vLLM):
    message construction, response/chunk text extraction, token-dict packing, and the
    empty-completion check. All extraction is `getattr`-based, so it works identically
    on the `openrouter` SDK's and the `openai` SDK's typed response objects."""

    @staticmethod
    def _content(user: str, images: list[B64Image] | None) -> Any:
        """User-turn content: a plain string, or OpenAI-style content parts when
        images are attached (base64 `data:` URLs)."""
        if not images:
            return user
        parts: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{img.mime};base64,{img.data}"}}
            for img in images
        ]
        parts.append({"type": "text", "text": user})
        return parts

    @staticmethod
    def _text(resp: Any) -> str:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        return getattr(msg, "content", None) or ""

    @staticmethod
    def _finish_reason(resp: Any) -> str | None:
        """First choice's `finish_reason` (e.g. 'stop' / 'length' / 'content_filter' /
        'error'), or None when there are no choices — surfaced in EmptyCompletionError so
        an empty completion's cause is visible in the retry-warning log."""
        choices = getattr(resp, "choices", None) or []
        return getattr(choices[0], "finish_reason", None) if choices else None

    @staticmethod
    def _usage_tokens_chat(usage: Any) -> dict:
        """Token counts from an OpenAI-style chat `usage` (best-effort — any may be None;
        vLLM reports `prompt_tokens_details.cached_tokens` only with prefix caching on)."""
        details = getattr(usage, "prompt_tokens_details", None)
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cache_input_tokens": getattr(details, "cached_tokens", None) if details else None,
        }

    def _single_messages(self, spec: CallSpec) -> list[dict]:
        """Single-shot messages in OpenAI chat format: leading system (if any) + the
        one user turn (with any images) — shared by the sync and async bodies."""
        messages: list[dict] = []
        if spec.system:
            messages.append({"role": "system", "content": spec.system})
        messages.append(
            {"role": "user", "content": self._content(spec.user, spec.images)}
        )
        return messages

    @staticmethod
    def _chat_messages(system: str, messages: list[dict]) -> list[dict]:
        """Multi-turn messages in OpenAI chat format: a leading system message (if any)
        then the {role, content} turns ('assistant' → assistant, else user)."""
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(
            {
                "role": "assistant" if m["role"] == "assistant" else "user",
                "content": _OpenAIChatBackend._content(m["content"], m.get("images")),
            }
            for m in messages
        )
        return out

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        text = ""
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            text += getattr(delta, "content", None) or ""
        return text

    def _checked_text(self, resp: Any, spec: CallSpec) -> str:
        """The completion text, or `EmptyCompletionError` (with the provider's diagnostic
        suffix — see `_empty_detail`) when it came back empty — shared by the sync and
        async bodies."""
        output_text = self._text(resp).strip()
        if not output_text:
            raise EmptyCompletionError(
                f"{self.provider} empty content (model={spec.model} call_site={spec.call_site} "
                f"finish_reason={self._finish_reason(resp)}{self._empty_detail(resp)})"
            )
        return output_text

    def _empty_detail(self, resp: Any) -> str:
        """Provider-specific diagnostic suffix for an empty completion ("" by default)."""
        return ""


class _OpenRouterBackend(_OpenAIChatBackend):
    """OpenRouter generation + embeddings (`provider=openrouter`), via the `openrouter`
    SDK. Owns the OpenRouter-only concepts: per-call provider-order pinning, the
    effort→reasoning mapping, and the raw-body / error-metadata diagnostics."""

    provider = "openrouter"

    def __init__(self, config: SystemConfig, usage: UsageTracker, api_key: str | None = None) -> None:
        super().__init__(config, usage)
        self._api_key = api_key
        self._client: OpenRouter | None = None

    def _get_client(self) -> OpenRouter:
        if self._client is None:
            self._client = _make_openrouter_client(self._api_key)
        return self._client

    @staticmethod
    def _effort_to_reasoning(effort: Effort) -> dict:
        """Map our Effort tier onto OpenRouter's `reasoning` arg. "off" maps to the
        lowest tier ("minimal") rather than disabling reasoning outright, since some
        endpoints (e.g. Gemini 3) mandate reasoning and reject `effort=none` with a
        400. The rest pass straight through (OpenRouter's `effort` accepts
        minimal/low/medium/high). Models without reasoning ignore it."""
        return {"effort": "minimal" if effort == "off" else effort}

    @staticmethod
    def _openrouter_error_detail(resp: Any) -> str:
        """Diagnostic suffix for an empty OpenRouter completion. Two cases:

        - If OpenRouter populated the response-level `error` (code + a message that usually
          carries the provider's raw text — rate-limit, capacity, content-filter, 5xx),
          surface it: that's the actionable cause.
        - On a `finish_reason=error` from an upstream Gemini fault, OpenRouter instead
          returns a NULL message (content/reasoning/refusal all null) and NO error body —
          there is no "why" in the response. The only handle is the generation `id` (look it
          up at openrouter.ai/activity or `GET /api/v1/generation?id=<id>`), so surface that
          plus the resolved model. The null message also rules out a refusal or a
          length/thinking truncation — this is purely provider-side.

        Best-effort; returns "" when nothing useful is present."""
        err = getattr(resp, "error", None)
        if err:
            code = getattr(err, "code", None)
            msg = getattr(err, "message", None)
            parts = ([f"code={code}"] if code is not None else []) + (
                [f"message={str(msg)[:800]}"] if msg else []
            )
            return (" upstream_error: " + " ".join(parts)) if parts else ""
        gen_id = getattr(resp, "id", None)
        resolved = getattr(resp, "model", None)
        parts: list[str] = []
        if gen_id:
            # Emit the generation-lookup URL inline so a failure is one click from its
            # native provider error. NOTE: this `GET /generation` endpoint 404s for most
            # `finish_reason=error` generations (OpenRouter doesn't persist a generation
            # that failed upstream) and has a ~20s propagation delay even for successes —
            # the web dashboard (openrouter.ai/activity) is the more reliable place to see
            # the native error. The URL is still worth surfacing: it resolves for some
            # failure modes and is the handle for the activity page.
            parts.append(f"gen_id={gen_id}")
            parts.append(f"lookup=https://openrouter.ai/api/v1/generation?id={gen_id}")
        if resolved:
            parts.append(f"resolved_model={resolved}")
        return (" no_error_body (" + " ".join(parts) + ")") if parts else ""

    def _empty_detail(self, resp: Any) -> str:
        return self._openrouter_error_detail(resp) + _raw_body_suffix()

    def _provider_kwarg(self, spec: "CallSpec | None" = None) -> dict:
        """`{"provider": {...}}` pinning generation to a provider order (no fallback), or `{}` when
        unset. A per-call `spec.provider_order` wins over the client-wide `config.llm_provider_order`,
        so one model (e.g. a cheaper semantic-filter judge) can be routed to specific providers while
        the rest of the client stays unpinned. Lets a run route deterministically to one provider so
        its prompt-cache and pricing are the ones actually used."""
        order = (spec.provider_order if spec else None) or getattr(self._config, "llm_provider_order", None)
        if not order:
            return {}
        return {"provider": {"order": list(order), "allow_fallbacks": False}}

    def _gen_call(self, spec: CallSpec) -> LLMResponse:
        """One OpenRouter chat call (no retry — the retry loop owns that)."""
        client = self._get_client()
        messages = self._single_messages(spec)
        reasoning = self._effort_to_reasoning(spec.effort)
        t0 = time.monotonic()
        resp = client.chat.send(
            model=spec.model, messages=messages, stream=False, # type: ignore
            temperature=spec.temperature, reasoning=reasoning, # type: ignore
            **self._provider_kwarg(spec),
        )
        latency_s = time.monotonic() - t0
        output_text = self._checked_text(resp, spec)
        toks = self._usage_tokens_chat(getattr(resp, "usage", None))
        return self._finish(spec, output_text, toks, latency_s, spec.model)

    async def _gen_acall(self, spec: CallSpec) -> LLMResponse:
        """Async twin of `_gen_call` — uses `client.chat.send_async`."""
        client = self._get_client()
        messages = self._single_messages(spec)
        reasoning = self._effort_to_reasoning(spec.effort)
        # TPM throttle (opt-in via SKUNK_MODEL_TPM): meter input tokens so throughput
        # stays under quota, separate from the RPM limiter. Per attempt (the retry loop
        # re-invokes this body), so each retry re-charges. `_tpm_settle` corrects the
        # estimate post-call.
        tpm_lim, est = await self._tpm_acquire(spec.model, spec.system, spec.user)
        extra = (
            {"max_tokens": spec.max_output_tokens}
            if spec.max_output_tokens is not None
            else {}
        )
        extra.update(self._provider_kwarg(spec))
        t0 = time.monotonic()
        coro = client.chat.send_async(
            model=spec.model, messages=messages, stream=False, # type: ignore
            temperature=spec.temperature, reasoning=reasoning, **extra, # type: ignore
        )
        resp = await (
            asyncio.wait_for(coro, spec.timeout_s) if spec.timeout_s is not None else coro
        )
        latency_s = time.monotonic() - t0
        output_text = self._checked_text(resp, spec)
        toks = self._usage_tokens_chat(getattr(resp, "usage", None))
        self._tpm_settle(tpm_lim, est, toks["input_tokens"])
        return self._finish(spec, output_text, toks, latency_s, spec.model)

    async def _gen_astream(self, spec: CallSpec) -> LLMResponse:
        """OpenRouter streaming body for `astream` — `client.chat.send_async` + `async for`."""
        client = self._get_client()
        or_messages = self._chat_messages(spec.system, list(spec.messages or ()))
        reasoning = self._effort_to_reasoning(spec.effort)
        extra = (
            {"max_tokens": spec.max_output_tokens}
            if spec.max_output_tokens is not None
            else {}
        )
        extra.update(self._provider_kwarg(spec))
        should_stop = spec.should_stop

        async def _consume() -> tuple[str, Any]:
            accumulated = ""
            stopped_text: str | None = None
            usage = None
            async with await client.chat.send_async(
                model=spec.model, messages=or_messages, stream=True, # type: ignore
                temperature=spec.temperature, reasoning=reasoning, **extra, # type: ignore
            ) as resp_stream:
                async for chunk in resp_stream:
                    accumulated += self._chunk_text(chunk)
                    usage = getattr(chunk, "usage", None) or usage
                    # OpenRouter sends the usage chunk LAST, so we must drain the stream
                    # to capture cost stats — breaking at `should_stop` drops them
                    # entirely. Record the cut point for the returned text but keep
                    # reading; the agent is prompted to emit one block then stop, so the
                    # stream usually ends right after (max_output_tokens + the request
                    # timeout bound the worst case).
                    if stopped_text is None and should_stop is not None and should_stop(accumulated):
                        stopped_text = accumulated
            return (accumulated if stopped_text is None else stopped_text), usage

        t0 = time.monotonic()
        if spec.timeout_s is not None:
            accumulated, usage = await asyncio.wait_for(_consume(), spec.timeout_s)
        else:
            accumulated, usage = await _consume()
        latency_s = time.monotonic() - t0
        if usage is None:
            _warn(
                spec.ctx,
                f"stream_no_usage cost undercounted (call_site={spec.call_site} model={spec.model})",
            )
        toks = self._usage_tokens_chat(usage)
        return self._finish(spec, accumulated, toks, latency_s, spec.model)

    def _embed_once(self, model: str, text: str) -> tuple[list[float], int]:
        """One OpenRouter embeddings call (e.g. Qwen3-Embedding-8B). Token count is the
        provider-reported `usage.prompt_tokens`."""
        resp = self._get_client().embeddings.generate(input=text, model=model)
        data = getattr(resp, "data", None) or []
        vector = list(data[0].embedding) if data and data[0].embedding else []
        if not vector:
            raise EmptyEmbeddingError(f"openrouter empty embedding (model={model})")
        usage = getattr(resp, "usage", None)
        in_tok = (getattr(usage, "prompt_tokens", None) or 0) if usage else 0
        return vector, in_tok


class _VLLMBackend(_OpenAIChatBackend):
    """Local vLLM servers speaking the OpenAI API (`provider=vllm`), via the `openai`
    SDK. vLLM serves ONE model per server process, so the base URL is resolved per
    model from `config.vllm_base_urls` (launch servers with qatfd's
    `scripts/run_vllm_servers.sh`, whose `--served-model-name` defaults to the model id
    so the map keys line up). `spec.effort` is deliberately ignored — reasoning control
    on local models goes through `config.vllm_extra_body` (e.g. `chat_template_kwargs`)
    — as is `spec.provider_order` (an OpenRouter routing concept with no vLLM
    analogue). Models routed here cost $0 even when `llm_prices` prices them: the
    facade registers every `vllm_base_urls` key as a `UsageTracker` free model."""

    provider = "vllm"

    def __init__(self, config: SystemConfig, usage: UsageTracker, api_key: str | None = None) -> None:
        super().__init__(config, usage)
        self._api_key = api_key

    def _clients_for(self, model: str) -> tuple[OpenAI, AsyncOpenAI]:
        base_url = self._config.vllm_base_urls.get(model)
        if base_url is None:
            raise RuntimeError(
                f"model {model!r} routed to vllm but has no vllm_base_urls entry "
                f"(configured: {sorted(self._config.vllm_base_urls)})"
            )
        return _make_vllm_clients(base_url, self._api_key)

    def _request_kwargs(self, spec: CallSpec) -> dict:
        """Per-call extras beyond (model, messages, temperature): the output cap when
        set, and `config.vllm_extra_body` merged into the request body (vLLM-specific
        knobs such as `chat_template_kwargs`)."""
        kwargs: dict = {}
        if spec.max_output_tokens is not None:
            kwargs["max_tokens"] = spec.max_output_tokens
        if self._config.vllm_extra_body:
            kwargs["extra_body"] = self._config.vllm_extra_body
        return kwargs

    def _gen_call(self, spec: CallSpec) -> LLMResponse:
        """One vLLM chat call (no retry — the retry loop owns that)."""
        client, _ = self._clients_for(spec.model)
        messages = self._single_messages(spec)
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=spec.model, messages=messages, temperature=spec.temperature, # type: ignore[arg-type]
            **self._request_kwargs(spec),
        )
        latency_s = time.monotonic() - t0
        output_text = self._checked_text(resp, spec)
        toks = self._usage_tokens_chat(getattr(resp, "usage", None))
        return self._finish(spec, output_text, toks, latency_s, spec.model)

    async def _gen_acall(self, spec: CallSpec) -> LLMResponse:
        """Async twin of `_gen_call`."""
        _, aclient = self._clients_for(spec.model)
        messages = self._single_messages(spec)
        # TPM throttle (opt-in via SKUNK_MODEL_TPM) — identical to the OpenRouter path.
        tpm_lim, est = await self._tpm_acquire(spec.model, spec.system, spec.user)
        t0 = time.monotonic()
        coro = aclient.chat.completions.create(
            model=spec.model, messages=messages, temperature=spec.temperature, # type: ignore[arg-type]
            **self._request_kwargs(spec),
        )
        resp = await (
            asyncio.wait_for(coro, spec.timeout_s) if spec.timeout_s is not None else coro
        )
        latency_s = time.monotonic() - t0
        output_text = self._checked_text(resp, spec)
        toks = self._usage_tokens_chat(getattr(resp, "usage", None))
        self._tpm_settle(tpm_lim, est, toks["input_tokens"])
        return self._finish(spec, output_text, toks, latency_s, spec.model)

    async def _gen_astream(self, spec: CallSpec) -> LLMResponse:
        """vLLM streaming body for `astream`. `stream_options.include_usage` requests
        the final usage chunk — OpenAI-spec servers omit usage by default when
        streaming (unlike OpenRouter, which always sends it)."""
        _, aclient = self._clients_for(spec.model)
        messages = self._chat_messages(spec.system, list(spec.messages or ()))
        should_stop = spec.should_stop

        async def _consume() -> tuple[str, Any]:
            accumulated = ""
            stopped_text: str | None = None
            usage = None
            stream = await aclient.chat.completions.create(
                model=spec.model, messages=messages, temperature=spec.temperature, # type: ignore[arg-type]
                stream=True, stream_options={"include_usage": True},
                **self._request_kwargs(spec),
            )
            try:
                async for chunk in stream:
                    accumulated += self._chunk_text(chunk)
                    usage = getattr(chunk, "usage", None) or usage
                    # The usage chunk arrives LAST (include_usage), so keep draining past
                    # `should_stop` and record the cut point for the returned text — same
                    # contract as the OpenRouter streaming body.
                    if stopped_text is None and should_stop is not None and should_stop(accumulated):
                        stopped_text = accumulated
            finally:
                # Runs on normal completion AND on wait_for cancellation, so a
                # timed-out stream still releases its connection.
                try:  # noqa: SIM105
                    await stream.close()
                except Exception:
                    pass
            return (accumulated if stopped_text is None else stopped_text), usage

        t0 = time.monotonic()
        if spec.timeout_s is not None:
            accumulated, usage = await asyncio.wait_for(_consume(), spec.timeout_s)
        else:
            accumulated, usage = await _consume()
        latency_s = time.monotonic() - t0
        if usage is None:
            _warn(
                spec.ctx,
                f"stream_no_usage cost undercounted (call_site={spec.call_site} model={spec.model})",
            )
        toks = self._usage_tokens_chat(usage)
        return self._finish(spec, accumulated, toks, latency_s, spec.model)

    def _embed_once(self, model: str, text: str) -> tuple[list[float], int]:
        """One vLLM embeddings call (server started with `--task embed`). Token count is
        the server-reported `usage.prompt_tokens`."""
        client, _ = self._clients_for(model)
        resp = client.embeddings.create(model=model, input=text)
        data = getattr(resp, "data", None) or []
        vector = list(data[0].embedding) if data and data[0].embedding else []
        if not vector:
            raise EmptyEmbeddingError(f"vllm empty embedding (model={model})")
        usage = getattr(resp, "usage", None)
        in_tok = (getattr(usage, "prompt_tokens", None) or 0) if usage else 0
        return vector, in_tok


class LLMClient:
    """The public LLM client — a facade that routes each call to a provider backend
    and owns the run's single `UsageTracker`. Generation routes PER CALL: a model
    listed in `config.vllm_base_urls` goes to the vLLM server that serves it; every
    other model uses `config.llm_provider` ("openrouter" | "vllm") — so one client can
    run e.g. the search agent on OpenRouter while the semantic filter judges on a
    local vLLM model. Embeddings route on `config.emb_provider` the same way. All
    backends share this client's rate-limit / retry / logging scaffolding and bill
    into `self.usage`."""

    def __init__(
        self,
        config: SystemConfig,
        *,
        openrouter_api_key: str | None = None,
        vllm_api_key: str | None = None,
    ) -> None:
        """`openrouter_api_key` / `vllm_api_key` override the environment
        (`OPENROUTER_API_KEY` / `VLLM_API_KEY`) — the constructor is the seam for
        callers that manage credentials themselves; env remains the default."""
        self._config = config
        self._openrouter_api_key = openrouter_api_key
        self._vllm_api_key = vllm_api_key
        self.usage = UsageTracker(
            default_model=config.llm_model,
            prices=config.llm_prices,
            # vLLM-routed models cost $0 regardless of the price table: routing is static
            # per model id (a `vllm_base_urls` key ALWAYS goes local on this client), so
            # the map's keys are exactly the models whose tokens were served locally —
            # and the same id can stay priced for OpenRouter runs of other configs.
            free_models=set(config.vllm_base_urls),
        )
        self._backends: dict[str, _LLMBackend] = {}

    def _backend(self, provider: str) -> _LLMBackend:
        """The (lazily built, cached) backend for a provider name. All backends share
        this client's `usage` tracker."""
        backend = self._backends.get(provider)
        if backend is None:
            if provider == "openrouter":
                backend = _OpenRouterBackend(self._config, self.usage, api_key=self._openrouter_api_key)
            elif provider == "vllm":
                backend = _VLLMBackend(self._config, self.usage, api_key=self._vllm_api_key)
            else:
                raise ValueError(f"unknown LLM provider {provider!r} (expected 'openrouter' or 'vllm')")
            self._backends[provider] = backend
        return backend

    def _backend_for_model(self, model: str) -> _LLMBackend:
        """Per-call generation routing: a model with a `vllm_base_urls` entry (exact
        match — keys must equal the server's --served-model-name) goes to vLLM;
        anything else uses the client-wide `llm_provider`. When `llm_provider` is
        "vllm", EVERY model must be mapped — an unmapped one fails here, naming the
        configured keys, rather than deep in an SDK with a connection error."""
        if model in self._config.vllm_base_urls:
            return self._backend("vllm")
        if self._config.llm_provider == "vllm":
            raise RuntimeError(
                f"llm_provider=vllm but model {model!r} has no vllm_base_urls entry "
                f"(configured: {sorted(self._config.vllm_base_urls)})"
            )
        return self._backend(self._config.llm_provider)

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
        provider_order: list[str] | None = None,
    ) -> LLMResponse:
        spec = CallSpec(
            system=system, user=user, images=images, temperature=temperature,
            effort=effort, ctx=ctx, call_site=call_site,
            model=model or self._config.llm_model,
            provider_order=provider_order,
        )
        return self._backend_for_model(spec.model).call(spec)

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
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Async twin of `call` for the request path. `max_output_tokens` /
        `timeout_s` mirror `astream`'s caps (None → provider default / no cap)."""
        spec = CallSpec(
            system=system, user=user, images=images, temperature=temperature,
            effort=effort, ctx=ctx, call_site=call_site,
            model=model or self._config.llm_model,
            max_output_tokens=max_output_tokens, timeout_s=timeout_s,
        )
        return await self._backend_for_model(spec.model).acall(spec)

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
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Multi-turn streaming call, accumulating chunks until `should_stop(acc)`
        or the stream ends. `messages` are the {role, content} turns after the
        system message ('assistant' → model role, else user). Lets multi-turn
        agents share this client's rate-limit + retry + logging; the event loop
        runs other tasks between chunks.

        `max_output_tokens` overrides the per-call output cap (None → provider
        default); `timeout_s` enforces a hard per-request wall-clock cap. Both are
        opt-in and currently used only by the search agent (see `MultiTurnAgent`)."""
        spec = CallSpec(
            system=system, messages=tuple(messages), temperature=temperature,
            effort=effort, ctx=ctx, call_site=call_site, should_stop=should_stop,
            model=model or self._config.llm_model,
            max_output_tokens=max_output_tokens, timeout_s=timeout_s,
        )
        return await self._backend_for_model(spec.model).astream(spec)

    def embed_query(
        self,
        text: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        ctx: ExecutionContext | None = None,
    ) -> list[float]:
        """Embed a single query string for vector search (see `_LLMBackend.embed_query`
        for the retry/usage envelope).

        Backends (per `config.emb_provider`, overridable via `provider`):
          - "openrouter": the OpenRouter embeddings endpoint (e.g. Qwen3-Embedding-8B).
          - "vllm": a local vLLM embedding server (`--task embed`); the base URL comes
            from `config.vllm_base_urls[model]`, same map as generation."""
        model = model or self._config.emb_model_id
        provider = provider or self._config.emb_provider
        return self._backend(provider).embed_query(text, model=model, ctx=ctx)
