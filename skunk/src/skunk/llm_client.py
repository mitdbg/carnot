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
    estimate_tokens,
    get_async_tpm_limiter,
    get_rate_limiter,
)
from skunk.constants import IMAGE_TOKENS_EST
from skunk.usage import UsageTracker

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI
    from openrouter import OpenRouter

    from skunk.common import B64Image
    from skunk.config import InferenceConfig
    from skunk.trace import Tracer

def _warn(tracer: Tracer | None, id: str, msg: str, data: dict | None) -> None:
    """Route a client warning onto the owning question's event stream when a tracer
    is in scope (retry/usage warnings become attributable per question); tracer-less
    callers (offline corpus prep) fall back to plain stderr."""
    if tracer is not None:
        tracer.emit(id, level="warning", kind="call", message=msg, data=data or {})
    else:
        print(msg, file=sys.stderr)

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
    # per-request wall-clock cap (asyncio.wait_for in `_gen_acall` raises builtin
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

def _estimate_prompt_tokens(messages: list[dict]) -> float:
    """Pre-call input-token estimate for the TPM bucket: `common.estimate_tokens` on each
    message's `content` text (exact BPE count, chars/4 above its size cutoff) plus a flat
    per-image charge. Approximate by design — it only paces throughput, it doesn't bill;
    `_tpm_settle` corrects against real usage post-call."""
    text_tokens = sum(estimate_tokens(m.get("content") or "") for m in messages)
    n_images = sum(len(m.get("images") or ()) for m in messages)
    return text_tokens + n_images * IMAGE_TOKENS_EST


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    # Cached (prompt-cache hit) input tokens, when the provider reports them.
    # Subset of input_tokens; None when unknown.
    cache_input_tokens: int | None = None
    # Thinking/reasoning tokens (billed at the output rate); None when unreported.
    thinking_tokens: int | None = None



@dataclass(frozen=True)
class CallSpec:
    """One generation request, provider-agnostic — the value threaded from the
    public `call` / `acall` methods into the per-provider bodies. `model` is always
    resolved by the public method (never empty inside a provider body). `messages`
    is expected to be a list of dictionaries with the following format:
    [
        {"role": "system" | "user" | "assistant", "content": str, "images": list[B64Image] | None},
    ]
    """
    model: str
    messages: list[dict]
    temperature: float = 0.0
    effort: Effort = "off"
    # Disable reasoning outright (OpenRouter `reasoning={"effort": "none"}`), overriding `effort`.
    # Unlike effort="off" — which maps to the cheapest reasoning tier because some endpoints
    # mandate reasoning — this asks the provider to emit NO thinking tokens at all, for call
    # sites whose reply is a single token (e.g. the semantic_filter TRUE/FALSE judge). Endpoints
    # that mandate reasoning (e.g. Gemini 3) reject it with a 400 — keep those call sites on
    # `effort` instead. (The SDK's chat `Reasoning` model silently DROPS an `enabled` key, so
    # `{"enabled": false}` is not a usable encoding here; `effort` is an open enum and "none"
    # passes through to the provider.)
    disable_reasoning: bool = False
    call_site: str = "llm"
    max_output_tokens: int | None = None
    timeout_s: float | None = None
    provider_order: list[str] | None = None
    usage_key: str = "default"


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
    if not body:
        return ""
    # A finish_reason=length empty completion can be ~2 KB of newlines/whitespace; kept verbatim
    # it turns this one-line diagnostic (and the trace viewer's rendering of it) into a huge blank
    # block, so collapse whitespace runs — the JSON tokens that matter for debugging survive.
    body = " ".join(body.split())
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
    (`_gen_call` / `_gen_acall` / `_embed_once`). Backends are built only by the `LLMClient`
    facade, which injects its own `UsageTracker` — every backend of a client bills into the
    one tracker, so a mixed OpenRouter+vLLM run still reports one usage/cost total."""

    provider: str = "?"

    def __init__(self, config: InferenceConfig, usage: UsageTracker, tracer: Tracer | None = None) -> None:
        self._config = config
        self.usage = usage
        self._tracer = tracer

    # --- entry points (called by the LLMClient facade with a resolved CallSpec) -------

    def call(self, spec: CallSpec) -> LLMResponse:
        return self._retry_call(lambda: self._gen_call(spec), self.provider, spec.model)

    async def acall(self, spec: CallSpec) -> LLMResponse:
        return await self._aretry_call(lambda: self._gen_acall(spec), self.provider, spec.model)

    def embed_query(self, text: str, *, model: str, usage_key: str = "default", http_headers: dict[str, str] | None = None) -> list[float]:
        """Embed a single query string for vector search, routing through the same
        rate-limit / retry / usage-accounting scaffolding as generation so embedding
        spend lands on this client's `usage` tracker (tokens, cost, calls). The provider
        body is `_embed_once`; if `self._tracer` is present, emits the uniform `call ...`
        envelope (so the embed call's latency shows in the per-question trace like every LLM
        call); usage is recorded regardless of the tracer."""

        def attempt() -> list[float]:
            t0 = time.monotonic()
            vector, in_tok = self._embed_once(model, text, http_headers=http_headers)
            latency_s = time.monotonic() - t0
            self.usage.add_embed(model, in_tok, usage_key)
            if self._tracer is not None:
                cost = self.usage.price_embed(model, in_tok)
                self._tracer.emit(
                    id="call",
                    kind="call",
                    data={
                        "call_site": "embed",
                        "model": model,
                        "provider": self.provider,
                        "latency_s": round(latency_s, 3),
                        "in_tok": in_tok,
                        "dim": len(vector),
                        "cost": cost
                    },
                )

            return vector

        # Embeddings share one process-wide "embed" rate bucket across all workers
        # (separate from the per-model generation buckets).
        return self._retry_call(attempt, self.provider, model, bucket="embed")

    # --- provider bodies (subclass responsibility) -------------------------------------

    def _gen_call(self, spec: CallSpec) -> LLMResponse:
        raise NotImplementedError(f"provider {self.provider!r} does not implement call")

    async def _gen_acall(self, spec: CallSpec) -> LLMResponse:
        raise NotImplementedError(f"provider {self.provider!r} does not implement acall")

    def _embed_once(self, model: str, text: str, http_headers: dict[str, str] | None = None) -> tuple[list[float], int]:
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
    ) -> R:
        """Drive `attempt()` with exponential-backoff retry, paced by a
        rate-limiter bucket. Retries only transient faults (`_is_retryable`: 429 / 5xx /
        transport blips); a non-429 4xx (bad request, auth, context overflow) raises
        immediately. `attempt` owns the API call, timing, parsing, and the success emit;
        `provider`/`model` are for the failure warning (routed onto `self._tracer`'s event
        stream when present — see `_warn`). `bucket` overrides the limiter (a shared
        process-wide bucket like `"embed"`); when None, the per-model `llm:<model>` bucket
        is used at the configured RPM."""
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
                    self._tracer,
                    id="llm_call_failed",
                    msg=(
                        f"llm_call_failed attempt={i + 1}/{max_retries + 1} provider={provider} "
                        f"model={model} error={type(e).__name__}: {e}{_error_detail(e)}"
                        f"{'' if stop else f'; retrying in {delay:.1f}s'}"
                    ),
                    data={
                        "attempt": i + 1,
                        "max_attempts": max_retries + 1,
                        "provider": provider,
                        "model": model,
                        "error": f"{type(e).__name__}: {e}{_error_detail(e)}",
                    }
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
                    self._tracer,
                    id="llm_call_failed",
                    msg=(
                        f"llm_call_failed attempt={i + 1}/{max_retries + 1} provider={provider} "
                        f"model={model} error={type(e).__name__}: {e}{_error_detail(e)}"
                        f"{'' if stop else f'; retrying in {delay:.1f}s'}"
                    ),
                    data={
                        "attempt": i + 1,
                        "max_attempts": max_retries + 1,
                        "provider": provider,
                        "model": model,
                        "error": f"{type(e).__name__}: {e}{_error_detail(e)}",
                    }
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
        call_site: str,
        usage_key: str,
    ) -> LLMResponse:
        """Shared tail for every generation path: emit the uniform `call ...` envelope
        (when `self._tracer` is set), accumulate the call into this client's `usage` tracker,
        and pack the provider-agnostic token dict (`_usage_tokens_chat`) into an `LLMResponse`.
        `toks` may omit any key — the chat dict has no `thinking_tokens`, hence the `.get`s.
        The single chokepoint every call path funnels through, so usage accounting lives here
        once rather than at each public method."""
        if self._tracer is not None:
            # Exact per-call USD (priced as `cost()` aggregates) → the trace viewer shows
            # per-step and cumulative spend without re-deriving it from a duplicate price table.
            cost = self.usage.price_call(
                model, toks["input_tokens"], toks.get("cache_input_tokens") or 0,
                toks["output_tokens"], toks.get("thinking_tokens") or 0,
            )
            self._tracer.emit(
                id="call",
                kind="call",
                data={
                    "call_site": call_site,
                    "model": model,
                    "provider": self.provider,
                    "temperature": temperature,
                    "effort": effort,
                    "latency_s": round(latency_s, 3),
                    "in_tok": toks["input_tokens"],
                    "out_tok": toks["output_tokens"],
                    "think_tok": toks.get("thinking_tokens"),
                    "cost": cost,
                },
            )
        resp = LLMResponse(
            text=text,
            latency_s=latency_s,
            input_tokens=toks["input_tokens"],
            output_tokens=toks["output_tokens"],
            cache_input_tokens=toks.get("cache_input_tokens"),
            thinking_tokens=toks.get("thinking_tokens"),
        )
        self.usage.add(resp, model, usage_key)
        return resp

    async def _tpm_acquire(
        self, model: str, messages: list[dict],
    ) -> tuple[AsyncTokenBudget | None, float]:
        """Acquire the per-model token budget (`config.llm_model_tpm`, falling back to
        `llm_default_tpm`) before a call, charging an up-front input-token estimate.
        Returns (limiter, estimate) so the caller can `_tpm_settle` the actual-vs-estimate
        delta afterward; (None, 0.0) when the model is unthrottled (paced by RPM alone)."""
        tpm = self._config.llm_model_tpm.get(model, self._config.llm_default_tpm)
        if not tpm:
            return None, 0.0
        est = _estimate_prompt_tokens(messages)
        lim = get_async_tpm_limiter(model, tpm)
        await lim.acquire(est)
        return lim, est

    @staticmethod
    def _tpm_settle(lim: AsyncTokenBudget | None, est: float, input_tokens: int | None) -> None:
        """Post-call correction: charge the actual-minus-estimated input tokens so the
        bucket tracks REAL usage (the pre-call estimate uses o200k_base, not the provider's
        tokenizer, and flat-estimates images and message overhead)."""
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
            call_site=spec.call_site, usage_key=spec.usage_key,
        )


class _OpenAIChatBackend(_LLMBackend):
    """Shared shaping for the two OpenAI-chat-flavored backends (OpenRouter and vLLM):
    message construction, response/chunk text extraction, token-dict packing, and the
    empty-completion check. All extraction is `getattr`-based, so it works identically
    on the `openrouter` SDK's and the `openai` SDK's typed response objects."""

    @staticmethod
    def _content(user: str, images: list[B64Image] | None) -> str | list[dict]:
        """Returns a plain string, or OpenAI-style content parts when images are attached (base64 `data:` URLs)."""
        if not images:
            return user
        parts: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{img.mime};base64,{img.data}"}}
            for img in images
        ]
        parts.append({"type": "text", "text": user})
        return parts

    @staticmethod
    def _role(role: str) -> str:
        """Returns the role as-is; maps unexpected roles to "user"."""
        return role if role in ["system", "assistant", "user"] else "user"

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

    def _chat_messages(self, messages: list[dict]) -> list[dict]:
        """Multi-turn messages in OpenAI chat format."""
        return [
            {
                "role": self._role(msg["role"]),
                "content": self._content(msg["content"], msg.get("images")),
            }
            for msg in messages
        ]

    def _checked_text(self, resp: Any, spec: CallSpec) -> str:
        """The completion text, or `EmptyCompletionError` (with the provider's diagnostic
        suffix — see `_empty_detail`) when it came back empty — shared by the sync and
        async bodies."""
        output_text = self._text(resp).strip()
        if not output_text:
            raise EmptyCompletionError(
                f"{self.provider} empty content (model={spec.model} call_site={spec.call_site} usage_key={spec.usage_key}"
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

    def __init__(self, config: InferenceConfig, usage: UsageTracker, tracer: Tracer | None = None, api_key: str | None = None) -> None:
        super().__init__(config, usage, tracer)
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

    def _provider_kwarg(self, spec: CallSpec | None = None) -> dict:
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
        messages = self._chat_messages(spec.messages)
        reasoning = {"effort": "none"} if spec.disable_reasoning else self._effort_to_reasoning(spec.effort)
        extra = (
            {"max_tokens": spec.max_output_tokens}
            if spec.max_output_tokens is not None
            else {}
        )
        extra.update(self._provider_kwarg(spec))
        if spec.timeout_s is not None:
            # Sync twin of the async path's asyncio.wait_for cap, enforced at the SDK
            # request layer. A tripped timeout raises httpx.TimeoutException — retryable.
            extra["timeout_ms"] = int(spec.timeout_s * 1000)
        t0 = time.monotonic()
        resp = client.chat.send(
            model=spec.model, messages=messages, stream=False, # type: ignore
            temperature=spec.temperature, reasoning=reasoning, **extra, # type: ignore
        )
        latency_s = time.monotonic() - t0
        output_text = self._checked_text(resp, spec)
        toks = self._usage_tokens_chat(getattr(resp, "usage", None))
        return self._finish(spec, output_text, toks, latency_s, spec.model)

    async def _gen_acall(self, spec: CallSpec) -> LLMResponse:
        """Async twin of `_gen_call` — uses `client.chat.send_async`."""
        client = self._get_client()
        messages = self._chat_messages(spec.messages)
        reasoning = {"effort": "none"} if spec.disable_reasoning else self._effort_to_reasoning(spec.effort)
        # TPM throttle (opt-in via SKUNK_MODEL_TPM): meter input tokens so throughput
        # stays under quota, separate from the RPM limiter. Per attempt (the retry loop
        # re-invokes this body), so each retry re-charges. `_tpm_settle` corrects the
        # estimate post-call. Estimated from the spec-format messages (`content` text +
        # `images`), the wire-format `messages` local hides images inside content-parts lists.
        tpm_lim, est = await self._tpm_acquire(spec.model, spec.messages)
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

    def _embed_once(self, model: str, text: str, http_headers: dict[str, str] | None = None) -> tuple[list[float], int]:
        """One OpenRouter embeddings call (e.g. Qwen3-Embedding-8B). Token count is the
        provider-reported `usage.prompt_tokens`. `http_headers` are sent on this request
        only (e.g. x-session-id for OpenRouter session attribution; OpenRouter ignores a
        body session_id on the embeddings endpoint)."""
        resp = self._get_client().embeddings.generate(input=text, model=model, http_headers=dict(http_headers) if http_headers else None)
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

    def __init__(self, config: InferenceConfig, usage: UsageTracker, tracer: Tracer | None = None, api_key: str | None = None) -> None:
        super().__init__(config, usage, tracer)
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
        messages = self._chat_messages(spec.messages)
        kwargs = self._request_kwargs(spec)
        if spec.timeout_s is not None:
            # Sync twin of the async path's asyncio.wait_for cap, enforced at the SDK
            # request layer. A tripped timeout raises APITimeoutError — retryable.
            kwargs["timeout"] = spec.timeout_s
        t0 = time.monotonic()
        resp = client.chat.completions.create(
            model=spec.model, messages=messages, temperature=spec.temperature, # type: ignore[arg-type]
            **kwargs,
        )
        latency_s = time.monotonic() - t0
        output_text = self._checked_text(resp, spec)
        toks = self._usage_tokens_chat(getattr(resp, "usage", None))
        return self._finish(spec, output_text, toks, latency_s, spec.model)

    async def _gen_acall(self, spec: CallSpec) -> LLMResponse:
        """Async twin of `_gen_call`."""
        _, aclient = self._clients_for(spec.model)
        messages = self._chat_messages(spec.messages)
        # TPM throttle (opt-in via SKUNK_MODEL_TPM) — identical to the OpenRouter path.
        tpm_lim, est = await self._tpm_acquire(spec.model, spec.messages)
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

    def _embed_once(self, model: str, text: str, http_headers: dict[str, str] | None = None) -> tuple[list[float], int]:
        """One vLLM embeddings call (server started with `--task embed`). Token count is
        the server-reported `usage.prompt_tokens`."""
        # http_headers unused: local vLLM has no per-session attribution
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
        config: InferenceConfig,
        *,
        tracer: Tracer | None = None,
        openrouter_api_key: str | None = None,
        vllm_api_key: str | None = None,
    ) -> None:
        """`openrouter_api_key` / `vllm_api_key` override the environment
        (`OPENROUTER_API_KEY` / `VLLM_API_KEY`) — the constructor is the seam for
        callers that manage credentials themselves; env remains the default."""
        self._config = config
        self._tracer = tracer
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

    @property
    def config(self) -> InferenceConfig:
        """The client's `InferenceConfig` (read-only). The seam for callers that need the
        client's model defaults or limits (e.g. `SemanticFilterTool` resolving its judge
        model's context window) without threading the config alongside the client."""
        return self._config

    def _backend(self, provider: str) -> _LLMBackend:
        """The (lazily built, cached) backend for a provider name. All backends share
        this client's `usage` tracker."""
        backend = self._backends.get(provider)
        if backend is None:
            if provider == "openrouter":
                backend = _OpenRouterBackend(self._config, self.usage, tracer=self._tracer, api_key=self._openrouter_api_key)
            elif provider == "vllm":
                backend = _VLLMBackend(self._config, self.usage, tracer=self._tracer, api_key=self._vllm_api_key)
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
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        disable_reasoning: bool = False,
        call_site: str = "llm",
        provider_order: list[str] | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        usage_key: str = "default",
    ) -> LLMResponse:
        spec = CallSpec(
            messages=list(messages),
            temperature=temperature, effort=effort, disable_reasoning=disable_reasoning,
            call_site=call_site,
            model=model or self._config.llm_model, usage_key=usage_key,
            provider_order=provider_order, max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )
        return self._backend_for_model(spec.model).call(spec)

    async def acall(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        effort: Effort = "off",
        disable_reasoning: bool = False,
        call_site: str = "llm",
        provider_order: list[str] | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        usage_key: str = "default",
    ) -> LLMResponse:
        """Async twin of `call` for the request path."""
        spec = CallSpec(
            messages=list(messages),
            temperature=temperature, effort=effort, disable_reasoning=disable_reasoning,
            call_site=call_site,
            model=model or self._config.llm_model, usage_key=usage_key,
            provider_order=provider_order, max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )
        return await self._backend_for_model(spec.model).acall(spec)

    def embed_query(
        self,
        text: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        usage_key: str = "default",
        http_headers: dict[str, str] | None = None,
    ) -> list[float]:
        """Embed a single query string for vector search (see `_LLMBackend.embed_query`
        for the retry/usage envelope).

        Backends (per `config.emb_provider`, overridable via `provider`):
          - "openrouter": the OpenRouter embeddings endpoint (e.g. Qwen3-Embedding-8B).
          - "vllm": a local vLLM embedding server (`--task embed`); the base URL comes
            from `config.vllm_base_urls[model]`, same map as generation."""
        model = model or self._config.emb_model_id
        provider = provider or self._config.emb_provider
        return self._backend(provider).embed_query(text, model=model, usage_key=usage_key, http_headers=http_headers)
