"""Shared runtime: the LLM client plus the cross-cutting types threaded between
operators and the orchestrator (`PageRef`, `AnnotatedValue`, `HarnessContext`).

All LLM traffic goes through the direct Gemini API (AI Studio) via `google-genai`,
authenticated by `GEMINI_API_KEY`. LLM calls are paced by the process-wide
token-bucket limiter named `"llm"` (see `_RATE_LIMITS` / `get_rate_limiter`); any
SDK exception is retried with exponential backoff up to `llm_max_retries` times."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from google import genai
from google.genai import types

from skunk import trace
from skunk.config import SkunkConfig

if TYPE_CHECKING:
    from skunk.prompted_call import PromptOverride

# Reasoning-effort knob, mapped onto Gemini's `thinking_level` enum. "off" means
# no thinking; "minimal" is the cheapest thinking tier.
Effort = Literal["off", "minimal", "low", "medium", "high"]
_EFFORT_VALUES = ("off", "minimal", "low", "medium", "high")

# Process-scoped logger: retries happen with no per-question ctx in scope (see
# `_retry_call`), so they go through stdlib logging rather than `ctx.emit`.
log = logging.getLogger(__name__)


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


def get_rate_limiter(name: str) -> _RateLimiter:
    """Process-wide token-bucket limiter for external service `name` (a key of
    `_RATE_LIMITS`), paced at its rpm — env override read ONCE at first use, then
    frozen for the run. All threads in this process share the named bucket.

    Scope is per process, NOT per API key: separate processes get independent
    buckets and do not coordinate, so a multi-process deployment would not be
    bounded by the underlying per-key cap. The eval harness is single-process."""
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(name)
        if lim is None:
            env_var, default = _RATE_LIMITS[name]
            lim = _RateLimiter(rate_per_sec=float(os.environ.get(env_var, default)) / 60.0)
            _LIMITERS[name] = lim
        return lim


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
        images: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        effort: "Effort" = "off",
        ctx: "HarnessContext | None" = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        if effort not in _EFFORT_VALUES:
            raise ValueError(f"effort must be one of {_EFFORT_VALUES}, got {effort!r}")
        return self._call_gemini(
            system, user, images, temperature, effort, ctx, call_site,
        )

    def _retry_call(self, do_call: "Callable[[], LLMResponse]") -> LLMResponse:
        """Run `do_call` under the rate limiter with exponential-backoff retry.
        `do_call` owns the API invocation, timing, parsing, and success emit."""
        limiter = get_rate_limiter("llm")
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
                log.warning(
                    "attempt %d/%d failed: %s: %s; sleeping %.1fs",
                    attempt + 1, max_retries + 1, type(e).__name__, e, delay,
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
        # ThinkingLevel is the only knob that hard-caps thinking spend for Gemini 3
        # (the legacy thinking_budget int is soft-bucketed).
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
    ) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=65535,
            temperature=temperature,
            thinking_config=LLMClient._effort_to_thinking_config(effort),
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
        images: list[tuple[str, str]] | None,
        temperature: float,
        effort: "Effort",
        ctx: "HarnessContext | None",
        call_site: str = "llm",
    ) -> LLMResponse:
        """Single Gemini call. `call_site` attributes the envelope log to the caller."""
        client = self._get_gemini_client()
        parts = self._gemini_parts(user, images)
        gen_config = self._gemini_config(system, temperature, effort)
        model = self._config.llm_model

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
                    call_site, "call",
                    model=model,
                    temperature=temperature,
                    effort=effort,
                    latency_s=round(latency_s, 3),
                    **toks,
                    input_text=system + "\n\n---\n\n" + user,
                    output_text=output_text,
                )
            return LLMResponse(
                text=output_text,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return self._retry_call(do)

    def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str | None = None,
        should_stop: "Callable[[str], bool] | None" = None,
        temperature: float = 0.0,
        ctx: "HarnessContext | None" = None,
        call_site: str = "llm",
    ) -> LLMResponse:
        """Multi-turn streaming call, accumulating chunks until `should_stop(acc)`
        or the stream ends. `messages` are the {role, content} turns after the
        system message ('assistant' → model role, else user). Lets multi-turn
        agents share this client's rate-limit + retry + logging. `temperature`
        defaults to 0.0 — agent loops are deterministic like every other call site."""
        client = self._get_gemini_client()
        model_id = (model or self._config.llm_model).removeprefix("google/")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]
        gen_config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
        )

        def do() -> LLMResponse:
            t0 = time.monotonic()
            resp_stream = client.models.generate_content_stream(
                model=model_id, contents=contents, config=gen_config,
            )
            accumulated = ""
            usage = None
            for chunk in resp_stream:
                accumulated += chunk.text or ""
                usage = getattr(chunk, "usage_metadata", None) or usage
                if should_stop is not None and should_stop(accumulated):
                    break
            try:  # noqa: SIM105
                resp_stream.close()
            except Exception:
                pass
            latency_s = time.monotonic() - t0
            toks = self._usage_tokens(usage)
            if ctx is not None:
                ctx.emit(
                    call_site, "call",
                    model=model_id,
                    temperature=temperature,
                    latency_s=round(latency_s, 3),
                    **toks,
                    output_text=accumulated,
                )
            return LLMResponse(
                text=accumulated,
                latency_s=latency_s,
                input_tokens=toks["input_tokens"],
                output_tokens=toks["output_tokens"],
            )

        return self._retry_call(do)


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
        import pandas as pd
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


@dataclass
class HarnessContext:
    question: str
    uid: str | None = None    # benchmark UID, when run from the eval harness; tags every event
    verbose: bool = False     # also echo orchestrator + operator events to the console
    log_path: str | None = None  # when set, stream this question's events to that file (live, flushed)
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events
    config: SkunkConfig = field(default_factory=SkunkConfig.from_env)
    llm_client: LLMClient | None = None  # inject a mock for tests; auto-created otherwise
    prompt_overrides: tuple[PromptOverride, ...] = ()  # corpus/few_shot/lesson overrides; operators pick out their own entries by name

    def __post_init__(self) -> None:
        if self.llm_client is None:
            self.llm_client = LLMClient(self.config)
        # The active (step_idx, op) frame is thread-local: a single ctx fans
        # branches out across worker threads (see Orchestrator._run_branches),
        # so each thread must see only the step it is currently running.
        self._step_stack = threading.local()
        # Per-question log file: opened when log_path is set, written line-per-event
        # and flushed live so a single question's stream lands in its own file (and
        # survives a crash). Guarded by a lock since branch threads share this ctx.
        self._logfile = None
        self._log_lock = threading.Lock()
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._logfile = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        """Close the per-question log file, if one was opened."""
        with self._log_lock:
            if self._logfile is not None:
                self._logfile.close()
                self._logfile = None

    @contextmanager
    def step(self, step_idx: int, op: str):
        """Scope the current operator step on this thread. Every `emit` inside
        the block is stamped with `(step_idx, op)`; nested steps stack. The
        orchestrator opens one per traced operator call (replaces the old
        `_step`/`begin` sentinel event as the trace/event join key)."""
        stack = getattr(self._step_stack, "stack", None)
        if stack is None:
            stack = []
            self._step_stack.stack = stack
        stack.append((step_idx, op))
        try:
            yield
        finally:
            stack.pop()

    def _current_step(self) -> tuple[int | None, str | None]:
        stack = getattr(self._step_stack, "stack", None)
        return stack[-1] if stack else (None, None)

    def emit(self, source: str, message: str, level: str | None = None, **fields: Any) -> None:
        """Record a request-scoped diagnostic event onto this question's event stream.

        Convention (see ARCHITECTURE.md "Logging & observability"):
        - `source` is the op / call-site name (e.g. "extract", "compute",
          "retrieve").
        - `message` is a STABLE event-key literal (snake_case, no
          interpolation). Every variable goes in `**fields`, never into the
          message string — that keeps events groupable/filterable.
        - Emit the fact at the layer that owns it, and only there: the
          orchestrator owns operator boundaries (the `("orchestrator", "step")`
          event), so operators do NOT emit their own "starting"/"done"; each
          operator emits only its own internal decisions. Use stdlib `logging`
          (not `emit`) from code that has no per-question ctx (build pipelines,
          offline prep).

        Severity defaults to "warning" for `*_failed` messages or any event
        carrying an `error` field, else "info"; pass `level=` to override. The
        event is always captured to `self.events` (per-question, consumed by the
        trace dump) and the JSONL sink; streamed to this question's `.log` file
        when one is open; and echoed to the console only when `verbose` (rendered
        via `trace.render_line`). The per-question file omits `uid` (implied by the
        filename); the shared console prepends it so interleaved lines stay
        attributable.
        """
        step_idx, op = self._current_step()
        if level is None:
            level = "warning" if (message.endswith("_failed") or "error" in fields) else "info"
        evt = {
            "source": source,
            "message": message,
            "step_idx": step_idx,
            "op": op,
            "level": level,
            **fields,
        }
        self.events.append(evt)
        trace.write_jsonl({"uid": self.uid, **evt})
        if self._logfile is not None:
            # close() only runs in the question's teardown, after all branch
            # threads have joined — no emit races it, so no re-check under the lock.
            line = trace.render_line(evt) + "\n"
            with self._log_lock:
                self._logfile.write(line)
                self._logfile.flush()
        if self.verbose:
            print(trace.render_line({"uid": self.uid, **evt}))
