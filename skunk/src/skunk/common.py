"""Shared runtime: the process-wide rate limiters plus the cross-cutting types
threaded between operators and the orchestrator (`PageRef`, `AnnotatedValue`,
`ExecutionContext`).

The LLM client itself lives in `skunk.llm_client`; this module owns the external
rate-limit infrastructure it (and the data-source tools) pace against — the
process-wide token-bucket limiters (see `_RATE_LIMITS` / `get_rate_limiter`)."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import os
import re
import threading
import time
from chromadb import Collection
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import tiktoken
from pydantic import BaseModel, ConfigDict, model_validator

from skunk import trace
from skunk.config import OrchestratorConfig
from skunk.constants import CHARS_PER_TOKEN_EST

if TYPE_CHECKING:
    from skunk.llm_client import LLMClient
    from skunk.prompted_call import PromptOverride
    from skunk.storage.document_map import DocumentMap

# Reasoning-effort knob, mapped onto Gemini's `thinking_level` enum. "off" means
# no thinking; "minimal" is the cheapest thinking tier.
Effort = Literal["off", "minimal", "low", "medium", "high"]
EFFORT_VALUES = ("off", "minimal", "low", "medium", "high")

@dataclass
class B64Image:
    """A single rendered page image ready to pass to an LLM: MIME type + base64 data."""

    mime: str
    data: str


def pdf_path_for(doc: str, pdf_dir: Path | str) -> Path:
    """The corpus PDF for a doc id: `<pdf_dir>/<doc>.pdf`. `doc` is the filename stem
    (e.g. "combined_statement__modern__2024__c40") — the same key the page index uses."""
    return Path(pdf_dir) / f"{doc}.pdf"


def render_page_b64(
    stem: str | None,
    page: int | None,
    *,
    pdf_dir: Path | str,
    renders_dir: Path | str | None = None,
    dpi: int = 300,
    fmt: str = "png",
    jpg_quality: int | None = None,
) -> B64Image | None:
    """The single PDF-page rasterizer for the repo: render one page to in-memory image bytes via
    PyMuPDF. `stem` is the doc-id stem naming the PDF (`<pdf_dir>/<stem>.pdf`). Returns None when
    the PDF doesn't exist; PyMuPDF errors propagate. `fitz` is imported lazily so importing
    `common` doesn't pull in PyMuPDF.

    When `renders_dir` is given, a pre-rendered PNG at `<renders_dir>/<stem>_<page>.png` (the page
    render cache) is served directly — skipping the PDF open. `dpi`/`fmt`/`jpg_quality` are
    ignored for a cache hit (the cached PNG's own resolution applies); on a miss we fall back to
    rasterizing the PDF."""
    if stem is None or page is None or int(page) <= 0:
        return None
    if renders_dir is not None:
        cached = Path(renders_dir) / f"{stem}_{int(page)}.png"
        if cached.exists():
            return B64Image(
                mime="image/png",
                data=base64.standard_b64encode(cached.read_bytes()).decode(),
            )
    pdf_path = pdf_path_for(stem, pdf_dir)
    if not pdf_path.exists():
        return None
    import fitz

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as doc:
        pix = doc[int(page) - 1].get_pixmap(matrix=mat)
    if fmt == "jpg":
        data = pix.tobytes(
            "jpg", jpg_quality=jpg_quality if jpg_quality is not None else 95
        )
        mime = "image/jpeg"
    else:
        data = pix.tobytes("png")
        mime = "image/png"
    return B64Image(mime=mime, data=base64.standard_b64encode(data).decode())


class _RateLimiter:
    """Process-wide token-bucket rate limiter with BOTH interfaces over ONE bucket.

    `acquire()` blocks the calling thread; `acquire_async()` yields to the event loop. Both
    draw from the same tokens, so a name's rate cap holds whether it's hit from sync code
    (offline build, the `requests`-based lookup tools, embeddings prep, the sync LLM path) or
    from async code (the async request-path LLM calls).

    Thread-safe: a `threading.Condition` guards the refill+deduct. The async path holds that
    lock only across the (non-awaiting) token math and sleeps via `asyncio.sleep` OUTSIDE it,
    so it never blocks its event loop. Sync waiters block on `cond.wait(timeout)` and re-check
    on timeout, so no notify is needed when the async path deducts (and vice versa)."""

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
        """Sync: block the calling thread until 1 slot is free, then deduct it."""
        with self._cond:
            while True:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._rate
                self._cond.wait(timeout=wait_s)

    async def acquire_async(self) -> None:
        """Async: yield to the loop until 1 slot is free, then deduct it. The lock is held
        only across the synchronous token math; the wait is `await asyncio.sleep` outside it."""
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait_s)


# Single source of truth for every external rate cap we pace against. Each row is
# `name: (env override, default rpm)`. The rpm is read ONCE, on first use of the
# name, and frozen for the rest of the run — buckets are created on demand and
# never rebuilt (rate limits are immutable for a run). To add a service: add a row
# and call `get_rate_limiter("<name>")` at the call site.
_RATE_LIMITS: dict[str, tuple[str, float]] = {
    # name          (env override,            default rpm)  # rationale
    "embed": (
        "SKUNK_EMBED_RPM",
        600.0,
    ),  # Gemini embeddings (search_agent.vector_search)
    "fred": ("SKUNK_FRED_RPM", 60.0),  # hard 120/min per API key; stay well under —
    # FRED escalates to an extended key-wide ban
    # (persistent 429s) once the cap is tripped.
    "bls": ("SKUNK_BLS_RPM", 50.0),  # 500/day (registered); smooth worker bursts
    "world_bank": (
        "SKUNK_WORLD_BANK_RPM",
        120.0,
    ),  # no published cap; stay a good citizen
    "tavily": ("SKUNK_TAVILY_RPM", 100.0),  # ~100/min on the dev tier
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
    """Process-wide token-bucket limiter for service `name`, paced at its rpm — read ONCE at
    first use, then frozen for the run. ONE bucket per name, serving BOTH sync `acquire()` and
    async `acquire_async()` callers (so the cap holds across sync+async use). Known services
    key off `_RATE_LIMITS` (env override); dynamic buckets (per-model `llm:<model>`) pass an
    explicit `rate_per_min`. All threads in this process share the named bucket.

    Scope is per process, NOT per API key: separate processes get independent
    buckets and do not coordinate, so a multi-process deployment would not be
    bounded by the underlying per-key cap. The eval harness is single-process."""
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(name)
        if lim is None:
            lim = _RateLimiter(rate_per_sec=_resolve_rate_per_sec(name, rate_per_min))
            _LIMITERS[name] = lim
        return lim


# ---------------------------------------------------------------------------
# Tokens-per-minute (TPM) throttle — async; per-model caps come from the
# InferenceConfig (`llm_model_tpm`, falling back to `llm_default_tpm`).
#
# The RPM limiter alone can't bound token throughput: one request can carry tens
# of thousands of tokens, so a request-paced run still blows a TPM quota (the
# full-text page-index filter pushed ~36M tok/min and 429-stormed). This bucket
# meters estimated *input* tokens per call. Mirrors `_RateLimiter.acquire_async`'s
# cross-loop safety (threading.Lock around refill+deduct, sleep outside the lock).
# Inert for any model whose effective TPM is falsy (None/0) — paced by RPM alone.
# Housed here so BOTH process-wide pacing registries (RPM above, TPM below) live
# in one module with one idiom.
# ---------------------------------------------------------------------------


class AsyncTokenBudget:
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
_ASYNC_TPM_LIMITERS: dict[str, AsyncTokenBudget] = {}


def get_async_tpm_limiter(model: str, tpm: float) -> AsyncTokenBudget:
    """Process-wide TPM bucket for `model`, paced at `tpm` tokens/min. Capacity is
    ~4s of budget so a few large concurrent requests can burst, then throttle."""
    with _ASYNC_TPM_LOCK:
        lim = _ASYNC_TPM_LIMITERS.get(model)
        if lim is None:
            rate_per_sec = tpm / 60.0
            lim = AsyncTokenBudget(rate_per_sec, capacity=max(rate_per_sec * 4.0, 256_000.0))
            _ASYNC_TPM_LIMITERS[model] = lim
        return lim


# ---------------------------------------------------------------------------
# Token estimation — the one place text is converted to a token count.
# ---------------------------------------------------------------------------

# Above this many characters (~1M tokens at ~4 chars/token), `estimate_tokens` falls back
# to the char-count heuristic: exact BPE encoding is fast and fits in modest memory (~8GB)
# up to roughly this size, but beyond it the encode cost outweighs the estimate's error.
_EXACT_TOKENIZE_MAX_CHARS = 4_000_000

# Lazily built on first `estimate_tokens` call so importing `common` doesn't pay the
# BPE-table load. Benign race: concurrent first calls both build; one assignment wins.
_token_encoder: tiktoken.Encoding | None = None


def estimate_tokens(s: str) -> int:
    """Input-token estimate for `s`: an exact BPE count (o200k_base) for strings up to
    `_EXACT_TOKENIZE_MAX_CHARS` chars, falling back to the ~4 chars/token heuristic
    (`CHARS_PER_TOKEN_EST`) beyond that. Still an estimate, not a billing figure — the
    provider's tokenizer may differ from o200k_base — but far tighter than chars/4."""
    if len(s) > _EXACT_TOKENIZE_MAX_CHARS:
        return len(s) // CHARS_PER_TOKEN_EST
    global _token_encoder
    if _token_encoder is None:
        _token_encoder = tiktoken.get_encoding("o200k_base")
    return len(_token_encoder.encode(s))


# Active (step_idx, op) frame for the current operator call. A ContextVar, not
# thread-local, because within a question the branches run as concurrent asyncio
# tasks on that question's event loop (one loop per worker thread): a
# `threading.local` would let those sibling tasks collide on the shared thread,
# whereas a ContextVar gives each `asyncio.Task` a copy-on-write snapshot, so a
# task that rebinds the frame is isolated from its siblings. Operator steps don't
# nest (the orchestrator opens exactly one per traced call), so this holds a
# single frame, not a stack.
_step_frame: contextvars.ContextVar[tuple[int, str, int | None] | None] = (
    contextvars.ContextVar("skunk_step_frame", default=None)
)


# Map a message's leading snake_case event key to a semantic `kind` (the role used
# by the trace viewer for color-coding). Call sites that pass `kind=` explicitly win;
# this only classifies the legacy one-liner emits that don't. Unknown keys → "note"
# (a neutral, informational event).
_KIND_BY_PREFIX: dict[str, str] = {
    "question": "user",
    "observation": "observation",
    "error": "error",
    "call": "call",
    "step": "step",
    "validation_failed": "error",
    # Harness lifecycle events — messages the loop injected into the agent's context
    # (low-steps warning) or actions it took after exhausting the step budget (the
    # forced terminal turn). Grouped under one `lifecycle` kind so the viewer can
    # color them distinctly and gate them behind a single toggle.
    "steps_low_warning": "lifecycle",
    "terminal_prompt": "lifecycle",
    "terminal_reply": "lifecycle",
    "terminal_commit": "lifecycle",
    "terminal_giveup": "lifecycle",
    "terminal_turn_failed": "lifecycle",
    "parallel_branch_failed": "error",
    "compute_needs_more": "note",
    "compute_partial_malformed": "note",
    "pool_update": "note",
    "compute_short_circuit": "note",
    "recovery_exhausted": "error",
}


def infer_kind(message: str) -> str:
    """Classify a one-liner `emit` message into a semantic `kind` from its leading
    event key (see `_KIND_BY_PREFIX`). Used when a call site doesn't pass `kind=`."""
    token = message.split(" ", 1)[0] if message else ""
    return _KIND_BY_PREFIX.get(token, "note")


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


# --- Cross-cutting runtime types threaded between operators and the orchestrator ---


@dataclass(frozen=True)
class PageRef:
    """Canonical page coordinate. Frozen so it's hashable — usable as a dict key
    and set member (e.g. the page-index catalog is keyed by `PageRef`)."""

    stem: str | None = None  # doc-id stem, e.g. "combined_statement__historical__cs-1872" (search-agent / page-index key)
    page: int | None = None  # 1-based PDF page index (canonical)

    @property
    def year(self) -> int | None:
        """The calendar year for a `YYYY-MM` slot, or None. Returns None when the slot is not a
        bare month (the rekeyed page index stores a doc-id stem here — its year is parsed from
        the stem via `corpus.parse_doc_id`, not from this coordinate)."""
        if not self.stem or not self.stem[:4].isdigit():
            return None
        return int(self.stem[:4])

    def __post_init__(self) -> None:
        if self.page is not None and self.stem is None:
            raise ValueError(
                f"PageRef with page={self.page} requires stem for parsed-JSON lookup"
            )

    def __repr__(self) -> str:
        parts = []
        if self.year:
            parts.append(f"year={self.year}")
        if self.stem:
            parts.append(f"stem={self.stem}")
        if self.page is not None:
            parts.append(f"page={self.page}")
        return f"PageRef({', '.join(parts)})"


def page_key_to_pageref(key: str) -> PageRef:
    """Parse a search-agent page key `"<stem>_<page>"` into a `PageRef`. The doc-id stem (which
    itself contains underscores, e.g. `"combined_statement__historical__cs-1872"`) goes in the
    `stem` slot; the trailing integer is the 1-based page index."""
    try:
        stem, page_str = key.rsplit("_", 1)
        return PageRef(stem=stem, page=int(page_str))
    except ValueError as e:
        raise ValueError(f"page key {key!r} not in '<stem>_<page>' form") from e


@dataclass(frozen=True)
class RetrievedDoc:
    """One retrieved page's content: a `PageRef` plus the page's text. Produced by the
    retrieve operator (search-agent backend) and read directly by compute — there is no
    separate extract step. The text is the cleaned per-page corpus text the search agent
    itself reads (`retrieve`'s `document_map`)."""

    ref: PageRef
    text: str


@dataclass(frozen=True)
class BranchRetrieval:
    """One retrieve branch's normalized result — the single retrieval contract the
    search-agent backend produces: the retrieved pages with their text, read directly by
    compute. Produced solely by `retrieve.run_retrieve_all`."""

    documents: tuple[RetrievedDoc, ...]

    @property
    def pages(self) -> tuple[PageRef, ...]:
        """The retrieved pages' refs (for recall reporting / the retrieved-page union)."""
        return tuple(d.ref for d in self.documents)


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

    `notes` is LLM-authored at extract time: prose capturing the page's textual
    context that bears on the question — footnotes, headnotes, comments, scope
    caveats, break-in-series notes, and the meaning of any print flags (p/r) —
    summarized as it pertains to the question, preferring the page's original
    wording. It is one note for the whole payload (a vector/table shares it), so
    it is context, not a per-cell discriminator; per-datum disambiguation lives
    in `description`. Empty for external lookups.

    `source` is the LLM-authored publisher/origin of an external lookup's value
    (e.g. the data provider the lookup agent pulled it from) — the external-lookup
    analog of the machine-stamped corpus provenance below, which it cannot fill.
    Empty for corpus extracts (whose provenance is `source_stem`/`pages`).

    Provenance (`source_stem`/`pages`/`requested_period`/`retrieve_key`/`obtained_visually`)
    is machine-stamped from the extract inputs — the source page refs and the
    retrieve branch — NOT authored by the LLM. `obtained_visually` records the
    machine fact that this value was read by the vision tier (the figure/chart
    fallback); it is internal provenance and never shown to any LLM (it is
    excluded from every prompt rendering). It is absent (None/empty) for
    external lookups. `source_stem` is the source document the value was read
    from, carried as that document's parsed-JSON/PDF filename stem (e.g.
    "combined_statement__historical__cs-1872"; the same stem `PageRef.stem`
    holds — deliberately NOT called doc_id, which in the search-agent layer
    names the retrieval unit, often a single page). It identifies the source
    document, not a date; the data window the value covers lives in
    `requested_period` (and the page's own data span), so any chronological
    ordering keys off period, not `source_stem`.
    """

    model_config = ConfigDict(frozen=True)

    description: str
    value: Any
    unit: str = ""
    notes: str = ""
    source: str = ""  # external-lookup publisher (LLM-authored); empty for corpus extracts
    kind: Literal["scalar", "vector", "table"] = "scalar"
    index_name: str | None = None
    row_name: str | None = None
    col_name: str | None = None

    # Provenance — copied from the source page/branch at extract time, never
    # LLM-written. Defaults keep external lookups and old payloads valid.
    source_stem: str | None = None  # source document's filename stem; None for external lookups
    pages: tuple[int, ...] = ()  # source PDF page(s); () when unattributable
    requested_period: str | None = None  # branch.period — data window requested
    retrieve_key: str | None = None  # branch.key — concept this datum serves
    obtained_visually: bool = False  # True when read via the vision tier (figure/chart fallback)

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
            if not (
                isinstance(v, dict)
                and all(isinstance(k, str) and is_prim(c) for k, c in v.items())
            ):
                raise ValueError("vector value must be flat dict[str, scalar]")
            if not self.index_name:
                raise ValueError("vector entry missing non-empty 'index_name'")
        else:  # table
            if not (
                isinstance(v, dict)
                and all(
                    isinstance(r, str)
                    and isinstance(row, dict)
                    and all(isinstance(k, str) and is_prim(c) for k, c in row.items())
                    for r, row in v.items()
                )
            ):
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


@dataclass(frozen=True)
class Final:
    """Compute produced the answer."""

    answer: str


@dataclass(frozen=True)
class NeedsMore:
    """Compute judged its inputs insufficient. `missing_reason`/`missing` describe what to
    gather next; the orchestrator keeps everything gathered so far and replans to add more."""

    missing_reason: str
    missing: list[str]


_PAD = "         "  # 9-space continuation indent for an entry's detail lines


def _provenance_str(e: AnnotatedValue) -> str:
    """One-line provenance for the schema view — only the fields that are set, so
    external lookups (no source_stem) stay uncluttered. Empty string when nothing is set."""
    parts: list[str] = []
    if e.source_stem:
        parts.append(f"source_stem={e.source_stem!r}")
    if e.pages:
        parts.append(f"pages={list(e.pages)!r}")
    if e.source:
        parts.append(f"source={e.source!r}")
    if e.requested_period:
        parts.append(f"requested_period={e.requested_period!r}")
    if e.retrieve_key:
        parts.append(f"retrieve_key={e.retrieve_key!r}")
    return " ".join(parts)


# Above this many axis labels, `_join_labels` elides the middle: a pathological frame
# (e.g. a multi-decade monthly series, 1000s of rows) would otherwise render thousands of
# labels into one untrimmed prompt line and push the codegen / replan request past the
# model's input-token ceiling. Generated code can still read the full `.frame.index` at
# runtime, so eliding the middle here loses nothing for selection logic.
_MAX_RENDERED_LABELS = 200
_LABEL_EDGE = 100


def _join_labels(labels: list[str]) -> str:
    """Comma-join axis labels, eliding the middle when there are more than
    `_MAX_RENDERED_LABELS` so one frame can't blow up the prompt size."""
    if len(labels) <= _MAX_RENDERED_LABELS:
        return ", ".join(labels)
    elided = len(labels) - 2 * _LABEL_EDGE
    head = ", ".join(labels[:_LABEL_EDGE])
    tail = ", ".join(labels[-_LABEL_EDGE:])
    return f"{head}, … ({elided} of {len(labels)} labels elided) …, {tail}"


def _describe_entry(i: int, e: AnnotatedValue) -> list[str]:
    """Full view of one `AnnotatedValue`: the meta line (kind/shape/dtypes), then — for
    non-scalars — the entire frame rendered cell-by-cell via `df.to_string()`. Values are
    included (not just the schema) so generated code can see NaN / "n/a" cells, sanity-check
    magnitudes, and key `.loc[...]` on the real labels. Scalars show in full. A
    `provenance:` line carries the machine-stamped source fields when present."""
    head = (
        f"  input_values[{i}]  description: {(e.description or '(no description)')!r}"
    )
    prov = _provenance_str(e)
    prov_lines = [f"{_PAD}provenance: {prov}"] if prov else []
    if e.notes:
        prov_lines.insert(0, f"{_PAD}notes: {e.notes!r}")
    if e.kind == "scalar":
        return [
            head,
            f"{_PAD}value={e.value!r}  (kind=scalar, unit={e.unit!r})",
            *prov_lines,
        ]

    df = e.frame
    n_rows, n_cols = df.shape
    if e.kind == "vector":
        meta = f"kind=vector, index_name={e.index_name!r}, unit={e.unit!r}, shape=({n_rows}, {n_cols})"
    else:
        meta = (
            f"kind=table, row_name={e.row_name!r}, col_name={e.col_name!r}, "
            f"unit={e.unit!r}, shape=({n_rows}, {n_cols})"
        )
    lines = [head, f"{_PAD}{meta}", *prov_lines]
    if n_rows == 0:
        lines.append(f"{_PAD}frame: (empty)")
        return lines

    if e.kind == "vector":
        lines.append(f"{_PAD}dtype: {df.dtypes.iloc[0]}")
    else:
        cols = _join_labels([str(c) for c in df.columns])
        dtypes = ", ".join(str(t) for t in df.dtypes)
        lines.append(f"{_PAD}columns: [{cols}]   dtypes: [{dtypes}]")

    # The whole frame, values included — the agent reads the actual cells (NaN/'n/a',
    # magnitudes, exact labels), not just the schema. The full frame also lives in the
    # exec env for the generated code to operate on. `float_format` forces FIXED-POINT,
    # full-precision rendering: pandas' default switches a column to scientific notation
    # (e.g. 452197.98 -> 4.521980e+05) once it holds a large value, hiding the cents and
    # leading the agent to transcribe rounded numbers. `format_float_positional` prints the
    # exact value without scientific notation (trim='-' drops only trailing zeros/point).
    # Rows are capped like `_join_labels` caps columns: a pathological frame (a multi-decade
    # monthly series, 1000s of rows) would otherwise render every row into one prompt and
    # blow past the input-token ceiling. Generated code still reads the full `.frame`.
    lines.append(f"{_PAD}frame:")
    if n_rows > _MAX_RENDERED_LABELS:
        head = df.iloc[:_LABEL_EDGE].to_string(
            float_format=lambda v: np.format_float_positional(v, trim="-")
        )
        tail = df.iloc[-_LABEL_EDGE:].to_string(
            float_format=lambda v: np.format_float_positional(v, trim="-"), header=False
        )
        elided = n_rows - 2 * _LABEL_EDGE
        rendered = f"{head}\n… ({elided} of {n_rows} labels elided) …\n{tail}"
    else:
        rendered = df.to_string(
            float_format=lambda v: np.format_float_positional(v, trim="-")
        )
    lines.extend(f"{_PAD}  {ln}" for ln in rendered.splitlines())
    return lines


def input_values_desc(input_values: list[AnnotatedValue]) -> str:
    """Render `input_values` for the codegen / re-planner prompts as a **full view** —
    meta (axis labels, dtypes) plus the entire frame rendered cell-by-cell, so the agent
    reads the actual data (NaN/'n/a' cells, magnitudes, exact labels), not just the schema.
    The full frames also live in the exec env for the generated code to operate on."""
    lines = [f"input_values ({len(input_values)} entries)"]
    for i, e in enumerate(input_values):
        lines.extend(_describe_entry(i, e))
    return "\n".join(lines)


def documents_desc(documents: list["RetrievedDoc"]) -> str:
    """Render retrieved pages as a plain-text context block for the compute / replan
    prompts: one delimited section per page, keyed by its page id. Read directly by the
    LLM — the model transcribes any numbers it needs into code (the pages are NOT in the
    exec environment)."""
    if not documents:
        return "(no pages retrieved)"
    blocks = [f"retrieved pages ({len(documents)})"]
    for d in documents:
        page_id = f"{d.ref.stem}_{d.ref.page}" if d.ref.stem else str(d.ref)
        blocks.append(f"=== page {page_id} ===\n{d.text}")
    return "\n\n".join(blocks)


def split_pool(
    pool: list,
) -> tuple[list["RetrievedDoc"], list[AnnotatedValue]]:
    """Split the orchestrator's mixed gathered pool into retrieved pages (from `retrieve`
    branches) and structured values (from `lookup_external` branches)."""
    docs = [x for x in pool if isinstance(x, RetrievedDoc)]
    values = [x for x in pool if isinstance(x, AnnotatedValue)]
    return docs, values


def pool_desc(pool: list) -> str:
    """Render the mixed gathered pool (retrieved pages + lookup values) for the replan
    prompt."""
    docs, values = split_pool(pool)
    parts = [documents_desc(docs)]
    if values:
        parts.append(input_values_desc(values))
    return "\n\n".join(parts)


@dataclass
class ExecutionContext:
    """Per-question execution state, owned by exactly one `Orchestrator` and never
    shared across questions or threads.
    """

    question: str
    config: OrchestratorConfig
    document_map: DocumentMap
    uid: str | None = (
        None  # benchmark UID, when run from the eval harness; tags every event
    )
    verbose: bool = False  # also echo orchestrator + operator events to the console
    log_path: str | None = (
        None  # when set, stream this question's events to that file as JSON lines (live, flushed)
    )
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events
    llm_client: LLMClient = None  # type: ignore[assignment]  # set in __post_init__; pass a mock to override
    prompt_overrides: tuple[
        PromptOverride, ...
    ] = ()  # corpus/few_shot/lesson overrides; operators pick out their own entries by name
    chroma_collection: Collection | None = None

    def __post_init__(self) -> None:
        if self.llm_client is None:
            # Local import: `skunk.llm_client` imports from this module, so a
            # top-level import would be circular.
            from skunk.llm_client import LLMClient

            self.llm_client = LLMClient(self.config.inference)
        # The active (step_idx, op) frame lives in the module-level `_step_frame`
        # ContextVar (per asyncio task), not on the instance — see its definition.
        # Per-question event log: opened when log_path is set, written as one JSON
        # line per event and flushed live so a single question's stream lands in its
        # own file (and survives a crash). No lock: this ctx is touched by one thread
        # only (see the class docstring), so the file write needs no synchronization.
        self._logfile = None
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._logfile = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115
        # Monotonic step-index allocator. Owned here (not by the caller) so the
        # ctx is the single source of step identities. Incremented synchronously
        # inside `step()` before any `await`, so the single-thread/cooperative
        # contract (see class docstring) makes it race-free without a lock.
        self._next_step_idx = 1
        # Query start, for the per-event `t` offset (seconds since this question
        # began). Stamped at ctx construction — the orchestrator builds the ctx
        # right before `execute()`, so this is ~"t=0 at the start of the query".
        self._t0 = time.monotonic()

    def close(self) -> None:
        """Close the per-question log file, if one was opened."""
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None

    @contextmanager
    def step(self, op: str, branch_id: int | None = None):
        """Open an observability step (a span) for one operator call, yielding its
        allocated `step_idx`. Every `emit` inside the block is auto-stamped with
        `(step_idx, op, branch_id)` via the `_step_frame` ContextVar, so callees emit
        without knowing their step — and per-branch steps (retrieve/extract/lookup) stamp
        their `branch_id` onto every event, so the trace viewer can group a rollout under
        its branch *while it streams*, not only once the boundary `step` event lands. The
        ctx allocates the index; the caller does not pass one in. Steps don't nest, so this
        rebinds a single frame (set/reset) rather than pushing a stack, and the
        contextmanager guarantees the frame is reset on every exit path. Per-task isolation
        comes from the ContextVar — see `_step_frame`."""
        step_idx = self._next_step_idx
        self._next_step_idx += 1
        token = _step_frame.set((step_idx, op, branch_id))
        try:
            yield step_idx
        finally:
            _step_frame.reset(token)

    def _current_step(self) -> tuple[int | None, str | None, int | None]:
        return _step_frame.get() or (None, None, None)

    def emit(
        self,
        message: str,
        level: str | None = None,
        *,
        kind: str | None = None,
        data: dict | None = None,
    ) -> None:
        """Record a request-scoped diagnostic event onto this question's event stream.

        Convention (see ARCHITECTURE.md "Logging & observability"):
        - There is no separate `source`/`op` argument: the active step's `op`
          (from `_step_frame`, stamped here) identifies who emitted. Out-of-step
          emits carry `op=None` and rely on the message alone.
        - `message` is a single human-readable string. Lead it with a stable
          snake_case event key, then interpolate any variables inline
          (`f"extracted tier=parsed_json n_entries={n}"`). Keep large blobs out
          of the *message* (it must stay a scannable one-liner) — but they may go in
          `data` (see below).
        - `kind` is the event's semantic role for the trace viewer's color-coding
          (`system` / `user` / `assistant` / `observation` / `error` / `call` /
          `plan` / `summary` / `step` / `note` / `lifecycle`). When omitted it is inferred from the
          message's leading event key (`infer_kind`), so legacy one-liners need no
          change.
        - `data` is an optional structured payload (the full system prompt, an
          assistant turn, structured observation blocks, a plan, a node summary).
          It is captured to `self.events` and the per-question `.jsonl` file for
          the viewer, but is **deliberately excluded** from the rendered console
          line (which stays the scannable one-liner). Pass JSON-friendly values;
          the file write falls back to `str` for anything else.
        - Emit the fact at the layer that owns it, and only there: the
          orchestrator owns operator boundaries (the `"step"` event), so operators
          do NOT emit their own "starting"/"done"; each operator emits only its own
          internal decisions. Use stdlib `logging` (not `emit`) from code that has
          no per-question ctx (build pipelines, offline prep).

        Severity defaults to "warning" when the message contains a `_failed` event
        key or an `error=` field, else "info"; pass `level=` to override. The event
        is always captured to `self.events` (per-question, consumed by the trace
        dump); streamed as a JSON line to this question's `.jsonl` file when one
        is open (the durable machine-readable record); and echoed to the console
        only when `verbose` (rendered via `trace.render_line`). The per-question
        file omits `uid` (implied by the filename); the shared console prepends it
        so interleaved lines stay attributable.
        """
        step_idx, op, branch_id = self._current_step()
        if level is None:
            level = (
                "warning" if ("_failed" in message or "error=" in message) else "info"
            )
        evt = {
            "message": message,
            "step_idx": step_idx,
            "op": op,
            "level": level,
            "kind": kind if kind is not None else infer_kind(message),
            # Seconds since the question began — the trace viewer's timeline axis
            # (start/end offsets, latencies). Skipped on the rendered line (the
            # console already carries a wall-clock HH:MM:SS).
            "t": round(time.monotonic() - self._t0, 3),
        }
        # Per-branch steps stamp branch_id on every event so the viewer can group a live
        # rollout under its branch before the boundary `step` event arrives.
        if branch_id is not None:
            evt["branch_id"] = branch_id
        if data is not None:
            evt["data"] = data
        self.events.append(evt)
        if self._logfile is not None:
            self._logfile.write(json.dumps(evt, default=str, ensure_ascii=False) + "\n")
            self._logfile.flush()
        if self.verbose:
            print(trace.render_line({"uid": self.uid, **evt}))


async def traced_step[T](
    ctx: ExecutionContext,
    op_name: str,
    fn: Callable[[], Awaitable[T]],
    *,
    branch_id: int | None = None,
    summary_metadata: dict[str, Any] | None = None,
) -> T:
    """Run `fn` inside a `ctx.step` frame and emit a boundary event with elapsed time."""
    from skunk.errors import MissingData, StepFailed
    from skunk.result import describe_value, summarize_value

    t0 = time.perf_counter()
    with ctx.step(op_name, branch_id=branch_id):
        try:
            result = await fn()
        except (StepFailed, MissingData) as e:
            err = f"MissingData: {e.reason}" if isinstance(e, MissingData) else str(e)
            elapsed = round(time.perf_counter() - t0, 3)
            ctx.emit(
                f"step elapsed_s={elapsed} output=(failed) error={err!r}",
                kind="step",
                data={"branch_id": branch_id, "elapsed_s": elapsed, "error": err},
            )
            raise
        elapsed = round(time.perf_counter() - t0, 3)
        summary = summarize_value(result)
        if summary_metadata:
            summary.update(summary_metadata)
        ctx.emit(
            f"step elapsed_s={elapsed} output={describe_value(result)!r}",
            kind="step",
            data={
                "branch_id": branch_id,
                "elapsed_s": elapsed,
                "summary": summary,
            },
        )
    return result
