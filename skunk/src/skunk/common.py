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

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from google import genai

from skunk import trace
from skunk.config import SkunkConfig

if TYPE_CHECKING:
    from skunk.llm_client import LLMClient
    from skunk.prompted_call import PromptOverride
    from skunk.page_index.data_model import ContentBlock

# Reasoning-effort knob, mapped onto Gemini's `thinking_level` enum. "off" means
# no thinking; "minimal" is the cheapest thinking tier.
Effort = Literal["off", "minimal", "low", "medium", "high"]
EFFORT_VALUES = ("off", "minimal", "low", "medium", "high")

HumanInterventionHandler = Callable[
    [str, str, str | None, list[str], dict[str, Any] | None],
    Awaitable[dict[str, Any]],
]


@dataclass(frozen=True)
class PendingHumanIntervention:
    response: Awaitable[dict[str, Any]]


@dataclass
class B64Image:
    """A single rendered page image ready to pass to an LLM: MIME type + base64 data."""

    mime: str
    data: str


def pdf_path_for(bulletin: str, pdf_dir: Path | str) -> Path:
    """'1953-06' -> <pdf_dir>/treasury_bulletin_1953_06.pdf. Inverse of the bulletin-id parse."""
    year, mon = bulletin.split("-")
    return Path(pdf_dir) / f"treasury_bulletin_{int(year):04d}_{int(mon):02d}.pdf"


def render_page_b64(
    month: str | None,
    page: int | None,
    *,
    pdf_dir: Path | str,
    dpi: int = 300,
    fmt: str = "png",
    jpg_quality: int | None = None,
) -> B64Image | None:
    """The single PDF-page rasterizer for the repo: render one page to in-memory image bytes via
    PyMuPDF. Returns None when the PDF doesn't exist; PyMuPDF errors propagate. No disk cache (the
    page store layers its own cache on top). `fitz` is imported lazily so importing `common`
    doesn't pull in PyMuPDF."""
    if month is None or page is None or int(page) <= 0:
        return None
    pdf_path = pdf_path_for(month, pdf_dir)
    if not pdf_path.exists():
        return None
    import fitz

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    with fitz.open(pdf_path) as doc:
        pix = doc[int(page) - 1].get_pixmap(matrix=mat)
    if fmt == "jpg":
        data = pix.tobytes("jpg", jpg_quality=jpg_quality if jpg_quality is not None else 95)
        mime = "image/jpeg"
    else:
        data = pix.tobytes("png")
        mime = "image/png"
    return B64Image(mime=mime, data=base64.standard_b64encode(data).decode())


# Process-scoped logger for code with no per-question ctx in scope (build
# pipelines, offline prep); request-path code uses `ctx.emit` instead.
log = logging.getLogger(__name__)


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
    "llm": ("SKUNK_LLM_RPM", 1000.0),  # Gemini generation; provider-side quota
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
    "terminal_giveup": "error",
    "terminal_turn_failed": "error",
    "terminal_commit": "note",
    "steps_low_warning": "note",
    "parallel_branch_failed": "error",
    "replan_dropped": "note",
    "codegen_missing": "note",
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


def chunk(seq: list, n: int) -> list[list]:
    """Split `seq` into consecutive sub-lists of at most `n` items."""
    return [seq[i : i + n] for i in range(0, len(seq), n)]


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None


def make_genai_client() -> genai.Client:
    """Build a direct-Gemini (AI Studio) genai.Client from `GEMINI_API_KEY`.
    Auth via api-key; no GCP project required.

    No client-side request timeout is set: HttpOptions.timeout doubles as a
    SERVER deadline (the API kills the request with 504 DEADLINE_EXCEEDED at
    the cutoff), which turned long-thinking calls — e.g. Gemini 3.x Pro vision
    reads that deliberate for minutes — into deterministic retry-storm failures.
    Slow calls are given however long the transport allows; transport-level
    connection faults still surface and are retried (`llm_client._is_retryable`)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (required for Gemini API)")
    return genai.Client(api_key=api_key)


# --- Cross-cutting runtime types threaded between operators and the orchestrator ---


@dataclass(frozen=True)
class PageRef:
    """Canonical page coordinate. Frozen so it's hashable — usable as a dict key
    and set member (e.g. the page-index catalog is keyed by `PageRef`)."""

    month: str | None = None  # "YYYY-MM" (a.k.a. bulletin in the page index)
    page: int | None = None  # 1-based PDF page index (canonical)

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


@dataclass(frozen=True)
class BlockRef:
    """One retrieved CONTENT BLOCK — the retriever's native output unit. `page` is the anchor
    page, `block_index` its position in `content_blocks` (None for a page kept wholesale).
    `member_refs` are the physical pages this block spans (anchor + any table-merge extra pages).
    `block` is excluded from identity so `BlockRef`s de-dupe on `page`/`block_index`/`member_refs`."""

    page: PageRef
    block_index: int | None
    member_refs: tuple[PageRef, ...]
    block: ContentBlock | None = field(default=None, compare=False)


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

    Provenance (`bulletin`/`pages`/`as_of`/`requested_period`/`retrieve_key`) is
    machine-stamped from the extract inputs — the source page refs and the
    retrieve branch — NOT authored by the LLM. It is absent (None/empty) for
    external lookups and for older payloads. `bulletin` is the issue the value
    was printed in ("YYYY-MM", lexically sortable = chronological); downstream
    compute uses it to sort/filter by publication date — e.g. to pick the
    latest non-revised vintage across several bulletins, where the LLM-written
    `description` of the same series+period can be identical across issues.
    """

    model_config = ConfigDict(frozen=True)

    description: str
    value: Any
    unit: str = ""
    kind: Literal["scalar", "vector", "table"] = "scalar"
    index_name: str | None = None
    row_name: str | None = None
    col_name: str | None = None

    # Provenance — copied from the source page/branch at extract time, never
    # LLM-written. Defaults keep external lookups and old payloads valid.
    bulletin: str | None = None  # source issue "YYYY-MM" (publication date)
    pages: tuple[int, ...] = ()  # source PDF page(s); () when unattributable
    as_of: str | list[str | None] | None = (
        None  # branch.as_of — pinned vintage requested (a list = per-period-entry pins)
    )
    requested_period: str | None = None  # branch.period — data window requested
    retrieve_key: str | None = None  # branch.key — concept this datum serves
    source_block_page: int | None = None
    source_block_index: int | None = None

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


_PAD = "         "  # 9-space continuation indent for an entry's detail lines


def _provenance_str(e: AnnotatedValue) -> str:
    """One-line provenance for the schema view — only the fields that are set, so
    external lookups (no bulletin) stay uncluttered. Empty string when nothing is set."""
    parts: list[str] = []
    if e.bulletin:
        parts.append(f"bulletin={e.bulletin!r}")
    if e.pages:
        parts.append(f"pages={list(e.pages)!r}")
    if e.as_of:
        parts.append(f"as_of={e.as_of!r}")
    if e.requested_period:
        parts.append(f"requested_period={e.requested_period!r}")
    if e.retrieve_key:
        parts.append(f"retrieve_key={e.retrieve_key!r}")
    return " ".join(parts)


def _describe_entry(i: int, e: AnnotatedValue) -> list[str]:
    """Schema view of one `AnnotatedValue`: the meta line, then (for non-scalars) the
    full index labels — load-bearing, they're what generated code keys `.loc[...]` on,
    and aren't otherwise in the prompt. Scalars show in full. A `provenance:` line
    carries the machine-stamped source fields when present."""
    head = (
        f"  input_values[{i}]  description: {(e.description or '(no description)')!r}"
    )
    prov = _provenance_str(e)
    prov_lines = [f"{_PAD}provenance: {prov}"] if prov else []
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
    return lines


def input_values_desc(input_values: list[AnnotatedValue]) -> str:
    """Render `input_values` for the codegen / re-planner prompts as a
    **schema view** — axis labels + dtypes, NOT a cell dump. Generated code operates
    on the frames symbolically (full frames live in the exec env), so it needs the
    labels (to write selections), not the interior grid."""
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
    uid: str | None = (
        None  # benchmark UID, when run from the eval harness; tags every event
    )
    verbose: bool = False  # also echo orchestrator + operator events to the console
    log_path: str | None = (
        None  # when set, stream this question's events to that file (live, flushed)
    )
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events
    config: SkunkConfig = field(default_factory=SkunkConfig.from_env)
    llm_client: LLMClient | None = (
        None  # auto-created in __post_init__; pass a mock to override
    )
    prompt_overrides: tuple[
        PromptOverride, ...
    ] = ()  # corpus/few_shot/lesson overrides; operators pick out their own entries by name
    human_intervention_handler: HumanInterventionHandler | None = None
    human_intervention_enabled: bool = False

    def __post_init__(self) -> None:
        if self.llm_client is None:
            # Local import: `skunk.llm_client` imports from this module, so a
            # top-level import would be circular.
            from skunk.llm_client import LLMClient

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
          `plan` / `summary` / `step` / `note`). When omitted it is inferred from the
          message's leading event key (`infer_kind`), so legacy one-liners need no
          change.
        - `data` is an optional structured payload (the full system prompt, an
          assistant turn, structured observation blocks, a plan, a node summary).
          It is captured to `self.events` and the durable JSONL sink for the viewer,
          but is **deliberately excluded** from the rendered console / `.log` line
          (which stays the scannable one-liner). Pass JSON-friendly values; the JSONL
          sink falls back to `str` for anything else.
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
        if data is not None:
            evt["data"] = data
        self.events.append(evt)
        trace.write_jsonl({"uid": self.uid, **evt})
        if self._logfile is not None:
            self._logfile.write(trace.render_line(evt) + "\n")
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
    with ctx.step(op_name):
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
