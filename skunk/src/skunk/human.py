"""Human-in-the-loop assistance — a modular middleware layer that routes specific
sub-tasks to a person instead of (or after) the model.

Design: the operators (`ExtractOp`, `LookupExternalOp`) are untouched. The orchestrator
calls this layer at its dispatch seam; everything here is gated by config flags
(`human_figure`, `human_verify_extract`, `human_lookup`) that default OFF, so with no flag
set this is inert. Three pieces, each a swappable seam:

  - `HumanAssistPolicy` decides *whether* a human is asked for a given operator call. Today
    it's all-or-nothing per flag; future nuance (verify only list-valued extractions, etc.)
    lives entirely here — no operator/orchestrator change.
  - `HumanChannel` is the I/O *how*. `ConsoleChannel` rasterizes the source page(s), prints
    the model's candidate(s), and reads a typed reply on stdin (blocking, dev/local).
    `BrokerChannel` instead delegates to the competition harness's async
    `HumanInterventionHandler`, so a worker resolves the request in the web UI and the
    branch's coroutine suspends without blocking the event loop. The orchestrator picks the
    channel by handler presence (broker under the server, console for a local CLI run).
  - `HumanAssist` is the facade the orchestrator holds: it joins policy + channel and
    returns `list[AnnotatedValue]` — the same currency every operator speaks.

On the `ConsoleChannel` a run blocks on the console one branch at a time (a module-level
`asyncio.Lock` serializes prompts across parallel branches), so target a few UIDs, not a
sweep. The `BrokerChannel` has no such limit — many interventions can be outstanding at once.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from dataclasses import dataclass, field
from typing import Literal, Protocol

from skunk.common import (
    AnnotatedValue,
    B64Image,
    ExecutionContext,
    HumanInterventionHandler,
    PageRef,
)
from skunk.errors import ParseError
from skunk.extract import _render_pages_b64, _stamp_provenance
from skunk.plan import Branch, LookupBranch, RetrieveBranch

# The JSON shape a human types to override the model — the same `AnnotatedValue` field set
# `LookupAgent.final_answer_doc` documents, so the human and the model speak one format.
_FIELDS = ("description", "value", "unit", "kind", "index_name", "row_name", "col_name")

# Sentinel "branch id" carried in the data-prep pool review's guidance. The optimistic
# register→resolve→recompute path keys overrides by branch id; the pool review isn't a single
# branch (data-prep coalesces across all of them), so it owns this reserved id. `recompute_answer`
# applies the human's edits onto the snapshot stored under it (the whole cleaned pool).
POOL_REVIEW_BRANCH_ID = -1


# The fields a human edits in the review overlay (the rest — kind/index_name/row_name/col_name
# and machine provenance — are preserved from the original extraction via `_src`).
_EDITABLE_FIELDS = ("description", "unit", "value")


def _candidate_dicts(entries: list[AnnotatedValue]) -> list[dict]:
    """The model's values for the review UI: the editable field set PLUS read-only `notes` (the
    LLM's extract-time page context), `source`/`retrieve_key` (an external lookup's publisher +
    target), and an `external` flag. A value with no corpus provenance (no `bulletin`/`pages`) is
    an external lookup, not a corpus extract — flagged so the UI can label it (priority) and show
    its target/src instead of a (nonexistent) source page. Display-only: the human edits only
    `_EDITABLE_FIELDS`; everything else rides back untouched via `_src`."""
    out: list[dict] = []
    for c in entries:
        d = c.model_dump(include=set(_FIELDS) | {"notes", "source", "retrieve_key"})
        d["external"] = not (c.bulletin or c.pages)
        out.append(d)
    return out


def apply_overrides(
    items: list[dict], base: list[AnnotatedValue]
) -> list[AnnotatedValue]:
    """Build a branch's revised entries from the human's edited review items, source-indexed so
    deletes/reorders are honored and provenance is preserved. Each item is
    `{"_src": <original index | None>, "description"?, "unit"?, "value"?}`:
      - `_src` in range → overlay the present editable fields onto `base[_src]` (keeps that
        entry's kind/index_name/row_name/col_name + machine provenance);
      - otherwise → construct a fresh `AnnotatedValue` from the given fields (a value the human
        added, rare).
    A `base` entry whose index never appears was deleted by the human and is dropped."""
    out: list[AnnotatedValue] = []
    for item in items:
        src = item.get("_src")
        fields = {f: item[f] for f in _EDITABLE_FIELDS if f in item}
        if isinstance(src, int) and 0 <= src < len(base):
            out.append(base[src].model_copy(update=fields))
        else:
            out.append(AnnotatedValue.model_validate(fields))
    return out


def _pagerefs_to_docstrings(refs: list[PageRef]) -> list[str]:
    """The UI's canonical source-doc strings for a set of page refs — its source-page viewer
    (`/api/source/{month}/page/{page}.png`) renders them. Shared by the blocking broker and the
    optimistic registration path so both speak one format."""
    return [
        f"Treasury Bulletin {p.month} PDF page {p.page}"
        for p in refs
        if p.month is not None and p.page is not None
    ]


def _value_page_attribution(
    entries: list[AnnotatedValue],
) -> tuple[list[PageRef], list[dict]]:
    """Map each extracted VALUE back to the actual page(s) it was read from — every
    `AnnotatedValue` carries its own machine-stamped `bulletin`+`pages` — so a review shows ONLY
    those pages, not the whole sem-filter survivor pool, and can label each page with the value(s)
    that came from it. Returns `(refs, page_values)` where `page_values` is
    `[{"month","page","values":[description,...]}]` in page order. `([], [])` when no entry is
    attributable (e.g. a multi-bulletin extract left provenance empty) — the caller then falls
    back to the block pool so the viewer is never empty."""
    refs: list[PageRef] = []
    page_values: list[dict] = []
    by_key: dict[tuple[str, int], list[str]] = {}
    for entry in entries:
        if not entry.bulletin or not entry.pages:
            continue
        for page in entry.pages:
            key = (entry.bulletin, page)
            descs = by_key.get(key)
            if descs is None:
                by_key[key] = descs = []  # same list object lands in page_values below
                refs.append(PageRef(month=entry.bulletin, page=page))
                page_values.append({"month": entry.bulletin, "page": page, "values": descs})
            if entry.description and entry.description not in descs:
                descs.append(entry.description)
    return refs, page_values


def _figure_value_template(branch: RetrieveBranch, entries: list[AnnotatedValue]) -> str:
    """The figure path's editor: a pre-filled `AnnotatedValue` JSON the human edits while reading
    the chart. The model's chart reads are unreliable, so they must NOT anchor the human as confirm-
    or-correct cards do — instead the box states the job. `description` is the RETRIEVAL TARGET
    (`branch.key`, the concept the value must serve), so it says what to read off the page, not what
    the model guessed. Each item keeps `_src` (its candidate index) so a recompute overlays the
    human's value onto that entry and preserves its kind + machine provenance; the model's value
    rides along as a shape scaffold the human overwrites. With no candidates it's a single fresh
    scalar. Pretty-printed JSON (an object for the fresh case, a list when scaffolded)."""
    target = branch.key or ""
    if not entries:
        return json.dumps({"description": target, "value": "", "unit": ""}, indent=2)
    items = [
        {"_src": i, "description": target, "value": e.value, "unit": e.unit, "kind": e.kind}
        for i, e in enumerate(entries)
    ]
    return json.dumps(items, indent=2)


def _branch_identity(branch: Branch) -> dict:
    """The branch's structural identity, carried in a review's guidance so a later recompute can
    target the right branch. Mirrors the `searched` block of the missing-data guidance."""
    if isinstance(branch, RetrieveBranch):
        return {
            "kind": "retrieve",
            "key": branch.key,
            "period": branch.period,
            "visual_only": branch.visual_only,
        }
    return {"kind": "lookup_external", "target": branch.target, "src": branch.src}


# Serializes console prompts across parallel branches — two `asyncio.gather`'d branches must
# never interleave on stdin. Module-level so every channel instance shares one lock.
_PROMPT_LOCK = asyncio.Lock()

HumanTask = Literal["verify_extract", "figure", "lookup"]


@dataclass
class HumanRequest:
    """One unit of work handed to a human: the framing, the model's current answer (if any),
    and the source pages to look at (empty for an external lookup). Pages are carried as
    refs, not pre-rendered images — each channel renders them only if it needs to (the
    console rasterizes to PNG; the web UI links to the live page viewer)."""

    task: HumanTask
    instruction: str
    candidates: list[AnnotatedValue]
    pages: list[PageRef] = field(default_factory=list)
    # Per-page value attribution [{month,page,values:[desc,...]}] — which extracted value(s) came
    # from each page, so the viewer can caption a page with its bulletin + value(s).
    page_values: list[dict] = field(default_factory=list)
    # Figure task only: a pre-filled AnnotatedValue JSON template (description = retrieval target)
    # the human edits while reading the chart, in place of confirm/correct candidate cards.
    value_template: str | None = None
    branch: Branch | None = None


class HumanChannel(Protocol):
    """How a `HumanRequest` is shown and a reply collected. Returns the human's
    `AnnotatedValue`s (provenance is re-stamped by the caller, not the channel)."""

    async def ask(
        self, req: HumanRequest, ctx: ExecutionContext
    ) -> list[AnnotatedValue]: ...


def _candidates_json(candidates: list[AnnotatedValue]) -> str:
    """The model's current answer as the field subset a human edits (no machine provenance)."""
    return json.dumps(
        [c.model_dump(include=set(_FIELDS)) for c in candidates], indent=2
    )


def _parse_reply(raw: str) -> list[AnnotatedValue]:
    """Parse a human's typed JSON (one object or a list) into `AnnotatedValue`s, raising
    `ParseError` on anything malformed so the channel can re-prompt with the detail."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(raw, f"not valid JSON: {e}") from e
    items = data if isinstance(data, list) else [data]
    if not items:
        raise ParseError(raw, "expected at least one AnnotatedValue object")
    out: list[AnnotatedValue] = []
    for item in items:
        try:
            out.append(AnnotatedValue.model_validate(item))
        except Exception as e:  # noqa: BLE001 — surface any validation error to the human
            raise ParseError(raw, f"not a valid AnnotatedValue: {e}") from e
    return out


class ConsoleChannel:
    """Blocking stdin/stdout channel. Rasterizes the source page(s) to temp PNGs and prints
    their paths, prints the candidate(s) + instruction, then reads a reply: a blank line
    accepts the candidates unchanged; otherwise paste a JSON object/array (terminated by a
    line `END`) to override. Stdin is read off the event loop via `asyncio.to_thread`."""

    async def ask(
        self, req: HumanRequest, ctx: ExecutionContext
    ) -> list[AnnotatedValue]:
        async with _PROMPT_LOCK:
            # Render page refs to b64 only here, where the terminal actually needs pixels.
            images, _ = _render_pages_b64(req.pages, ctx) if req.pages else ([], [])
            paths = self._dump_images(images)
            while True:
                self._render_prompt(req, paths)
                raw = await asyncio.to_thread(self._read_stdin)
                stripped = raw.strip()
                if not stripped:
                    if req.candidates:
                        ctx.emit(
                            f"human_response action=accept n={len(req.candidates)}",
                            kind="user",
                        )
                        return list(req.candidates)
                    print("  (no candidate to accept — please paste a value)")
                    continue
                try:
                    parsed = _parse_reply(stripped)
                except ParseError as e:
                    print(f"  ✗ {e.detail}\n  Try again.")
                    continue
                ctx.emit(f"human_response action=override n={len(parsed)}", kind="user")
                return parsed

    @staticmethod
    def _dump_images(images: list[B64Image]) -> list[str]:
        paths: list[str] = []
        for img in images:
            suffix = ".png" if img.mime.endswith("png") else ".jpg"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, prefix="skunk_human_"
            ) as fh:
                fh.write(base64.b64decode(img.data))
                paths.append(fh.name)
        return paths

    @staticmethod
    def _render_prompt(req: HumanRequest, image_paths: list[str]) -> None:
        bar = "=" * 72
        print(f"\n{bar}\n🧑  HUMAN ASSIST [{req.task}]\n{bar}")
        print(req.instruction)
        for i, p in enumerate(image_paths, 1):
            print(f"  image {i}/{len(image_paths)}: {p}")
        if req.value_template is not None:
            # Figure path: the pre-filled template (description = retrieval target) is what to edit,
            # not the model's unreliable read.
            print("\nTemplate to fill (description = retrieval target):")
            print(req.value_template)
        elif req.candidates:
            print("\nModel's current answer:")
            print(_candidates_json(req.candidates))
        print(
            "\nReply: press Enter to accept as-is, or paste a JSON object/array of "
            "AnnotatedValue\nfields and finish with a line containing only END.\n"
        )

    @staticmethod
    def _read_stdin() -> str:
        """Read until EOF or a lone `END` sentinel; a single blank first line returns ''
        (= accept). Runs in a worker thread so the event loop isn't blocked."""
        import sys

        lines: list[str] = []
        for line in sys.stdin:
            if line.rstrip("\n") == "END":
                break
            if not lines and line.strip() == "":
                return ""
            lines.append(line)
        return "".join(lines)


class BrokerChannel:
    """Async, server-mediated channel: delegates to the competition harness's
    `HumanInterventionHandler` (`ctx.human_intervention_handler`) instead of blocking on
    stdin. Awaiting the handler suspends only this branch's coroutine — the event loop keeps
    running every other branch and question — so the system never blocks on human feedback;
    the verified value still gates this branch (compute awaits it). The handler creates a
    `HumanIntervention` a worker claims/resolves in the web UI; the worker's corrected
    `AnnotatedValue`(s) ride back JSON-encoded in the response's `response` field (an empty
    response = "accept the model's candidates as-is")."""

    def __init__(self, handler: HumanInterventionHandler) -> None:
        self._handler = handler

    async def ask(
        self, req: HumanRequest, ctx: ExecutionContext
    ) -> list[AnnotatedValue]:
        # Page refs as the UI's canonical doc strings so its existing source-page viewer
        # (`/api/source/{month}/page/{page}.png`) renders them — no pixels shipped over the wire.
        source_docs = [
            f"Treasury Bulletin {p.month} PDF page {p.page}"
            for p in req.pages
            if p.month is not None and p.page is not None
        ]
        guidance = {
            "task": req.task,
            "candidates": _candidate_dicts(req.candidates),
            "fields": list(_FIELDS),
            "page_values": req.page_values,
        }
        if req.value_template is not None:
            guidance["value_template"] = req.value_template
        ctx.emit(
            f"human_request task={req.task} candidates={len(req.candidates)} "
            f"n_pages={len(source_docs)} via=broker",
            kind="user",
        )
        result = await self._handler(
            req.task, req.instruction, ctx.question, source_docs, guidance
        )
        raw = (result.get("response") or "").strip()
        if not raw:
            # Accept-as-is: no correction submitted → keep the model's candidates.
            ctx.emit(
                f"human_response action=accept n={len(req.candidates)}", kind="user"
            )
            return list(req.candidates)
        try:
            parsed = _parse_reply(raw)
        except ParseError as e:
            # The worker already resolved in the UI — unlike the console we cannot re-prompt.
            # Fall back to the model's candidates (if any) rather than crashing the branch;
            # surface the reason in the trace. A lookup has no candidate, so re-raise there.
            ctx.emit(
                f"human_response action=parse_failed detail={e.detail!r}", kind="user"
            )
            if req.candidates:
                return list(req.candidates)
            raise
        ctx.emit(f"human_response action=override n={len(parsed)}", kind="user")
        return parsed


class HumanAssistPolicy:
    """Decides *whether* a human is consulted for a given operator call. The future-nuance
    seam: today all-or-nothing per flag; tomorrow e.g. verify only list-valued extractions
    (`any(e.kind != "scalar" for e in entries)`) — no caller change needed."""

    def verify_extract(
        self,
        branch: RetrieveBranch,
        entries: list[AnnotatedValue],
        cfg,
    ) -> bool:
        # Figure questions (visual_only) gate on human_figure — the model reads charts
        # unreliably so the human produces the answer. Every other extraction gates on
        # human_verify_extract — the human confirms/corrects the OCR/table read, regardless
        # of value shape (scalar/vector/table all get the same review).
        if branch.visual_only:
            return cfg.human_figure
        return cfg.human_verify_extract

    def human_lookup(self, branch: LookupBranch, cfg) -> bool:
        return cfg.human_lookup


class HumanAssist:
    """Facade the orchestrator holds. Joins policy + channel and returns the same
    `list[AnnotatedValue]` the operators do, so wiring it in is a one-line wrap per branch."""

    def __init__(
        self,
        channel: HumanChannel | None = None,
        policy: HumanAssistPolicy | None = None,
    ) -> None:
        self._channel = channel or ConsoleChannel()
        self._policy = policy or HumanAssistPolicy()

    def wants_verify(
        self,
        branch: RetrieveBranch,
        entries: list[AnnotatedValue],
        ctx: ExecutionContext,
    ) -> bool:
        """Cheap, side-effect-free policy check — lets the orchestrator skip opening a
        traced step (and any console prompt) when no human is consulted."""
        return self._policy.verify_extract(branch, entries, ctx.config)

    def wants_lookup(self, branch: LookupBranch, ctx: ExecutionContext) -> bool:
        # Disabled: an external lookup's value(s) are rolled into the data-prep agent's output and
        # reviewed in the single data-prep pool review (verify_extract), so the per-lookup human
        # hook (optimistic register_lookup + blocking human_lookup) is redundant. The lookup agent
        # still runs; only the separate review is suppressed. (`human_lookup` policy flag retained.)
        return False

    def wants_pool_review(self, ctx: ExecutionContext) -> bool:
        """Whether the cleaned data-prep pool gets a human review this round. Gated by the same
        flags that gated the former per-extract verify — `human_verify_extract` (non-visual) or
        `human_figure` (chart/figure reads) — now applied once to the data-prep output (which holds
        both kinds of value) instead of once per branch."""
        return ctx.config.human_verify_extract or ctx.config.human_figure

    async def verify_extract(
        self,
        entries: list[AnnotatedValue],
        pages: list[PageRef],
        branch: RetrieveBranch,
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Show the rendered source page(s) + the model's candidates and return the human's
        answer, re-stamped with the branch/page provenance the operators stamp. Assumes the
        caller already gated on `wants_verify`."""
        # Show ONLY the pages the values were actually read from (per-value provenance), not the
        # whole retrieved set; fall back to the retrieved pages when nothing is attributable.
        refs, page_values = _value_page_attribution(entries)
        if not refs:
            refs, page_values = list(pages), []
        instruction = (
            "This answer must be read off the figure/chart on the page(s) below — the "
            "model is unreliable here. Give the correct value(s)."
            if branch.visual_only
            else "Confirm or correct the value(s) the model extracted, checking them "
            "against the source page(s) below."
        )
        ctx.emit(
            f"human_request task={'figure' if branch.visual_only else 'verify_extract'} "
            f"candidates={[e.description for e in entries]!r} n_pages={len(refs)}",
            kind="user",
        )
        reply = await self._channel.ask(
            HumanRequest(
                task="figure" if branch.visual_only else "verify_extract",
                instruction=instruction,
                candidates=entries,
                pages=refs,
                page_values=page_values,
                value_template=(
                    _figure_value_template(branch, entries) if branch.visual_only else None
                ),
                branch=branch,
            ),
            ctx,
        )
        # Re-stamp machine provenance from the source pages + branch (the human authors only
        # the semantic fields), matching how extract stamps its own output.
        return _stamp_provenance(reply, refs, branch)

    async def human_lookup(
        self, branch: LookupBranch, ctx: ExecutionContext
    ) -> list[AnnotatedValue]:
        """Run the external lookup as a human task. Assumes the caller already gated on
        `wants_lookup`."""
        instruction = (
            f"Perform this external lookup and report the value(s):\n  target: "
            f"{branch.target}\n  source hint: {branch.src or '(any authoritative source)'}"
        )
        ctx.emit(f"human_request task=lookup target={branch.target!r}", kind="user")
        return await self._channel.ask(
            HumanRequest(
                task="lookup",
                instruction=instruction,
                candidates=[],
                pages=[],
                branch=branch,
            ),
            ctx,
        )

    # ---- Optimistic (non-blocking) registration ----------------------------------------
    # These open a human review and return immediately; the branch keeps the LLM result and
    # the question completes. A human resolve later drives a server-side recompute. Each is a
    # no-op when no register hook is wired (local CLI), so the caller can invoke unconditionally.

    def register_pool_review(
        self,
        pool: list[AnnotatedValue],
        ctx: ExecutionContext,
        *,
        source_values: list[AnnotatedValue] | None = None,
    ) -> str | None:
        """Open ONE review of the data-prep agent's output — the cleaned value pool compute is
        about to read — supplied with the PDF page(s) those values were pulled from. Replaces the
        former per-branch extract reviews: a single review fires per round instead of one per
        extract. Keyed by the sentinel `POOL_REVIEW_BRANCH_ID` so a resolve recomputes over the
        human-edited pool via the same register→resolve→recompute path. Assumes the caller gated
        on `wants_pool_review`; no-op when no register hook is wired (local CLI).

        `source_values` (the PRE-clean pool) is the page-attribution source: data-prep coalesces
        values across bulletins and collapses `bulletin` to a RANGE ("1985-03..1987-06"), which
        can't map a page to a single renderable PDF. The pre-clean values still carry one bulletin
        each, so the viewer resolves their pages. Falls back to `pool` when not given."""
        register = ctx.human_review_register
        if register is None:
            return None
        # Show the pages the values were pulled from (per-value provenance). Attribute from the
        # pre-clean values (single bulletin each) so coalesced range-bulletins don't break the
        # viewer. Lookup-derived values carry no page provenance and simply appear as cards with
        # no page; that's expected.
        refs, page_values = _value_page_attribution(
            source_values if source_values is not None else pool
        )
        instruction = (
            "Review the data the agent cleaned and is about to compute over. Confirm or correct "
            "the value(s), checking them against the source page(s) below."
        )
        guidance = {
            "task": "verify_extract",
            "branch_id": POOL_REVIEW_BRANCH_ID,
            "branch": {},
            "candidates": _candidate_dicts(pool),
            "fields": list(_FIELDS),
            "page_values": page_values,
        }
        review_id = register(
            "verify_extract", instruction, ctx.question, _pagerefs_to_docstrings(refs), guidance
        )
        ctx.emit(
            f"human_review_registered task=verify_extract branch_id={POOL_REVIEW_BRANCH_ID} "
            f"review_id={review_id} candidates={len(pool)} n_pages={len(refs)}",
            kind="user",
        )
        return review_id

    def register_lookup(
        self,
        entries: list[AnnotatedValue],
        branch: LookupBranch,
        bid: int,
        ctx: ExecutionContext,
    ) -> str | None:
        """Open a review of the lookup agent's value(s) for an external lookup, so a human can
        confirm/correct them. `entries` are the agent's result (the review's candidates)."""
        register = ctx.human_review_register
        if register is None:
            return None
        instruction = (
            f"Confirm or correct the value(s) found for this external lookup:\n  target: "
            f"{branch.target}\n  source hint: {branch.src or '(any authoritative source)'}"
        )
        guidance = {
            "task": "lookup",
            "branch_id": bid,
            "branch": _branch_identity(branch),
            "candidates": _candidate_dicts(entries),
            "fields": list(_FIELDS),
        }
        review_id = register("lookup", instruction, ctx.question, [], guidance)
        ctx.emit(
            f"human_review_registered task=lookup branch_id={bid} "
            f"review_id={review_id} candidates={len(entries)}",
            kind="user",
        )
        return review_id
