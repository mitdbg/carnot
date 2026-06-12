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
from skunk.extract import _blocks_to_pagerefs, _render_pages_b64, _stamp_provenance
from skunk.plan import Branch, LookupBranch, RetrieveBranch

# The JSON shape a human types to override the model — the same `AnnotatedValue` field set
# `LookupAgent.final_answer_doc` documents, so the human and the model speak one format.
_FIELDS = ("description", "value", "unit", "kind", "index_name", "row_name", "col_name")


# The fields a human edits in the review overlay (the rest — kind/index_name/row_name/col_name
# and machine provenance — are preserved from the original extraction via `_src`).
_EDITABLE_FIELDS = ("description", "unit", "value")


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


def _branch_identity(branch: Branch) -> dict:
    """The branch's structural identity, carried in a review's guidance so a later recompute can
    target the right branch. Mirrors the `searched` block of the missing-data guidance."""
    if isinstance(branch, RetrieveBranch):
        return {
            "kind": "retrieve",
            "key": branch.key,
            "period": branch.period,
            "as_of": branch.as_of,
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
        if req.candidates:
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
            "candidates": [c.model_dump(include=set(_FIELDS)) for c in req.candidates],
            "fields": list(_FIELDS),
        }
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
        # unreliably so the human produces the answer. Everything else gates on
        # human_verify_extract — the human confirms/corrects an OCR/table read — but only
        # for vector/table-shaped reads: scalar extractions are cheap to trust and not worth a
        # human's attention, so they never open a review regardless of the flag.
        if branch.visual_only:
            return cfg.human_figure
        if all(e.kind == "scalar" for e in entries):
            return False
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
        return self._policy.human_lookup(branch, ctx.config)

    async def verify_extract(
        self,
        entries: list[AnnotatedValue],
        blocks: list,
        branch: RetrieveBranch,
        ctx: ExecutionContext,
    ) -> list[AnnotatedValue]:
        """Show the rendered source page(s) + the model's candidates and return the human's
        answer, re-stamped with the branch/page provenance the operators stamp. Assumes the
        caller already gated on `wants_verify`."""
        refs = _blocks_to_pagerefs(blocks)
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

    def register_verify(
        self,
        entries: list[AnnotatedValue],
        blocks: list,
        branch: RetrieveBranch,
        bid: int,
        ctx: ExecutionContext,
    ) -> str | None:
        """Open a review of the model's extracted value(s) against the source page(s), carrying
        the branch identity (keyed by `bid`) so a resolve can recompute. Assumes the caller
        already gated on `wants_verify`."""
        register = ctx.human_review_register
        if register is None:
            return None
        task = "figure" if branch.visual_only else "verify_extract"
        refs = _blocks_to_pagerefs(blocks)
        instruction = (
            "This answer must be read off the figure/chart on the page(s) below — the "
            "model is unreliable here. Give the correct value(s)."
            if branch.visual_only
            else "Confirm or correct the value(s) the model extracted, checking them "
            "against the source page(s) below."
        )
        guidance = {
            "task": task,
            "branch_id": bid,
            "branch": _branch_identity(branch),
            "candidates": [c.model_dump(include=set(_FIELDS)) for c in entries],
            "fields": list(_FIELDS),
        }
        review_id = register(
            task, instruction, ctx.question, _pagerefs_to_docstrings(refs), guidance
        )
        ctx.emit(
            f"human_review_registered task={task} branch_id={bid} "
            f"review_id={review_id} candidates={len(entries)} n_pages={len(refs)}",
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
            "candidates": [c.model_dump(include=set(_FIELDS)) for c in entries],
            "fields": list(_FIELDS),
        }
        review_id = register("lookup", instruction, ctx.question, [], guidance)
        ctx.emit(
            f"human_review_registered task=lookup branch_id={bid} "
            f"review_id={review_id} candidates={len(entries)}",
            kind="user",
        )
        return review_id
