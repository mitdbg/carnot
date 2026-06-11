"""Human-in-the-loop assistance — a modular middleware layer that routes specific
sub-tasks to a person instead of (or after) the model.

Design: the operators (`ExtractOp`, `LookupExternalOp`) are untouched. The orchestrator
calls this layer at its dispatch seam; everything here is gated by config flags
(`human_figure`, `human_verify_extract`, `human_lookup`) that default OFF, so with no flag
set this is inert. Three pieces, each a swappable seam:

  - `HumanAssistPolicy` decides *whether* a human is asked for a given operator call. Today
    it's all-or-nothing per flag; future nuance (verify only list-valued extractions, etc.)
    lives entirely here — no operator/orchestrator change.
  - `HumanChannel` is the I/O *how*. `ConsoleChannel` (default) renders the source page(s),
    prints the model's candidate(s), and reads a typed reply on stdin. The client/server
    harness UI can later implement the same interface without touching anything else.
  - `HumanAssist` is the facade the orchestrator holds: it joins policy + channel and
    returns `list[AnnotatedValue]` — the same currency every operator speaks.

With any flag on, a run blocks on the console one branch at a time (a module-level
`asyncio.Lock` serializes prompts across parallel branches), so target a few UIDs, not a
sweep.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
from dataclasses import dataclass, field
from typing import Literal, Protocol

from skunk.common import AnnotatedValue, B64Image, ExecutionContext
from skunk.errors import ParseError
from skunk.extract import _blocks_to_pagerefs, _render_pages_b64, _stamp_provenance
from skunk.plan import Branch, LookupBranch, RetrieveBranch

# The JSON shape a human types to override the model — the same `AnnotatedValue` field set
# `LookupAgent.final_answer_doc` documents, so the human and the model speak one format.
_FIELDS = ("description", "value", "unit", "kind", "index_name", "row_name", "col_name")

# Serializes console prompts across parallel branches — two `asyncio.gather`'d branches must
# never interleave on stdin. Module-level so every channel instance shares one lock.
_PROMPT_LOCK = asyncio.Lock()

HumanTask = Literal["verify_extract", "figure", "lookup"]


@dataclass
class HumanRequest:
    """One unit of work handed to a human: the framing, the model's current answer (if any),
    and the source-page images to look at (empty for an external lookup)."""

    task: HumanTask
    instruction: str
    candidates: list[AnnotatedValue]
    images: list[B64Image] = field(default_factory=list)
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
    """Blocking stdin/stdout channel. Renders each image to a temp PNG and prints its path,
    prints the candidate(s) + instruction, then reads a reply: a blank line accepts the
    candidates unchanged; otherwise paste a JSON object/array (terminated by a line `END`)
    to override. Stdin is read off the event loop via `asyncio.to_thread`."""

    async def ask(
        self, req: HumanRequest, ctx: ExecutionContext
    ) -> list[AnnotatedValue]:
        async with _PROMPT_LOCK:
            paths = self._dump_images(req.images)
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
        # human_verify_extract — the human confirms/corrects an OCR/table read.
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
        # best-effort render; skip the page store entirely when there are no source pages.
        images, _ = _render_pages_b64(refs, ctx) if refs else ([], [])
        instruction = (
            "This answer must be read off the figure/chart on the page(s) below — the "
            "model is unreliable here. Give the correct value(s)."
            if branch.visual_only
            else "Confirm or correct the value(s) the model extracted, checking them "
            "against the source page(s) below."
        )
        ctx.emit(
            f"human_request task={'figure' if branch.visual_only else 'verify_extract'} "
            f"candidates={[e.description for e in entries]!r} n_images={len(images)}",
            kind="user",
        )
        reply = await self._channel.ask(
            HumanRequest(
                task="figure" if branch.visual_only else "verify_extract",
                instruction=instruction,
                candidates=entries,
                images=images,
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
                images=[],
                branch=branch,
            ),
            ctx,
        )
