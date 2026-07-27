from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import yaml

from skunk.common import B64Image, Effort, ExecutionContext
from skunk.errors import ParseError

if TYPE_CHECKING:
    from skunk.llm_client import LLMResponse


Section = Literal["corpus", "few_shots", "lessons"]


@dataclass(frozen=True)
class PromptOverride:
    section: Section
    targets: tuple[str, ...]  # agent names; "*" = all
    content: str | tuple[str, ...]  # str for corpus/lessons; tuple for few_shots


def load_prompt_overrides(path: str | Path) -> tuple[PromptOverride, ...]:
    """Read a YAML override file → tuple of `PromptOverride`."""
    data = yaml.safe_load(Path(path).read_text())
    if not data or "overrides" not in data:
        return ()

    out: list[PromptOverride] = []
    for raw in data["overrides"]:
        section = raw["section"]
        if section not in ("corpus", "few_shots", "lessons"):
            raise ValueError(f"unknown prompt override section: {section!r}")
        targets = tuple(raw.get("targets") or ())
        if not targets:
            raise ValueError(f"override missing targets: {raw!r}")

        content: str | tuple[str, ...]
        if section == "few_shots":
            items = raw.get("content") or []
            if not isinstance(items, list):
                raise ValueError(f"few_shots content must be a list: {raw!r}")
            content = tuple(str(item) for item in items)
        else:
            content = str(raw.get("content") or "")

        out.append(PromptOverride(section=section, targets=targets, content=content))

    return tuple(out)


def _identity(raw: str, _ctx: ExecutionContext) -> str:
    return raw


@dataclass(frozen=True)
class _PromptParts:
    corpus: str
    few_shots: tuple[str, ...]
    lessons: tuple[str, ...]


def _build_tail(parts: _PromptParts) -> str:
    sections: list[str] = []
    if parts.corpus:
        sections.append("\n## Dataset\n" + parts.corpus)
    if parts.few_shots:
        sections.append("\n## Few-shot examples\n" + "\n\n".join(parts.few_shots))
    if parts.lessons:
        sections.append(
            "\n## Lessons learned\n"
            + "\n".join(f"- {lesson}" for lesson in parts.lessons)
        )
    if not sections:
        return ""
    return (
        "\nThe dataset section below enumerates the corpus-specific conventions and points to a few\n"
        "worked examples.\n" + "".join(sections)
    )


def _gather_overrides(overrides: tuple[PromptOverride, ...], name: str) -> _PromptParts:
    corpus_parts: list[str] = []
    few_shots: list[str] = []
    lessons: list[str] = []
    for o in overrides:
        if "*" not in o.targets and name not in o.targets:
            continue
        if o.section == "corpus" and isinstance(o.content, str) and o.content:
            corpus_parts.append(o.content)
        elif o.section == "few_shots" and isinstance(o.content, tuple):
            few_shots.extend(o.content)
        elif o.section == "lessons" and isinstance(o.content, str):
            for line in o.content.splitlines():
                stripped = line.strip()
                if stripped:
                    lessons.append(stripped)
    return _PromptParts(
        corpus="\n\n".join(corpus_parts),
        few_shots=tuple(few_shots),
        lessons=tuple(lessons),
    )


class PromptedCall[T]:
    """One LLM-prompted call-site, generic over the type `T` of its parsed result.
    Owns prompt assembly, effort resolution, LLM dispatch, logging, and parsing, so
    call-sites carry only their domain logic. `call()` returns `parse(raw_text, ctx)`
    (default parser returns raw text → a bare `PromptedCall` is `PromptedCall[str]`).
    `name` is the override-registry / effort-override key.
    """

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        default_effort: Effort = "off",
        parse: Callable[[str, ExecutionContext], T] = _identity,  # type: ignore[assignment]
        output_instruction: str | None = None,
        max_parse_retries: int = 1,
    ) -> None:
        self.name = name
        self._system_prompt = system_prompt
        self._default_effort = default_effort
        self._parse = parse
        self._output_instruction = output_instruction
        self._max_parse_retries = max_parse_retries

    def assemble_system_prompt(self, ctx: ExecutionContext) -> str:
        """The full system prompt this call-site sends: the declared `system_prompt`
        plus the override tail (`corpus` / `few_shots` / `lessons` sections gathered
        from `ctx.prompt_overrides` by `name`). Public — agent loops render it
        themselves when they own the transport (e.g. a rollout backend)."""
        return self._system_prompt + _build_tail(
            _gather_overrides(ctx.prompt_overrides, self.name)
        )

    def _resolve_effort(self, ctx: ExecutionContext, effort: Effort | None) -> Effort:
        if effort is not None:
            return effort
        return cast(
            Effort, ctx.config.inference.effort_overrides.get(self.name, self._default_effort)
        )

    def _resolve_model(self, ctx: ExecutionContext) -> str:
        return ctx.config.inference.model_overrides.get(self.name, ctx.config.inference.llm_model)

    def _compose_user(self, user: str, retry: ParseError | None) -> str:
        parts = [user]
        if retry is not None:
            parts.append(
                f"Your previous reply could not be parsed:\n```\n{retry.raw}\n```\n\n"
                f"Error: {retry.detail}\nFix the issue and return a valid response."
            )
        if self._output_instruction:
            parts.append(self._output_instruction)
        return "\n\n".join(parts)

    async def call_multi_turn(
        self,
        ctx: ExecutionContext,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        effort: Effort | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        on_response: Callable[[LLMResponse], None] | None = None,
        usage_key: str = "default",
    ) -> tuple[T, int]:
        """One agent-loop turn over an existing `messages` transcript: assemble the
        system prompt, send to the llm_client, and parse the response. On a `ParseError`,
        the failed reply + fix-it prompt are appended to a COPY of `messages` and the
        turn re-streams (up to `max_parse_retries`) — the caller's list never sees the
        retry exchange. `model` overrides the per-site resolution (the agent layer passes
        its own agent model; effort/model overrides otherwise resolve exactly as `call`)."""
        eff = self._resolve_effort(ctx, effort)
        system = self.assemble_system_prompt(ctx)
        resolved_model = model or self._resolve_model(ctx)
        messages = list(messages)  # work on a copy so callers don't see retry exchanges
        attempt = 0
        while True:
            resp = await ctx.llm_client.acall(
                system=system,
                messages=messages,
                model=resolved_model,
                temperature=temperature,
                effort=eff,
                ctx=ctx,
                call_site=self.name,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
                usage_key=usage_key,
            )
            if on_response is not None:
                on_response(resp)
            try:
                # Usage fields are `int | None` (a provider CAN omit usage even on a 200);
                # coalesce to 0 so a usage-less response degrades the tracker, not the step.
                return self._parse(resp.text, ctx), (resp.input_tokens or 0) + (resp.output_tokens or 0)
            except ParseError as e:
                if attempt >= self._max_parse_retries:
                    raise
                ctx.emit(
                    f"parse_retry call_site={self.name} attempt={attempt + 1} error={e.detail!r}"
                )
                messages.append({"role": "assistant", "content": resp.text})
                messages.append(
                    {"role": "user", "content": self._compose_user("", e).strip()}
                )
                attempt += 1

    async def call(
        self,
        ctx: ExecutionContext,
        user: str = "",
        *,
        images: list[B64Image] | None = None,
        temperature: float = 0.0,
        effort: Effort | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
        on_response: Callable[[LLMResponse], None] | None = None,
        usage_key: str = "default",
    ) -> T:
        """Single-shot: assemble system+user, resolve effort/model, invoke `acall`,
        parse into a typed result; retries on `ParseError` up to `max_parse_retries`
        times (at temperature 1.0 — see the loop). `on_response` (when given)
        observes every raw `LLMResponse` — one per attempt, parse-retries included —
        so call-sites can record per-call latency/token stats without re-deriving
        them from the envelope log. Agent loops use `call_multi_turn` instead."""
        eff = self._resolve_effort(ctx, effort)
        system = self.assemble_system_prompt(ctx)
        model = self._resolve_model(ctx)
        # Capture the operator's full LLM I/O for the trace viewer — the `call`
        # envelope LLMClient logs carries only latency/tokens, not the text. System
        # + the base user go once; each attempt's assistant reply is emitted as it
        # arrives. The multi-turn branch above never reaches here, and those agents
        # emit their own per-turn system/assistant via `MultiTurnAgent`, so there is
        # no double-logging.
        ctx.emit(
            f"prompt_system call_site={self.name} chars={len(system)}",
            kind="system",
            data={"text": system},
        )
        base_user = self._compose_user(user, None)
        ctx.emit(
            f"prompt_user call_site={self.name} chars={len(base_user)}",
            kind="user",
            data={"text": base_user},
        )
        attempt = 0
        retry: ParseError | None = None
        while True:
            # First attempt honors the caller's temperature; parse-retries run at 1.0.
            # A low-temp reply that won't parse tends to regenerate verbatim even when
            # re-prompted with the error, so jumping to full temperature — not just
            # re-asking — is what actually breaks a degenerate output (e.g.
            # prose-as-JSON, an unescaped backslash).
            attempt_temp = temperature if attempt == 0 else 1.0
            messages = [{"role": "user", "content": self._compose_user(user, retry), "images": images}]
            resp = await ctx.llm_client.acall(
                system,
                messages,
                temperature=attempt_temp,
                effort=eff,
                ctx=ctx,
                call_site=self.name,
                model=model,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
                usage_key=usage_key,
            )
            if on_response is not None:
                on_response(resp)
            ctx.emit(
                f"assistant call_site={self.name} chars={len(resp.text)}",
                kind="assistant",
                data={"text": resp.text},
            )
            try:
                return self._parse(resp.text, ctx)
            except ParseError as e:
                if attempt >= self._max_parse_retries or not e.retryable:
                    raise
                ctx.emit(
                    f"parse_retry call_site={self.name} attempt={attempt + 1} "
                    f"next_temp=1.0 error={e.detail!r}"
                )
                retry = e
                attempt += 1
