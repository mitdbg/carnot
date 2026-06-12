from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from skunk.common import B64Image, Effort, ExecutionContext
from skunk.errors import ParseError


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

    def _assemble_system_prompt(self, ctx: ExecutionContext) -> str:
        return self._system_prompt + _build_tail(
            _gather_overrides(ctx.prompt_overrides, self.name)
        )

    def _resolve_effort(self, ctx: ExecutionContext, effort: Effort | None) -> Effort:
        if effort is not None:
            return effort
        return cast(
            Effort, ctx.config.effort_overrides.get(self.name, self._default_effort)
        )

    def _resolve_model(self, ctx: ExecutionContext) -> str:
        return ctx.config.model_overrides.get(self.name, ctx.config.llm_model)

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

    async def call(
        self,
        ctx: ExecutionContext,
        user: str = "",
        *,
        messages: list[dict] | None = None,
        images: list[B64Image] | None = None,
        temperature: float = 0.0,
        effort: Effort | None = None,
        should_stop: Callable[[str], bool] | None = None,
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> T:
        """Assemble the prompt, resolve effort, invoke the LLM, then parse into a
        typed result.

        Single-shot (default): assembles system+user, calls `acall`, retries on
        `ParseError` up to `max_parse_retries` times.

        Multi-turn (pass `messages`): assembles system, streams via `astream` with
        optional `should_stop`; no retry — the caller owns the loop.
        """
        eff = self._resolve_effort(ctx, effort)
        system = self._assemble_system_prompt(ctx)
        if messages is not None:
            model = ctx.config.agent_model_id or self._resolve_model(ctx)
            messages = list(
                messages
            )  # work on a copy so callers don't see retry exchanges
            attempt = 0
            while True:
                resp = await ctx.llm_client.astream(
                    system=system,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    should_stop=should_stop,
                    effort=eff,
                    ctx=ctx,
                    call_site=self.name,
                    max_output_tokens=max_output_tokens,
                    timeout_s=timeout_s,
                )
                try:
                    return self._parse(resp.text, ctx)
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
            # First attempt honors the caller's temperature (usually 0.0 for determinism);
            # parse-retries ESCALATE it. A temp-0 reply that won't parse tends to regenerate
            # verbatim even when re-prompted with the error, so nudging temperature — not just
            # re-asking — is what actually breaks a degenerate output (e.g. prose-as-JSON, an
            # unescaped backslash). 0.0 → 0.4 → 0.6 → 0.8, capped at 1.0.
            attempt_temp = temperature if attempt == 0 else min(1.0, 0.4 + 0.2 * (attempt - 1))
            resp = await ctx.llm_client.acall(
                system,
                self._compose_user(user, retry),
                images=images,
                temperature=attempt_temp,
                effort=eff,
                ctx=ctx,
                call_site=self.name,
                model=model,
                max_output_tokens=max_output_tokens,
                timeout_s=timeout_s,
            )
            ctx.emit(
                f"assistant call_site={self.name} chars={len(resp.text)}",
                kind="assistant",
                data={"text": resp.text},
            )
            try:
                return self._parse(resp.text, ctx)
            except ParseError as e:
                if attempt >= self._max_parse_retries:
                    raise
                ctx.emit(
                    f"parse_retry call_site={self.name} attempt={attempt + 1} "
                    f"next_temp={min(1.0, 0.4 + 0.2 * attempt):.1f} error={e.detail!r}"
                )
                retry = e
                attempt += 1
