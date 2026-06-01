"""PromptedCall — the runtime helper each LLM-prompted call-site HAS-A. It renders
a Jinja `system_prompt`, runs the LLM call, and parses the reply into a typed result.

The override registry (`prompt_overrides.yaml`, selected per call-site by name)
supplies the template vars `corpus`, `few_shots`, `lessons`, and `default_tail`
(the pre-rendered conventional trailing block). Templates wanting that layout end
with `{{ default_tail }}`; others reference the raw vars. A `template_vars` provider
adds runtime entries.

Templates needing a literal `{{`/`{%` must wrap it in `{% raw %}…{% endraw %}`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from jinja2 import Environment, StrictUndefined

from skunk.common import HarnessContext
from skunk.errors import ParseError

if TYPE_CHECKING:
    from skunk.common import Effort


# Supplementary prompt content (corpus / few_shots / lessons) loaded from YAML and
# attached to `HarnessContext.prompt_overrides`. `PromptedCall` selects each
# call-site's slices by name ("*" = all).
Section = Literal["corpus", "few_shots", "lessons"]


@dataclass(frozen=True)
class PromptOverride:
    section: Section
    targets: tuple[str, ...]                          # agent names; "*" = all
    content: str | tuple[str, ...]                    # str for corpus/lessons; tuple[str, ...] for few_shots


def load_prompt_overrides(path: str | Path) -> tuple[PromptOverride, ...]:
    """Read a YAML override file → tuple of `PromptOverride`. Each entry has
    `section` (corpus/few_shots/lessons), `targets` (agent names or "*"), and
    `content` (a string, or a list of strings for few_shots)."""
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

# Yields extra Jinja vars from `ctx` at render time (e.g. an agent's `max_steps`).
TemplateVarsProvider = Callable[["HarnessContext"], dict]


def _identity(raw: str, ctx: HarnessContext) -> str:
    """Default parser — return the raw model text unchanged (`PromptedCall[str]`)."""
    return raw

# `StrictUndefined` makes a typo'd `{{ var }}` fail loudly; `keep_trailing_newline`
# preserves the trailing `\n` of triple-quoted templates.
_ENV = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)


def _build_default_tail(
    corpus: str,
    few_shots: tuple[str, ...],
    lessons: tuple[str, ...],
) -> str:
    """The conventional `## Dataset` / `## Few-shot examples` / `## Lessons learned`
    trailing block. Empty when no overrides target this call-site."""
    parts: list[str] = []
    if corpus:
        parts.append("\n## Dataset\n" + corpus)
    if few_shots:
        parts.append("\n## Few-shot examples\n" + "\n\n".join(few_shots))
    if lessons:
        parts.append("\n## Lessons learned\n" + "\n".join(f"- {lesson}" for lesson in lessons))
    return "".join(parts)


def _gather_overrides(
    overrides: tuple[PromptOverride, ...], name: str
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Select this call-site's (corpus, few_shots, lessons) override slices from the
    registry, concatenating multiple entries per section in declaration order."""
    corpus_parts: list[str] = []
    few_shots: list[str] = []
    lessons: list[str] = []
    for o in overrides:
        # `"*"` targets every call-site; otherwise match this call-site by name
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
    return "\n\n".join(corpus_parts), tuple(few_shots), tuple(lessons)


class PromptedCall[T]:
    """One LLM-prompted call-site, generic over the type `T` of its parsed result.
    Owns prompt assembly, effort resolution, LLM dispatch, logging, and parsing, so
    call-sites carry only their domain logic. `call()` returns `parse(raw_text, ctx)`
    (default parser returns raw text → a bare `PromptedCall` is `PromptedCall[str]`).
    `name` is the override-registry / effort-override key.

    The `parse` hook may raise a call-site control-flow signal or `ParseError`; the
    raw response is logged before parse runs, so a parse failure is still traced."""

    def __init__(
        self,
        *,
        name: str,
        system_prompt: str,
        default_effort: "Effort" = "off",
        template_vars: TemplateVarsProvider | None = None,
        parse: Callable[[str, HarnessContext], T] = _identity,  # type: ignore[assignment]
        max_parse_retries: int = 1,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.default_effort = default_effort
        self._template_vars = template_vars
        self._parse = parse
        self._max_parse_retries = max_parse_retries

    def assemble_system_prompt(self, ctx: HarnessContext) -> str:
        corpus, few_shots, lessons = _gather_overrides(ctx.prompt_overrides, self.name)
        variables = {
            **(self._template_vars(ctx) if self._template_vars else {}),
            "corpus": corpus,
            "few_shots": few_shots,
            "lessons": lessons,
            "default_tail": _build_default_tail(corpus, few_shots, lessons),
        }
        return _ENV.from_string(self.system_prompt).render(**variables)

    def resolve_effort(self, ctx: HarnessContext, effort: "Effort | None") -> "Effort":
        """Resolution priority: explicit arg > config override > class default."""
        if effort is not None:
            return effort
        return ctx.config.effort_overrides.get(self.name, self.default_effort)

    @staticmethod
    def _retry_message(user: str, e: ParseError) -> str:
        """Re-prompt body for a parse retry: original user message + the
        unparseable reply + the error. No history accumulation."""
        return (
            f"{user}\n\n"
            f"Your previous reply could not be parsed:\n```\n{e.raw}\n```\n\n"
            f"Error: {e.detail}\n"
            "Fix the issue and return a valid response."
        )

    def call(
        self,
        ctx: HarnessContext,
        user: str,
        *,
        images: list[tuple[str, str]] | None = None,
        temperature: float = 0.0,
        effort: "Effort | None" = None,
    ) -> T:
        """Assemble the prompt, resolve effort, invoke the LLM, then parse into a
        typed result. A `ParseError` re-prompts up to `max_parse_retries` times
        (the only retry this layer owns); any other parse exception (e.g.
        `MissingData`) is application-level and propagates immediately."""
        eff = self.resolve_effort(ctx, effort)
        system = self.assemble_system_prompt(ctx)  # identical across retries; render once
        message = user
        attempt = 0
        while True:
            resp = ctx.llm_client.call(
                system,
                message,
                images=images,
                temperature=temperature,
                effort=eff,
                ctx=ctx,
                call_site=self.name,
            )
            try:
                return self._parse(resp.text, ctx)
            except ParseError as e:
                if attempt >= self._max_parse_retries:
                    raise
                ctx.emit(self.name, "parse_retry", attempt=attempt + 1, error=e.detail)
                message = self._retry_message(user, e)
                attempt += 1
