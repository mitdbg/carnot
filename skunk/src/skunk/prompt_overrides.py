"""Prompt overrides plumbing.

Each subagent owns its core SYSTEM prompt in its own module. Three sections
are appended at runtime from override entries collected on the
HarnessContext:

  - corpus     — dataset-specific background paragraphs.
  - few_shots  — list of pre-formatted example strings; the operator
                 concatenates them verbatim under a "Few-shot examples" header.
                 The YAML author controls the formatting.
  - lessons    — short bullet strings about gotchas / things to watch for.

A `PromptOverride` is keyed by `(section, targets)` and carries either a
text blob (corpus / lessons) or a tuple of example strings (few_shots).
The targets field is a tuple of agent names, with the special wildcard
"*" meaning "applies to every agent". Multiple overrides for the same
section + agent concatenate in declaration order.

Overrides ship in a YAML file (see `load_prompt_overrides`). Top-level
runners load one and attach it to `HarnessContext.prompt_overrides`; CLI
or test code can add entries by appending more PromptOverride tuples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Section = Literal["corpus", "few_shots", "lessons"]


@dataclass(frozen=True)
class PromptOverride:
    section: Section
    targets: tuple[str, ...]                          # agent names; "*" = all
    content: str | tuple[str, ...]                    # str for corpus/lessons; tuple[str, ...] for few_shots


def _matches(o: PromptOverride, agent: str) -> bool:
    return "*" in o.targets or agent in o.targets


def gather_corpus(overrides: tuple[PromptOverride, ...], agent: str) -> str:
    """Concatenate all `corpus` overrides addressed to `agent`, in order."""
    parts = [o.content for o in overrides
             if o.section == "corpus" and _matches(o, agent)
             and isinstance(o.content, str) and o.content]
    return "\n\n".join(parts)


def gather_few_shots(overrides: tuple[PromptOverride, ...], agent: str) -> tuple[str, ...]:
    """Return the pre-formatted few-shot strings addressed to `agent`."""
    out: list[str] = []
    for o in overrides:
        if o.section == "few_shots" and _matches(o, agent) and isinstance(o.content, tuple):
            out.extend(o.content)
    return tuple(out)


def gather_lessons(overrides: tuple[PromptOverride, ...], agent: str) -> tuple[str, ...]:
    """Lessons are bullet strings. Each content blob may contain one or more
    bullets separated by newlines."""
    out: list[str] = []
    for o in overrides:
        if o.section == "lessons" and _matches(o, agent) and isinstance(o.content, str):
            for line in o.content.splitlines():
                line = line.strip()
                if line:
                    out.append(line)
    return tuple(out)


def render_lessons_block(lessons: tuple[str, ...]) -> str:
    """Format a lessons tuple as a markdown bullet list under a header.
    Returns "" if there are no lessons (so callers can join unconditionally)."""
    if not lessons:
        return ""
    return "## Lessons learned\n" + "\n".join(f"- {l}" for l in lessons)


def render_corpus_block(corpus: str) -> str:
    """Format the corpus blurb under a header. Returns "" when empty."""
    if not corpus:
        return ""
    return f"## Dataset\n{corpus}"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_prompt_overrides(path: str | Path) -> tuple[PromptOverride, ...]:
    """Read a YAML override file and return the list of overrides.

    YAML shape:

      overrides:
        - section: corpus
          targets: [planner]
          content: |
            the OfficeQA harness over the U.S. Treasury Monthly Bulletins,
            spanning 1939–2025.
        - section: few_shots
          targets: [planner]
          content:
            - |
              ### Example
              Q: ...
              Plan: ```json {...} ```
            - |
              ### Example
              Q: ...
        - section: lessons
          targets: ["*"]
          content: |
            - watch out for FY vs CY conventions
            - Treasury Bulletin printed-page numbers are not PDF page numbers
    """
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
