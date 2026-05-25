"""Prompt overrides plumbing.

Each executor owns its core SYSTEM prompt (a Jinja template) in its own
module. Three named sections of supplementary content flow in from override
entries on the HarnessContext:

  - corpus     — dataset-specific background paragraphs.
  - few_shots  — list of pre-formatted example strings; rendered verbatim.
                 The YAML author controls the formatting.
  - lessons    — short bullet strings about gotchas / things to watch for.

A `PromptOverride` is keyed by `(section, targets)` and carries either a
text blob (corpus / lessons) or a tuple of example strings (few_shots).
The targets field is a tuple of agent names, with the special wildcard
"*" meaning "applies to every agent". Multiple overrides for the same
section + agent concatenate in declaration order.

The `gather_*` helpers below return per-call-site values. `PromptedCall`
exposes them in template scope as `corpus`, `few_shots`, `lessons` (plus
a pre-rendered `default_tail` that bundles all three under the
conventional `## Dataset` / `## Few-shot examples` / `## Lessons learned`
headers). Subclass templates interpolate these wherever they want — this
module no longer hard-codes section layout; see `skunk.prompted_call`.

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
