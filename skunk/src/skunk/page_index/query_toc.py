"""Query path — ToC chapter pick over the flat concept tree.

The first pass of the page-index query path
(`ToC pick → year filter → semantic filter → candidate set`). A single
LLM call picks up to two canonical chapter(s) from the tree; every page
under the picked chapter(s) becomes a candidate. Used by
`page_index.query.PageIndexRetriever`.

The concept tree is the flat shape produced by `merge.build_tree`:

    {"chapters": {"<canonical>": {"n_pages", "description",
                                  "examples", "pages": [...]}}}

Telemetry: each retrieve call emits one `RetrieveTrace` containing one
`LevelTrace` for the LLM pick.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skunk.common import LLMClient

from .schema import PageCatalogRow


# Probe-trace excerpt cap. Long enough for a human reviewer to read the
# prompt/response intent without flooding the trace.
_EXCERPT_MAX_LEN = 800


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class LevelTrace:
    """One LLM call's worth of telemetry."""
    level: str                            # "parent_pick"
    input_count: int                      # number of candidate items shown to the LLM
    input_chars: int                      # length of the user message
    input_tokens: int | None = None
    output_chars: int = 0
    output_tokens: int | None = None
    latency_s: float = 0.0
    output_count: int = 0
    prompt_excerpt: str = ""              # truncated to _EXCERPT_MAX_LEN
    response_excerpt: str = ""            # truncated to _EXCERPT_MAX_LEN


@dataclass
class RetrieveTrace:
    uid: str | None
    retrieve_idx: int
    concept: str
    period: str | None
    catalog_size: int
    candidate_count: int = 0
    levels: list[LevelTrace] = field(default_factory=list)
    top_k: list[dict[str, Any]] = field(default_factory=list)
    total_walk_s: float = 0.0
    # The chapter(s) the LLM picked. Set by `one_shot_parent_chapter_retrieve`
    # so downstream steps can attribute candidates without re-running the
    # LLM. Empty list when the LLM picked nothing valid; one or two chapter
    # names otherwise.
    picked_chapters: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_catalog(catalog_dir: Path) -> list[PageCatalogRow]:
    """Load every per-bulletin catalog `.jsonl` from `catalog_dir`."""
    rows: list[PageCatalogRow] = []
    for f in sorted(catalog_dir.glob("*.jsonl")):
        for line in f.open():
            rows.append(PageCatalogRow.from_json(line))
    return rows


def load_concept_tree(path: Path) -> dict[str, Any]:
    """Load the flat concept tree produced by `merge.build_tree`."""
    return json.loads(Path(path).read_text())


def _safe_json(text: str) -> dict[str, Any] | None:
    """Parse JSON object from a (possibly fenced) LLM response."""
    from .util import safe_json_loads
    obj = safe_json_loads(text, context="query_toc")
    return obj if isinstance(obj, dict) else None


def _chapter_pages(tree: dict[str, Any], chapter: str) -> list[tuple[str, int]]:
    """Deduped (bulletin, page) list for one chapter in the flat tree."""
    data = tree.get("chapters", {}).get(chapter)
    if not data:
        return []
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, int]] = []
    for p in data.get("pages", []):
        k = (p["bulletin"], p["page"])
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

_PARENT_PICK_SYSTEM = """You pick which Treasury Bulletin chapter most
likely contains the answer to the user's question.

## Input

The user message is a single JSON object:

  {"question": "<natural-language question>",
   "concept":  "<concept tag>",
   "period":   "<period>",
   "chapters": [
     {"chapter":     "<canonical chapter name>",
      "n_pages":     <int>,
      "description": "<scope statement>",
      "examples":    ["<sub-area name>", ...]},
     ...
   ]}

Each `chapters[]` entry has:
  - `chapter`: the canonical chapter name (your output MUST be this)
  - `n_pages`: total pages in the chapter
  - `description`: a scope statement of what the chapter covers
  - `examples`: concrete sub-area / topic names that live in this
    chapter. These are string-match anchors for questions that
    reference specific eras, programs, or table topics by name.

Use BOTH `description` and `examples` to match the question. The
description gives the chapter's abstract framing; the examples surface
specific sub-areas (e.g. era-specific programs, named tables) that the
description may not mention.

## Output

Output a SINGLE JSON object (no prose, no fences):
  {"picked": ["<exact chapter label>", ...]}

Rules:
  - Return the best-matching chapter. Return a SECOND chapter only
    when the question genuinely straddles two chapters and you are
    unsure which holds the answer — at most TWO labels, best first.
    A precise downstream filter reads every page you return, so a
    spurious second chapter only adds cost. When in doubt, return one.
  - Use the EXACT `chapter` value(s) shown; do NOT emit anything from a
    `description` or `examples` list.
  - A bare string (one label) is also accepted.
"""


async def one_shot_parent_chapter_retrieve(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str | None,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """One LLM call → up to 2 canonical chapters → union of their pages.

    Accepting both a bare string and a one-element list absorbs LLM
    output variance without re-prompting. Out-of-vocabulary picks are
    dropped silently; whatever survives is returned.
    """
    chapters_data = tree.get("chapters", {})
    trace = RetrieveTrace(
        uid=uid, retrieve_idx=retrieve_idx,
        concept=concept, period=period,
        catalog_size=sum(c.get("n_pages", 0) for c in chapters_data.values()),
    )
    t_walk = time.monotonic()

    listing: list[dict[str, Any]] = [
        {
            "chapter": chapter,
            "n_pages": data.get("n_pages", 0),
            "description": data.get("description", ""),
            "examples": data.get("examples", []),
        }
        for chapter, data in chapters_data.items()
    ]
    listing.sort(key=lambda x: -x["n_pages"])

    level_trace = LevelTrace(level="parent_pick",
                             input_count=len(listing), input_chars=0)
    user = json.dumps(
        {"question": question, "concept": concept,
         "period": period, "chapters": listing},
        ensure_ascii=False, indent=1,
    )
    level_trace.input_chars = len(user)
    level_trace.prompt_excerpt = user[:_EXCERPT_MAX_LEN]

    t0 = time.monotonic()
    resp = await llm.acall(system=_PARENT_PICK_SYSTEM, user=user, temperature=0.0)
    level_trace.latency_s = time.monotonic() - t0
    level_trace.output_chars = len(resp.text)
    level_trace.input_tokens = resp.input_tokens
    level_trace.output_tokens = resp.output_tokens
    level_trace.response_excerpt = resp.text[:_EXCERPT_MAX_LEN]

    obj = _safe_json(resp.text) or {}
    raw_picked = obj.get("picked", "")
    if isinstance(raw_picked, str):
        raw_picks = [raw_picked]
    elif isinstance(raw_picked, list):
        raw_picks = [str(x).strip() for x in raw_picked if str(x).strip()]
    else:
        raw_picks = []
    valid = {c.lower(): c for c in chapters_data}
    picked_chapters: list[str] = []
    seen_picks: set[str] = set()
    for raw in raw_picks[:2]:
        ch = valid.get(raw.strip().lower())
        if ch and ch not in seen_picks:
            picked_chapters.append(ch)
            seen_picks.add(ch)
    trace.levels.append(level_trace)

    trace.picked_chapters = list(picked_chapters)
    if not picked_chapters:
        trace.total_walk_s = time.monotonic() - t_walk
        return [], trace

    seen_pages: set[tuple[str, int]] = set()
    top: list[dict[str, Any]] = []
    for chapter in picked_chapters:
        for (b, p) in _chapter_pages(tree, chapter):
            if (b, p) in seen_pages:
                continue
            seen_pages.add((b, p))
            top.append({"bulletin": b, "page": p,
                        "reason": f"chapter={chapter}"})
    trace.candidate_count = len(top)
    level_trace.output_count = len(picked_chapters)
    trace.top_k = top[:50]
    trace.total_walk_s = time.monotonic() - t_walk
    return top, trace
