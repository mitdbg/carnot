"""One-shot parent-chapter retriever over the flat concept tree.

A single LLM call picks one (or up to two) canonical chapter(s) from
the tree; every page under the picked chapter(s) is returned as the
prediction. Used by the production retrieve operator
(`src/skunk/retrieve.py`) and by `eval/eval_retrieve.py`.

The concept tree is the flat shape produced by `merge.build_tree`:

    {"chapters": {"<canonical>": {"n_pages", "description",
                                  "examples", "pages": [...]}}}

Telemetry: each retrieve call emits one `RetrieveTrace` containing one
`LevelTrace` for the LLM pick.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from skunk.common import LLMClient

from .schema import PageCatalogRow


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
    prompt_excerpt: str = ""              # first 800 chars of user prompt
    response_excerpt: str = ""            # first 800 chars of raw response


@dataclass
class RetrieveTrace:
    uid: str | None
    retrieve_idx: int
    concept: str
    period: str
    catalog_size: int
    candidate_count: int = 0
    levels: list[LevelTrace] = field(default_factory=list)
    top_k: list[dict[str, Any]] = field(default_factory=list)
    total_walk_s: float = 0.0
    # The chapter(s) the LLM picked. Set by `one_shot_parent_chapter_retrieve`
    # so downstream steps (e.g. BM25 rerank) can find the per-chapter index
    # without re-running the LLM. Empty list when the LLM picked nothing
    # valid; one or two chapter names otherwise.
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
    obj = safe_json_loads(text, context="retrieve_probe")
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
  {"picked": "<exact chapter label>"}

Rules:
  - Return exactly ONE chapter label, the best match.
  - Use the EXACT `chapter` value shown; do NOT emit anything from a
    `description` or `examples` list.
"""


def one_shot_parent_chapter_retrieve(
    tree: dict[str, Any],
    question: str,
    concept: str,
    period: str,
    llm: LLMClient,
    catalog_index: dict[tuple[str, int], PageCatalogRow],
    *,
    uid: str | None = None,
    retrieve_idx: int = 0,
) -> tuple[list[dict[str, Any]], RetrieveTrace]:
    """One LLM call → up to 2 canonical chapters → union of their pages.

    The LLM is asked for exactly one chapter, but the validator accepts a
    one-element list as well. Out-of-vocabulary picks are dropped; the
    function returns whatever survives.
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
    level_trace.prompt_excerpt = user[:800]

    t0 = time.monotonic()
    resp = llm.call(system=_PARENT_PICK_SYSTEM, user=user, temperature=0.0)
    level_trace.latency_s = time.monotonic() - t0
    level_trace.output_chars = len(resp.text)
    level_trace.input_tokens = resp.input_tokens
    level_trace.output_tokens = resp.output_tokens
    level_trace.response_excerpt = resp.text[:800]

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


# Back-compat alias used by `eval/eval_retrieve.py --retriever one-shot-section`.
one_shot_section_retrieve = one_shot_parent_chapter_retrieve


def write_trace_jsonl(trace: RetrieveTrace, path: Path) -> None:
    """Append one RetrieveTrace as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(trace), ensure_ascii=False))
        f.write("\n")
