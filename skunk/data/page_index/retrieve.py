"""Lightweight Treasury Bulletin page-index retriever.

Self-contained alongside `concept_tree.json` in this directory. Given
just the user's OfficeQA question, the module:

  0. **Mini-plan** — one LLM call extracts a list of `(concept, period)`
     retrieve branches from the question. Compound questions (multi-
     period comparisons, multi-series aggregations) fan out into
     multiple branches.
  1. **L1 chapter pick** (per branch) — one LLM call picks a canonical
     chapter using the chapter `description` + `examples` baked into
     the tree.
  2. **Period filter** (per branch) — bulletin-month window check: a
     page passes if its publication month is within
     `[period_start, period_end + 12 months]`. The +12mo lag captures
     retrospective tables that report on a closed period in a later
     issue.
  3. **Union** the surviving `(bulletin, page)` pairs across branches.

On the 32-UID dev sample, L1+period preserves the L1 recall ceiling
(**~98.6%**) at a mean of ~2,700 pages per branch (median 1,800; range
186–13,247). No vector index, no embeddings — pure LLM + symbolic
filter.

Usage
-----

```python
from pathlib import Path
from data.page_index_old.retrieve import RetrievalIndex

idx = RetrievalIndex.load(Path("data/page_index_old/concept_tree.json"))

# Caller supplies an LLM callable: (system_prompt, user_prompt) -> text
def my_llm(system: str, user: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content

# Single call: question in, candidate pages out.
pages = idx.retrieve("What was the OASI trust fund balance at end of CY1953?", my_llm)
# pages: list[tuple[str, int]]   e.g. [("1953-12", 35), ("1954-01", 37), ...]

# For callers that already have (concept, period) from their own planner:
pages = idx.retrieve_branch(question, concept, period, my_llm)
```

Dependencies: stdlib only.
"""

from __future__ import annotations

import calendar
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Period parser (DSL grammar from DSL.md, copied here so the shipped module
# stays self-contained against the shipped tree).
# ---------------------------------------------------------------------------

_POINT_RE = re.compile(
    r"^(?:"
    r"CY(?P<cy>\d{4})"
    r"|FY(?P<fy>\d{4})"
    r"|Q(?P<q>[1-4])-(?P<qy>\d{4})"
    r"|(?P<ymd>\d{4}-\d{2}-\d{2})"
    r"|(?P<ym>\d{4}-\d{2})"
    r"|(?P<y>\d{4})"
    r")$"
)


def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _parse_point(p: str) -> tuple[str, str]:
    m = _POINT_RE.match(p)
    if not m:
        raise ValueError(f"unrecognized period point: {p!r}")
    if m.group("cy"):
        y = int(m.group("cy"))
        return f"{y:04d}-01-01", f"{y:04d}-12-31"
    if m.group("fy"):
        y = int(m.group("fy"))
        # Pre-1977: FY ends June 30. 1977 onward: FY ends Sep 30.
        if y < 1977:
            return f"{y - 1:04d}-07-01", f"{y:04d}-06-30"
        return f"{y - 1:04d}-10-01", f"{y:04d}-09-30"
    if m.group("q"):
        q, y = int(m.group("q")), int(m.group("qy"))
        starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        sm, sd = starts[q]
        em, ed = ends[q]
        return f"{y:04d}-{sm:02d}-{sd:02d}", f"{y:04d}-{em:02d}-{ed:02d}"
    if m.group("ymd"):
        s = m.group("ymd")
        return s, s
    if m.group("ym"):
        s = m.group("ym")
        y, mo = int(s[:4]), int(s[5:7])
        return f"{y:04d}-{mo:02d}-01", f"{y:04d}-{mo:02d}-{_last_day(y, mo):02d}"
    y = int(m.group("y"))
    return f"{y:04d}-01-01", f"{y:04d}-12-31"


def period_to_intervals(period: str) -> list[tuple[str, str]]:
    """Parse a period string into a list of `(start_iso, end_iso)` intervals.

    Grammar:
      point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY
                    | YYYY"-"MM"-"DD | YYYY"-"MM | YYYY
      range       ::= point ".." point      (inclusive)
      enumeration ::= point ("," point)+
    """
    if ".." in period:
        a, b = period.split("..", 1)
        a_start, _ = _parse_point(a)
        _, b_end = _parse_point(b)
        if a_start > b_end:
            raise ValueError(f"range start > end: {period!r}")
        return [(a_start, b_end)]
    if "," in period:
        return [_parse_point(p.strip()) for p in period.split(",") if p.strip()]
    return [_parse_point(period)]


# ---------------------------------------------------------------------------
# L1 chapter-pick prompt
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """You decompose a U.S. Treasury Bulletin analyst question into one
or more retrieve branches. Each branch names a single data series and a
single time period; the downstream retriever runs each branch
independently and unions the results.

For each branch, emit:
  - `concept`: a short noun-phrase tag for the data series being asked
    about (snake_case or a short phrase). Examples:
    "outstanding_public_debt_by_holder",
    "unemployment_trust_fund_balance",
    "2-year Treasury note auction tenders",
    "national defense expenditures".
  - `period`: a period string in this grammar:
        point       ::= "CY"YYYY | "FY"YYYY | "Q"[1-4]"-"YYYY
                       | YYYY"-"MM"-"DD | YYYY"-"MM | YYYY
        range       ::= point ".." point   (inclusive)
        enumeration ::= point ("," point)+
    Examples: "CY1946", "FY1991", "1990-09", "FY1990..FY1995",
    "CY1991, CY1996".

Fan out one branch per (series, distinct period) pair the question
asks about. A "compare X between 1934 and 1946" question becomes two
branches; a "list top sources of revenue in FY1991" question is one
branch; a "compare debt vs deficit for FY1991" question is two
branches (one per concept).

Output a SINGLE JSON object (no prose, no markdown fences):
  {"branches": [
    {"concept": "<noun phrase>", "period": "<period string>"},
    ...
  ]}

Examples:

  Q: "What was the unemployment trust fund balance at end of CY1946?"
  → {"branches": [
       {"concept": "unemployment_trust_fund_balance", "period": "CY1946"}
     ]}

  Q: "Compare public-works spending in 1934 versus 1946."
  → {"branches": [
       {"concept": "public_works_expenditures", "period": "FY1934"},
       {"concept": "public_works_expenditures", "period": "FY1946"}
     ]}

  Q: "What was the FY1991 outstanding public debt and total interest paid?"
  → {"branches": [
       {"concept": "outstanding_public_debt", "period": "FY1991"},
       {"concept": "interest_paid_on_public_debt", "period": "FY1991"}
     ]}
"""


_PICK_SYSTEM = """You pick which Treasury Bulletin chapter most likely contains the answer
to the user's question.

You will see the question and a list of canonical chapters. Each entry has:
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

Output a SINGLE JSON object (no prose, no fences):
  {"picked": "<exact chapter label>"}

Rules:
  - Return exactly ONE chapter label, the best match.
  - Use the EXACT `chapter` value shown; do NOT emit anything from a
    `description` or `examples` list.
"""


# ---------------------------------------------------------------------------
# Bulletin-month period window
# ---------------------------------------------------------------------------

_PUBLISH_LAG_MONTHS = 12


def _add_months_to_iso(iso_end: str, months: int) -> str:
    """`'YYYY-MM-DD' + N months` → ISO end-of-window string."""
    y, m, d = int(iso_end[:4]), int(iso_end[5:7]), int(iso_end[8:10])
    total = y * 12 + (m - 1) + months
    ny, nm = divmod(total, 12)
    return f"{ny:04d}-{nm + 1:02d}-{d:02d}"


def _bulletin_in_period_window(
    bulletin: str, period_intervals: list[tuple[str, str]],
) -> bool:
    """True iff the bulletin's publication month falls within any period
    interval extended by `_PUBLISH_LAG_MONTHS` to capture retrospective
    tables (e.g., FY1991 data often appears in 1991-09 / 1992-03 issues)."""
    try:
        y, m = bulletin.split("-")
        b_iso = f"{int(y):04d}-{int(m):02d}-15"
    except (ValueError, AttributeError):
        return False
    for p_start, p_end in period_intervals:
        w_end = _add_months_to_iso(p_end, _PUBLISH_LAG_MONTHS)
        if not (b_iso < p_start or b_iso > w_end):
            return True
    return False


def _strip_code_fence(s: str) -> str:
    """LLMs sometimes wrap JSON in a ```json fence; strip it."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


# ---------------------------------------------------------------------------
# Retrieval index
# ---------------------------------------------------------------------------

LLMCallable = Callable[[str, str], str]
"""(system_prompt, user_prompt) -> response_text. Caller wires this to
their preferred LLM (OpenRouter, Gemini, OpenAI, etc.)."""


@dataclass
class RetrievalIndex:
    """In-memory view of the shipped tree. Use `RetrievalIndex.load(path)`."""
    chapters: dict[str, dict[str, Any]]   # name → {n_pages, description, examples, pages}
    _listing: list[dict[str, Any]]        # sorted listing for the L1 prompt
    _chapter_lower: dict[str, str]        # lowercased lookup for validating LLM picks

    @classmethod
    def load(cls, tree_path: Path) -> "RetrievalIndex":
        tree = json.loads(Path(tree_path).read_text())
        chapters = tree.get("chapters", {})
        if not chapters:
            raise ValueError(f"{tree_path} has no `chapters` key")
        listing = [
            {
                "chapter": name,
                "n_pages": data.get("n_pages", 0),
                "description": data.get("description", ""),
                "examples": data.get("examples", []),
            }
            for name, data in chapters.items()
        ]
        listing.sort(key=lambda x: -x["n_pages"])
        return cls(
            chapters=chapters,
            _listing=listing,
            _chapter_lower={name.lower(): name for name in chapters},
        )

    def pick_chapter(
        self, question: str, concept: str, period: str, llm: LLMCallable,
    ) -> str | None:
        """One LLM call → chosen chapter name (or None if the LLM emits
        something off-vocab)."""
        user = (
            f"Question: {question}\n\n"
            f"Concept: {concept}\n"
            f"Period:  {period}\n\n"
            f"Chapters ({len(self._listing)}):\n"
            f"{json.dumps(self._listing, ensure_ascii=False, indent=1)}\n"
        )
        try:
            resp = llm(_PICK_SYSTEM, user)
            obj = json.loads(_strip_code_fence(resp))
        except (json.JSONDecodeError, Exception):  # noqa: BLE001
            return None
        raw = obj.get("picked", "") if isinstance(obj, dict) else ""
        if isinstance(raw, list) and raw:
            raw = raw[0]
        return self._chapter_lower.get(str(raw).strip().lower())

    def retrieve_branch(
        self,
        question: str,
        concept: str,
        period: str,
        llm: LLMCallable,
    ) -> list[tuple[str, int]]:
        """L1 chapter pick + period mask → candidate `(bulletin, page)` pairs.

        Empty list if the L1 pick fails or the period is unparseable.
        Page order within the result mirrors the chapter's `pages` list
        in the tree (which is roughly chronological by bulletin).
        """
        picked = self.pick_chapter(question, concept, period, llm)
        if not picked:
            return []
        try:
            intervals = period_to_intervals(period) if period else []
        except ValueError:
            intervals = []
        out: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for p in self.chapters[picked].get("pages", []):
            b = p["bulletin"]
            if intervals and not _bulletin_in_period_window(b, intervals):
                continue
            key = (b, int(p["page"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def plan_branches(
        self, question: str, llm: LLMCallable,
    ) -> list[dict[str, str]]:
        """One LLM call → list of `{concept, period}` retrieve branches.

        Returns `[]` on parse failure or empty plan; caller can treat
        the question as having no retrievable structure (rare).
        """
        try:
            resp = llm(_PLAN_SYSTEM, f"Question: {question}")
            obj = json.loads(_strip_code_fence(resp))
        except (json.JSONDecodeError, Exception):  # noqa: BLE001
            return []
        raw = obj.get("branches", []) if isinstance(obj, dict) else []
        branches: list[dict[str, str]] = []
        if not isinstance(raw, list):
            return []
        for b in raw:
            if not isinstance(b, dict):
                continue
            concept = str(b.get("concept") or "").strip()
            period = str(b.get("period") or "").strip()
            if concept and period:
                branches.append({"concept": concept, "period": period})
        return branches

    def retrieve(
        self,
        question: str,
        llm: LLMCallable,
        *,
        max_workers: int = 4,
    ) -> list[tuple[str, int]]:
        """End-to-end: question → mini-plan → per-branch retrieve → union.

        One LLM call to extract the retrieve branches, then one LLM call
        per branch for the chapter pick (run in parallel up to
        `max_workers`). Periods are filtered deterministically. Returns
        the union of `(bulletin, page)` candidates across branches,
        sorted by bulletin then page.
        """
        branches = self.plan_branches(question, llm)
        if not branches:
            return []
        pages: set[tuple[str, int]] = set()
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = [
                ex.submit(self.retrieve_branch,
                          question, b["concept"], b["period"], llm)
                for b in branches
            ]
            for f in futures:
                try:
                    pages.update(f.result())
                except Exception:  # noqa: BLE001
                    continue
        return sorted(pages)
