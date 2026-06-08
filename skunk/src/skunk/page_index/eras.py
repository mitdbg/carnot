"""Build the era-keyed concept tree — timeline segmentation + per-era chapter finalization.

Treasury reporting conventions stay stable for long stretches, then shift, so a single global
chapter tree would conflate era-variants. This module segments the corpus timeline into ERAS
within which the ToC schema is stable, then builds one canonical table of contents per era — so
the query picker can scope chapters to a question's period. `build_concept_tree` runs both
phases and returns the `concept_tree.json` dict.

Segmentation (`segment_corpus`) — pure-LLM, chunk-and-merge (chunk size is just a context
batch, not a claim about era length):
  1. `segment_chunk` (map) — split the bulletins into batches of `_CHUNK_BULLETINS`; each batch
     is ONE LLM call that partitions it into contiguous segments where the schema shifts.
  2. `_reduce` (recursive merge) — pairwise-combine adjacent batches (32→64→128…); one
     `era_merge` LLM call per blind batch seam decides whether the two sides are one era.

Finalization (`_finalize_era`) — one `chapter_finalize` LLM call per era reconciles its raw
headings into canonical chapters (with deduped secondary chapters as picker metadata); plain
Python then files each placed page under its chapter as page ranges.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError, model_validator

from skunk.common import ExecutionContext, strip_code_fence
from skunk.errors import ParseError
from skunk.prompted_call import PromptedCall

from .toc_index import TOC_CHAPTER_FIELDS

_WS = re.compile(r"\s+")


def _norm_key(name: str) -> str:
    """Lowercased, whitespace-collapsed, punctuation-stripped grouping key — the unit the
    digest dedups on and the finalize maps raw headings to canonical chapters with."""
    return _WS.sub(" ", name.strip().lower()).strip(" .-")

# Bulletins per segmentation call — a context-window batch (≈32 issues' chapter-name sets
# fit comfortably in one call), NOT an assumption about how long an era lasts.
_CHUNK_BULLETINS = 32


class EraSpan(BaseModel):
    """One era: an inclusive `YYYY-MM` span and a short label. Spans returned by
    `segment_corpus` are contiguous and gapless across the corpus month range."""
    start: str   # "YYYY-MM" inclusive
    end: str     # "YYYY-MM" inclusive
    label: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> "EraSpan":
        if self.start > self.end:
            raise ValueError(f"era start after end: {self.start!r}..{self.end!r}")
        return self


@dataclass
class _Seg:
    """A contiguous run of bulletins sharing one schema. `counts` maps each top-level chapter
    to the number of the segment's issues it heads; `child_names` are the names seen as some
    chapter's sub-section (children) anywhere in the run. `core` — the schema the seam-merge
    compares — is the recurring top-level chapters with those sub-sections removed, so a
    sub-section that occasionally leaks to the top level never counts as a schema entry."""
    bulletins: list[str]       # member bulletin ids, in date order
    label: str
    counts: dict[str, int]     # top-level chapter -> number of the segment's issues it heads
    child_names: set[str]      # names that appear as a sub-section (child) somewhere in the run
    n: int                     # number of issues in the segment

    @property
    def start(self) -> str:
        return self.bulletins[0]

    @property
    def end(self) -> str:
        return self.bulletins[-1]

    @property
    def core(self) -> list[str]:
        """The recurring schema — top-level chapters in at least half the issues, minus any
        name that is really a sub-section (appears as a child elsewhere)."""
        return sorted(c for c, k in self.counts.items()
                      if 2 * k >= self.n and c not in self.child_names)


# ---------------------------------------------------------------------------
# Per-bulletin digest (no LLM)
# ---------------------------------------------------------------------------

def _bulletin_digest(gather: dict) -> list[dict]:
    """Corpus gather → each issue's table of contents, oldest first:
    `{"bulletin", "chapters": [{"name", "children": [...]}]}`. Names (and children) are
    normalized with `_norm_key` (the same key the merge clusters on) and de-duplicated, so
    surface noise — a trailing period, capitalization, whitespace — never looks like a schema
    change. Carrying `children` lets the segmenter treat sub-sections as secondary explicitly
    (a name that is some chapter's child is a sub-section, not a top-level schema entry)."""
    out: list[dict] = []
    for b in sorted(gather):
        seen: set[str] = set()
        chapters: list[dict] = []
        for c in gather[b].get("chapters", []):
            name = _norm_key(c["name"])
            if not name or name in seen:
                continue
            seen.add(name)
            children = list(dict.fromkeys(k for ch in c.get("children", []) if (k := _norm_key(ch))))
            chapters.append({"name": name, "children": children})
        out.append({"bulletin": b, "chapters": chapters})
    return out


def _prev_month(m: str) -> str:
    """`YYYY-MM` one month earlier (for gapless era spans)."""
    y, mo = int(m[:4]), int(m[5:7])
    return f"{y - 1:04d}-12" if mo == 1 else f"{y:04d}-{mo - 1:02d}"


def _to_eras(segs: list[_Seg]) -> list[EraSpan]:
    """Final segments → gapless `EraSpan` list: each era runs to the month before the next
    era begins (the last era to its final bulletin month)."""
    spans: list[EraSpan] = []
    for i, s in enumerate(segs):
        end = _prev_month(segs[i + 1].start) if i + 1 < len(segs) else s.end
        spans.append(EraSpan(start=s.start, end=max(end, s.start), label=s.label))
    return spans


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

class _ChunkSeg(BaseModel):
    start: str        # bulletin YYYY-MM where this segment begins
    label: str = ""


class _ChunkResult(BaseModel):
    segments: list[_ChunkSeg] = Field(default_factory=list)


class _MergeDecision(BaseModel):
    merge: bool
    label: str = ""   # label for the combined era when merge is true


def _json_parser(model):
    def parse(raw: str, _ctx: ExecutionContext):
        try:
            return model.model_validate_json(strip_code_fence(raw).strip())
        except ValidationError as e:
            raise ParseError(raw, str(e)) from e
    return parse


_SEGMENT_SYSTEM = (
    """\
You are given the table of contents of consecutive issues of a periodical in date order, each a
list of chapters.

"""
    + TOC_CHAPTER_FIELDS + """

Each issue here lists its top-level chapters with their `children`; the page labels are omitted.

You are dividing the issues into eras. Each era share ONE schema for top-level table of contents that a reader
can use to unambiguously pinpoint the chapter that holds a queried table or figure. Two issues fit the same
table of contents schema when their top-level chapters line up once you reconcile OCR errors and spelling variants,
wording and synonyms. If a chapter is renamed, or a subsection is promoted to the top level, you may reconcile them
using the most inclusive chapter title. Judge from top-level chapters only: a chapter's
`children`, and any name that appears as some chapter's child, are sub-sections can be used as a reference but should not
participate in the schema.

Start a new era at the issue where the table of contents can no longer fit the running era's ToC
without that kind of ambiguity — the material has been reorganized so a single ToC can no longer
route each question to one chapter. Consecutive issues usually fit, so most batches are a single
era.

## Output

{"segments": [{"start": "<YYYY-MM>", "label": "<short label naming the era's top-level chapters>"}, ...]}

The first segment starts at the batch's first issue; date order.
"""
)

_MERGE_SYSTEM = """\
You are building the table of contents a reader uses to find answers: shown one era's top-level
chapters, the reader must unambiguously pinpoint the single chapter that holds a queried table or
figure. If a concept were split across two chapters, or one chapter blurred two concepts, the
reader could choose wrong and miss the page — a permanent loss.

Two adjacent eras were separated only because they fell in different processing batches. You get
each side's month span, label, and top-level chapters.

Merge them when both fit ONE top-level table-of-contents schema: their chapters line up once you
reconcile OCR errors and spelling variants, wording and synonyms. If a chapter is renamed, or a
sub-section is promoted to the top level, reconcile them using the most inclusive chapter title. A
chapter present on only one side is fine — it simply becomes a chapter of the combined ToC.

Keep them separate only when the two organize the material so differently that no single ToC can
serve both: one side's chapter maps to several of the other's with no clean correspondence, so a
query could not be pinpointed to one unambiguous chapter.

## Output

{"merge": <bool>, "label": "<combined era label — required when merge is true>"}
"""

_segment_call: PromptedCall[_ChunkResult] = PromptedCall(
    name="era_segment",
    system_prompt=_SEGMENT_SYSTEM,
    parse=_json_parser(_ChunkResult),
    output_instruction='Output a single bare JSON object {"segments": [...]} — no fences, no prose.',
)
_merge_call: PromptedCall[_MergeDecision] = PromptedCall(
    name="era_merge",
    system_prompt=_MERGE_SYSTEM,
    parse=_json_parser(_MergeDecision),
    output_instruction='Output a single bare JSON object {"merge": ..., "label": ...} — no fences, no prose.',
)


def _tally(tocs: list[list[dict]]) -> tuple[dict[str, int], set[str]]:
    """Over a run of issues' tables of contents → ({top-level chapter -> # issues it heads},
    {names seen as a sub-section}). A name is counted top-level per issue it heads, and added
    to the child set wherever it appears under a parent."""
    counts: dict[str, int] = {}
    child_names: set[str] = set()
    for toc in tocs:
        for c in toc:
            counts[c["name"]] = counts.get(c["name"], 0) + 1
            child_names.update(c.get("children", []))
    return counts, child_names


async def segment_chunk(ctx: ExecutionContext, chunk: list[dict]) -> list[_Seg]:
    """One `era_segment` call → the batch's contiguous segments. Segment starts are
    snapped to actual batch issues (the first is forced to the batch's first issue), and
    each segment tallies its top-level chapters and sub-sections (for its `core` schema)."""
    res = await _segment_call.call(ctx, json.dumps({"issues": chunk}, ensure_ascii=False, indent=1))
    ids = [d["bulletin"] for d in chunk]
    toc = {d["bulletin"]: d["chapters"] for d in chunk}
    labels = {s.start: s.label for s in res.segments if s.start in toc}
    starts = sorted({ids[0]} | set(labels))
    segs: list[_Seg] = []
    for i, st in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        members = [b for b in ids if st <= b and (nxt is None or b < nxt)]
        counts, child_names = _tally([toc[b] for b in members])
        segs.append(_Seg(bulletins=members, label=labels.get(st, ""),
                         counts=counts, child_names=child_names, n=len(members)))
    return segs


async def _combine(ctx: ExecutionContext, left: list[_Seg], right: list[_Seg]) -> list[_Seg]:
    """Concatenate two adjacent segment lists, reconciling the one blind adjacency (the
    seam between `left[-1]` and `right[0]`) with a single `era_merge` call that compares the
    two segments' recurring `core` schemas. Within-batch boundaries are trusted (the
    detection call only splits on persistent shifts)."""
    a, b = left[-1], right[0]
    user = json.dumps(
        {"left": {"span": [a.start, a.end], "label": a.label, "chapters": a.core},
         "right": {"span": [b.start, b.end], "label": b.label, "chapters": b.core}},
        ensure_ascii=False, indent=1,
    )
    dec = await _merge_call.call(ctx, user)
    if dec.merge:
        counts = dict(a.counts)
        for c, k in b.counts.items():
            counts[c] = counts.get(c, 0) + k
        merged = _Seg(bulletins=a.bulletins + b.bulletins, label=dec.label or a.label,
                      counts=counts, child_names=a.child_names | b.child_names, n=a.n + b.n)
        return left[:-1] + [merged] + right[1:]
    return left + right


async def _reduce(
    ctx: ExecutionContext, blocks: list[list[_Seg]], sem: asyncio.Semaphore | None,
) -> list[_Seg]:
    """Recursively pairwise-combine adjacent blocks (32→64→128…), reconciling each blind
    batch seam, until one segment list remains. Combines at a level are independent → run
    in parallel, bounded by `sem`."""
    async def _guarded(left: list[_Seg], right: list[_Seg]) -> list[_Seg]:
        async with (sem or nullcontext()):
            return await _combine(ctx, left, right)

    while len(blocks) > 1:
        tasks = [_guarded(blocks[i], blocks[i + 1]) for i in range(0, len(blocks) - 1, 2)]
        combined = await asyncio.gather(*tasks)
        nxt = list(combined)
        if len(blocks) % 2 == 1:
            nxt.append(blocks[-1])
        blocks = nxt
    return blocks[0] if blocks else []


async def segment_corpus(
    ctx: ExecutionContext, gather: dict, *, sem: asyncio.Semaphore | None = None,
) -> list[EraSpan]:
    """Full segmentation: chunk the bulletins into batches of `_CHUNK_BULLETINS`, segment
    each batch (parallel map), recursively merge across the blind batch seams → a gapless
    `EraSpan` partition of the corpus. LLM calls are bounded by `sem` (the build's shared
    concurrency limit). Empty gather → no eras."""
    digest = _bulletin_digest(gather)
    if not digest:
        return []
    chunks = [digest[i:i + _CHUNK_BULLETINS] for i in range(0, len(digest), _CHUNK_BULLETINS)]

    async def _seg(chunk: list[dict]) -> list[_Seg]:
        async with (sem or nullcontext()):
            return await segment_chunk(ctx, chunk)

    blocks = await asyncio.gather(*[_seg(c) for c in chunks])
    segs = await _reduce(ctx, list(blocks), sem)
    return _to_eras(segs)


# ---------------------------------------------------------------------------
# Per-era finalization — one canonical table of contents per era
# ---------------------------------------------------------------------------

def partition_gather_by_era(gather: dict, eras: list[EraSpan]) -> list[tuple[EraSpan, dict]]:
    """Bucket the corpus gather into one sub-gather per era, by each bulletin's `YYYY-MM`
    falling within `[era.start, era.end]` (inclusive). Returns `[(era, sub_gather), ...]` in
    era order; bulletins outside every span are dropped (the eras cover the corpus)."""
    buckets: list[tuple[EraSpan, dict]] = [(e, {}) for e in eras]
    for bulletin, data in gather.items():
        for era, sub in buckets:
            if era.start <= bulletin <= era.end:
                sub[bulletin] = data
                break
    return buckets


def _aggregate(gather: dict) -> dict:
    """An era's distinct top-level chapter headings → finalize stats:
    `{norm_key: {display, n_pages, n_issues, children:set, listed_as_child_of:[...]}}`."""
    child_parents: dict[str, set[str]] = {}
    for data in gather.values():
        for c in data["chapters"]:
            for child in c["children"]:
                child_parents.setdefault(_norm_key(child), set()).add(c["name"])
    agg: dict[str, dict] = {}
    for data in gather.values():
        for c in data["chapters"]:
            k = _norm_key(c["name"])
            e = agg.setdefault(
                k, {"display": c["name"], "n_pages": 0, "n_issues": 0, "children": set()})
            e["n_pages"] += c["n_pages"]
            e["n_issues"] += 1
            e["children"].update(c["children"])
    for k, e in agg.items():
        e["listed_as_child_of"] = sorted(child_parents.get(k, set()) - {e["display"]})
    return agg


def _contiguous_ranges(pages: list[int]) -> list[tuple[int, int]]:
    """Sorted page numbers → list of inclusive `(start, end)` runs of consecutive pages."""
    ranges: list[tuple[int, int]] = []
    for p in pages:
        if ranges and p == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], p)
        else:
            ranges.append((p, p))
    return ranges


class _FinalChapter(BaseModel):
    name: str                                          # canonical top-level chapter name
    members: list[str] = Field(default_factory=list)   # raw headings folded into it
    secondary: list[str] = Field(default_factory=list)  # deduped sub-sections under it


class _FinalizeResult(BaseModel):
    chapters: list[_FinalChapter] = Field(default_factory=list)


_FINALIZE_SYSTEM = """\
You are building the table of contents a reader uses to find answers within ONE era of a
periodical — a span of issues that share one reporting structure.

You get the era's raw top-level chapter headings observed across its issues, each with its total
page count, how many issues it appeared in, the sub-section headings seen beneath it, and any
chapters that list IT as a sub-section elsewhere ("listed_as_child_of").

Produce the era's canonical table of contents: group the raw headings into the real top-level
CHAPTERS — one clearly-named bucket per chapter — so a reader can pinpoint the single chapter
that holds a queried table or figure.

Reconcile into ONE chapter: OCR and spelling variants, wording and synonyms, a renamed chapter,
and a heading that is really a sub-section of another (fold it under its parent). Name each
chapter with the most inclusive, standard wording among its members. Do NOT create form-based
catch-alls ("Special Reports", "Miscellaneous", "Other", "Appendix", "Glossary") — route their
members into the topical chapter they belong to.

For each canonical chapter also return its SECONDARY chapters — the deduped sub-section headings
under it (reconcile OCR/wording there too) — as metadata a reader scans to confirm the chapter
holds a topic.

## Output

{"chapters": [{"name": "<canonical>", "members": ["<raw heading>", ...], "secondary": ["<sub-section>", ...]}, ...]}

Every input heading appears in exactly one chapter's `members`.
"""


def _parse_finalize(raw: str, _ctx: ExecutionContext) -> _FinalizeResult:
    try:
        return _FinalizeResult.model_validate_json(strip_code_fence(raw).strip())
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e


_finalize_call: PromptedCall[_FinalizeResult] = PromptedCall(
    name="chapter_finalize",
    system_prompt=_FINALIZE_SYSTEM,
    parse=_parse_finalize,
    output_instruction='Output a single bare JSON object {"chapters": [...]} — no fences, no prose.',
)


async def _finalize_era(
    ctx: ExecutionContext, gather: dict, *, sem: asyncio.Semaphore | None
) -> dict:
    """One `chapter_finalize` call → the era's canonical chapters, then file every placed page
    under its chapter as page ranges. Returns the on-disk `{name: Chapter}` dict for the era:
    `{n_pages, pages (ranges)}` — `description`/`examples` are added afterward by the describe
    pass."""
    agg = _aggregate(gather)
    if not agg:
        return {}
    payload = [{"name": e["display"], "n_pages": e["n_pages"], "n_issues": e["n_issues"],
                "children": sorted(e["children"])[:10], "listed_as_child_of": e["listed_as_child_of"]}
               for e in sorted(agg.values(), key=lambda e: -e["n_pages"])]
    async with (sem or nullcontext()):
        res = await _finalize_call.call(ctx, json.dumps({"chapters": payload}, ensure_ascii=False, indent=1))

    # `ch.secondary` (sub-section children) is a build-time signal that helps the LLM fold
    # sub-sections into their parent here; it is NOT persisted — the describe pass populates the
    # picker-facing description/examples from sampled table titles instead.
    raw_to_canon: dict[str, str] = {}
    for ch in res.chapters:
        for m in ch.members:
            raw_to_canon[_norm_key(m)] = ch.name
    for e in agg.values():               # any heading the LLM omitted → its own chapter
        raw_to_canon.setdefault(_norm_key(e["display"]), e["display"])

    # File each placed page under its canonical chapter as page ranges.
    pages: dict[str, dict[str, set[int]]] = {}
    for bulletin, data in gather.items():
        for page_str, raw in data["assign"].items():
            canon = raw_to_canon.get(_norm_key(raw), raw)
            pages.setdefault(canon, {}).setdefault(bulletin, set()).add(int(page_str))

    chapters: dict[str, dict] = {}
    for canon, by_bulletin in pages.items():
        ranges = [{"bulletin": b, "start": s, "end": e}
                  for b in sorted(by_bulletin)
                  for s, e in _contiguous_ranges(sorted(by_bulletin[b]))]
        chapters[canon] = {
            "n_pages": sum(len(p) for p in by_bulletin.values()),
            "pages": ranges,
        }
    return chapters


# ── Pass 4: describe (picker-facing scope blurb) ───────────────────────────────
# Modeled on the pre-refactor merger's describe pass, but grounded in the chapter's ACTUAL
# content: one LLM call per era summarizes a sample of the real table titles on each chapter's
# pages into a discriminative `description` + an `examples` list — the string-match anchors the
# query picker matches a question against (the old flat tree got 95%+ ToC recall off exactly
# these). Children/members are not used here — table titles describe a chapter uniformly,
# including the many eras whose printed ToC carried no sub-sections at all.

# Most-frequent table titles fed to the describe LLM per chapter — the recurring monthly tables
# rank first; one-off OCR-garble titles fall below the cut.
_DESCRIBE_TITLE_CAP = 40


class _DescribedChapter(BaseModel):
    description: str = ""
    examples: list[str] = Field(default_factory=list)


class _DescribeResult(BaseModel):
    chapters: dict[str, _DescribedChapter] = Field(default_factory=dict)


_DESCRIBE_SYSTEM = """\
You write a scope description and a short example list for each U.S. Treasury Bulletin canonical
chapter, within ONE era of the periodical.

You receive each chapter's name and a sample of the ACTUAL TABLE TITLES that appear on its pages
across the era's issues (most frequent first). For each chapter produce:
  1. `description` — concrete prose stating what the chapter covers, summarizing the kinds of data
     its tables report. Stop when another phrase wouldn't help a retriever tell this chapter from
     the others (a sentence is often enough; chapters with many sub-areas may need 2-3).
  2. `examples` — 4-10 concrete sub-area / topic names drawn from the table titles that: cover the
     chapter's distinct sub-areas (not 5 variants of one topic); stay short — the topic only, with
     any leading table number ("Table 3.- ") and trailing "(Continued)" stripped (e.g. "Receipts
     by Principal Sources", not "Table 1.- Receipts by Principal Sources"); prefer era- or
     program-specific entries. These are the retriever's string-match anchors for topic vocabulary.

Both fields are shown to a downstream retriever LLM that picks the one chapter holding a queried
table — keep them complementary, not redundant.

## Output

A single bare JSON object, no prose or fences:
  {"chapters": {"<canonical>": {"description": "<scope>", "examples": ["<sub-area>", ...]}, ...}}

Every input chapter appears as a key, using its EXACT name.
"""


def _parse_describe(raw: str, _ctx: ExecutionContext) -> _DescribeResult:
    try:
        return _DescribeResult.model_validate_json(strip_code_fence(raw).strip())
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e


_describe_call: PromptedCall[_DescribeResult] = PromptedCall(
    name="chapter_describe",
    system_prompt=_DESCRIBE_SYSTEM,
    parse=_parse_describe,
    output_instruction='Output a single bare JSON object {"chapters": {...}} — no fences, no prose.',
)


def _chapter_titles(chapter: dict, titles_by_page: dict[tuple[str, int], list[str]]) -> list[str]:
    """The chapter's most-frequent table titles (top `_DESCRIBE_TITLE_CAP`), gathered from the
    catalog titles on every page in its ranges. Frequency-ranked so recurring monthly tables lead
    and one-off OCR garble falls off."""
    cnt: Counter[str] = Counter()
    for pr in chapter["pages"]:
        for pg in range(pr["start"], pr["end"] + 1):
            for t in titles_by_page.get((pr["bulletin"], pg), ()):
                if (t := t.strip()):
                    cnt[t] += 1
    return [t for t, _ in cnt.most_common(_DESCRIBE_TITLE_CAP)]


async def _describe_era(
    ctx: ExecutionContext, chapters: dict,
    titles_by_page: dict[tuple[str, int], list[str]], *, sem: asyncio.Semaphore | None,
) -> dict:
    """Pass 4 — one `chapter_describe` call fills `description` + `examples` on each of the era's
    chapters, summarizing a sample of the real table titles on its pages. In-place on the
    `_finalize_era` dict; a chapter the LLM omits keeps its empty defaults (the picker still has
    its name + n_pages)."""
    if not chapters:
        return chapters
    payload = [{"chapter": name, "table_titles": _chapter_titles(c, titles_by_page)}
               for name, c in chapters.items()]
    async with (sem or nullcontext()):
        res = await _describe_call.call(
            ctx, json.dumps({"chapters": payload}, ensure_ascii=False, indent=1))
    for name, c in chapters.items():
        d = res.chapters.get(name)
        c["description"] = d.description if d else ""
        c["examples"] = list(d.examples) if d else []
    return chapters


async def build_concept_tree(
    ctx: ExecutionContext, gather: dict,
    titles_by_page: dict[tuple[str, int], list[str]], *, sem: asyncio.Semaphore | None = None,
) -> dict:
    """Segment the corpus into eras, then build each era's canonical table of contents from its
    chapter headings (one `chapter_finalize` call per era, run in parallel), and describe each
    chapter from a sample of its pages' table titles (one `chapter_describe` call per era) → the
    era-keyed `concept_tree.json` dict, validatable with `data_model.ConceptTree`."""
    eras = await segment_corpus(ctx, gather, sem=sem)

    async def _one(era: EraSpan, sub: dict) -> dict:
        chapters = await _finalize_era(ctx, sub, sem=sem) if sub else {}
        chapters = await _describe_era(ctx, chapters, titles_by_page, sem=sem)
        return {"span": [era.start, era.end], "label": era.label, "chapters": chapters}

    out = await asyncio.gather(
        *[_one(era, sub) for era, sub in partition_gather_by_era(gather, eras)])
    return {"eras": list(out)}
