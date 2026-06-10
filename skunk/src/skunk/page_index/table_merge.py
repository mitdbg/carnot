"""Build pass: cross-page TABLE merge, keyed on the empty-labels signal.

A table block with NO column labels and NO row labels is, in this corpus, a genuine
continuation — a fragment whose headers live on an earlier page (modern "PDO-1
(Continued)" pages) or a "Footnotes to Table X" spillover. Everything else, including
the thousands of "(Continued)"-titled pages, restates its own labels and must NOT merge
(the retired page-level merge pass failed exactly by trusting captions). The signal is
sparse by construction: ~49 of 82K table blocks corpus-wide.

For each candidate, one cheap LLM call (the run's default flash model, thinking off)
decides which earlier block it belongs to — resilient to OCR-mangled titles and odd
layouts where a deterministic title match would misfire. The model sees the fragment
page's actual text plus the recent blocks' metadata, and may answer null (leave it
standalone). Application is annotation-only and lossless: the parent block gains the
fragment's page in `extra_pages` (so retrieval expands a selected block to every page of
the table), the fragment block records `merged_into`; nothing is deleted from the scans.

Driven per bulletin by the pipeline's `table_merge` stage. Requires per-page scan rows —
bulletins still carrying legacy page-level `continuation_pages` are skipped until
un-merged (`scripts/unmerge_rescan.py`).
"""

from __future__ import annotations

from skunk.common import ExecutionContext, parse_json_response
from skunk.errors import ParseError
from skunk.prompted_call import PromptedCall

# Earlier content pages offered as merge targets. Fragments continue their immediate
# predecessor, but a footnotes page can trail a table that itself spans a few pages.
_PARENT_PAGES_BACK = 4

# Page-text excerpt shown for the fragment (enough to expose "Continued from Table X" /
# footnote markers without paying for whole dense pages).
_FRAGMENT_TEXT_CHARS = 1800

_SYSTEM_PROMPT = """\
You link orphaned Treasury Bulletin table fragments back to their parent table.

The CANDIDATE block is a table detected with NO column labels and NO row labels — usually one of:
  - a continuation fragment: data rows of a multi-page table whose headers were only printed on an
    earlier page (e.g. a page headed "Continued from Table PDO-1" with bare data rows);
  - a footnotes spillover: a block of footnotes to a table on an earlier page (its title usually
    names the parent, e.g. "Footnotes to Table PDO-4");
  - or neither: a self-contained exhibit/facsimile/odd block that merely failed label extraction.

You are given the candidate's metadata, an excerpt of its page's actual text, and the blocks of the
same page and the few preceding pages (each with a `block_id`). Pick the ONE block the candidate is
a continuation/footnote of, or null:
  - The parent must be the table the fragment's rows/footnotes actually belong to — match on table
    number/name in the fragment text when present (tolerate OCR garbling), else on content.
  - When the same table appears on several preceding pages (a "(Continued)" run), pick the NEAREST
    preceding occurrence.
  - Answer null when the candidate is self-contained, or you are not confident which block is the
    parent. Wrongly merging is worse than leaving a fragment standalone.

## Output

A single JSON object, no prose, no markdown fences:
  {"parent": "<block_id exactly as listed>"}  or  {"parent": null}"""


def find_candidates(scans: dict[str, dict]) -> list[tuple[int, int]]:
    """`(page, block_index)` of every unprocessed merge candidate in one bulletin's scans:
    a TABLE block with both label lists empty, not already judged (`merged_into` set or
    `extra_pages` non-empty marks a processed parent/fragment)."""
    out: list[tuple[int, int]] = []
    for p, s in scans.items():
        if s.get("page_role") != "content":
            continue
        for bi, b in enumerate(s.get("blocks", [])):
            if (
                b.get("kind") == "table"
                and not b.get("column_headers")
                and not b.get("row_headers")
                and b.get("merged_into") is None
                and not b.get("extra_pages")
            ):
                out.append((int(p), bi))
    return sorted(out)


def _block_line(page: int, bi: int, b: dict) -> str:
    labels = (
        "has own labels"
        if (b.get("column_headers") or b.get("row_headers"))
        else "NO labels"
    )
    summary = f" — {b['summary']}" if b.get("summary") else ""
    return f"[{page}#{bi}] {b.get('kind')}: {b.get('title') or '(untitled)'} ({labels}){summary}"


def _parent_listing(
    scans: dict[str, dict], page: int, block_index: int
) -> list[tuple[str, tuple[int, int]]]:
    """Merge-target blocks for one candidate: earlier blocks on the same page, then the blocks
    of the up-to-`_PARENT_PAGES_BACK` preceding content pages (nearest first). Already-merged
    fragments are excluded — a chained fragment should link straight to the labeled root block
    (whose `extra_pages` then accumulates the whole run)."""
    out: list[tuple[str, tuple[int, int]]] = []

    def add(p: int, upto: int | None = None) -> None:
        blocks = scans.get(str(p), {}).get("blocks", [])
        for bi, b in enumerate(blocks):
            if upto is not None and bi >= upto:
                break
            if b.get("merged_into") is not None:
                continue
            if p == page and bi == block_index:
                continue
            out.append((_block_line(p, bi, b), (p, bi)))

    add(page, upto=block_index)
    earlier = sorted(
        (
            int(p)
            for p, s in scans.items()
            if int(p) < page and s.get("page_role") == "content"
        ),
        reverse=True,
    )[:_PARENT_PAGES_BACK]
    for p in earlier:
        add(p)
    return out


async def resolve_parent(
    ctx: ExecutionContext,
    bulletin: str,
    scans: dict[str, dict],
    page: int,
    block_index: int,
    page_text: str,
) -> tuple[int, int] | None:
    """One flash call: the candidate fragment's parent `(page, block_index)`, or None to
    leave it standalone. Invalid ids ride PromptedCall's reprompt loop."""
    parents = _parent_listing(scans, page, block_index)
    if not parents:
        return None
    by_id = {f"{p}#{bi}": (p, bi) for _, (p, bi) in parents}

    block = scans[str(page)]["blocks"][block_index]
    user = "\n".join(
        [
            f"Issue: {bulletin} Treasury Bulletin.",
            f"Candidate (PDF page {page}, block {block_index}): "
            f"{block.get('title') or '(untitled)'}"
            + (f" — {block['summary']}" if block.get("summary") else ""),
            "",
            f"Candidate page text (excerpt):\n{page_text[:_FRAGMENT_TEXT_CHARS]}",
            "",
            "Possible parent blocks (same page above the candidate, then preceding pages, nearest first):",
            *(line for line, _ in parents),
        ]
    )

    def _parse(text: str, _ctx: ExecutionContext) -> tuple[int, int] | None:
        obj = parse_json_response(text)
        if not isinstance(obj, dict) or "parent" not in obj:
            raise ParseError(text, 'expected {"parent": "<block_id>" | null}')
        parent = obj["parent"]
        if parent is None:
            return None
        if str(parent) not in by_id:
            raise ParseError(
                text,
                f"unknown block_id {parent!r} — return one of the listed ids or null",
            )
        return by_id[str(parent)]

    call: PromptedCall[tuple[int, int] | None] = PromptedCall(
        name="table_merge",
        system_prompt=_SYSTEM_PROMPT,
        parse=_parse,
        default_effort="off",
        output_instruction=(
            'Output ONLY a JSON object {"parent": "<block_id>"} or {"parent": null} — no prose.'
        ),
    )
    return await call.call(ctx, user, temperature=0.0)


def apply_merge(
    scans: dict[str, dict], fragment: tuple[int, int], parent: tuple[int, int]
) -> None:
    """Annotate one resolved merge: the fragment block records its parent's page, and —
    when the parent is on a DIFFERENT page — the parent block appends the fragment's page
    to `extra_pages` (a same-page parent already covers the fragment's render/text). The
    scans stay lossless; the catalog projection is what hides merged fragments."""
    (fp, fbi), (pp, pbi) = fragment, parent
    scans[str(fp)]["blocks"][fbi]["merged_into"] = pp
    if pp != fp:
        extra = scans[str(pp)]["blocks"][pbi].setdefault("extra_pages", [])
        if fp not in extra:
            extra.append(fp)
