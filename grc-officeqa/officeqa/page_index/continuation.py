"""Build pass: prune over-long `is_continuation` chains.

The scan sets `is_continuation` on a page whose table is header-less — its column headers sit
on the PREVIOUS page and aren't restated, so the page can't be read on its own. The flag is
self-describing: at access time a continuation page pulls in its predecessor chain
(`data_model.continuation_chain`), so the build NEVER merges or folds anything.

Genuine continuations are short. A run of MORE THAN TWO consecutive flagged pages almost
always means the scan over-fired — a self-contained reprint that restates its headers got
mis-flagged, gluing unrelated pages into one long chain. This pass reviews each such run with
one cheap LLM call (the run's header page + each flagged page's text) and clears
`is_continuation` on the pages that actually stand on their own, breaking the chain.
Annotation-only: it only ever flips a page's `is_continuation` to False in the scan file.

Driven per bulletin by the pipeline's `continuation_check` stage; short (1-2 page) chains are
trusted as-is and never cost a call.
"""

from __future__ import annotations

from skunk.common import ExecutionContext, parse_json_response
from skunk.errors import ParseError
from skunk.prompted_call import PromptedCall

# Runs of more than this many consecutive flagged pages are reviewed; 1-2 page continuations
# are trusted as-is (the common genuine case).
_MAX_TRUSTED_CHAIN = 2

# Page-text excerpt shown per page — enough to expose whether the page restates its own column
# headers without paying for whole dense tables.
_PAGE_TEXT_CHARS = 1500

_SYSTEM_PROMPT = """\
You audit a run of pages flagged as header-less table continuations.

A page is a TRUE continuation only when its table opens straight into data columns with NO
column headers of its own — the labels that make its numbers interpretable are on an earlier
page and are NOT repeated here. A page that restates its own column headers, or starts a
different self-contained table/exhibit, is NOT a continuation, however its caption reads.

You are given the HEADER page (the labeled page the run continues from) and, in document order,
the pages currently flagged as continuations, each with an excerpt of its text. Decide which of
the flagged pages are TRUE continuations of the header page's table. A long run is usually an
over-flag: typically only the first page or two genuinely lack headers; later pages restate them
and should be dropped from the run.

## Output

A single JSON object, no prose, no markdown fences:
  {"continuations": [<page number>, ...]}

List ONLY page numbers from the flagged set that are TRUE continuations; omit every page that
stands on its own."""


def find_long_chains(scans: dict[str, dict]) -> list[list[int]]:
    """Maximal runs of physically-consecutive content pages all flagged `is_continuation`, longer
    than `_MAX_TRUSTED_CHAIN`. Each returned run lists the flagged pages only (ascending); its
    header page is the content page immediately before the run's first page."""
    flagged = sorted(
        int(p)
        for p, s in scans.items()
        if s.get("page_role") == "content" and s.get("is_continuation")
    )
    runs: list[list[int]] = []
    cur: list[int] = []
    for p in flagged:
        if cur and p == cur[-1] + 1:
            cur.append(p)
        else:
            if len(cur) > _MAX_TRUSTED_CHAIN:
                runs.append(cur)
            cur = [p]
    if len(cur) > _MAX_TRUSTED_CHAIN:
        runs.append(cur)
    return runs


async def review_chain(
    ctx: ExecutionContext,
    bulletin: str,
    run: list[int],
    texts: dict[int, str],
) -> list[int]:
    """One flash call: the subset of `run` that is genuinely a header-less continuation of the
    run's header page. Pages in `run` NOT returned should have their flag cleared. Invalid /
    out-of-set page numbers ride PromptedCall's reprompt loop."""
    header_page = run[0] - 1
    lines = [
        f"Issue: {bulletin}.",
        f"HEADER page {header_page}:",
        (texts.get(header_page, "")[:_PAGE_TEXT_CHARS] or "(no text)"),
        "",
        "Flagged continuation pages (in order):",
    ]
    for p in run:
        lines += [f"--- page {p} ---", (texts.get(p, "")[:_PAGE_TEXT_CHARS] or "(no text)")]
    user = "\n".join(lines)

    valid = set(run)

    def _parse(text: str, _ctx: ExecutionContext) -> list[int]:
        obj = parse_json_response(text)
        if not isinstance(obj, dict) or "continuations" not in obj:
            raise ParseError(text, 'expected {"continuations": [page, ...]}')
        out = obj["continuations"]
        if not isinstance(out, list):
            raise ParseError(text, '"continuations" must be a list of page numbers')
        keep: list[int] = []
        for x in out:
            try:
                p = int(x)
            except (TypeError, ValueError) as e:
                raise ParseError(text, f"non-integer page {x!r}") from e
            if p not in valid:
                raise ParseError(
                    text, f"page {p} is not in the flagged set {sorted(valid)}"
                )
            keep.append(p)
        return keep

    call: PromptedCall[list[int]] = PromptedCall(
        name="continuation_check",
        system_prompt=_SYSTEM_PROMPT,
        parse=_parse,
        default_effort="off",
        output_instruction='Output ONLY {"continuations": [page, ...]} — no prose.',
    )
    return await call.call(ctx, user, temperature=0.0)


def apply_review(scans: dict[str, dict], run: list[int], genuine: list[int]) -> list[int]:
    """Clear `is_continuation` on every page of `run` NOT in `genuine` (in place). Returns the
    cleared pages — the catalog patch reprojects exactly these rows."""
    keep = set(genuine)
    cleared: list[int] = []
    for p in run:
        if p not in keep:
            scans[str(p)]["is_continuation"] = False
            cleared.append(p)
    return cleared
