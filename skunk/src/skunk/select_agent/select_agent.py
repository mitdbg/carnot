"""SelectAgent — the precision stage as an iterative agent over the sem-filter survivors.

It mirrors the SearchAgent's architecture (a `MultiTurnAgent`: one fenced block per step,
a ```json``` final answer of `{"page_keys": [...]}`, a redactable block trajectory) but runs
in a different regime: stage-1 page-index retrieval (ToC + date filter + semantic filter) has
already flagged the most-likely candidate blocks, which are SEEDED as the agent's opening
observation. The agent's job is to verify and select among them — reading pages, viewing
figures, tiling a recurring series across issues — and it has full-corpus reach via the
catalog (`search_corpus`) if the flagged set proves insufficient.

No ChromaDB, no embeddings: the corpus index is the in-memory page-index catalog, page
content is read through the `PageStore`. The selection rules (exact-qualifier match, tiling,
source-pin / most-recent tie-break) are carried over from the block-selection tournament.
"""

from __future__ import annotations

from typing import Any

from skunk.common import (
    B64Image,
    ExecutionContext,
    PageRef,
    SemPoolEntry,
    page_key_to_pageref,
    pageref_to_doc_key,
)
from skunk.config import SkunkConfig
from skunk.local_python_executor import CodeOutput
from skunk.multi_turn_agent import (
    Block,
    ChunkBlock,
    ImageBlock,
    MultiTurnAgent,
    TextBlock,
)
from skunk.page_index.data_model import CATALOG_ROW_FIELDS, PageCatalogRow
from skunk.page_index.store import PageStore
from skunk.select_agent.select_tools import (
    EMPTY_GREP_MESSAGE,
    EMPTY_RESULT_MESSAGE,
    GREP_RESULT_TAG,
    PRUNE_RESULT_TAG,
    READ_DOCUMENT_RESULT_TAG,
    SEARCH_RESULT_TAG,
    VIEW_FIGURE_RESULT_TAG,
    CatalogView,
    GrepCorpusTool,
    PruneTool,
    ReadDocumentTool,
    SearchCorpusTool,
    ViewFigureTool,
    block_id,
)


def _render_candidate(entry: SemPoolEntry) -> tuple[str, str, str]:
    """One flagged candidate rendered as `(block_id, doc_id, line)`. Mirrors
    `select_tools.render_block_line` so the seed and `search_corpus` results read alike."""
    doc_id = pageref_to_doc_key(entry.ref.page)
    bi = entry.ref.block_index if entry.ref.block_index is not None else 0
    bid = block_id(doc_id, bi)
    dates = f"{entry.interval[0]}..{entry.interval[1]}" if entry.interval else "none"
    line = (
        f"[{bid}] {entry.ref.page.month} p.{entry.ref.page.page}  dates={dates}  "
        f"{entry.kind}: {entry.title or '(untitled)'}"
    )
    cols = list(entry.cols)
    rows = list(entry.rows or entry.rows_tail)
    if cols:
        line += f"\n    cols: {', '.join(cols)}"
    if rows:
        line += f"\n    rows: {', '.join(rows)}"
    if entry.summary:
        line += f"\n    summary: {entry.summary}"
    return bid, doc_id, line


class SelectAgent(MultiTurnAgent):
    name = "select_agent"
    # Matches the SearchAgent: dense Treasury tables are ~1.5 chars/token, so keep the char
    # budget conservative under the model's ~1M-token input ceiling.
    context_budget_chars: int = 1_300_000
    warn_steps_remaining = 2

    briefing = (
        "You are the SELECTION stage of a research-question pipeline over the U.S. Treasury "
        "Bulletin corpus. An extraction step will read the pages you select and pull the "
        "numbers from them, so your job is to choose the SMALLEST set of pages that together "
        "carry the data the question needs — with the exact series, qualifiers, unit, and "
        "time basis it asks for.\n\n"
        "A first-pass retrieval has already FLAGGED the most-likely candidate blocks; they "
        "appear as your first observation, each with its block id, source page, data-date "
        "span, table title, column/row labels, and a short summary (no numbers). Treat the "
        "flagged set as high-precision and start there. You ALSO have full-corpus reach: "
        "`search_corpus` queries the catalog of every page in the corpus, so use it to find "
        "additional candidates only when the flagged set is insufficient (e.g. to reach an "
        "earlier/later reprint that completes a period). Use `read_document` to confirm a "
        "candidate's text before committing, `view_figure` for chart pages, and `prune` "
        "aggressively on candidates you rule out to keep your context focused.\n\n"
        "## Catalog schema\n\n"
        + CATALOG_ROW_FIELDS
        + "\n\n## Selection rules\n\n"
        "- Match the question's EXACT wording — the precise series with every qualifier, "
        "total vs subtotal, unit, and time basis. The flagged summaries are paraphrases; the "
        "question governs.\n"
        "- Confirm the requested dates exist as rows at the needed granularity (monthly rows "
        "vs an annual / fiscal-year roll-up).\n"
        "- Recurring series: the same table recurs across consecutive issues with a shifting "
        "data window (sometimes revised). Keep the FEWEST prints whose windows together cover "
        "the requested period at the needed granularity — one print when one suffices, else a "
        "tiling of the same table reaching the period's first and last months.\n"
        "- If the question pins a source (\"as reported in the <Month Year> Bulletin\", \"as of "
        "<date>\"), select the blocks that best match it. When the same figure is restated "
        "across issues and all else is equal, prefer the most recent issue.\n"
        "- Drop a candidate when nothing in it — title, labels, or summary — could carry the "
        "target series, or when a kept print of the same table already contains everything it "
        "would contribute."
    )

    final_answer_doc = """\
A JSON object with the `doc_id`s of the pages you selected, under the key "page_keys":
```json
{"page_keys": ["1946_11_41", "1947_01_38"]}
```
Use each `doc_id` exactly as it appears in the candidate listing / search results. Select at
least one page; if the question's data is genuinely on none of them, you are out of luck —
do not invent pages."""

    def __init__(
        self,
        config: SkunkConfig,
        catalog: dict[PageRef, PageCatalogRow],
        page_store: PageStore,
        candidates: list[SemPoolEntry],
    ):
        self.config = config
        self._catalog = catalog
        self._candidates = candidates

        # Full text of the flagged pages — the surface grep_corpus searches.
        survivor_texts: dict[str, str] = {}
        for entry in candidates:
            doc_id = pageref_to_doc_key(entry.ref.page)
            if doc_id in survivor_texts:
                continue
            text = page_store.text(entry.ref.page)
            if text is not None:
                survivor_texts[doc_id] = text
        self._view = CatalogView(catalog, survivor_texts, page_store)

        # Per-question prune state shared by the tools and read for render redaction.
        self._pruned_doc_ids: set[str] = set()
        self._pruned_block_ids: set[str] = set()

        self.max_output_tokens = config.search_agent_max_output_tokens
        self.request_timeout_s = config.search_agent_request_timeout_s

        tools = [
            SearchCorpusTool(self._view, self._pruned_doc_ids, self._pruned_block_ids),
            GrepCorpusTool(self._view, self._pruned_doc_ids, config.grep_max_output_tokens),
            ReadDocumentTool(
                self._view,
                config.agent_max_pages_per_tool_call,
                config.read_document_max_output_chars,
            ),
            ViewFigureTool(self._view),
            PruneTool(self._pruned_doc_ids, self._pruned_block_ids),
        ]
        super().__init__(
            tools,
            max_steps=config.agent_max_steps,
            max_misfires=config.agent_max_misfires,
        )

    # ------------------------------------------------------------------
    # Seed + redaction
    # ------------------------------------------------------------------

    def _seed_blocks(self) -> list[Block]:
        """The flagged candidates as the opening (pruneable) observation."""
        blocks: list[Block] = [
            TextBlock(
                f"Flagged candidates from first-pass retrieval ({len(self._candidates)} "
                "block(s)), most likely to be relevant — start here:"
            )
        ]
        for entry in self._candidates:
            bid, doc_id, line = _render_candidate(entry)
            blocks.append(ChunkBlock(chunk_id=bid, doc_id=doc_id, text=line))
        return blocks

    def _block_is_visible(self, block: Block) -> bool:
        if isinstance(block, ChunkBlock):
            if block.chunk_id is not None and block.chunk_id in self._pruned_block_ids:
                return False
            if block.doc_id in self._pruned_doc_ids:
                return False
        return True

    def _blocks_from_output(self, out: CodeOutput) -> list[Block]:
        blocks: list[Block] = []
        stdout_s = (out.logs or "").strip()
        if stdout_s:
            blocks.append(TextBlock(f"[stdout]\n{stdout_s}"))
        output = out.output

        if isinstance(output, dict) and output.get(SEARCH_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["chunks"]:
                blocks.append(TextBlock(EMPTY_RESULT_MESSAGE))
            else:
                blocks.extend(
                    ChunkBlock(chunk_id=c["chunk_id"], doc_id=c["doc_id"], text=c["text"])
                    for c in output["chunks"]
                )
            return blocks

        if isinstance(output, dict) and output.get(GREP_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            elif not output["groups"]:
                blocks.append(TextBlock(EMPTY_GREP_MESSAGE))
            else:
                for group in output["groups"]:
                    blocks.append(
                        ChunkBlock(
                            chunk_id=None,
                            doc_id=group["doc_id"],
                            text=f"{group['header']}\n{group['text']}",
                        )
                    )
            if output.get("truncation_note"):
                blocks.append(TextBlock(output["truncation_note"]))
            return blocks

        if isinstance(output, dict) and output.get(READ_DOCUMENT_RESULT_TAG):
            blocks.extend(
                ChunkBlock(chunk_id=None, doc_id=d["doc_id"], text=d["text"])
                for d in output["docs"]
            )
            return blocks

        if isinstance(output, dict) and output.get(VIEW_FIGURE_RESULT_TAG):
            if output.get("error"):
                blocks.append(TextBlock(f"[error]\n{output['error']}"))
            else:
                caption = (
                    f"[full-page image of doc_id={output['doc_id']} "
                    f"(contains <figure id={output['figure_id']}>)]"
                )
                blocks.append(
                    ImageBlock(
                        doc_id=output["doc_id"],
                        figure_id=output["figure_id"],
                        image=B64Image(mime=output["mime"], data=output["data"]),
                        text=caption,
                    )
                )
            return blocks

        if isinstance(output, dict) and output.get(PRUNE_RESULT_TAG):
            blocks.append(
                TextBlock(
                    f"[result]\nPruned {output['new_doc_count']} page(s) and "
                    f"{output['new_block_count']} block(s). {output['total_docs']} page(s) "
                    f"and {output['total_blocks']} block(s) now excluded."
                )
            )
            return blocks

        result_s = "" if output is None else str(output).strip()
        if result_s and result_s != stdout_s and result_s not in stdout_s:
            blocks.append(TextBlock(f"[result]\n{result_s}"))
        if not blocks:
            blocks.append(TextBlock("[no output]"))
        return blocks

    # ------------------------------------------------------------------
    # Final answer + entry point
    # ------------------------------------------------------------------

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, dict) or "page_keys" not in payload:
            return 'Final answer must be a JSON object with a "page_keys" list.'
        keys = self._page_keys_from_payload(payload)
        if not keys:
            return "Select at least one page (a non-empty page_keys list)."
        bad: list[str] = []
        for key in keys:
            try:
                ref = page_key_to_pageref(key)
            except ValueError:
                bad.append(key)
                continue
            if ref not in self._catalog:
                bad.append(key)
        if bad:
            return (
                f"These page_keys are not real catalog pages: {bad[:8]}. Use a doc_id exactly "
                "as it appears in the candidate listing or search_corpus results."
            )
        return None

    async def retrieve(
        self,
        ctx: ExecutionContext,
        question: str,
        *,
        branch_key: str | None = None,
        branch_period: str | None = None,
    ) -> list[str]:
        parts = [f"Question: {question}"]
        if branch_key:
            parts.append(f"Selection target: {branch_key}")
        if branch_period:
            parts.append(f"Time period (of the data): {branch_period}")
        payload = await self.call(ctx, "\n".join(parts))
        return self._page_keys_from_payload(payload)

    @staticmethod
    def _page_keys_from_payload(payload: Any) -> list[str]:
        keys = payload.get("page_keys") or [] if isinstance(payload, dict) else []
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]
