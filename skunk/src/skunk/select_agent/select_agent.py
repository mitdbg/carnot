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
from skunk.page_index.data_model import PageCatalogRow
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
)


class SelectAgent(MultiTurnAgent):
    name = "select_agent"
    # Matches the SearchAgent: dense Treasury tables are ~1.5 chars/token, so keep the char
    # budget conservative under the model's ~1M-token input ceiling.
    context_budget_chars: int = 1_300_000
    warn_steps_remaining = 2

    briefing = (
        "You are a helpful assistant for retrieving relevant information from a large "
        "collection of documents. You will be given a question, and a shortlist of candidate "
        "pages that likely (but not always) carry relevant information. Some questions require "
        "looking up information that is not contained in the corpus; that is handled by a "
        "separate agent, so you should only focus on retrieving relevant documents for the "
        "remainder of the question. You do not need to retrieve everything in a single tool "
        "call: use early steps to explore documents of potential relevance, then refine your "
        "searches in later steps based on what you find. Use `prune(...)` aggressively on "
        "pages and blocks you have ruled out, to keep later searches focused and your context "
        "window manageable.\n\n"
        "## Hints\n\n"
        "- Match the question's EXACT wording — the precise series with every qualifier, "
        "total vs subtotal, unit, and time basis. The catalog summaries are paraphrases; the "
        "question governs.\n"
        "- Confirm the requested dates exist at the needed granularity (monthly rows "
        "vs an annual / fiscal-year roll-up).\n"
        "-  Most questions should be answered by the most "
        "contemporaneous print, but you must check for later revisions. Revisions are "
        "clearly marked as such on the pages or notes. Otherwise, later prints may contain changes in "
        "accounting methods or classifications, and must not be used.\n"
        "- If the question pins a source (\"as reported in the <Month Year> Bulletin\", \"as of "
        "<date>\"), select the blocks that best match it.\n"
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
        """The candidate PAGES (distinct, deduped) as the opening observation — page ids
        only, no summaries. Deliberately just a list: first-pass retrieval flags pages that
        very likely hold the right table, and the agent discovers everything else through its
        tools (read_document / search_corpus / grep_corpus)."""
        seen: list[str] = []
        seen_set: set[str] = set()
        for entry in self._candidates:
            doc_id = pageref_to_doc_key(entry.ref.page)
            if doc_id not in seen_set:
                seen_set.add(doc_id)
                seen.append(doc_id)
        return [
            TextBlock(
                f"Candidate pages from first-pass retrieval ({len(seen)}). With high "
                "probability the right table is in one of these pages, and they are likely "
                f"excellent starting points:\n{', '.join(seen)}"
            )
        ]

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

    async def retrieve(self, ctx: ExecutionContext, question: str) -> list[str]:
        """One rollout for the whole question (no per-branch hints) — the full question
        carries every series/period the selected pages must cover."""
        payload = await self.call(ctx, f"Question: {question}")
        return self._page_keys_from_payload(payload)

    @staticmethod
    def _page_keys_from_payload(payload: Any) -> list[str]:
        keys = payload.get("page_keys") or [] if isinstance(payload, dict) else []
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]
