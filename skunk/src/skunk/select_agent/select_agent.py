"""SelectAgent — the precision stage as an iterative agent over the sem-filter survivors.

It mirrors the SearchAgent's architecture (a `MultiTurnAgent`: one fenced block per step,
a ```json``` final answer of `{"pages": [{"doc_id", "target"}, ...]}`, a redactable block trajectory) but runs
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
    AnnotatedValue,
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
    context_budget_chars: int = 1_500_000
    warn_steps_remaining = 5
    # Collapse all but the 5 most recent tool results to a placeholder in the render; pruned
    # corpus content and the agent's own findings are preserved separately (prune state +
    # any future checkpoint summary), so stale raw observations are the only thing dropped.
    visible_observations: int | None = 5
    # Tighter than the search agent's shared budget: selection over a shortlist needs less
    # exploration than open-corpus retrieval.
    max_steps: int | None = 20

    briefing = (
        "You are a helpful assistant for retrieving relevant information from a large "
        "collection of documents. You will be given a question, a set of tools, and a shortlist of candidate "
        "pages that likely (but not always) carry relevant information. Your ONLY output is a set of page ids: never compute, "
        "calculate, or answer the question yourself. Use early steps to explore documents of potential relevance, then"
        "refine your searches in later steps based on what you find. Use `prune(...)` aggressively on "
        "pages and blocks you have ruled out, to keep later searches focused and your context "
        "window manageable. Stop exploration and commit when you have enough information, do not"
        "overthink . Some questions are multi-hop or require looking up external information;"
        "if you suspect this from inspecting the likely candidate set, commit all information you can find"
        "in the corpus even if they only help make partial progress. \n\n"

        "Write down thoughts and observations about the what you have seen as code comments in your tool call blocks"
        "-- examples include summarizing what the previous observation showed, what you concluded, pages you may want "
        "too revisit later, and why you are making this call, so a human can follow your thoughts\n\n"

        "## Hints\n\n"
        "- Match the question's EXACT wording — the precise series with every qualifier, "
        "total vs subtotal, unit, and time basis.\n"
        "- Confirm the requested dates exist at the needed granularity (monthly rows "
        "vs an annual / fiscal-year roll-up).\n"
        "- A statistic usually has a few months of reporting delay, so the first issue "
        "reporting period P is typically dated a few months later than P.\n"
        "-  Most questions should be answered by the most contemporaneous print, but you should check for later"
        " revisions to be sure. Revisions are often published under the same series in later prints and clearly marked"
        " next to the data or in footnotes. Otherwise, later prints may contain changes in accounting methods or"
        " classifications, and should not be used.\n"
        "- Multi-period span: when the requested period spans more than one print's data "
        "window, select precisely the prints whose data windows together cover the entire span\n"
        "- If the question pins a source (\"as reported in the <Month Year> Bulletin\", \"as of "
        "<date>\"), select the blocks that best match it.\n"
        "- A page's `doc_id` (e.g. `1980_04_85`) is its PDF-index key, NOT its printed footer "
        "label. Emit the `doc_id` of the page you actually read, copied verbatim from the "
        "candidate list or a tool result — never a printed page number and never an adjacent id.\n"
    )

    final_answer_doc = """\
A JSON object under the key "pages": a list of the pages you selected, each an object with its
`doc_id` and a `target` — ONE natural-language expression naming exactly what to retrieve from
that page (the precise series, qualifier, unit, and the dates/periods to pull):
```json
{"pages": [
  {"doc_id": "1946_11_41", "target": "gross public debt outstanding, end of month, Jan–Jun 1946"},
  {"doc_id": "1947_01_38", "target": "gross public debt outstanding, end of month, Jul–Dec 1946"}
]}
```
Copy each `doc_id` VERBATIM from the candidate listing or a tool result. The `target` describes WHAT to read and should
not be an answer or a computation — name the rows/columns and dates to transcribe."""

    def __init__(
        self,
        config: SkunkConfig,
        catalog: dict[PageRef, PageCatalogRow],
        page_store: PageStore,
        candidates: list[SemPoolEntry],
        prior_values: list[AnnotatedValue] | None = None,
    ):
        self.config = config
        self._catalog = catalog
        self._candidates = candidates
        # Data earlier replan attempts already gathered (empty on the first sweep). Surfaced
        # in the seed so the agent doesn't re-select pages for values already in hand.
        self._prior_values = prior_values or []

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
            max_steps=self.max_steps,
            max_misfires=config.agent_max_misfires,
        )

    # ------------------------------------------------------------------
    # Seed + redaction
    # ------------------------------------------------------------------

    def _seed_blocks(self) -> list[Block]:
        """The candidate PAGES (distinct, deduped) as the opening observation — page ids
        only, no summaries. Deliberately just a list: first-pass retrieval flags pages that
        very likely hold the right table, and the agent discovers everything else through its
        tools (read_document / search_corpus / grep_corpus). On a replan sweep, a second block
        lists the data earlier attempts already gathered, so the agent narrows to what's left."""
        seen: list[str] = []
        seen_set: set[str] = set()
        for entry in self._candidates:
            doc_id = pageref_to_doc_key(entry.ref.page)
            if doc_id not in seen_set:
                seen_set.add(doc_id)
                seen.append(doc_id)
        blocks: list[Block] = [
            TextBlock(
                f"Candidate pages from first-pass retrieval ({len(seen)}). With high "
                "probability the right table is in one of these pages, and they are likely "
                f"excellent starting points:\n{', '.join(seen)}"
            )
        ]
        prior = self._prior_values_block()
        if prior is not None:
            blocks.append(prior)
        return blocks

    def _prior_values_block(self) -> Block | None:
        """One block summarizing what earlier replan attempts already gathered (description +
        source issue/pages), instructing the agent not to re-select pages just to re-obtain
        them. None on the first sweep (nothing gathered yet)."""
        if not self._prior_values:
            return None
        lines: list[str] = []
        for v in self._prior_values:
            if v.bulletin:
                pgs = " " + ", ".join(f"p{p}" for p in v.pages) if v.pages else ""
                src = f" [{v.bulletin}{pgs}]"
            elif v.source:
                src = f" [{v.source}]"
            else:
                src = ""
            lines.append(f"- {v.description}{src}")
        return TextBlock(
            "Data ALREADY gathered by earlier attempts (it is in hand — do NOT select pages "
            "solely to re-obtain these; select pages only for the data still missing):\n"
            + "\n".join(lines)
        )

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
        if not isinstance(payload, dict) or "pages" not in payload:
            return 'Final answer must be a JSON object with a "pages" list.'
        pages = payload.get("pages")
        if not isinstance(pages, list) or not pages:
            return "Select at least one page (a non-empty pages list)."
        bad: list[str] = []
        no_target: list[str] = []
        for p in pages:
            if not isinstance(p, dict) or not p.get("doc_id"):
                return 'Each entry in "pages" must be an object with "doc_id" and "target".'
            key = str(p["doc_id"])
            if not str(p.get("target") or "").strip():
                no_target.append(key)
            try:
                ref = page_key_to_pageref(key)
            except ValueError:
                bad.append(key)
                continue
            if ref not in self._catalog:
                bad.append(key)
        if bad:
            return (
                f"These doc_ids are not real catalog pages: {bad[:8]}. Use a doc_id exactly "
                "as it appears in the candidate listing or search_corpus results."
            )
        if no_target:
            return (
                f"These pages have no `target`: {no_target[:8]}. Give each page one "
                "natural-language target naming the series, qualifier, unit, and dates to read."
            )
        return None

    async def retrieve(self, ctx: ExecutionContext, question: str) -> list[tuple[str, str]]:
        """One rollout for the whole question (no per-branch hints) — the full question
        carries every series/period the selected pages must cover. Returns (doc_id, target)
        pairs: the per-page retrieval target the agent wrote drives that page's extraction."""
        payload = await self.call(ctx, f"Question: {question}")
        return self._pages_from_payload(payload)

    @staticmethod
    def _pages_from_payload(payload: Any) -> list[tuple[str, str]]:
        """(doc_id, target) per selected page, deduped on doc_id (a repeated page's targets
        are joined with '; '). Tolerates a bare string/list of page_keys for robustness."""
        raw = payload.get("pages") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            return []
        merged: dict[str, str] = {}
        for p in raw:
            if not isinstance(p, dict) or not p.get("doc_id"):
                continue
            doc_id = str(p["doc_id"])
            target = str(p.get("target") or "").strip()
            if doc_id in merged and target:
                merged[doc_id] = f"{merged[doc_id]}; {target}" if merged[doc_id] else target
            else:
                merged.setdefault(doc_id, target)
        return list(merged.items())
