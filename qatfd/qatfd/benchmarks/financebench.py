"""FinanceBench benchmark (Patronus AI open-source subset).

Questions: the FinanceBench open-source JSONL (`financebench_open_source.jsonl`) — 150
  human-annotated questions over SEC filings (10-K / 10-Q / 8-K / earnings). Each record has
  a free-form gold `answer` (e.g. "$1577.00") plus `evidence[]`, where each evidence item names
  the source `doc_name` and a ZERO-indexed `evidence_page_num`. Only the OPEN_SOURCE subset is
  public; the full benchmark (10,231 questions) is gated behind Patronus.
Corpus + index: the 368 source PDFs in `pdfs/`, indexed at the PAGE level (matching KARL) into a
  Chroma collection, one element per page. The doc map (page_key -> page text) is rebuilt from the
  embedding metadata, so it matches exactly the text the vectors were built from.
Scoring: LLM-as-judge, single-nugget correctness (see judge.py) — the official benchmark grades
  by hand (n=2,400 in the paper), so there is no deterministic scorer to port.
Gold docs / recall: gold relevance is labeled at the (doc, page) level, so gold_docs are page keys
  ("{doc_name}::p{page_num}") and recall is reported both at page and document granularity.
Held-out test set: `test_ids_path` if given, else ALL 150 questions (run with --split test). KARL's
  Table 2 lists FinanceBench as 150 and there is no published 100-question subset, so the whole
  open-source set is the comparison set.
"""

from __future__ import annotations

import glob
import json
import os

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_single_nugget
from qatfd.config import FinanceBenchConfig
from qatfd.constants import FINANCE_BENCH
from qatfd.paths import resolve_under_skunk
from qatfd.types import Question

# Separator joining a document name and its (zero-indexed) page number into a page key. Chosen so
# it cannot collide with FinanceBench doc_names (which use only [A-Za-z0-9_-]); the embedding job
# (compute_financebench_element_embeddings.py) and the chroma adapter (create_vector_db.py) MUST
# use the identical key, or retrieved doc_ids won't line up with gold for recall.
_PAGE_SEP = "::p"


def page_key(doc_name: str, page_num: int) -> str:
    """Page-level key "{doc_name}::p{page_num}" (page_num zero-indexed, as in FinanceBench)."""
    return f"{doc_name}{_PAGE_SEP}{int(page_num)}"


def _page_key_to_doc(key: str) -> str:
    """Collapse a page key back to its document name (drop the trailing ::pN)."""
    return key.split(_PAGE_SEP)[0]


def financebench_recall_metrics(retrieved: list[str] | None, gold_page_keys: list[str]) -> dict[str, float]:
    """Recall at two granularities over page keys: `page_recall` (exact page hit) and `doc_recall`
    (right document, page ignored). The gap isolates "found the right filing but not the right
    page". Pure (no resources) so it can be recomputed offline from a stored report.csv."""
    page_recall = doc_recall(retrieved, gold_page_keys)
    ret_docs = None if retrieved is None else [_page_key_to_doc(k) for k in retrieved]
    gold_docs = [_page_key_to_doc(k) for k in gold_page_keys]
    return {"page_recall": page_recall, "doc_recall": doc_recall(ret_docs, gold_docs)}


class FinanceBenchBenchmark(Benchmark):
    name = FINANCE_BENCH
    config: FinanceBenchConfig

    # FinanceBench answers are short financial facts (a value, often with units, or a brief phrase).
    # The judge is lenient on formatting, but a concise direct answer keeps it unambiguous.
    answer_format_hint = (
        "Answer with the specific financial fact requested — the numeric value (with its units, e.g. "
        "USD millions, %, or a ratio) or the short phrase that directly answers the question. Be "
        "concise: state the value plainly without lengthy working or restating the question."
    )

    def __init__(self, config: FinanceBenchConfig) -> None:
        config.chromadb_dir = str(resolve_under_skunk(config.chromadb_dir))
        config.questions_path = str(resolve_under_skunk(config.questions_path))
        config.fb_metadata_glob = str(resolve_under_skunk(config.fb_metadata_glob))
        if config.test_ids_path:
            config.test_ids_path = str(resolve_under_skunk(config.test_ids_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_skunk(config.prompts_path))
        if config.pdf_dir:
            config.pdf_dir = str(resolve_under_skunk(config.pdf_dir))
        super().__init__(config)

    # ---- questions ------------------------------------------------------------

    @staticmethod
    def _gold_page_keys(rec: dict) -> list[str]:
        """Page keys for every evidence item, de-duplicated in first-seen order. Each evidence item
        carries `doc_name` + zero-indexed `evidence_page_num`; the record's top-level `doc_name` is
        the fallback for the (rare) item that omits its own."""
        keys: list[str] = []
        seen: set[str] = set()
        for ev in rec.get("evidence", []):
            doc = str(ev.get("doc_name") or rec.get("doc_name") or "")
            page = ev.get("evidence_page_num")
            if not doc or page is None:
                continue
            k = page_key(doc, int(page))
            if k not in seen:
                seen.add(k)
                keys.append(k)
        return keys

    def load_questions(self) -> list[Question]:
        questions: list[Question] = []
        with open(self.config.questions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gold_keys = self._gold_page_keys(rec)
                questions.append(
                    Question(
                        qid=str(rec["financebench_id"]),
                        text=str(rec["question"]),
                        gold=str(rec.get("answer", "")),
                        gold_docs=gold_keys,
                        meta={
                            "company": rec.get("company", ""),
                            "doc_name": rec.get("doc_name", ""),
                            "question_type": rec.get("question_type", ""),
                            "justification": rec.get("justification", ""),
                            "gold_docs_unique": sorted({_page_key_to_doc(k) for k in gold_keys}),
                        },
                    )
                )
        return questions

    # ---- retrieval substrate --------------------------------------------------

    def _build_document_map(self) -> dict[str, str]:
        """page_key -> page text, reconstructed from the embedding-job metadata (metadata_rank*.json:
        unique_element_id -> {page_key, cleaned, element_id}). A page is a single element, so its text
        is that element's `cleaned` (concatenated by element_id if a page was ever split)."""
        paths = sorted(glob.glob(self.config.fb_metadata_glob))
        if not paths:
            raise FileNotFoundError(
                f"no FinanceBench element metadata found at {self.config.fb_metadata_glob}; "
                f"expected metadata_rank*.json produced by compute_financebench_element_embeddings.py."
            )
        by_doc: dict[str, dict[int, str]] = {}
        for p in paths:
            with open(p) as f:
                meta = json.load(f)
            for entry in meta.values():
                key = str(entry["page_key"])
                eid = int(entry.get("element_id", 0))
                by_doc.setdefault(key, {})[eid] = entry.get("cleaned", "")
        return {key: "\n\n".join(elements[k] for k in sorted(elements)) for key, elements in by_doc.items()}

    def _build_resources(self) -> BenchmarkResources:
        collection = self._open_chroma_collection()
        return BenchmarkResources(
            chroma_collection=collection,
            document_map=self._build_document_map(),
            config=self.config,
        )

    # ---- scoring + metrics ----------------------------------------------------

    async def score(self, question: Question, predicted: str, ctx) -> dict:
        return await judge_single_nugget(
            ctx, question=question.text, gold=question.gold, predicted=predicted, model=self.config.judge_model
        )

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        return financebench_recall_metrics(retrieved, question.gold_docs)

    def test_qids(self) -> set[str]:
        if self.config.test_ids_path:
            assert os.path.exists(self.config.test_ids_path), (
                f"test_ids_path {self.config.test_ids_path} does not exist; expected a JSON object with "
                f"a 'query_ids' field listing the held-out FinanceBench qids."
            )
            with open(self.config.test_ids_path) as f:
                data = json.load(f)
            assert "query_ids" in data, f"test_ids_path {self.config.test_ids_path} must have a 'query_ids' field."
            return {str(q) for q in data["query_ids"]}

        # no explicit split: the whole 150-question open-source set is the held-out comparison set.
        return {q.qid for q in self.load_questions()}
