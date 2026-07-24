"""FreshStack benchmark (the KARLBench sub-benchmark; Thakur et al.).

FreshStack tests procedural reasoning over RECENT technical software documentation. KARL uses the
`langchain` topic (203 StackOverflow-derived questions over 49,514 docs from the langchain ecosystem);
each FreshStack topic is a SEPARATE corpus, so we expose `topic` as a config knob (test = langchain,
dev = laravel) and build one Chroma collection per topic.

Questions: the FreshStack queries JSONL (`{topic}/queries.jsonl`). Per record: `query_id`,
  `query_title` (the StackOverflow title), `query_text` (the full question body), and `nuggets[]`,
  where each nugget carries `text` (a GPT-4o-generated decompositional fact) plus `relevant_corpus_ids`
  / `non_relevant_corpus_ids` (corpus `_id`s supporting / not supporting that fact). The question we
  pose to a system is the title + body.
Gold nuggets: the nugget `text`s — one nugget per decompositional fact. The score is nugget-completion
  recall (each fact graded support / partial / not_support by the LLM judge in judge.py), exactly as
  KARL grades FreshStack ("convert ground-truth answers into fixed nuggets ... prior to evaluation").
Corpus + index: the topic's corpus JSONL (`freshstack/{topic}/corpus.jsonl`), one embedded element
  per corpus record (a byte-range slice of a source file), embedded with Qwen3-Embedding-0.6B into
  the `freshstack-{topic}-qwen-0.6b` Chroma collection (KARL retrieves FreshStack with Qwen3-0.6B,
  k=10), stored under benchmarks/freshstack/{topic}/chromadb. The retrieval unit (`doc_id`) is the
  source FILE (`file_id` = the `_id` minus its byte-range suffix; see CORPUS_MODEL.md — before
  2026-07-09 it was the chunk `_id` itself; existing collections are migrated in place by
  scripts/migrate_doc_ids.py). The corpus is small (~50K chunks), so the doc map (file_id -> file
  text, slices joined in byte order) is rebuilt from corpus.jsonl into RAM — the file is the source
  of the vectors.
Gold docs / recall: each nugget's `relevant_corpus_ids` reference corpus `_id`s (chunks), so
  gold_docs stay chunk `_id`s and `recall_metrics` collapses both sides to file_ids — `file_recall`
  is the primary (and only final-answer-computable) retrieval metric.
Dev/test split: by TOPIC — dev = all laravel queries, test = all langchain queries
  (benchmarks/freshstack/freshstack_splits.json, via benchmarks.splits_path; see scripts/make_splits.py).
"""

from __future__ import annotations

import json
import os

from skunk.common import ExecutionContext

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_nugget_recall
from qatfd.config import FreshstackConfig
from qatfd.constants import FRESHSTACK
from qatfd.keys import freshstack_byte_range, freshstack_file_id
from qatfd.paths import resolve_under_benchmarks
from qatfd.types import Question

# Grader persona for the nugget judge: FreshStack answers are technical software facts (code,
# configuration, library APIs, error resolutions), so a software-engineering evaluator judges
# whether the predicted answer entails each decompositional fact.
_JUDGE_SYSTEM = (
    "You are a careful answer-evaluation judge for technical software-engineering questions (code, "
    "libraries, APIs, configuration, and error resolution). Each decompositional fact is one piece "
    "of the correct answer; judge whether the predicted answer entails that fact, allowing for "
    "equivalent code, paraphrase, and formatting differences."
)


class FreshstackBenchmark(Benchmark):
    name = FRESHSTACK
    config: FreshstackConfig

    # FreshStack questions are answered with a technical explanation (often including code/config), and
    # the score is nugget recall, so the answer should cover every supported fact — breadth over brevity.
    answer_format_hint = (
        "Answer the technical question thoroughly using the documents: explain the resolution and "
        "include the relevant code, configuration, commands, or API usage the documents support. Cover "
        "every supported fact the question calls for — prefer completeness over a one-line answer."
    )

    def __init__(self, config: FreshstackConfig) -> None:
        # `topic` is the single source of truth: derive the data paths + collection from it unless
        # explicitly overridden, so running the dev topic is just `benchmarks.topic=laravel`.
        topic = config.topic
        if not config.questions_path:
            config.questions_path = f"{config.data_dir}/{topic}/queries.jsonl"
        if not config.corpus_path:
            config.corpus_path = f"{config.data_dir}/{topic}/corpus.jsonl"
        if not config.storage.collection_name:
            config.storage.collection_name = f"freshstack-{topic}-qwen-0.6b"

        # all benchmark data (questions, corpus, prompts) resolves under qatfd/benchmarks/.
        config.questions_path = str(resolve_under_benchmarks(config.questions_path))
        config.corpus_path = str(resolve_under_benchmarks(config.corpus_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_benchmarks(config.prompts_path))
        super().__init__(config)

    # ---- questions ------------------------------------------------------------

    @staticmethod
    def _gold_docs(rec: dict) -> list[str]:
        """Corpus `_id`s supporting any nugget, de-duplicated in first-seen order. These reference
        corpus documents directly (they ARE the Chroma row ids), so they are the gold for doc-recall."""
        docs: list[str] = []
        seen: set[str] = set()
        for nug in rec.get("nuggets", []):
            for cid in nug.get("relevant_corpus_ids", []):
                c = str(cid)
                if c and c not in seen:
                    seen.add(c)
                    docs.append(c)
        return docs

    @staticmethod
    def _question_text(rec: dict) -> str:
        """The question posed to a system: the StackOverflow title followed by the full body."""
        title = str(rec.get("query_title", "")).strip()
        body = str(rec.get("query_text", "")).strip()
        if title and body:
            return f"{title}\n\n{body}"
        return title or body

    def load_questions(self) -> list[Question]:
        questions: list[Question] = []
        path = self.config.questions_path
        assert path is not None  # normalized to a concrete path in __init__
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # one nugget per gold decompositional fact (KARL grades FreshStack by nugget completion).
                nuggets = [str(n["text"]).strip() for n in rec.get("nuggets", []) if str(n.get("text", "")).strip()]
                questions.append(
                    Question(
                        qid=str(rec["query_id"]),
                        text=self._question_text(rec),
                        gold=" ".join(nuggets),
                        gold_docs=self._gold_docs(rec),
                        meta={
                            "nuggets": nuggets,
                            "n_nuggets": len(nuggets),
                            "query_title": rec.get("query_title", ""),
                            "tags": (rec.get("metadata") or {}).get("tags", []),
                        },
                    )
                )
        return questions

    # ---- retrieval substrate --------------------------------------------------

    def _build_document_map(self) -> dict[str, str]:
        """`file_id -> source-file text`, reassembled from the topic's corpus.jsonl (the ~50K-chunk
        corpus fits in RAM, and this file is exactly the text the vectors were built from). Each
        corpus record is a byte-range slice of a source file; a file's text is its slices joined in
        start_byte order (the slices tile each file with only whitespace-sized gaps, so ordered
        concatenation is a faithful reconstruction). The file_id is the Chroma `doc_id`, so
        retrieved doc_ids look up directly here."""
        path = self.config.corpus_path
        assert path is not None  # normalized to a concrete path in __init__
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FreshStack corpus not found at {path}; expected the topic's corpus.jsonl "
                f"(download with engaging-scripts/download_freshstack.py)."
            )
        by_file: dict[str, list[tuple[int, str]]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = str(rec["_id"])
                span = freshstack_byte_range(cid)
                by_file.setdefault(freshstack_file_id(cid), []).append(
                    (span[0] if span else 0, str(rec.get("text", "")))
                )
        return {fid: "\n".join(text for _, text in sorted(slices)) for fid, slices in by_file.items()}

    def _build_resources(self) -> BenchmarkResources:
        collection = self._open_chroma_collection()
        return BenchmarkResources(
            chroma_collection=collection,
            document_map=self._build_document_map(),
        )

    # ---- scoring + metrics ----------------------------------------------------

    async def score(self, question: Question, predicted: str, ctx: ExecutionContext) -> dict:
        return await judge_nugget_recall(
            ctx,
            question=question.text,
            nuggets=question.meta.get("nuggets", []),
            predicted=predicted,
            model=self.config.judge_model,
            judge_system=_JUDGE_SYSTEM,
            partial_credit=self.config.partial_credit,
        )

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        # Gold relevance is labeled at the CHUNK level (a nugget's relevant_corpus_ids are corpus
        # `_id`s), but the retrieval unit systems return is the source FILE (since 2026-07-09; see
        # CORPUS_MODEL.md), so recall is at file granularity: a gold file counts as found when it
        # was retrieved. Retrieved ids are collapsed through file_id too — a no-op on file ids —
        # so the metric also recomputes identically from pre-migration report.csv rows (whose
        # retrieved ids are chunk `_id`s). Exact-chunk recall is no longer computable from the
        # final answer; a trace-based diagnostic (did the agent ever surface the exact gold slice)
        # would have to recompute it offline from traces/<qid>.jsonl.
        ret_files = None if retrieved is None else [freshstack_file_id(c) for c in retrieved]
        gold_files = [freshstack_file_id(g) for g in question.gold_docs]
        return {"file_recall": doc_recall(ret_files, gold_files)}
