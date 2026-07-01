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
Corpus + index: the topic's corpus JSONL (`{topic}/corpus.jsonl`), one element per document, embedded
  with Qwen3-Embedding-0.6B into the `qwen-freshstack-{topic}-0.6b` Chroma collection (KARL retrieves
  FreshStack with Qwen3-0.6B, k=10). The corpus is small (~50K docs), so the doc map (_id -> text) is
  read straight from corpus.jsonl into RAM — the file is the source of truth for the embedded text.
Gold docs / recall: each nugget's `relevant_corpus_ids` reference corpus `_id`s DIRECTLY (the corpus
  `_id`, e.g. "azure-openai/LICENSE.md_0_1140", is the Chroma row id), so gold_docs are the union of
  those ids and recall is a plain `doc_recall` over retrieved `_id`s — no key remapping needed.
Held-out test set: `test_ids_path` if given, else ALL of the topic's questions (run with --split test).
"""

from __future__ import annotations

import json
import os

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_nugget_recall
from qatfd.config import FreshstackConfig
from qatfd.constants import FRESHSTACK
from qatfd.paths import resolve_under_skunk
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


def _file_id(doc_id: str) -> str:
    """The source FILE a chunk belongs to: the chunk `_id` minus its trailing "_{start}_{end}" byte
    range, e.g. "azure-openai/LICENSE.md_0_1140" -> "azure-openai/LICENSE.md". (File paths can contain
    underscores, so we only strip when the last two underscore-separated fields are both integers.)
    Mirrors `_file_id` in compute_freshstack_embeddings.py — keep the two identical."""
    parts = doc_id.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return doc_id


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
        if not config.chromadb_collection:
            config.chromadb_collection = f"freshstack-{topic}-qwen-0.6b"
        # Each topic lives in its OWN slim store; derive the per-topic dir from the default shared
        # value (override chromadb_dir explicitly, or chromadb_host, to point elsewhere).
        if config.chromadb_dir in (None, ".chromadb"):
            config.chromadb_dir = f".chromadb-freshstack-{topic}-qwen-0.6b"

        config.chromadb_dir = str(resolve_under_skunk(config.chromadb_dir))
        config.questions_path = str(resolve_under_skunk(config.questions_path))
        config.corpus_path = str(resolve_under_skunk(config.corpus_path))
        if config.test_ids_path:
            config.test_ids_path = str(resolve_under_skunk(config.test_ids_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_skunk(config.prompts_path))
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
        with open(self.config.questions_path) as f:
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
        """`_id -> document text`, read straight from the topic's corpus.jsonl (the ~50K-doc corpus
        fits in RAM, and this file is exactly the text the vectors were built from). The corpus `_id`
        is the Chroma row id, so retrieved doc_ids look up directly here."""
        path = self.config.corpus_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FreshStack corpus not found at {path}; expected the topic's corpus.jsonl "
                f"(download with engaging-scripts/download_freshstack.py)."
            )
        doc_map: dict[str, str] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc_map[str(rec["_id"])] = str(rec.get("text", ""))
        return doc_map

    def _build_resources(self) -> BenchmarkResources:
        collection = self._open_chroma_collection()
        return BenchmarkResources(
            chroma_collection=collection,
            document_map=self._build_document_map(),
            config=self.config,
        )

    # ---- scoring + metrics ----------------------------------------------------

    async def score(self, question: Question, predicted: str, ctx) -> dict:
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
        # `_id`s = the retrieved doc_ids), so `doc_recall` is exact-chunk recall. `file_recall`
        # collapses both sides to their source file (drop the `_id`'s byte-range suffix), isolating
        # "found the right file but not the exact chunk". Pure (file_id derives from the id string,
        # no resources), so it recomputes offline from a stored report.csv.
        chunk_recall = doc_recall(retrieved, question.gold_docs)
        ret_files = None if retrieved is None else [_file_id(c) for c in retrieved]
        gold_files = [_file_id(g) for g in question.gold_docs]
        return {"doc_recall": chunk_recall, "file_recall": doc_recall(ret_files, gold_files)}
