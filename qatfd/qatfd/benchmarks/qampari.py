"""QAMPARI benchmark (Tel Aviv University; the KARLBench sub-benchmark).

Questions: the QAMPARI test JSONL (`test_data.jsonl`) — 1000 open-domain questions over Wikipedia,
  each with MANY answers (>=5, ~14.7 on average). Per record: `qid`, `question_text`, and
  `answer_list[]`, where every answer carries `answer_text`, `aliases`, and `proof[]` (each proof
  names the supporting Wikipedia article via `found_in_url`). QAMPARI tests EXHAUSTIVE entity search
  — success depends on comprehensive recall, not on finding a single supporting mention.
Gold nuggets: the gold answer entities themselves — one nugget per `answer_text`. The score is
  nugget-completion recall (KARL "treats each entity as a separate nugget"), graded by the LLM judge
  in judge.py so an alias / paraphrase of the entity still counts.
Corpus + index: the FULL QAMPARI chunked Wikipedia (~25.9M ~100-word passages), embedded with
  Qwen3-Embedding-0.6B into a Chroma collection (see compute_qampari_embeddings.py). We index the
  whole corpus — KARL's "chunks containing a gold answer entity" subset can't be reliably
  reproduced (the answer entities are common strings that match most of Wikipedia, and the
  entity-LINK annotations that would scope it aren't in the chunk metadata), so — like TREC-BioGen
  — we embed everything and let retrieval do the work. The same index serves both the dev and test
  question splits. The chunk-text doc map is served lazily from the Chroma `documents` column
  (a 25.9M-entry dict would be far too large for RAM), keyed by chunk_id.
Gold docs / recall: a proof identifies the supporting Wikipedia ARTICLE, not a corpus chunk (the
  proof `pid` is a per-question id, never a corpus chunk_id), and the released chunk `url` is a
  curid URL while proofs use title-slug URLs — so article identity is the normalized Wikipedia
  TITLE. gold_docs are normalized article titles; recall = fraction of gold articles whose chunks
  were retrieved (`doc_recall`), collapsing retrieved chunk_ids -> article title via the title in
  the Chroma metadata (a small per-question `.get` on just the retrieved ids).
Held-out test set: `test_ids_path` if given, else ALL 1000 questions (run with --split test).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from urllib.parse import unquote, urlparse

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_nugget_recall
from qatfd.config import QampariConfig
from qatfd.constants import QAMPARI
from qatfd.paths import resolve_under_skunk
from qatfd.types import Question

# Grader persona for the nugget judge: QAMPARI answers are encyclopedic entities (people, films,
# places), so a neutral evaluator — not TREC-BioGen's biomedical one — judges entity presence.
_JUDGE_SYSTEM = (
    "You are a careful answer-evaluation judge for open-domain, multi-answer questions over "
    "encyclopedic (Wikipedia) text. Each decompositional fact is one gold answer entity; judge "
    "whether the predicted answer names that entity (an alias or alternate surface form counts)."
)


def _norm_title(title: str) -> str:
    """Canonical form of a Wikipedia article title: underscores -> spaces, whitespace collapsed,
    lowercased. The single join key between a proof's article and a corpus chunk's article."""
    return " ".join(str(title).replace("_", " ").split()).lower()


def _article_from_url(url: str) -> str:
    """Normalized article identity for a proof's `found_in_url`. Proofs use title-slug URLs
    (".../wiki/The_Helen_Morgan_Story"), so take the slug after `/wiki/`, url-decode it, and
    normalize. Falls back to the raw (normalized) URL for the rare non-`/wiki/` form."""
    path = urlparse(str(url)).path
    marker = "/wiki/"
    if marker in path:
        return _norm_title(unquote(path.split(marker, 1)[1]))
    return _norm_title(unquote(url))


class _ChromaDocMap:
    """Lazy `chunk_id -> passage text`, served from the Chroma `documents` column so eval never
    materializes all ~25.9M chunk texts in RAM (a plain dict would be tens of GB). The row id IS the
    chunk_id (set at index-build time), and the systems only do keyed lookups (`.get` / `[]` / `in`),
    never iterate — so this stands in for the dict. A small LRU keeps recently-read chunks hot.
    Mirrors TREC-BioGen's lazy doc map."""

    def __init__(self, collection, cache_size: int = 4096) -> None:
        self._collection = collection
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cap = cache_size

    def get(self, chunk_id, default=None):
        key = str(chunk_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        got = self._collection.get(ids=[key], include=["documents"])
        docs = got.get("documents") or []
        text = docs[0] if docs and docs[0] else None
        if text is None:
            return default
        self._cache[key] = text
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return text

    def __getitem__(self, chunk_id):
        text = self.get(chunk_id)
        if text is None:
            raise KeyError(chunk_id)
        return text

    def __contains__(self, chunk_id) -> bool:
        return self.get(chunk_id) is not None


class QampariBenchmark(Benchmark):
    name = QAMPARI
    config: QampariConfig

    # QAMPARI questions have many correct answers; the metric is entity recall, so the answer must
    # be an exhaustive list. Push for breadth/recall over a single best answer.
    answer_format_hint = (
        "This question has MANY correct answers. List every entity that answers it that the "
        "documents support — as many as you can find — as a comma-separated list of entity names. "
        "Do not stop at one answer; prefer completeness (recall) over brevity, and include an "
        "entity whenever the documents support it."
    )

    def __init__(self, config: QampariConfig) -> None:
        config.chromadb_dir = str(resolve_under_skunk(config.chromadb_dir))
        config.questions_path = str(resolve_under_skunk(config.questions_path))
        if config.test_ids_path:
            config.test_ids_path = str(resolve_under_skunk(config.test_ids_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_skunk(config.prompts_path))
        # the chroma collection, set in _build_resources; recall_metrics reads the `title` metadata of
        # the retrieved chunk_ids from it to collapse them to Wikipedia articles.
        self._collection = None
        super().__init__(config)

    # ---- questions ------------------------------------------------------------

    @staticmethod
    def _gold_articles(rec: dict) -> list[str]:
        """Normalized titles of every Wikipedia article cited by any answer's proof, de-duplicated
        in first-seen order. These are the gold supporting documents for retrieval recall."""
        articles: list[str] = []
        seen: set[str] = set()
        for ans in rec.get("answer_list", []):
            for pr in ans.get("proof", []):
                url = pr.get("found_in_url")
                if not url:
                    continue
                art = _article_from_url(url)
                if art and art not in seen:
                    seen.add(art)
                    articles.append(art)
        return articles

    def load_questions(self) -> list[Question]:
        questions: list[Question] = []
        with open(self.config.questions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                answers = rec.get("answer_list", [])
                # one nugget per gold answer entity (KARL: "each entity is a separate nugget").
                nuggets = [str(a["answer_text"]).strip() for a in answers if str(a.get("answer_text", "")).strip()]
                # aliases per answer, for reference / debugging (the judge handles aliases semantically).
                aliases = {str(a.get("answer_text", "")): list(a.get("aliases", [])) for a in answers}
                questions.append(
                    Question(
                        qid=str(rec["qid"]),
                        text=str(rec["question_text"]),
                        gold=", ".join(nuggets),
                        gold_docs=self._gold_articles(rec),
                        meta={
                            "nuggets": nuggets,
                            "n_nuggets": len(nuggets),
                            "aliases": aliases,
                        },
                    )
                )
        return questions

    # ---- retrieval substrate --------------------------------------------------

    def _build_resources(self) -> BenchmarkResources:
        collection = self._open_chroma_collection()
        self._collection = collection
        # ~25.9M chunks is far too much for an in-RAM {chunk_id: text} dict, so serve the passage text
        # lazily + cached from the chroma `documents` column (only the chunks a question actually reads
        # get materialized). The systems only do keyed lookups, so the lazy mapping is a drop-in.
        return BenchmarkResources(
            chroma_collection=collection,
            document_map=_ChromaDocMap(collection),
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
        # Relevance is labeled at the Wikipedia-article level (a proof cites a supporting article), so
        # collapse retrieved chunk_ids to their article title and report article-level doc_recall. The
        # title lives in each chunk's chroma metadata, so look up just the retrieved ids (one .get).
        if retrieved is None:
            ret_articles: list[str] | None = None
        elif not retrieved:
            ret_articles = []
        else:
            got = self._collection.get(ids=list(retrieved), include=["metadatas"])
            id_to_title = {
                str(cid): _norm_title((meta or {}).get("title", ""))
                for cid, meta in zip(got.get("ids") or [], got.get("metadatas") or [])
            }
            ret_articles = [id_to_title.get(str(c), "") for c in retrieved]
        return {"doc_recall": doc_recall(ret_articles, question.gold_docs)}
