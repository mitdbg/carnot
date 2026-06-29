"""Semantic filter: an LLM TRUE/FALSE judge applied per document.

Used two ways:
  - As an agent TOOL (`SemanticFilterTool`) the SearchAgent can call mid-loop (system #3).
  - As a fixed one-shot post-step over the agent's K files (system #4).

The judging rubric mirrors carnot's `sem_filter.yaml`. The core is synchronous
(`filter_docs`) because the tool runs inside skunk's `LocalPythonExecutor`, which
calls tools synchronously; the sync `LLMClient.call` is used per document, fanned
out over a small thread pool.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from skunk.multi_turn_agent import Tool

SEMFILTER_RESULT_TAG = "__semfilter_result__"

_SEMFILTER_SYSTEM = (
    "You determine whether a document satisfies a filter condition. You are given a "
    "filter condition and a document. Output TRUE if the document satisfies the "
    "condition, and FALSE otherwise. Your reply must begin with exactly TRUE or FALSE."
)


def _parse_bool(text: str) -> bool:
    """True/False from the judge reply. Recall-safe: default TRUE (keep) if unclear."""
    t = (text or "").strip().lower()
    if t.startswith("false"):
        return False
    if t.startswith("true"):
        return True
    if "false" in t and "true" not in t:
        return False
    return True


def _judge_one(llm_client, predicate: str, item_text: str, model: str, ctx) -> bool:
    user = f"Filter Condition: {predicate}\n\nDocument:\n{item_text}"
    try:
        resp = llm_client.call(
            system=_SEMFILTER_SYSTEM, user=user, temperature=0.0, model=model, ctx=ctx, call_site="semfilter"
        )
    except Exception:
        return True  # recall-safe: keep on error
    return _parse_bool(resp.text)


def filter_docs(
    llm_client,
    predicate: str,
    doc_ids: list[str],
    document_map: dict[str, str],
    model: str,
    *,
    ctx=None,
    max_workers: int = 8,
) -> list[str]:
    """Return the subset of `doc_ids` whose document text satisfies `predicate`,
    preserving input order."""
    if not doc_ids:
        return []
    texts = [document_map.get(d, "") or "" for d in doc_ids]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(doc_ids))) as pool:
        verdicts = list(pool.map(lambda it: _judge_one(llm_client, predicate, it, model, ctx), texts))
    return [d for d, keep in zip(doc_ids, verdicts, strict=True) if keep]


class SemanticFilterTool(Tool):
    name = "semantic_filter"
    doc = """\
### semantic_filter(doc_ids: list[str], predicate: str)
Keep only the documents whose full text satisfies a natural-language `predicate`.
Each document in `doc_ids` is judged independently by an LLM (TRUE/FALSE) against
the predicate; the tool returns the `doc_id`s that passed. Use this to narrow a set
of candidate documents down to those actually relevant before reading them. This tool
is more powerful than vector search for identifying semantically relevant documents,
but it is also more expensive. Only execute it on document sets of <=1,000 documents.

```python
semantic_filter(doc_ids=["doc_id_1", "doc_id_2"], predicate="discusses topic X in relation to Y")
```"""

    def __init__(self, llm_client, document_map: dict[str, str], model: str, ctx=None) -> None:
        self._llm_client = llm_client
        self._document_map = document_map
        self._model = model
        self._ctx = ctx

    def __call__(self, doc_ids: list[str], predicate: str) -> dict:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        kept = filter_docs(self._llm_client, predicate, list(doc_ids), self._document_map, self._model, ctx=self._ctx)
        return {SEMFILTER_RESULT_TAG: True, "kept_doc_ids": kept, "n_in": len(doc_ids), "n_out": len(kept)}
