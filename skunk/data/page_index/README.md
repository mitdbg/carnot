# `data/page_index/` — shipped Treasury Bulletin retriever

A two-layer retriever from a planner-emitted retrieve branch
(`concept`, `period`, plus the user `question`) to a list of candidate
`(bulletin, page)` pages. Built by the three-phase pipeline at
`src/skunk/page_index/pipeline.py`.

## Contents

| Path | Description |
|---|---|
| `concept_tree.json` | The flat tree. 8 canonical chapters, 77,789 indexed pages. Each chapter has a `description`, a list of `examples` (sub-area names), and its full `pages` list. |
| `retrieve.py` | Self-contained Python module exposing `RetrievalIndex.retrieve_branch(question, concept, period, llm)` → list of `(bulletin, page)` pairs. Stdlib only. |
| `manifest.json` | Build metadata: git sha, model versions, page/chapter counts. |

The build also produces per-bulletin catalog rows (`catalog/`) and L1
chapter spans (`l1/`); those are build-time intermediates and not
shipped. They live under `cache/page_index_v3/` if you rebuild.

## Retrieval architecture

Top-level: question string in, list of `(bulletin, page)` candidates
out. Inside the module:

0. **Mini-plan** — one LLM call decomposes the OfficeQA question into
   `(concept, period)` branches. Compound questions
   ("compare X between 1934 and 1946", "list top sources of revenue
   in FY1991") fan out into multiple branches that run in parallel.
1. **L1 chapter pick** (per branch) — one LLM call selects a canonical
   chapter from the question + concept tag, using `description` +
   `examples` per chapter as context.
2. **Period filter** (per branch) — bulletin-month window check: a
   page passes if its publication month is within
   `[period_start, period_end + 12 months]`. The +12mo lag captures
   retrospective tables that report on a closed period in a later
   issue.
3. **Union** the surviving `(bulletin, page)` pairs across branches.

No vector index, no embeddings — pure L1 + period. Headline numbers on
the 101-UID dev set (32 UIDs held out per
`eval/test_set_uids.json`):

| Metric | Value |
|---|---|
| Micro-recall | **~98.6%** at the L1 ceiling (period mask is recall-preserving) |
| Mean candidate pages per branch | ~2,700 (median ~1,800) |
| LLM cost per branch | ~$0.0005 (one chapter-pick call) |

Earlier experiments layered a vector top-K on top of L1+period. The
vector ranking turned out to be subtractive at any K ≤ 1000 against
this corpus + blob shape (cost 3 pp recall to halve page count), so
the vector layer is not shipped.

## Tree shape

```json
{
  "chapters": {
    "Federal Debt": {
      "n_pages": 28503,
      "description": "Extensive data on the public debt of the United States, ...",
      "examples": [
        "Public Debt Operations",
        "Ownership of Federal Securities",
        "Market Quotations on Treasury Securities",
        "Average Yields of Long-Term Bonds",
        "U.S. Savings Bonds and Notes"
      ],
      "pages": [
        {"bulletin": "1972-05", "page": 50,
         "key_phrases": ["Budget Receipts", "Principal Sources"]},
        ...
      ]
    },
    ...
  }
}
```

## Use it

```python
from pathlib import Path
from data.page_index.retrieve import RetrievalIndex

idx = RetrievalIndex.load(Path("data/page_index/concept_tree.json"))

# Bring your own LLM as a (system_prompt, user_prompt) -> text callable.
def my_llm(system: str, user: str) -> str:
    return openai_client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0,
    ).choices[0].message.content

# Single call: OfficeQA question in, candidate pages out.
pages = idx.retrieve(
    "What was the OASI trust fund balance at end of CY1953?",
    my_llm,
)
# -> list[tuple[str, int]]  e.g. [("1953-12", 35), ("1954-01", 37), ...]
```

If you already have `(concept, period)` from your own planner, skip
the mini-plan step:

```python
pages = idx.retrieve_branch(question, concept, period, my_llm)
```

The module has no third-party dependencies — drop it next to
`concept_tree.json` and it works.

## Reproduce

```
python -m skunk.page_index.pipeline \
    --output-dir cache/page_index_v3/ \
    --workers 16
```

Pipeline stages: `build_catalog → extract_l1 → place_pages → merge_chapters → manifest`.
Total cost ≈ $0.55, ~10 min wall-clock with 16 workers. To promote a
new build, copy `concept_tree.json` and `manifest.json` from the cache
dir over this one.

## Don't write to this directory during development.

This is the shipped artifact. New builds should land in
`cache/page_index_v3/`. The `eval/eval_retrieve.py` harness reads the
tree from here by default.
