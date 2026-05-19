"""retrieve operator — placeholder pending integration of an external
retrieve implementation.

Honors `ctx.config.golden_pages` (the `--golden` eval path), so the dev
loop with `eval/eval_e2e.py --golden` keeps working. Any non-golden call
raises `NotImplementedError`: this operator is intentionally unwired
while we slot in someone else's retriever.

The previous page-index logic (chapter pick → year filter → optional
BM25 rerank) is preserved under
`skunk.page_index.retrieve_prototype.PageIndexRetrievePrototype` for
regression comparison.
"""

from __future__ import annotations

from skunk.models import HarnessContext, PageRef
from skunk.plan import RetrieveBranch


class RetrieveExecutor:
    """Placeholder retrieve operator. Golden-bypass only; raises on any
    real retrieval call until an external retriever is wired in."""

    def run(self, ctx: HarnessContext, branch: RetrieveBranch) -> list[PageRef]:
        if ctx.config.golden_pages is not None:
            ctx.emit("retrieve", "golden bypass",
                     n_pages=len(ctx.config.golden_pages),
                     refs=[str(r) for r in ctx.config.golden_pages])
            return ctx.config.golden_pages
        raise NotImplementedError(
            "RetrieveExecutor is a placeholder pending external-retriever "
            f"integration; cannot handle key={branch.key!r} period={branch.period!r}. "
            "Pass `--golden` to the eval harness, or use "
            "`skunk.page_index.retrieve_prototype.PageIndexRetrievePrototype` "
            "directly for the legacy page-index pipeline."
        )
