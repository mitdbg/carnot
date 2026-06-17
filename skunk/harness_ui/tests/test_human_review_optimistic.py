"""Optimistic (non-blocking) human review: the kind-gated trigger, the register hooks, and the
deterministic recompute path that revises an answer after a review resolves."""

from __future__ import annotations

import asyncio

import pytest

from skunk.common import AnnotatedValue, ExecutionContext, Final
from skunk.config import SkunkConfig
from skunk.human import (
    POOL_REVIEW_BRANCH_ID,
    HumanAssist,
    HumanAssistPolicy,
    apply_overrides,
)
from skunk.plan import LookupBranch, RetrieveBranch


def _scalar(v=5):
    return AnnotatedValue(description="d", value=v, kind="scalar")


def _table():
    return AnnotatedValue(
        description="d", value={"r": {"c": 1}}, kind="table", row_name="r", col_name="c"
    )


def _vector():
    return AnnotatedValue(
        description="d", value={"2020": 1.0}, kind="vector", index_name="year"
    )


# ── verify trigger gates only on the flag, not the value shape ──────────────────
@pytest.mark.parametrize("entry", [_scalar(), _table(), _vector()])
def test_all_extract_kinds_open_a_review_when_flag_on(entry) -> None:
    cfg = SkunkConfig(human_verify_extract=True)
    pol = HumanAssistPolicy()
    assert pol.verify_extract(RetrieveBranch(key="x"), [entry], cfg) is True


def test_verify_extract_flag_off_means_no_review_even_for_tables() -> None:
    cfg = SkunkConfig(human_verify_extract=False)
    pol = HumanAssistPolicy()
    assert pol.verify_extract(RetrieveBranch(key="x"), [_table()], cfg) is False


def test_obtained_visually_uses_figure_flag_regardless_of_kind() -> None:
    # A vision-read value gates on human_figure regardless of value kind; the trigger is
    # the machine-stamped `obtained_visually`, not the branch's `visual_only` prediction.
    pol = HumanAssistPolicy()
    branch = RetrieveBranch(key="x")
    vis = AnnotatedValue(description="d", value=5, kind="scalar", obtained_visually=True)
    assert pol.verify_extract(branch, [vis], SkunkConfig(human_figure=True)) is True
    assert pol.verify_extract(branch, [vis], SkunkConfig(human_figure=False)) is False


# ── register hooks ─────────────────────────────────────────────────────────────
def _ctx_with_register(captured: list[dict]) -> ExecutionContext:
    def register(task, instruction, question, source_docs, guidance):
        captured.append(
            {
                "task": task,
                "instruction": instruction,
                "question": question,
                "source_docs": source_docs,
                "guidance": guidance,
            }
        )
        return "rev-1"

    return ExecutionContext(
        question="q",
        config=SkunkConfig(),
        llm_client=object(),  # type: ignore[arg-type]
        human_review_register=register,
    )


def _table_with_pages() -> AnnotatedValue:
    return AnnotatedValue(
        description="d",
        value={"r": {"c": 1}},
        kind="table",
        row_name="r",
        col_name="c",
        bulletin="1954-02",
        pages=(17,),
    )


def test_register_pool_review_carries_candidates_and_pulled_pages() -> None:
    captured: list[dict] = []
    ctx = _ctx_with_register(captured)
    assist = HumanAssist()
    try:
        rid = assist.register_pool_review([_table_with_pages()], ctx)
    finally:
        ctx.close()
    assert rid == "rev-1"
    g = captured[0]["guidance"]
    # The single data-prep pool review: verify_extract task, keyed by the sentinel id, no single
    # branch identity, candidates = the cleaned pool, source docs = the pages the values came from.
    assert captured[0]["task"] == "verify_extract"
    assert g["branch_id"] == POOL_REVIEW_BRANCH_ID
    assert g["branch"] == {}
    assert g["candidates"][0]["value"] == {"r": {"c": 1}}
    assert captured[0]["source_docs"] == ["Treasury Bulletin 1954-02 PDF page 17"]


def test_register_lookup_carries_agent_candidates() -> None:
    captured: list[dict] = []
    ctx = _ctx_with_register(captured)
    assist = HumanAssist()
    branch = LookupBranch(target="CPI Dec 2020", src="FRED")
    try:
        assist.register_lookup([_scalar("260.474")], branch, 1, ctx)
    finally:
        ctx.close()
    assert captured[0]["task"] == "lookup"
    assert captured[0]["guidance"]["branch"]["target"] == "CPI Dec 2020"
    assert captured[0]["guidance"]["candidates"][0]["value"] == "260.474"


def test_register_is_noop_without_a_hook() -> None:
    ctx = ExecutionContext(question="q", config=SkunkConfig(), llm_client=object())  # type: ignore[arg-type]
    try:
        assert HumanAssist().register_pool_review([_table()], ctx) is None
    finally:
        ctx.close()


# ── override apply (source-indexed) + recompute ─────────────────────────────────
def test_apply_overrides_preserves_provenance_and_structure() -> None:
    base = [
        AnnotatedValue(
            description="orig",
            value={"2020": 1.0},
            kind="vector",
            index_name="year",
            unit="pct",
            bulletin="1954-02",
            pages=(17,),
        )
    ]
    # Human edits only description + value (decluttered editor sends _src + editable fields).
    items = [{"_src": 0, "description": "fixed", "value": {"2020": 9.0}}]
    out = apply_overrides(items, base)
    assert out[0].value == {"2020": 9.0}
    assert out[0].description == "fixed"
    # kind/index_name + provenance preserved from base (not sent by the editor).
    assert out[0].kind == "vector"
    assert out[0].index_name == "year"
    assert out[0].bulletin == "1954-02"
    assert out[0].pages == (17,)


def test_apply_overrides_drops_deleted_entries() -> None:
    base = [
        AnnotatedValue(description="a", value=1, kind="scalar"),
        AnnotatedValue(description="b", value=2, kind="scalar"),
        AnnotatedValue(description="c", value=3, kind="scalar"),
    ]
    # The human deleted the middle box: only _src 0 and 2 come back.
    items = [{"_src": 0, "value": 1}, {"_src": 2, "value": 30}]
    out = apply_overrides(items, base)
    assert [e.value for e in out] == [1, 30]


def test_recompute_answer_applies_override_and_keeps_other_branches(
    monkeypatch,
) -> None:
    from skunk import orchestrator as orch_mod

    seen: dict[str, list] = {}

    class FakeCompute:
        async def run(self, entries, ctx, concept_explanations=()):
            seen["entries"] = list(entries)
            return Final("ANSWER=" + ",".join(str(e.value) for e in entries))

    monkeypatch.setattr(orch_mod, "ComputeOp", FakeCompute)

    state = orch_mod.RecomputeState(
        question="q",
        order=[0, 1],
        entries_by_branch={
            0: [
                AnnotatedValue(
                    description="a",
                    value=1,
                    kind="scalar",
                    bulletin="1954-02",
                    pages=(5,),
                )
            ],
            1: [AnnotatedValue(description="b", value=2, kind="scalar")],
        },
        extra_entries=[AnnotatedValue(description="extra", value=9, kind="scalar")],
        explanations=[],
    )
    # Human corrects only branch 0's value (10 instead of 1) via a source-indexed item.
    overrides = {0: [{"_src": 0, "value": 10}]}
    ctx = ExecutionContext(question="q", config=SkunkConfig(), llm_client=object())  # type: ignore[arg-type]
    try:
        answer = asyncio.run(orch_mod.recompute_answer(state, overrides, ctx))
    finally:
        ctx.close()
    # branch 0 overridden -> 10; branch 1 unchanged -> 2; extra entry preserved -> 9.
    assert answer == "ANSWER=10,2,9"
    # provenance from the cached branch-0 entry survived the override.
    assert seen["entries"][0].bulletin == "1954-02"
    assert seen["entries"][0].pages == (5,)
