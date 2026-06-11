from __future__ import annotations

import asyncio
import sys
import types

import pytest

from skunk.common import AnnotatedValue, BlockRef, ExecutionContext, PageRef
from skunk.config import SkunkConfig
from skunk.errors import MissingData, StepFailed
from skunk.orchestrator import BranchOutcome, Orchestrator
from skunk.page_index.data_model import ContentBlock, PageCatalogRow
from skunk.page_index.query import PageIndexRetriever
from skunk.plan import Plan, PlanDiff, Planner, RetrieveBranch
from skunk.retrieve import RetrieveOp
from skunk_reasoner import _structured_reasoning_payload


class _Planner:
    def __init__(self) -> None:
        self.replan_calls: list[
            tuple[
                list[AnnotatedValue],
                str,
                list[str],
                list[tuple[int, list[str]]],
            ]
        ] = []

    async def plan(self, _question, _ctx) -> Plan:
        return Plan(branches=[RetrieveBranch(key="value")])

    async def replan(
        self,
        _ctx,
        _plan,
        entries,
        _failed,
        reason,
        missing,
        human_resolutions,
    ) -> PlanDiff:
        self.replan_calls.append(
            (
                list(entries),
                reason,
                list(missing),
                list(human_resolutions),
            )
        )
        return PlanDiff()


class _Explainer:
    async def run(self, _ctx, *, question):
        return []


class _Compute:
    def __init__(self) -> None:
        self.calls = 0
        self.inputs: list[list[AnnotatedValue]] = []

    async def run(self, entries, _ctx, *, concept_explanations):
        self.calls += 1
        self.inputs.append(list(entries))
        if self.calls == 1:
            raise MissingData("need value", ["value"])
        assert any(entry.value == "42" for entry in entries)
        return "final"


async def _empty_branches(_branches, _branch_ids):
    page = PageRef(month="1954-02", page=17)
    block = BlockRef(
        page=page,
        block_index=2,
        member_refs=(page,),
        block=ContentBlock(
            kind="table",
            title="Example Table",
            column_headers=["Amount"],
            row_headers=["Example row"],
            summary="Example table used by the human-intervention test.",
        ),
    )
    return [
        BranchOutcome(
            branch=branch,
            entries=[
                AnnotatedValue(
                    description="Example value",
                    value={"Example row": {"Amount": 7}},
                    unit="millions",
                    kind="table",
                    row_name="Category",
                    col_name="Measure",
                    bulletin="1954-02",
                    pages=(17,),
                    source_block_page=17,
                    source_block_index=2,
                )
            ],
            error=None,
            blocks=[block],
        )
        for branch in _branches
    ]


def test_missing_data_requires_human_before_replan() -> None:
    calls: list[tuple[str, str, str | None, list[str], dict]] = []

    async def human_handler(kind, instructions, context, source_docs, guidance):
        calls.append((kind, instructions, context, source_docs, guidance))
        return {"response": "42", "source_docs": ["source"]}

    planner = _Planner()
    orchestrator = Orchestrator(
        "q1",
        llm_client=object(),  # type: ignore[arg-type]
        human_intervention_handler=human_handler,
    )
    orchestrator._planner = planner
    orchestrator._explainer = _Explainer()
    compute = _Compute()
    orchestrator._compute = compute
    orchestrator._run_branches = _empty_branches  # type: ignore[method-assign]
    page = PageRef(month="1954-02", page=17)
    orchestrator._retrieved_blocks = [
        BlockRef(page=page, block_index=None, member_refs=(page,), block=None)
    ]

    try:
        result = asyncio.run(orchestrator.execute())
    finally:
        orchestrator.ctx.close()

    assert result == "final"
    assert calls[0][0] == "missing_data"
    assert "need value" in calls[0][1]
    assert "value" in calls[0][1]
    assert calls[0][4]["recovery_round"] == 1
    assert calls[0][4]["partial_plan"][0]["key"] == "value"
    previous = calls[0][4]["previous_round_plan"][0]
    assert previous["branch_id"] == 0
    assert previous["searched"]["key"] == "value"
    assert previous["blocks"][0]["bulletin"] == "1954-02"
    assert previous["blocks"][0]["page"] == 17
    assert previous["blocks"][0]["block_index"] == 2
    assert previous["blocks"][0]["title"] == "Example Table"
    assert previous["output"]["type"] == "values"
    assert previous["output"]["values"][0] == {
        "description": "Example value",
        "unit": "millions",
        "value_kind": "table",
        "value": {"Example row": {"Amount": 7}},
        "index_name": None,
        "row_name": "Category",
        "col_name": "Measure",
        "bulletin": "1954-02",
        "pages": [17],
        "source_block_page": 17,
        "source_block_index": 2,
    }
    assert previous["output_step"] == "extract"
    assert previous["status"] == "ok"
    assert previous["execution_status"] == "executed"
    assert previous["outcome_status"] == "succeeded"
    assert calls[0][3] == ["Treasury Bulletin 1954-02 PDF page 17"]
    assert calls[0][4]["likely_pages"] == [
        {"bulletin": "1954-02", "page": 17}
    ]
    assert planner.replan_calls == []
    assert compute.calls == 2
    assert compute.inputs[1][-1].description == "Human-provided value for: value (sources: source)"
    assert compute.inputs[1][-1].value == "42"
    assert (
        _structured_reasoning_payload(orchestrator.ctx.events, [])["summary"][
            "replan_count"
        ]
        == 0
    )


def test_previous_round_plan_marks_carried_and_failed_branches() -> None:
    calls: list[dict] = []

    class Planner(_Planner):
        async def replan(
            self,
            _ctx,
            _plan,
            entries,
            _failed,
            reason,
            missing,
            human_resolutions,
        ) -> PlanDiff:
            self.replan_calls.append(
                (
                    list(entries),
                    reason,
                    list(missing),
                    list(human_resolutions),
                )
            )
            if len(self.replan_calls) == 1:
                return PlanDiff(add=[RetrieveBranch(key="new value")])
            return PlanDiff()

    class Compute:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, _entries, _ctx, *, concept_explanations):
            self.calls += 1
            if self.calls == 1:
                raise MissingData("need original value", ["original value"])
            if self.calls == 2:
                raise MissingData("human value was insufficient", ["new value"])
            if self.calls == 3:
                raise MissingData("need another value", ["new value"])
            return "final"

    async def human_handler(_kind, _instructions, _context, _source_docs, guidance):
        calls.append(guidance)
        return {"response": "42", "source_docs": []}

    async def run_branches(branches, _branch_ids):
        if branches[0].key == "value":
            return [BranchOutcome(branch=branches[0], entries=[], error=None)]
        error = StepFailed(
            "retrieve",
            "no matching blocks",
            details={
                "considered_pages": [{"bulletin": "1954-02", "page": 9}]
            },
        )
        return [BranchOutcome(branch=branches[0], entries=None, error=error)]

    orchestrator = Orchestrator(
        "q1",
        llm_client=object(),  # type: ignore[arg-type]
        human_intervention_handler=human_handler,
    )
    planner = Planner()
    orchestrator._planner = planner
    orchestrator._explainer = _Explainer()
    orchestrator._compute = Compute()
    orchestrator._run_branches = run_branches  # type: ignore[method-assign]

    try:
        result = asyncio.run(orchestrator.execute())
    finally:
        orchestrator.ctx.close()

    assert result == "final"
    assert len(planner.replan_calls) == 1
    assert planner.replan_calls[0][1] == "human value was insufficient"
    assert planner.replan_calls[0][2] == ["new value"]
    assert planner.replan_calls[0][3] == [(0, ["original value"])]
    assert planner.replan_calls[0][0][0].description == (
        "Human-provided value for: original value"
    )
    assert calls[1]["previous_round_plan"][0]["execution_status"] == "carried_forward"
    assert calls[1]["previous_round_plan"][0]["outcome_status"] == "succeeded"
    assert calls[1]["previous_round_plan"][1]["execution_status"] == "executed"
    assert calls[1]["previous_round_plan"][1]["outcome_status"] == "failed"
    assert calls[1]["previous_round_plan"][1]["considered_pages"] == [
        {"bulletin": "1954-02", "page": 9}
    ]
    assert calls[1]["failed_branches"][0]["branch_id"] == 1
    assert calls[1]["failed_branches"][0]["considered_pages"] == [
        {"bulletin": "1954-02", "page": 9}
    ]


def test_document_only_feedback_reruns_targeted_retrieval_before_replanning() -> None:
    branch_runs: list[tuple[list[int], dict[int, list[str]] | None]] = []

    class Planner(_Planner):
        async def replan(self, *_args, **_kwargs) -> PlanDiff:
            raise AssertionError("successful human-directed retrieval must not replan")

    class Compute:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, entries, _ctx, *, concept_explanations):
            self.calls += 1
            if self.calls == 1:
                raise MissingData("need judiciary outlays", ["judiciary outlays"])
            assert entries[-1].description == "Judiciary outlays"
            assert entries[-1].value == "100"
            return "final"

    async def human_handler(_kind, _instructions, _context, _source_docs, _guidance):
        return {
            "response": "",
            "source_docs": ["Treasury Bulletin 1986-06 PDF"],
            "retrieval_directives": [
                {
                    "branch_id": 0,
                    "documents": ["Treasury Bulletin 1986-06 PDF"],
                }
            ],
        }

    async def run_branches(branches, branch_ids, *, document_scopes=None):
        branch_runs.append((list(branch_ids), document_scopes))
        if document_scopes:
            return [
                BranchOutcome(
                    branch=branches[0],
                    entries=[
                        AnnotatedValue(
                            description="Judiciary outlays",
                            value="100",
                            bulletin="1986-06",
                        )
                    ],
                    error=None,
                )
            ]
        return [
            BranchOutcome(
                branch=branches[0],
                entries=None,
                error=StepFailed("retrieve", "no matching blocks"),
            )
        ]

    orchestrator = Orchestrator(
        "q1",
        llm_client=object(),  # type: ignore[arg-type]
        human_intervention_handler=human_handler,
    )
    orchestrator._planner = Planner()
    orchestrator._explainer = _Explainer()
    orchestrator._compute = Compute()
    orchestrator._run_branches = run_branches  # type: ignore[method-assign]

    try:
        result = asyncio.run(orchestrator.execute())
    finally:
        orchestrator.ctx.close()

    assert result == "final"
    assert branch_runs == [
        ([0], None),
        ([0], {0: ["1986-06"]}),
    ]
    directed_events = [
        event
        for event in orchestrator.ctx.events
        if event["message"] == "human_directed_retrieval"
    ]
    assert directed_events[0]["data"]["directives"] == [
        {"branch_id": 0, "bulletins": ["1986-06"]}
    ]


def test_page_index_human_scope_uses_only_selected_bulletins(tmp_path) -> None:
    retriever = object.__new__(PageIndexRetriever)
    rows = [
        PageCatalogRow(
            bulletin="1985-12",
            page=4,
            content_blocks=[ContentBlock(kind="table", title="Old")],
        ),
        PageCatalogRow(
            bulletin="1986-06",
            page=7,
            content_blocks=[ContentBlock(kind="table", title="Target")],
        ),
        PageCatalogRow(
            bulletin="1987-06",
            page=9,
            content_blocks=[ContentBlock(kind="table", title="New")],
        ),
    ]
    retriever._catalog = {row.ref: row for row in rows}
    retriever._catalog_size = len(rows)
    seen_pages: list[PageRef] = []

    async def semantic_filter(pages, _branches, _ctx):
        seen_pages.extend(pages)
        return {page: [True] for page in pages}

    async def select_blocks(refs, _pdf_dir, _ctx, _question, _branch):
        row = retriever._catalog[refs[0]]
        return [
            BlockRef(
                page=row.ref,
                block_index=0,
                member_refs=(row.ref,),
                block=row.content_blocks[0],
            )
        ]

    retriever._semantic_filter = semantic_filter
    retriever.select_blocks = select_blocks
    ctx = ExecutionContext(
        question="Judiciary outlays",
        config=SkunkConfig(pdf_dir=tmp_path),
        llm_client=object(),  # type: ignore[arg-type]
    )
    try:
        result = asyncio.run(
            retriever.retrieve_all(
                ctx,
                [RetrieveBranch(key="judiciary outlays")],
                document_scopes=[["1986-06"]],
            )
        )
    finally:
        ctx.close()

    assert seen_pages == [PageRef(month="1986-06", page=7)]
    assert result[0][0].page == PageRef(month="1986-06", page=7)


def test_search_agent_discards_pages_outside_human_scope(monkeypatch) -> None:
    class SearchAgent:
        def __init__(self, **_kwargs):
            pass

        async def retrieve(self, *_args, **_kwargs):
            return ["1985_12_4", "1986_06_7", "1987_06_9"]

    search_agent_module = types.ModuleType("skunk.search_agent")
    search_agent_module.SearchAgent = SearchAgent
    monkeypatch.setitem(sys.modules, "skunk.search_agent", search_agent_module)
    op = RetrieveOp(SkunkConfig(retriever="search_agent"))
    op._ensure_resources = lambda _config: (object(), {})  # type: ignore[method-assign]
    ctx = ExecutionContext(
        question="Judiciary outlays",
        config=SkunkConfig(retriever="search_agent"),
        llm_client=object(),  # type: ignore[arg-type]
    )
    try:
        refs = asyncio.run(
            op._run_search_agent(
                ctx,
                RetrieveBranch(key="judiciary outlays"),
                required_bulletins=["1986-06"],
            )
        )
    finally:
        ctx.close()

    assert refs == [PageRef(month="1986-06", page=7)]


def test_replanner_prompt_maps_human_resolution_to_input() -> None:
    class Prompt:
        def __init__(self) -> None:
            self.message = ""

        async def call(self, _ctx, message, *, temperature):
            self.message = message
            return PlanDiff()

    class Context:
        question = "What is the value?"

    prompt = Prompt()
    planner = Planner()
    planner._replan_prompt = prompt  # type: ignore[method-assign]

    asyncio.run(
        planner.replan(
            Context(),  # type: ignore[arg-type]
            Plan(branches=[RetrieveBranch(key="value")]),
            [
                AnnotatedValue(
                    description="Human-provided value for: original value",
                    value="42",
                )
            ],
            [],
            "human value was insufficient",
            ["new value"],
            [(0, ["original value"])],
        )
    )

    assert (
        "prior missing ['original value'] -> input_values[0]"
        in prompt.message
    )
    assert "missing:     ['new value']" in prompt.message


def test_missing_data_without_handler_does_not_replan() -> None:
    planner = _Planner()
    orchestrator = Orchestrator(
        "q1",
        llm_client=object(),  # type: ignore[arg-type]
    )
    orchestrator._planner = planner
    orchestrator._explainer = _Explainer()
    orchestrator._compute = _Compute()
    orchestrator._run_branches = _empty_branches  # type: ignore[method-assign]

    try:
        with pytest.raises(MissingData, match="need value"):
            asyncio.run(orchestrator.execute())
    finally:
        orchestrator.ctx.close()

    assert planner.replan_calls == []
