"""HumanAssist routed through the async broker channel (BrokerChannel).

Covers the unification of the `human.py` verification gates (verify_extract / figure /
lookup) with the competition harness's `HumanInterventionHandler`: the model's candidate is
shipped to the handler, the worker's corrected `AnnotatedValue`(s) ride back JSON-encoded in
`response`, an empty response means "accept as-is", and provenance is re-stamped on return.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from skunk.common import AnnotatedValue, ExecutionContext, PageRef
from skunk.config import SkunkConfig
from skunk.errors import ParseError
from skunk.human import BrokerChannel, HumanAssist, HumanRequest
from skunk.plan import LookupBranch, RetrieveBranch


def _ctx() -> ExecutionContext:
    return ExecutionContext(
        question="What were total receipts?",
        config=SkunkConfig(),
        llm_client=object(),  # type: ignore[arg-type]
    )


def _handler(captured: list[dict], result: dict):
    async def handler(kind, instructions, context, source_docs, guidance):
        captured.append(
            {
                "kind": kind,
                "instructions": instructions,
                "context": context,
                "source_docs": source_docs,
                "guidance": guidance,
            }
        )
        return result

    return handler


def test_broker_channel_parses_corrected_value() -> None:
    captured: list[dict] = []
    handler = _handler(
        captured,
        {
            "response": json.dumps(
                {"description": "Total receipts", "value": 43, "unit": "millions"}
            )
        },
    )
    req = HumanRequest(
        task="verify_extract",
        instruction="Confirm or correct the value.",
        candidates=[
            AnnotatedValue(description="Total receipts", value=42, unit="millions")
        ],
        pages=[PageRef(stem="1954-02", page=17)],
    )
    ctx = _ctx()
    try:
        out = asyncio.run(BrokerChannel(handler).ask(req, ctx))
    finally:
        ctx.close()

    assert [v.value for v in out] == [43]
    call = captured[0]
    assert call["kind"] == "verify_extract"
    assert call["context"] == "What were total receipts?"
    assert call["source_docs"] == ["Treasury Bulletin 1954-02 PDF page 17"]
    assert call["guidance"]["task"] == "verify_extract"
    assert call["guidance"]["candidates"][0]["value"] == 42
    assert "value" in call["guidance"]["fields"]


def test_broker_channel_empty_response_accepts_candidates() -> None:
    candidates = [AnnotatedValue(description="Total receipts", value=42)]
    req = HumanRequest(
        task="verify_extract", instruction="...", candidates=candidates, pages=[]
    )
    ctx = _ctx()
    try:
        out = asyncio.run(BrokerChannel(_handler([], {"response": ""})).ask(req, ctx))
    finally:
        ctx.close()
    assert out == candidates


def test_broker_channel_parse_failure_falls_back_to_candidates() -> None:
    candidates = [AnnotatedValue(description="Total receipts", value=42)]
    req = HumanRequest(
        task="verify_extract", instruction="...", candidates=candidates, pages=[]
    )
    ctx = _ctx()
    try:
        out = asyncio.run(
            BrokerChannel(_handler([], {"response": "not json at all"})).ask(req, ctx)
        )
    finally:
        ctx.close()
    assert out == candidates  # malformed correction ignored, branch not crashed


def test_broker_channel_parse_failure_without_candidates_raises() -> None:
    req = HumanRequest(task="lookup", instruction="...", candidates=[], pages=[])
    ctx = _ctx()
    try:
        with pytest.raises(ParseError):
            asyncio.run(
                BrokerChannel(_handler([], {"response": "not json"})).ask(req, ctx)
            )
    finally:
        ctx.close()


def _source_page() -> PageRef:
    return PageRef(stem="1954-02", page=17)


def test_human_assist_verify_extract_restamps_provenance_via_broker() -> None:
    captured: list[dict] = []
    handler = _handler(
        captured,
        {
            "response": json.dumps(
                {"description": "Total receipts", "value": 43, "unit": "millions"}
            )
        },
    )
    assist = HumanAssist(channel=BrokerChannel(handler))
    branch = RetrieveBranch(key="total receipts", period="1954-02")
    entries = [AnnotatedValue(description="Total receipts", value=42, unit="millions")]
    ctx = _ctx()
    try:
        out = asyncio.run(assist.verify_extract(entries, [_source_page()], branch, ctx))
    finally:
        ctx.close()

    assert out[0].value == 43
    # Provenance is machine-stamped from the source page + branch, not the human.
    assert out[0].bulletin == "1954-02"
    assert out[0].pages == (17,)
    assert out[0].retrieve_key == "total receipts"
    assert captured[0]["kind"] == "verify_extract"


def test_human_assist_figure_branch_uses_figure_kind() -> None:
    captured: list[dict] = []
    handler = _handler(
        captured,
        {"response": json.dumps({"description": "Peak", "value": 7})},
    )
    assist = HumanAssist(channel=BrokerChannel(handler))
    branch = RetrieveBranch(key="peak of the chart")
    ctx = _ctx()
    try:
        # A value the vision tier produced (`obtained_visually`) drives the figure kind,
        # regardless of the branch's `visual_only` prediction.
        out = asyncio.run(
            assist.verify_extract(
                [AnnotatedValue(description="Peak", value=3, obtained_visually=True)],
                [_source_page()],
                branch,
                ctx,
            )
        )
    finally:
        ctx.close()
    assert out[0].value == 7
    assert captured[0]["kind"] == "figure"


