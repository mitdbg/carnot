"""Focused unit tests for the human-in-the-loop middleware (`skunk.human`).

No external services and no PDFs: policy gating, reply parsing, the console channel's
accept/override stdin protocol, and provenance re-stamping all run on empty blocks /
fake stdin. Runs under pytest if installed, or standalone: `python3 tests/test_human.py`.
"""

from __future__ import annotations

import asyncio
import io
import sys

from skunk.common import AnnotatedValue, ExecutionContext
from skunk.config import SkunkConfig
from skunk.errors import ParseError
from skunk.human import (
    ConsoleChannel,
    HumanAssist,
    HumanAssistPolicy,
    HumanRequest,
    _candidates_json,
    _parse_reply,
)
from skunk.plan import LookupBranch, RetrieveBranch


def _ctx(cfg: SkunkConfig) -> ExecutionContext:
    # Dummy llm_client so __post_init__ doesn't build a real one (no API key needed).
    return ExecutionContext(question="q", config=cfg, llm_client=object())


def _cfg(**flags: bool) -> SkunkConfig:
    base = dict(human_figure=False, human_verify_extract=False)
    base.update(flags)
    return SkunkConfig(**base)


# ---- policy gating ---------------------------------------------------------------


def test_policy_figure_gates_on_human_figure():
    # The figure gate triggers on a value the vision tier actually produced
    # (`obtained_visually`), not the branch's up-front `visual_only` prediction.
    p = HumanAssistPolicy()
    br = RetrieveBranch(key="k")
    vis = [AnnotatedValue(description="d", value=1, obtained_visually=True)]
    assert p.verify_extract(br, vis, _cfg(human_figure=True)) is True
    assert p.verify_extract(br, vis, _cfg(human_verify_extract=True)) is False
    assert p.verify_extract(br, vis, _cfg()) is False


def test_policy_verify_gates_on_human_verify_extract():
    p = HumanAssistPolicy()
    txt = RetrieveBranch(key="k", visual_only=False)
    assert p.verify_extract(txt, [], _cfg(human_verify_extract=True)) is True
    assert p.verify_extract(txt, [], _cfg(human_figure=True)) is False
    assert p.verify_extract(txt, [], _cfg()) is False


# ---- reply parsing ---------------------------------------------------------------


def test_parse_reply_single_object():
    out = _parse_reply('{"description": "d", "value": 18, "unit": "count"}')
    assert len(out) == 1 and out[0].value == 18 and out[0].description == "d"


def test_parse_reply_list():
    out = _parse_reply(
        '[{"description": "a", "value": 1}, {"description": "b", "value": 2}]'
    )
    assert [v.value for v in out] == [1, 2]


def test_parse_reply_rejects_bad_json():
    for bad in (
        "not json",
        "[]",
        '{"value": 1}',
    ):  # malformed / empty / missing description
        try:
            _parse_reply(bad)
        except ParseError:
            continue
        raise AssertionError(f"expected ParseError for {bad!r}")


def test_candidates_json_drops_provenance():
    av = AnnotatedValue(
        description="d",
        value=18,
        unit="count",
        retrieve_key="secret",
        doc_id="combined_statement__modern__1990__c10",
    )
    dumped = _candidates_json([av])
    assert (
        "secret" not in dumped
        and "combined_statement__modern__1990__c10" not in dumped
        and '"value": 18' in dumped
    )


# ---- console channel stdin protocol ----------------------------------------------


def _ask(channel, req, cfg, stdin_text: str):
    saved = sys.stdin
    sys.stdin = io.StringIO(stdin_text)
    try:
        return asyncio.run(channel.ask(req, _ctx(cfg)))
    finally:
        sys.stdin = saved


def test_console_blank_accepts_candidates():
    cand = [AnnotatedValue(description="d", value=13)]
    req = HumanRequest(task="figure", instruction="i", candidates=cand)
    out = _ask(ConsoleChannel(), req, _cfg(human_figure=True), "\n")
    assert out == cand  # accepted unchanged


def test_console_override_parses_reply():
    cand = [AnnotatedValue(description="d", value=13)]
    req = HumanRequest(task="figure", instruction="i", candidates=cand)
    out = _ask(
        ConsoleChannel(),
        req,
        _cfg(human_figure=True),
        '{"description": "d", "value": 18}\nEND\n',
    )
    assert len(out) == 1 and out[0].value == 18


# ---- HumanAssist facade: re-stamping + lookup ------------------------------------


class _FakeChannel:
    def __init__(self, reply):
        self.reply = reply
        self.requests: list[HumanRequest] = []

    async def ask(self, req, ctx):
        self.requests.append(req)
        return list(self.reply)


def test_verify_extract_restamps_branch_provenance():
    # Empty blocks → no rendering needed; branch fields still get stamped on the reply.
    branch = RetrieveBranch(
        key="cpi level", period="2020-01", as_of="2020-02", visual_only=True
    )
    reply = [AnnotatedValue(description="human answer", value=18)]
    assist = HumanAssist(channel=_FakeChannel(reply))
    cfg = _cfg(human_figure=True)
    assert assist.wants_verify(branch, [], _ctx(cfg)) is True
    out = asyncio.run(assist.verify_extract(reply, [], branch, _ctx(cfg)))
    assert len(out) == 1
    assert out[0].value == 18
    assert out[0].retrieve_key == "cpi level"
    assert out[0].requested_period == "2020-01"
    assert out[0].as_of == "2020-02"


if __name__ == "__main__":
    fns = [
        v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
