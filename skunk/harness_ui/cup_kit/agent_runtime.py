"""WS subscription / reconnect / retry plumbing for the OfficeQA Cup agent.

Teams writing a Cup agent should NOT need to read or edit this file —
edit ``reference_agent.py`` instead, which exposes ``solve()`` and
``_process_round()``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from websockets.exceptions import WebSocketException

from cup_kit.client import CupAPIError, CupClient
from cup_kit.protocol import (
    QuestionFragment,
    RoundStatus,
    SubmitRejectionReason,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentAnswer:
    """One question's worth of output from your ``solve`` function.

    - ``answer``: the final answer string.
    - ``reasoning``: chain-of-thought; **must be ≥100 characters** for
      agent submissions (server-enforced).
    - ``source_docs``: optional list of doc references. Empty list is
      accepted; if present, each entry must be non-empty and ≤512 chars.
    """

    answer: str
    reasoning: str
    source_docs: list[str] = field(default_factory=list)


Solve = Callable[[str], Awaitable[AgentAnswer]]
ProcessRound = Callable[
    [
        CupClient,
        Solve,
        dict[tuple[int, str], QuestionFragment],
        dict[tuple[int, str], AgentAnswer],
        set[tuple[int, str]],
    ],
    Awaitable[None],
]


# Rejection reasons that resubmitting the same payload can't fix.
# RATE_LIMITED + ROUND_NOT_ACTIVE are recoverable on retry; everything
# else is terminal for this question.
TERMINAL_SUBMIT_REASONS: frozenset[SubmitRejectionReason] = frozenset(
    set(SubmitRejectionReason)
    - {
        SubmitRejectionReason.RATE_LIMITED,
        SubmitRejectionReason.ROUND_NOT_ACTIVE,
    }
)


CupClientFactory = Callable[[str, str], CupClient]


def _default_cup_client_factory(base_url: str, team_token: str) -> CupClient:
    return CupClient(base_url, team_token)


async def run_agent(
    base_url: str,
    team_token: str,
    solve: Solve,
    process_round: ProcessRound,
    *,
    reconnect_backoff_s: float = 1.0,
    cup_client_factory: CupClientFactory = _default_cup_client_factory,
) -> None:
    """Drive the canonical answer/work loop against the Cup API.

    Subscribes to ``WS /v1/team/events``, accumulates the per-round
    question list from ``round_started`` events, and calls
    ``process_round`` whenever new work might be available. State
    survives reconnects; recoverable rejections leave the question for
    the next pass.
    """
    questions_by_key: dict[tuple[int, str], QuestionFragment] = {}
    answers_by_key: dict[tuple[int, str], AgentAnswer] = {}
    handled: set[tuple[int, str]] = set()

    while True:
        try:
            cup = cup_client_factory(base_url, team_token)
            async with cup:
                if questions_by_key:
                    await process_round(cup, solve, questions_by_key, answers_by_key, handled)
                async for ev in cup.events():
                    if ev.type == "round_started":
                        for q in ev.questions:
                            questions_by_key[(ev.round_num, q.question_id)] = q
                        # Honor the submission grace window if the
                        # server set one (matches the audience
                        # 3-2-1-Go countdown). Well-behaved agents
                        # sleep instead of taking ROUND_NOT_YET_OPEN
                        # rejections and burning a token.
                        if ev.opens_at is not None:
                            now = datetime.now(tz=ev.opens_at.tzinfo)
                            wait_s = (ev.opens_at - now).total_seconds()
                            if wait_s > 0:
                                await asyncio.sleep(wait_s)
                        await process_round(cup, solve, questions_by_key, answers_by_key, handled)
                    elif ev.type == "round_state":
                        if ev.status == RoundStatus.ACTIVE:
                            await process_round(
                                cup,
                                solve,
                                questions_by_key,
                                answers_by_key,
                                handled,
                            )
                    elif ev.type == "submission_scored":
                        logger.info(
                            "scored %s: %s (%s pts)",
                            ev.question_id,
                            "correct" if ev.correct else "wrong",
                            ev.points_awarded,
                        )
        except (
            CupAPIError,
            httpx.HTTPError,
            OSError,
            asyncio.IncompleteReadError,
            WebSocketException,
        ) as e:
            logger.warning("WS dropped, reconnecting in %ss: %s", reconnect_backoff_s, e)
        await asyncio.sleep(reconnect_backoff_s)
