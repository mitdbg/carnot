"""OfficeQA Cup reference agent — the file you edit.

Two things to customize:

1. ``solve(prompt)`` — your agent. Return an :class:`AgentAnswer`
   with your answer text, reasoning (≥100 chars), and source-doc
   citations (optional).

2. ``_process_round(...)`` — the per-question state machine. Edit if
   you want parallelism (``asyncio.gather`` over unhandled keys) or a
   custom answer order.

Run locally against the practice harness::

    # Terminal A
    python practice_server.py

    # Terminal B
    export CUP_BASE_URL=http://127.0.0.1:8765
    export CUP_TEAM_TOKEN=anything
    python reference_agent.py

On competition day, swap two env vars to point at the live cup::

    export CUP_BASE_URL=https://<live-cup-url>
    export CUP_TEAM_TOKEN=<your-team-token>
    python reference_agent.py

Same code path — only the destination differs.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from cup_kit.agent_runtime import (
    AgentAnswer,
    Solve,
    TERMINAL_SUBMIT_REASONS,
    run_agent,
)
from cup_kit.client import CupAPIError, CupClient
from cup_kit.protocol import QuestionFragment

logger = logging.getLogger(__name__)


async def solve(prompt: str) -> AgentAnswer:
    """**Replace this with your agent.**

    Given the question ``prompt``, return an :class:`AgentAnswer` with:

    - ``answer``: your final answer string.
    - ``reasoning``: chain-of-thought, ≥100 characters
      (server-enforced for ``submission_type="agent"``).
    - ``source_docs``: optional list of doc references. May be empty;
      if present, each entry must be non-empty and ≤512 chars.

    The placeholder body below clears server validation but always
    answers ``"placeholder answer"`` — useful for verifying connect →
    submit → score before wiring in a real model.
    """
    del prompt
    return AgentAnswer(
        answer="placeholder answer",
        reasoning=(
            "Reference agent placeholder reasoning. Replace this with "
            "your agent's actual chain-of-thought, including the "
            "passages from source_docs that justify the final answer. "
            "The server requires at least 100 characters for agent "
            "submissions; this string clears that floor."
        ),
        source_docs=["placeholder.pdf#p1"],
    )


async def _process_round(
    cup: CupClient,
    solve: Solve,
    questions_by_key: dict[tuple[int, str], QuestionFragment],
    answers_by_key: dict[tuple[int, str], AgentAnswer],
    handled: set[tuple[int, str]],
) -> None:
    """One pass over every known question in the active round.

    Edit me to answer in parallel (replace the for-loop with
    ``asyncio.gather`` after factoring the body into a helper) or in
    a custom order (sort/filter ``questions_by_key.items()``).
    Idempotent — handled keys are skipped; recoverable rejections
    (RATE_LIMITED, ROUND_NOT_ACTIVE) leave the key for the next pass.
    """
    for key, q in questions_by_key.items():
        if key in handled:
            continue
        try:
            ans = answers_by_key.get(key)
            if ans is None:
                ans = await solve(q.prompt)
                answers_by_key[key] = ans
            resp = await cup.submit(
                q.question_id,
                ans.answer,
                reasoning=ans.reasoning,
                source_docs=ans.source_docs,
            )
            if resp.accepted:
                handled.add(key)
                continue
            if resp.reason in TERMINAL_SUBMIT_REASONS:
                logger.warning("submit terminally rejected for %s: %s", q.question_id, resp.reason)
                handled.add(key)
        except (CupAPIError, httpx.HTTPError) as e:
            logger.warning("transient error processing %s: %s", q.question_id, e)


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(
        run_agent(
            os.environ["CUP_BASE_URL"],
            os.environ["CUP_TEAM_TOKEN"],
            solve,
            _process_round,
        )
    )


if __name__ == "__main__":
    _main()
