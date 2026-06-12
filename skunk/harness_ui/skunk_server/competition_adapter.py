"""Sole adapter between the Skunk server and the Cup API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx
from websockets.exceptions import WebSocketException

from cup_kit.client import CupAPIError, CupClient
from cup_kit.protocol import RoundStatus, SubmissionType

from skunk_server.domain import AnswerCandidate, SubmissionRecord
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry

logger = logging.getLogger(__name__)
StatusCallback = Callable[[], None]


class CompetitionAdapter:
    def __init__(
        self,
        base_url: str,
        team_token: str,
        registry: TaskRegistry,
        queues: TaskQueues,
        publish_status: StatusCallback,
        reconnect_backoff_s: float = 1.0,
    ) -> None:
        self._base_url = base_url
        self._team_token = team_token
        self._registry = registry
        self._queues = queues
        self._publish_status = publish_status
        self._reconnect_backoff_s = reconnect_backoff_s

    async def listen(self) -> None:
        while True:
            try:
                self._registry.set_connection("connecting")
                self._publish_status()
                async with CupClient(self._base_url, self._team_token) as cup:
                    current = await cup.get_current_round()
                    await self._apply_round(
                        current.round_num,
                        current.status,
                        current.questions,
                        current.ends_at,
                        None,
                        current.resubmits_left,
                        "loaded current round",
                    )
                    self._registry.set_connection("connected")
                    self._publish_status()
                    async for event in cup.events():
                        if event.type == "round_started":
                            await self._apply_round(
                                event.round_num,
                                RoundStatus.ACTIVE,
                                event.questions,
                                event.ends_at,
                                event.opens_at,
                                None,
                                f"round {event.round_num} started",
                            )
                        elif event.type == "round_state":
                            if event.status == RoundStatus.ACTIVE:
                                self._registry.update_round(
                                    round_num=event.round_num,
                                    status=event.status.value,
                                    event=f"round state {event.status.value}",
                                )
                            else:
                                self._registry.close_round(event.round_num, event.status.value)
                            self._publish_status()
                        elif event.type == "submission_scored":
                            self._registry.record_score(
                                event.question_id,
                                event.submission_id,
                                event.correct,
                                event.points_awarded,
                            )
                            self._publish_status()
            except asyncio.CancelledError:
                raise
            except (
                CupAPIError,
                httpx.HTTPError,
                OSError,
                asyncio.IncompleteReadError,
                WebSocketException,
            ) as error:
                logger.warning("Cup connection dropped: %s", error)
                self._registry.set_connection("disconnected", str(error))
                self._publish_status()
                await asyncio.sleep(self._reconnect_backoff_s)

    async def submit(
        self,
        submission: SubmissionRecord,
        candidate: AnswerCandidate,
    ):
        task = self._registry.get(submission.task_id)
        if task is None:
            raise KeyError(submission.task_id)
        async with CupClient(self._base_url, self._team_token) as cup:
            return await cup.submit(
                task.question_id,
                candidate.answer_text,
                reasoning=candidate.reasoning,
                source_docs=candidate.source_docs,
                submission_type=SubmissionType(candidate.submission_type),
            )

    async def _apply_round(
        self,
        round_num,
        status,
        questions,
        ends_at,
        opens_at,
        resubmits_left,
        event,
    ) -> None:
        self._registry.update_round(
            round_num=round_num,
            status=status.value,
            ends_at=ends_at,
            opens_at=opens_at,
            resubmits_left=resubmits_left,
            event=event,
        )
        if status == RoundStatus.ACTIVE:
            for question in questions:
                task, created = self._registry.create_task(
                    question.round_num,
                    question.question_id,
                    question.prompt,
                    ends_at,
                )
                if created:
                    self._queues.enqueue_agent(task.task_id)
        else:
            self._registry.close_round(round_num, status.value)
        self._publish_status()
