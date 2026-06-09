"""Async HTTP + WebSocket client for the OfficeQA Cup Competitor API.

Thin httpx wrapper around the four endpoints a team's agent calls
during a round. Business rejections come back as 200s with
``accepted=False`` and a typed ``reason``; teams branch on
``isinstance`` of the union return type. :class:`CupAPIError` is
reserved for HTTP 4xx/5xx (auth + server-side errors) so a team can
distinguish "fix your request body" from "your answer was rejected by
the round rules".
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Annotated, Any

import httpx
import websockets
from pydantic import Discriminator, TypeAdapter

from cup_kit.protocol import (
    RoundCurrentResponse,
    SubmissionType,
    SubmitAcceptedResponse,
    SubmitRejectedResponse,
    TeamEvent,
    TeamStatusResponse,
)

_SubmitResponse = Annotated[
    SubmitAcceptedResponse | SubmitRejectedResponse,
    Discriminator("accepted"),
]
_SUBMIT_ADAPTER: TypeAdapter[_SubmitResponse] = TypeAdapter(_SubmitResponse)
_TEAM_EVENT_ADAPTER: TypeAdapter[TeamEvent] = TypeAdapter(TeamEvent)


class CupAPIError(Exception):
    """HTTP-level failure from the Cup API. Business rejections do NOT
    raise — they come back as ``accepted=False`` response bodies."""

    def __init__(self, status_code: int, body: Any) -> None:
        super().__init__(f"Cup API returned HTTP {status_code}: {body!r}")
        self.status_code = status_code
        self.body = body


class CupClient:
    """Async client for the OfficeQA Cup Competitor API.

    Usage::

        async with CupClient(base_url, team_token) as cup:
            rnd = await cup.get_current_round()
            for q in rnd.questions:
                resp = await cup.submit(
                    q.question_id,
                    my_answer(q.prompt),
                    reasoning=my_reasoning,
                    source_docs=["doc1.pdf#p4"],
                )
    """

    def __init__(
        self,
        base_url: str,
        team_token: str,
        *,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._team_token = team_token
        # Authorization is the standard header. X-Cup-Auth is an
        # accepted fallback for environments where reverse proxies
        # strip Authorization; sending both is harmless.
        headers = {
            "Authorization": f"Bearer {team_token}",
            "X-Cup-Auth": f"Bearer {team_token}",
        }
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

    async def __aenter__(self) -> CupClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, *, json: Any = None) -> Any:
        resp = await self._client.request(method, path, json=json)
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise CupAPIError(resp.status_code, body)
        return resp.json()

    async def get_current_round(self) -> RoundCurrentResponse:
        body = await self._request("GET", "/v1/round/current")
        return RoundCurrentResponse.model_validate(body)

    async def get_team_status(self) -> TeamStatusResponse:
        body = await self._request("GET", "/v1/team/status")
        return TeamStatusResponse.model_validate(body)

    async def submit(
        self,
        question_id: str,
        answer_text: str,
        *,
        reasoning: str,
        source_docs: list[str],
        submission_type: SubmissionType | str = SubmissionType.AGENT,
    ) -> SubmitAcceptedResponse | SubmitRejectedResponse:
        type_value = submission_type.value if isinstance(submission_type, SubmissionType) else submission_type
        body = await self._request(
            "POST",
            "/v1/submit",
            json={
                "question_id": question_id,
                "answer_text": answer_text,
                "reasoning": reasoning,
                "source_docs": source_docs,
                "submission_type": type_value,
            },
        )
        return _SUBMIT_ADAPTER.validate_python(body)

    async def events(self) -> AsyncIterator[TeamEvent]:
        """Async iterator over server-pushed team events.

        Yields typed ``RoundStartedEvent`` / ``RoundStateEvent`` /
        ``SubmissionScoredEvent`` instances as they arrive on
        ``WS /v1/team/events``. The first frame after the WS handshake
        is a ``{"type": "connected", ...}`` ack which is consumed
        internally and NOT yielded.
        """
        async for raw in self._open_ws("/v1/team/events"):
            if raw.get("type") == "connected":
                continue
            yield _TEAM_EVENT_ADAPTER.validate_python(raw)

    async def _open_ws(self, path: str) -> AsyncIterator[dict[str, Any]]:
        url = self._build_ws_url(path)
        headers = {
            "Authorization": f"Bearer {self._team_token}",
            "X-Cup-Auth": f"Bearer {self._team_token}",
        }
        async with websockets.connect(url, additional_headers=headers) as ws:
            async for msg in ws:
                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8")
                yield json.loads(msg)

    def _build_ws_url(self, path: str) -> str:
        if self._base_url.startswith("https://"):
            ws_base = "wss://" + self._base_url[len("https://") :]
        elif self._base_url.startswith("http://"):
            ws_base = "ws://" + self._base_url[len("http://") :]
        else:
            ws_base = self._base_url
        return ws_base + path
