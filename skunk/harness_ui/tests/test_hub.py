from __future__ import annotations

import asyncio
import json

from skunk_server.api import event_frames
from skunk_server.hub import StreamHub
from skunk_server.task_registry import TaskRegistry


def _data(frame: str) -> dict:
    assert frame.startswith("data: ")
    return json.loads(frame[len("data: "):])


def test_stream_hub_fanout_and_drop_oldest() -> None:
    hub = StreamHub(lambda: {"round": {}, "tasks": []}, maxsize=2)

    # status fan-out: only registered subs receive frames, serialized once.
    status_q = hub.add_status_sub()
    hub.publish_status()
    assert json.loads(status_q.get_nowait()[len("data: "):]) == {"round": {}, "tasks": []}

    # event fan-out is per task_id; publish_event takes a BATCH of events.
    a = hub.add_event_sub("t1")
    b = hub.add_event_sub("t2")
    hub.publish_event("t1", "att", [{"seq": 0, "message": "hi"}])
    assert a.get_nowait() == ("att", {"seq": 0, "message": "hi"})
    assert b.empty()

    # bounded queue drops the OLDEST when full (maxsize=2); each event in a batch enqueues separately.
    hub.publish_event("t1", "att", [{"seq": 1}, {"seq": 2}, {"seq": 3}])  # overflow -> seq 1 dropped
    drained = [a.get_nowait()[1]["seq"] for _ in range(a.qsize())]
    assert drained == [2, 3]

    # cleanup removes the empty task bucket.
    hub.remove_event_sub("t1", a)
    hub.remove_event_sub("t2", b)
    assert hub._event_subs == {}  # noqa: SLF001 - white-box cleanup check


def test_registry_event_seq_and_backfill() -> None:
    registry = TaskRegistry()
    task, _ = registry.create_task(1, "q", "Question")
    attempt = registry.begin_attempt(task.task_id, "agent")
    assert attempt is not None

    first = registry.append_attempt_event(task.task_id, attempt.attempt_id, {"message": "a", "kind": "note"})
    second = registry.append_attempt_event(task.task_id, attempt.attempt_id, {"message": "b", "kind": "note"})
    assert first is not None and first["seq"] == 0
    assert second is not None and second["seq"] == 1

    backfill, next_seq = registry.snapshot_task_events(task.task_id)
    assert next_seq == 2
    assert [(seq, event["message"]) for (seq, _a, event) in backfill] == [(0, "a"), (1, "b")]

    # unknown attempt -> None (the gate the worker relies on); unknown task -> empty backfill.
    assert registry.append_attempt_event(task.task_id, "nope", {"message": "x"}) is None
    assert registry.snapshot_task_events("999:absent") == ([], 0)


def test_event_stream_backfills_then_streams_live_without_dupes() -> None:
    # Drive the SSE generator directly (httpx's ASGITransport buffers responses, so it can't
    # consume an open-ended event-stream). `anext` advances frame-by-frame.
    registry = TaskRegistry()
    hub = StreamHub(lambda: {"round": {}, "tasks": []})
    task, _ = registry.create_task(1, "q", "Question")
    attempt = registry.begin_attempt(task.task_id, "agent")
    assert attempt is not None
    # two events exist before anyone connects -> they must be back-filled.
    registry.append_attempt_event(task.task_id, attempt.attempt_id, {"message": "e0", "kind": "note"})
    registry.append_attempt_event(task.task_id, attempt.attempt_id, {"message": "e1", "kind": "note"})

    async def run() -> None:
        gen = event_frames(registry, hub, task.task_id, heartbeat=0.05)
        try:
            assert _data(await anext(gen))["event"]["message"] == "e0"  # backfill
            assert _data(await anext(gen))["event"]["message"] == "e1"

            # a stale duplicate (seq already back-filled) must be skipped; the genuinely new
            # event must come through. Both are queued before we pull the next frame.
            hub.publish_event(task.task_id, attempt.attempt_id, [{"seq": 1, "message": "dup"}])
            live = registry.append_attempt_event(task.task_id, attempt.attempt_id, {"message": "e2", "kind": "note"})
            hub.publish_event(task.task_id, attempt.attempt_id, [live])

            frame = _data(await anext(gen))
            assert frame["event"]["message"] == "e2"
            assert frame["event"]["seq"] == 2
        finally:
            await gen.aclose()  # runs the generator's finally -> removes the subscriber
        assert task.task_id not in hub._event_subs  # noqa: SLF001

    asyncio.run(run())
