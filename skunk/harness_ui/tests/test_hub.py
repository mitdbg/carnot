from __future__ import annotations

import json

from skunk_server.hub import StreamHub


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
