from __future__ import annotations

import json

from skunk_server import file_tailer
from skunk_server.file_stream import FileSink, events_path
from skunk_server.file_tailer import FileTailer
from skunk_server.hub import StreamHub


def test_filesink_copies_event_and_creates_round_file(tmp_path) -> None:
    sink = FileSink(lambda: {"round": {}, "tasks": []}, tmp_path)
    try:
        sink.roll_round(7)
        round_path = events_path(tmp_path, 7)
        assert round_path.exists()
        assert round_path.read_text() == ""

        event = {"message": "hello"}
        sink.write_event("t1", "a1", event)
        sink.flush()

        assert event == {"message": "hello"}
        record = json.loads(round_path.read_text().splitlines()[0])
        assert record["event"] == {"message": "hello", "seq": 0}
    finally:
        sink.close()


def test_file_tailer_backfill_skips_partial_trailing_line(tmp_path) -> None:
    path = events_path(tmp_path)
    path.write_bytes(
        b'{"task_id":"t1","attempt_id":"a","event":{"seq":0,"message":"ok"}}\n'
        b'{"task_id":"t1","attempt_id":"a","event":{"seq":1,"message":"partial"}}'
    )
    tailer = FileTailer(tmp_path)

    events, next_seq = tailer.backfill("t1")

    assert next_seq == 1
    assert [event["message"] for _seq, _attempt, event in events] == ["ok"]


def test_file_tailer_skips_inactive_events_before_compaction(tmp_path, monkeypatch) -> None:
    path = events_path(tmp_path)
    path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "t1", "attempt_id": "a", "event": {"seq": 0}}),
                json.dumps({"task_id": "t2", "attempt_id": "a", "event": {"seq": 1}}),
            ]
        )
        + "\n"
    )
    tailer = FileTailer(tmp_path)
    hub = StreamHub(tailer.status_provider)
    tailer.hub = hub
    queue = hub.add_event_sub("t1")
    compacted: list[int] = []

    def compact(event):
        compacted.append(event["seq"])
        return dict(event)

    monkeypatch.setattr(file_tailer, "_compact_event", compact)

    tailer._poll_events()  # noqa: SLF001 - focused tailer unit test

    assert compacted == [0]
    assert queue.get_nowait()[1]["seq"] == 0
