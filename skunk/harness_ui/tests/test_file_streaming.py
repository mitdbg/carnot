from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from skunk_server import file_tailer
from skunk_server.file_stream import FileSink, _json_default, events_path
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


# ── status snapshot serialization (regression: replan-approval review never reached the UI) ──
# A blocking replan_approval review injects its `guidance` (the data-prep output + plan branches)
# into the status snapshot. `AnnotatedValue.value` is `Any`, so a compute result can be a
# non-JSON-native leaf (numpy scalar / Decimal / datetime). `publish_status` json.dumps the whole
# snapshot and swallows any error, so without a `default=` fallback the snapshot silently failed to
# write for as long as that review was OPEN — freezing the status feed and hiding the "Approve
# Replan" button. The fix mirrors `write_event`: `json.dumps(..., default=_json_default)`.


def test_json_default_handles_non_native_leaves() -> None:
    class Color(Enum):
        RED = "red"

    @dataclass
    class Point:
        x: int

    assert _json_default(datetime(2026, 6, 17, tzinfo=timezone.utc)).startswith("2026-06-17")
    assert _json_default(Color.RED) == "red"
    assert _json_default(Point(1)) == {"x": 1}
    # last-resort str() — the case a numpy scalar / Decimal AnnotatedValue.value hits
    assert _json_default(Decimal("908000000")) == "908000000"


def test_publish_status_writes_replan_review_with_non_serializable_guidance(tmp_path) -> None:
    """A PROCESSING task whose OPEN replan_approval review carries a non-JSON-native value in its
    guidance must still serialize and be written to status.json — so the review (and its button)
    reaches the browser. Before the fix this json.dumps raised, publish_status swallowed it, and
    status.json was never written."""
    snapshot = {
        "round": {"round_num": 1, "status": "ACTIVE"},
        "tasks": [
            {
                "task_id": "1:dais_synth_0-5_jpy",
                "status": "PROCESSING",
                "reviews": [
                    {
                        "review_id": "r1",
                        "kind": "replan_approval",
                        "guidance": {
                            "data_prep_output": [
                                # mirrors AnnotatedValue.model_dump with an Any value that
                                # pydantic mode="json" leaves as a raw (non-serializable) leaf
                                {"description": "yen total", "value": Decimal("908000000")},
                            ],
                        },
                    }
                ],
            }
        ],
    }
    sink = FileSink(lambda: snapshot, tmp_path)
    try:
        sink.publish_status()
    finally:
        sink.close()

    written = (tmp_path / "status.json").read_text()
    assert written, "status.json was not written — publish_status silently dropped the snapshot"
    parsed = json.loads(written)
    review = parsed["tasks"][0]["reviews"][0]
    assert review["kind"] == "replan_approval"
    # the non-native leaf is preserved (as its string form), not dropped
    assert review["guidance"]["data_prep_output"][0]["value"] == "908000000"


def test_publish_status_one_bad_leaf_does_not_blank_the_whole_feed(tmp_path) -> None:
    """Serialization is whole-snapshot, so a single replan review's bad leaf must not blank out
    every other task's status (which is what froze the UI for the whole round)."""

    class Weird:
        def __repr__(self) -> str:
            return "weird-value"

    snapshot = {
        "round": {"round_num": 2, "status": "ACTIVE"},
        "tasks": [
            {"task_id": "ready-1", "status": "READY", "reviews": [{"kind": "verify_extract"}]},
            {
                "task_id": "replan-1",
                "status": "PROCESSING",
                "reviews": [{"kind": "replan_approval", "guidance": {"x": Weird()}}],
            },
        ],
    }
    sink = FileSink(lambda: snapshot, tmp_path)
    try:
        sink.publish_status()
    finally:
        sink.close()

    parsed = json.loads((tmp_path / "status.json").read_text())
    assert {t["task_id"] for t in parsed["tasks"]} == {"ready-1", "replan-1"}
    assert parsed["tasks"][1]["reviews"][0]["guidance"]["x"] == "weird-value"


def test_publish_status_serializes_real_replan_review_from_registry(tmp_path) -> None:
    """End-to-end through the registry: a replan_approval review created on a real task, with a
    non-serializable AnnotatedValue-style guidance value, must serialize into status.json via a
    summary-style provider that emits each open review's `guidance` verbatim (as the real
    server.summary() does)."""
    from skunk_server.task_registry import TaskRegistry

    reg = TaskRegistry()
    reg.update_round(round_num=1, status="ACTIVE")
    task, _ = reg.create_task(round_num=1, question_id="dais_synth_0-5_jpy", prompt="q")
    attempt = reg.begin_attempt(task.task_id, "worker-1")
    guidance = {"data_prep_output": [{"description": "yen", "value": Decimal("908000000")}]}
    reg.create_review(task.task_id, attempt.attempt_id, "replan_approval", "approve", "Q", [], guidance)

    def provider() -> dict:
        return {
            "round": reg.round_state(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "status": t.status.value,
                    "reviews": [
                        {"kind": r.kind, "guidance": r.guidance} for r in t.open_reviews
                    ],
                }
                for t in reg.list_tasks()
            ],
        }

    sink = FileSink(provider, tmp_path)
    try:
        sink.publish_status()
    finally:
        sink.close()

    parsed = json.loads((tmp_path / "status.json").read_text())
    review = parsed["tasks"][0]["reviews"][0]
    assert review["kind"] == "replan_approval"
    assert review["guidance"]["data_prep_output"][0]["value"] == "908000000"
