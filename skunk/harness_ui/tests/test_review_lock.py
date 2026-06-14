"""Per-UID review lock: registry acquire/release/expiry-sweep semantics and the broker's
notify-on-change wrappers. The lock gates which browser may open the review overlay; it never
touches the optimistic agent path."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from skunk_server.domain import utc_now
from skunk_server.human_work_broker import HumanWorkBroker
from skunk_server.task_registry import REVIEW_LOCK_TTL_S, TaskRegistry


def _task(registry: TaskRegistry) -> str:
    task, _ = registry.create_task(1, "Q1", "prompt")
    return task.task_id


def test_acquire_is_exclusive() -> None:
    registry = TaskRegistry()
    task_id = _task(registry)

    acquired, holder, changed = registry.acquire_review_lock(task_id, "alice")
    assert (acquired, holder, changed) == (True, "alice", True)
    assert registry.get(task_id).active_lock_holder() == "alice"

    # A second operator can't take it while alice holds it.
    acquired, holder, changed = registry.acquire_review_lock(task_id, "bob")
    assert (acquired, holder, changed) == (False, "alice", False)


def test_same_client_refresh_is_not_a_change() -> None:
    registry = TaskRegistry()
    task_id = _task(registry)
    registry.acquire_review_lock(task_id, "alice")
    before = registry.get(task_id).review_lock_expires_at

    # Re-acquiring (the heartbeat) refreshes the lease but isn't a holder change, so it must not
    # bump the version / churn the status stream.
    acquired, holder, changed = registry.acquire_review_lock(task_id, "alice")
    assert (acquired, holder, changed) == (True, "alice", False)
    assert registry.get(task_id).review_lock_expires_at >= before


def test_release_frees_for_others() -> None:
    registry = TaskRegistry()
    task_id = _task(registry)
    registry.acquire_review_lock(task_id, "alice")

    # A non-holder release is a no-op; the holder's release frees it.
    assert registry.release_review_lock(task_id, "bob") is False
    assert registry.release_review_lock(task_id, "alice") is True
    assert registry.get(task_id).active_lock_holder() is None
    assert registry.acquire_review_lock(task_id, "bob") == (True, "bob", True)


def test_expired_lock_is_swept_and_reacquirable() -> None:
    registry = TaskRegistry()
    task_id = _task(registry)
    registry.acquire_review_lock(task_id, "alice")

    # Simulate alice's tab dying: push the lease into the past. The lock now reads as free...
    task = registry.get(task_id)
    task.review_lock_expires_at = utc_now() - timedelta(seconds=1)
    assert task.active_lock_holder() is None

    # ...the sweeper clears the stale holder and reports it, and bob can take it.
    assert registry.sweep_review_locks() == [task_id]
    assert registry.get(task_id).review_lock_holder is None
    assert registry.acquire_review_lock(task_id, "bob") == (True, "bob", True)


def test_ttl_constant_is_sane() -> None:
    # Heartbeat (~7s) must comfortably beat the TTL so a live reviewer never loses the lock.
    assert REVIEW_LOCK_TTL_S >= 15


def test_broker_notifies_only_on_holder_change() -> None:
    async def scenario():
        registry = TaskRegistry()
        notifies = 0

        def publish_status():
            nonlocal notifies
            notifies += 1

        broker = HumanWorkBroker(registry, None, publish_status)
        broker.start(asyncio.get_running_loop())
        task_id = _task(registry)

        assert broker.acquire_review_lock(task_id, "alice") == (True, "alice")
        await asyncio.sleep(0)  # let the scheduled publish run
        assert notifies == 1

        # Heartbeat (same client) → no holder change → no extra publish.
        assert broker.acquire_review_lock(task_id, "alice") == (True, "alice")
        await asyncio.sleep(0)
        assert notifies == 1

        # Contender denied → no publish.
        assert broker.acquire_review_lock(task_id, "bob") == (False, "alice")
        await asyncio.sleep(0)
        assert notifies == 1

        # Release → publish.
        assert broker.release_review_lock(task_id, "alice") is True
        await asyncio.sleep(0)
        assert notifies == 2

    asyncio.run(scenario())


def test_acquire_unknown_task_is_safe() -> None:
    registry = TaskRegistry()
    broker = HumanWorkBroker(registry, None, lambda: None)
    # No event loop started: a vanished task must short-circuit before any _notify.
    assert broker.acquire_review_lock("9:GONE", "alice") == (False, None)
