"""P2-1：run lease 与崩溃恢复 worker。"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from app.agent.state import InMemoryRunStore, new_run
from app.agent.worker import recover_once


class _LLM:
    def complete(self, system, user):
        return '{"final":"recovered"}'


def _expired_run(store: InMemoryRunStore):
    run = new_run("恢复任务")
    run.lease_owner = "dead-worker"
    run.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
    store.create(run)
    return run


def test_recover_expired_running_run():
    store = InMemoryRunStore()
    run = _expired_run(store)
    recovered = recover_once(
        store=store, llm_factory=lambda: _LLM(),
        tools_factory=lambda **kwargs: {},
    )
    assert recovered == [run.id]
    assert store.get(run.id).status == "completed"
    assert store.get(run.id).final == "recovered"


def test_active_lease_is_not_recovered():
    store = InMemoryRunStore()
    run = new_run("活跃任务")
    run.lease_owner = "live-worker"
    run.lease_until = datetime.now(timezone.utc) + timedelta(minutes=1)
    store.create(run)
    called = {"n": 0}

    def llm_factory():
        called["n"] += 1
        return _LLM()

    assert recover_once(store=store, llm_factory=llm_factory,
                        tools_factory=lambda **kwargs: {}) == []
    assert called["n"] == 0 and store.get(run.id).status == "running"


def test_two_workers_do_not_execute_same_run():
    store = InMemoryRunStore()
    run = _expired_run(store)
    calls = {"n": 0}
    barrier = threading.Barrier(2)

    class SlowLLM:
        def complete(self, system, user):
            calls["n"] += 1
            time.sleep(0.05)
            return '{"final":"once"}'

    def worker():
        barrier.wait()
        recover_once(store=store, llm_factory=lambda: SlowLLM(),
                     tools_factory=lambda **kwargs: {})

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1
    assert store.get(run.id).status == "completed"


def test_pending_action_is_completed_on_recovery():
    store = InMemoryRunStore()
    run = new_run("待执行动作")
    run.lease_owner = "dead-worker"
    run.lease_until = datetime.now(timezone.utc) - timedelta(seconds=1)
    run.pending_action = {"step_index": 0, "thought": "查", "tool": "kb_search",
                          "args": {"query": "q"}}
    store.create(run)
    called = {"n": 0}

    def kb_search(query, top_k=5):
        called["n"] += 1
        return [{"source": "s", "content": "ok"}]

    recover_once(
        store=store, llm_factory=lambda: _LLM(),
        tools_factory=lambda **kwargs: {"kb_search": kb_search},
    )
    assert called["n"] == 1
    assert store.get(run.id).status == "completed"
    assert [s.action for s in store.list_steps(run.id)][0] == "kb_search"
