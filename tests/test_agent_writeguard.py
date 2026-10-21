"""P1 写工具前置门：审批、scope、非幂等超时 unknown_outcome（仅 test double）。"""
from __future__ import annotations

import time

from pydantic import BaseModel

from app.agent.react import resume_run, run_agent
from app.agent.state import InMemoryRunStore
from app.tools import TOOL_SPECS, TOOLS, ToolSpec


class TicketArgs(BaseModel):
    ticket: str
    idempotency_key: str | None = None


class ScriptedLLM:
    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def complete(self, system, user):
        out = self.outs[min(self.i, len(self.outs) - 1)]
        self.i += 1
        return out


def _spec(**kwargs) -> ToolSpec:
    defaults = dict(
        name="create_ticket", func=lambda ticket: {"ok": True},
        description="test double", args_model=TicketArgs,
        side_effect=True, idempotent=False, requires_approval=True,
        idempotency_key_field="idempotency_key",
    )
    defaults.update(kwargs)
    return ToolSpec(**defaults)


def _create(calls: dict, **spec_kwargs):
    store = InMemoryRunStore()
    spec = _spec(**spec_kwargs)

    def create_ticket(ticket):
        calls["n"] += 1
        return {"ticket": ticket}

    llm = ScriptedLLM(['{"tool":"create_ticket","args":{"ticket":"T1"}}'])
    result = run_agent("建单", llm=llm, tools={"create_ticket": create_ticket},
                       tool_specs={"create_ticket": spec}, store=store, max_steps=3)
    assert result["status"] == "awaiting_approval"
    return store, result["run_id"], create_ticket, spec


def test_write_tool_requires_approval_and_scope():
    calls = {"n": 0}
    store, run_id, create_ticket, spec = _create(calls)
    run = store.get(run_id)
    resumed = resume_run(
        run_id, decision="approve", approval_token=run.approval_token,
        llm=ScriptedLLM(['{"final":"done"}']), tools={"create_ticket": create_ticket},
        tool_specs={"create_ticket": spec}, store=store, scopes={"agent:research"},
    )
    assert resumed.status == "completed" and calls["n"] == 1
    assert "create_ticket" not in TOOLS and "create_ticket" not in TOOL_SPECS


def test_write_tool_reject_does_not_execute():
    calls = {"n": 0}
    store, run_id, _, _ = _create(calls)
    run = store.get(run_id)
    rejected = resume_run(
        run_id, decision="reject", approval_token=run.approval_token,
        llm=None, store=store, reason="no",
    )
    assert rejected.status == "rejected" and calls["n"] == 0


def test_idempotent_write_retries_same_args(monkeypatch):
    calls = {"n": 0}
    keys: list[str | None] = []

    def flaky(ticket, idempotency_key=None):
        calls["n"] += 1
        keys.append(idempotency_key)
        if calls["n"] == 1:
            raise OSError("temporary")
        return {"ticket": ticket}

    monkeypatch.setattr("app.agent.react.time.sleep", lambda _: None)
    store = InMemoryRunStore()
    spec = _spec(idempotent=True, max_retries=1, retry_on=("unavailable",))
    result = run_agent(
        "建单", llm=ScriptedLLM(['{"tool":"create_ticket","args":{"ticket":"T1"}}']),
        tools={"create_ticket": flaky}, tool_specs={"create_ticket": spec},
        store=store, max_steps=3,
    )
    run = store.get(result["run_id"])
    resumed = resume_run(
        result["run_id"], decision="approve", approval_token=run.approval_token,
        llm=ScriptedLLM(['{"final":"done"}']), tools={"create_ticket": flaky},
        tool_specs={"create_ticket": spec}, store=store,
    )
    assert resumed.status == "completed" and calls["n"] == 2
    assert len(keys) == 2 and keys[0] == keys[1] and keys[0]


def test_non_idempotent_write_timeout_is_unknown_outcome():
    calls = {"n": 0}

    def slow(ticket):
        calls["n"] += 1
        time.sleep(0.2)
        return {"ticket": ticket}

    store = InMemoryRunStore()
    spec = _spec(timeout_s=0.01)
    result = run_agent(
        "建单", llm=ScriptedLLM(['{"tool":"create_ticket","args":{"ticket":"T1"}}']),
        tools={"create_ticket": slow}, tool_specs={"create_ticket": spec},
        store=store, max_steps=3,
    )
    assert result["status"] == "awaiting_approval"
    run = store.get(result["run_id"])
    resumed = resume_run(
        result["run_id"], decision="approve", approval_token=run.approval_token,
        llm=ScriptedLLM(['{"final":"done"}']), tools={"create_ticket": slow},
        tool_specs={"create_ticket": spec}, store=store,
    )
    assert resumed.status == "completed" and calls["n"] == 1
    assert any(s.action == "unknown_outcome:create_ticket" for s in store.list_steps(run.id))
