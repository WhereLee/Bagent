"""P3-1：工具操作账本状态、重放、未知态阻断与 API。"""
from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.agent.react import resume_run, run_agent
from app.agent.state import InMemoryRunStore, ToolOperation, new_run
from app.api.main import app
from app.api.routes import agent as agent_route
from app.tools import ToolSpec


class ScriptedLLM:
    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def complete(self, system, user):
        out = self.outs[min(self.i, len(self.outs) - 1)]
        self.i += 1
        return out


class TicketArgs(BaseModel):
    ticket: str


def _spec():
    return ToolSpec(
        name="create_ticket", func=lambda ticket: {"ok": True},
        description="test", args_model=TicketArgs, side_effect=True,
        requires_approval=True, idempotent=False,
    )


def _key(run_id, step, args):
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{run_id}:{step}:create_ticket:{digest}"


def _awaiting_write():
    store = InMemoryRunStore()
    llm = ScriptedLLM(['{"tool":"create_ticket","args":{"ticket":"T1"}}'])
    run = run_agent("建单", llm=llm, tools={"create_ticket": lambda ticket: {"ok": True}},
                    tool_specs={"create_ticket": _spec()}, store=store, max_steps=3)
    assert run["status"] == "awaiting_approval"
    return store, run


def test_ledger_replays_succeeded_operation_without_execution():
    store, run = _awaiting_write()
    key = _key(run["run_id"], 0, {"ticket": "T1"})
    store.begin_operation(ToolOperation(
        operation_id="op1", run_id=run["run_id"], step_index=0,
        tool="create_ticket", idempotency_key=key, args_hash="h",
    ))
    store.mark_operation(key, "succeeded", result={"ok": True, "replayed": True})
    called = {"n": 0}

    def create_ticket(ticket):
        called["n"] += 1
        return {"ok": True}

    current = store.get(run["run_id"])
    result = resume_run(
        run["run_id"], decision="approve", approval_token=current.approval_token,
        llm=ScriptedLLM(['{"final":"done"}']), tools={"create_ticket": create_ticket},
        tool_specs={"create_ticket": _spec()}, store=store,
    )
    assert result.status == "completed" and called["n"] == 0
    assert store.list_steps(result.id)[0].action == "create_ticket"


def test_ledger_unknown_blocks_execution():
    store, run = _awaiting_write()
    key = _key(run["run_id"], 0, {"ticket": "T1"})
    store.begin_operation(ToolOperation(
        operation_id="op1", run_id=run["run_id"], step_index=0,
        tool="create_ticket", idempotency_key=key, args_hash="h",
    ))
    called = {"n": 0}

    def create_ticket(ticket):
        called["n"] += 1
        return {"ok": True}

    current = store.get(run["run_id"])
    result = resume_run(
        run["run_id"], decision="approve", approval_token=current.approval_token,
        llm=ScriptedLLM(['{"final":"done"}']), tools={"create_ticket": create_ticket},
        tool_specs={"create_ticket": _spec()}, store=store,
    )
    assert called["n"] == 0
    assert any(s.action == "unknown_outcome:create_ticket" for s in store.list_steps(result.id))


def test_operations_api(monkeypatch):
    store = InMemoryRunStore()
    run = new_run("ops")
    store.create(run)
    store.begin_operation(ToolOperation(
        operation_id="op1", run_id=run.id, step_index=0, tool="create_ticket",
        idempotency_key="k1", args_hash="h",
    ))
    monkeypatch.setattr(agent_route, "get_run_store", lambda: store)
    client = TestClient(app)
    res = client.get(f"/agent/runs/{run.id}/operations")
    assert res.status_code == 200
    assert res.json()["operations"][0]["operation_id"] == "op1"
