"""P0 集成：真实 Postgres 上验证 run/step 落库与审批恢复。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, update

from app.agent.react import resume_run, start_run
from app.agent.state import PostgresRunStore, new_run
from app.db.models import AgentRun as AgentRunRow
from app.db.models import AgentStep as AgentStepRow
from app.db.session import get_session, init_schema
from app.tools import TOOL_SPECS

pytestmark = pytest.mark.integration


class ScriptedLLM:
    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def complete(self, system, user):
        out = self.outs[min(self.i, len(self.outs) - 1)]
        self.i += 1
        return out


def _cleanup(run_id: str) -> None:
    session = get_session()
    try:
        session.execute(delete(AgentStepRow).where(AgentStepRow.run_id == run_id))
        session.execute(delete(AgentRunRow).where(AgentRunRow.id == run_id))
        session.commit()
    finally:
        session.close()


def test_agent_run_approval_roundtrip_against_real_db():
    init_schema()
    store = PostgresRunStore()
    run = None
    try:
        run = start_run(
            "查一下", llm=ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}']),
            tools={"research": lambda topic, **kwargs: {"ok": True}},
            tool_specs=TOOL_SPECS, store=store, max_steps=4,
        )
        assert run.status == "awaiting_approval"
        assert store.list_steps(run.id) == []

        resumed = resume_run(
            run.id, decision="approve", approval_token=run.approval_token,
            llm=ScriptedLLM(['{"final":"done"}']),
            tools={"research": lambda topic, **kwargs: {"ok": True}},
            tool_specs=TOOL_SPECS, store=store,
        )
        assert resumed.status == "completed"
        assert [s.action for s in store.list_steps(run.id)] == ["research", "final"]
        assert resumed.token_output >= 0
    finally:
        if run is not None:
            _cleanup(run.id)


def test_agent_lease_roundtrip_against_real_db():
    init_schema()
    store = PostgresRunStore()
    run = store.create(new_run("lease", user_id="u1", scopes=["agent:read"]))
    try:
        acquired = store.acquire_lease(run.id, "worker-a", 30)
        assert acquired
        assert store.acquire_lease(run.id, "worker-b", 30) is False
        assert store.renew_lease(run.id, "worker-a", 30)
        assert store.release_lease(run.id, "worker-a") is None
        session = get_session()
        try:
            session.execute(
                update(AgentRunRow).where(AgentRunRow.id == run.id).values(
                    lease_owner="dead",
                    lease_until=datetime.now(timezone.utc) - timedelta(seconds=1),
                )
            )
            session.commit()
        finally:
            session.close()
        assert run.id in [r.id for r in store.list_recoverable(limit=5)]
        assert store.acquire_lease(run.id, "worker-b", 30) is True
    finally:
        _cleanup(run.id)


def test_agent_reject_against_real_db():
    init_schema()
    store = PostgresRunStore()
    run = None
    called = {"n": 0}

    def research(topic, **kwargs):
        called["n"] += 1
        return {}

    try:
        run = start_run(
            "查一下", llm=ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}']),
            tools={"research": research}, tool_specs=TOOL_SPECS, store=store,
        )
        rejected = resume_run(
            run.id, decision="reject", approval_token=run.approval_token,
            llm=None, tools=None, tool_specs=TOOL_SPECS, store=store, reason="no",
        )
        assert rejected.status == "rejected"
        assert called["n"] == 0
    finally:
        if run is not None:
            _cleanup(run.id)
