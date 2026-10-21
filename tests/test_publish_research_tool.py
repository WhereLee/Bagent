"""P2-2：research publish 真实写工具的契约与调用路径（不触真实 DB）。"""
from __future__ import annotations

from app import tools
from app.agent.react import resume_run, run_agent
from app.agent.state import InMemoryRunStore
from app.tools import TOOL_SPECS


class _Doc:
    def to_markdown(self):
        return "# doc"

    def to_dict(self):
        return {"topic": "t"}


def test_publish_research_is_approval_gated_write_tool():
    spec = tools.TOOL_SPECS["publish_research"]
    assert spec.side_effect is True
    assert spec.requires_approval is True
    assert spec.idempotent is True
    assert spec.required_scope == "agent:research"


def test_publish_research_calls_existing_publish(monkeypatch):
    seen = {}

    class Session:
        def commit(self):
            seen["committed"] = True

        def close(self):
            seen["closed"] = True

    class Embedder:
        def encode_documents(self, docs):
            return [[0.0] * 8 for _ in docs]

    monkeypatch.setattr("app.db.session.get_session", lambda: Session())
    monkeypatch.setattr("app.ingestion.embedder.get_embedder", lambda: Embedder())
    monkeypatch.setattr("app.research.session.get", lambda sid: _Doc())

    def fake_publish(doc, *, session, embedder, event_id=None):
        seen["event_id"] = event_id
        return 2

    monkeypatch.setattr("app.research.publish.publish", fake_publish)
    out = tools.publish_research("sid", event_id=7)
    assert out["published"] == 2 and out["event_id"] == 7
    assert seen["committed"] is True and seen["closed"] is True


def test_publish_research_pauses_for_approval():
    store = InMemoryRunStore()
    called = {"n": 0}

    def publish(session_id, event_id=None, event_name=None, event_type=None):
        called["n"] += 1
        return {"published": 1}

    class LLM:
        def __init__(self):
            self.i = 0

        def complete(self, system, user):
            self.i += 1
            return ('{"tool":"publish_research","args":{"session_id":"s1"}}'
                    if self.i == 1 else '{"final":"done"}')

    run = run_agent(
        "publish", llm=LLM(), tools={"publish_research": publish},
        tool_specs=TOOL_SPECS, store=store, max_steps=3,
    )
    assert run["status"] == "awaiting_approval" and called["n"] == 0
    current = store.get(run["run_id"])
    result = resume_run(
        run["run_id"], decision="approve", approval_token=current.approval_token,
        llm=LLM(), tools={"publish_research": publish}, tool_specs=TOOL_SPECS,
        store=store,
    )
    assert result.status == "completed" and called["n"] == 1
    assert [s.action for s in store.list_steps(result.id)][0] == "publish_research"


def test_research_tool_returns_session_id(monkeypatch):
    monkeypatch.setattr("app.generation.llm.get_llm", lambda: object())
    monkeypatch.setattr("app.research.agent.research", lambda *args, **kwargs: _Doc())
    monkeypatch.setattr("app.research.session.create", lambda doc: "sid-123")
    out = tools.research("topic")
    assert out["session_id"] == "sid-123"
