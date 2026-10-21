"""P0 Agent API：run 创建、审批恢复、拒绝、取消与旧 /agent 兼容。"""
from __future__ import annotations

import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.agent.state import InMemoryRunStore
from app.api.main import app
from app.api.routes import agent as agent_route


class ScriptedLLM:
    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def complete(self, system, user):
        out = self.outs[min(self.i, len(self.outs) - 1)]
        self.i += 1
        return out


def _install(monkeypatch, store, llms, tools=None, faithfulness=False):
    monkeypatch.setattr(agent_route, "get_run_store", lambda: store)
    monkeypatch.setattr(agent_route, "get_llm", lambda: llms.pop(0))
    monkeypatch.setattr(agent_route, "_tools", lambda user_id=None, scopes=None: tools or {
        "research": lambda topic, **kwargs: {"ok": True},
    })
    monkeypatch.setattr(agent_route, "get_settings",
                        lambda: SimpleNamespace(faithfulness_enabled=faithfulness))


def test_create_then_approve(monkeypatch):
    store = InMemoryRunStore()
    llms = [
        ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}']),
        ScriptedLLM(['{"final":"done"}']),
    ]
    _install(monkeypatch, store, llms)
    client = TestClient(app)

    resp = client.post("/agent/runs", json={"goal": "查一下", "max_steps": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "awaiting_approval"
    assert body["pending_action"]["tool"] == "research"
    token = body["approval_token"]

    resp2 = client.post(
        f"/agent/runs/{body['run_id']}/approve",
        json={"step_index": 0, "approval_token": token},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["status"] == "completed"
    assert [s["action"] for s in body2["steps"]] == ["research", "final"]

    fetched = client.get(f"/agent/runs/{body['run_id']}").json()
    assert "approval_token" not in fetched
    repeated = client.post(
        f"/agent/runs/{body['run_id']}/approve",
        json={"step_index": 0, "approval_token": token},
    )
    assert repeated.status_code == 409


def test_reject_does_not_execute(monkeypatch):
    store = InMemoryRunStore()
    called = {"n": 0}

    def research(topic, **kwargs):
        called["n"] += 1
        return {}

    llms = [ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])]
    _install(monkeypatch, store, llms, tools={"research": research})
    client = TestClient(app)

    created = client.post("/agent/runs", json={"goal": "查一下"}).json()
    resp = client.post(
        f"/agent/runs/{created['run_id']}/reject",
        json={"step_index": 0, "approval_token": created["approval_token"], "reason": "no"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    assert called["n"] == 0


def test_cancel(monkeypatch):
    store = InMemoryRunStore()
    llms = [ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])]
    _install(monkeypatch, store, llms)
    client = TestClient(app)

    created = client.post("/agent/runs", json={"goal": "查一下"}).json()
    resp = client.post(
        f"/agent/runs/{created['run_id']}/cancel",
        json={"approval_token": created["approval_token"]},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_bad_token_and_step(monkeypatch):
    store = InMemoryRunStore()
    llms = [ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])]
    _install(monkeypatch, store, llms)
    client = TestClient(app)

    assert client.get("/agent/runs/missing").status_code == 404
    created = client.post("/agent/runs", json={"goal": "查一下"}).json()
    bad_token = client.post(
        f"/agent/runs/{created['run_id']}/approve",
        json={"step_index": 0, "approval_token": "bad"},
    )
    assert bad_token.status_code == 403
    bad_step = client.post(
        f"/agent/runs/{created['run_id']}/approve",
        json={"step_index": 9, "approval_token": created["approval_token"]},
    )
    assert bad_step.status_code == 409


def test_query_agent_returns_quality_fields(monkeypatch):
    from app.api.routes import query as query_route

    monkeypatch.setattr(query_route, "build_tools", lambda **kwargs: {
        "kb_search": lambda **kwargs: [],
        "web_search": lambda **kwargs: [],
    })
    monkeypatch.setattr(query_route, "get_llm", lambda: object())
    monkeypatch.setattr("app.agent.react.run_agent", lambda *args, **kwargs: {
        "status": "completed", "run_id": "r1", "final": "raw", "steps": [],
        "answer": {"text": "clean", "sources": [{"source": "s"}],
                   "grounded": True, "low_confidence": False,
                   "invalid_citations": []},
    })

    out = query_route._query_agent("q", top_k=5)
    assert out["text"] == "clean"
    assert out["sources"] == [{"source": "s"}]
    assert out["grounded"] is True and out["run_id"] == "r1"


def test_agent_stream_and_replay_after_approval(monkeypatch):
    store = InMemoryRunStore()
    llms = [ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])]
    _install(monkeypatch, store, llms)
    client = TestClient(app)

    with client.stream("POST", "/agent/runs/stream", json={"goal": "查一下"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        run_id = resp.headers["x-run-id"]
        token = resp.headers["x-approval-token"]
        body = "".join(resp.iter_text())

    assert "event: run_started" in body
    assert "event: step_started" in body
    assert "event: approval_required" in body
    assert "approval_token" not in body and "args" not in body

    llms.append(ScriptedLLM(['{"final":"done"}']))
    approved = client.post(
        f"/agent/runs/{run_id}/approve",
        json={"step_index": 0, "approval_token": token},
    )
    assert approved.status_code == 200 and approved.json()["status"] == "completed"

    with client.stream("GET", f"/agent/runs/{run_id}/events") as events:
        assert events.status_code == 200
        replay = "".join(events.iter_text())
    assert "event: run_completed" in replay
    assert "event: tool_result" in replay

    with client.stream(
        "GET", f"/agent/runs/{run_id}/events", headers={"Last-Event-ID": "3"}
    ) as events:
        after = "".join(events.iter_text())
    assert "id: 2" not in after and "id: 4" in after


def test_stream_disconnect_does_not_cancel_run(monkeypatch):
    store = InMemoryRunStore()
    llms = [ScriptedLLM([
        '{"tool":"kb_search","args":{"query":"q"}}',
        '{"final":"ok"}',
    ])]
    _install(monkeypatch, store, llms, tools={
        "kb_search": lambda query, top_k=5: (time.sleep(0.2), [])[1],
    })
    client = TestClient(app)

    with client.stream("POST", "/agent/runs/stream", json={"goal": "q"}) as resp:
        run_id = resp.headers["x-run-id"]
        next(resp.iter_lines())

    deadline = time.time() + 2.0
    while time.time() < deadline:
        run = store.get(run_id)
        if run is not None and run.status == "completed":
            break
        time.sleep(0.02)
    assert store.get(run_id).status == "completed"


def test_legacy_agent_entry(monkeypatch):
    store = InMemoryRunStore()
    llms = [ScriptedLLM(['{"final":"ok"}'])]
    _install(monkeypatch, store, llms)
    client = TestClient(app)

    resp = client.post("/agent", json={"goal": "问题"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["status"] == "completed"
    assert body["result"]["run_id"]
    assert body["metrics"]["completed_rate"] == 1.0
