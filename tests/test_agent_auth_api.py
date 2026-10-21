"""P1 Auth API：JWT、body user_id 不可信、run owner、scope 二次校验。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
from fastapi.testclient import TestClient

from app import auth
from app.agent.state import InMemoryRunStore
from app.api.main import app
from app.api.routes import agent as agent_route

SECRET = "s" * 32


class ScriptedLLM:
    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def complete(self, system, user):
        out = self.outs[min(self.i, len(self.outs) - 1)]
        self.i += 1
        return out


def _token(sub: str, scopes: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({
        "sub": sub, "scope": scopes, "iss": "issuer", "aud": "audience",
        "exp": now + timedelta(minutes=5),
    }, SECRET, algorithm="HS256")


def _install(monkeypatch, store, llms, tools, seen):
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(
        auth_enabled=True, jwt_secret=SECRET, jwt_alg="HS256",
        jwt_issuer="issuer", jwt_audience="audience",
        oidc_jwks_url="", oidc_issuer="",
    ))
    monkeypatch.setattr(agent_route, "get_settings",
                        lambda: SimpleNamespace(faithfulness_enabled=False))
    monkeypatch.setattr(agent_route, "get_run_store", lambda: store)
    monkeypatch.setattr(agent_route, "get_llm", lambda: llms.pop(0))

    def fake_tools(user_id=None, scopes=None):
        seen["user_id"] = user_id
        seen["scopes"] = scopes
        return tools

    monkeypatch.setattr(agent_route, "_tools", fake_tools)


def test_body_user_id_ignored_and_owner_enforced(monkeypatch):
    store = InMemoryRunStore()
    llms = [ScriptedLLM(['{"final":"ok"}'])]
    seen = {}
    _install(monkeypatch, store, llms, {"kb_search": lambda **kwargs: []}, seen)
    client = TestClient(app)
    h1 = {"Authorization": f"Bearer {_token('u1', 'agent:read')}"}
    h2 = {"Authorization": f"Bearer {_token('u2', 'agent:read')}"}

    created = client.post("/agent/runs", headers=h1,
                          json={"goal": "问题", "user_id": "attacker"})
    assert created.status_code == 200
    body = created.json()
    assert body["user_id"] == "u1"
    assert seen["user_id"] == "u1" and seen["scopes"] == frozenset({"agent:read"})

    listing = client.get("/agent/runs", headers=h2)
    assert listing.status_code == 200 and listing.json()["runs"] == []
    assert client.get("/agent/runs", headers=h1).json()["runs"][0]["run_id"] == body["run_id"]
    assert client.get(f"/agent/runs/{body['run_id']}", headers=h2).status_code == 403
    assert client.get(f"/agent/runs/{body['run_id']}", headers=h1).status_code == 200
    assert client.get(f"/agent/runs/{body['run_id']}/events", headers=h2).status_code == 403
    assert client.get(f"/agent/runs/{body['run_id']}/events", headers=h1).status_code == 200


def test_no_token_401(monkeypatch):
    store = InMemoryRunStore()
    _install(monkeypatch, store, [], {}, {})
    client = TestClient(app)
    assert client.get("/agent/runs/missing").status_code == 401


def test_cross_user_approval_403(monkeypatch):
    store = InMemoryRunStore()
    llms = [
        ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}']),
        ScriptedLLM(['{"final":"done"}']),
    ]
    seen = {}
    _install(monkeypatch, store, llms,
             {"research": lambda topic, **kwargs: {"ok": True}}, seen)
    client = TestClient(app)
    h1 = {"Authorization": f"Bearer {_token('u1', 'agent:research')}"}
    h2 = {"Authorization": f"Bearer {_token('u2', 'agent:research')}"}

    created = client.post("/agent/runs", headers=h1, json={"goal": "研究"}).json()
    assert created["status"] == "awaiting_approval"
    denied = client.post(
        f"/agent/runs/{created['run_id']}/approve", headers=h2,
        json={"step_index": 0, "approval_token": created["approval_token"]},
    )
    assert denied.status_code == 403
    allowed = client.post(
        f"/agent/runs/{created['run_id']}/approve", headers=h1,
        json={"step_index": 0, "approval_token": created["approval_token"]},
    )
    assert allowed.status_code == 200 and allowed.json()["status"] == "completed"


def test_scope_denied_at_runtime(monkeypatch):
    store = InMemoryRunStore()
    llms = [ScriptedLLM([
        '{"tool":"research","args":{"topic":"x"}}',
        '{"final":"ok"}',
    ])]
    seen = {}
    _install(monkeypatch, store, llms,
             {"research": lambda topic, **kwargs: {"ok": True}}, seen)
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {_token('u1', 'agent:read')}"}

    body = client.post("/agent/runs", headers=headers, json={"goal": "研究"}).json()
    assert body["status"] == "completed"
    assert any(s["action"] == "forbidden:research" for s in body["steps"])
