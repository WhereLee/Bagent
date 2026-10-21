"""API 路由清单与就绪探针的 hermetic 测试（不连 DB）。"""
from __future__ import annotations

from app.api.main import app
from app.api.routes import health
from app.observability.middleware import EXEMPT_PATHS


def test_ready_is_exempt_from_auth_and_rate_limit():
    assert "/ready" in EXEMPT_PATHS


def test_query_stream_sse_frames(monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.routes import query as query_route

    def fake_stream(*args, **kwargs):
        yield "start", {"query": "q"}
        yield "token", {"text": "你好"}
        yield "done", {"text": "你好", "sources": []}

    monkeypatch.setattr(query_route, "stream_answer", fake_stream)
    client = TestClient(app)   # 不用 with：避免触发 lifespan 连 DB
    with client.stream("POST", "/query/stream", json={"query": "q"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"
        body = "".join(resp.iter_text())
    assert "event: start" in body and "event: done" in body
    assert "data:" in body


def test_known_routes_registered():
    paths = {r.path for r in app.routes}
    expected = {
        "/health", "/metrics", "/ready",
        "/documents", "/stats",
        "/ingest", "/delete", "/search",
        "/query", "/chat", "/query/stream", "/chat/stream",
        "/memory", "/memory/promote", "/memory/invalidate", "/memory/feedback",
        "/research", "/research/{sid}", "/research/{sid}/refine", "/research/{sid}/publish",
        "/events", "/events/{event_id}/playbook",
        "/mcp/tools", "/mcp/call", "/workbench",
        "/agent", "/agent/runs", "/agent/runs/stream",
        "/agent/runs/{run_id}", "/agent/runs/{run_id}/events",
        "/agent/runs/{run_id}/operations",
        "/agent/runs/{run_id}/steps/{step_index}",
        "/agent/runs/{run_id}/approve", "/agent/runs/{run_id}/reject",
        "/agent/runs/{run_id}/cancel",
    }
    assert expected <= paths


class _Conn:
    def execute(self, stmt):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Engine:
    def connect(self):
        return _Conn()


class _BrokenEngine:
    def connect(self):
        raise RuntimeError("db down")


def test_ready_ok(monkeypatch):
    monkeypatch.setattr(health, "get_engine", lambda: _Engine())
    assert health.ready()["status"] == "ready"


def test_ready_503_when_db_unavailable(monkeypatch):
    monkeypatch.setattr(health, "get_engine", lambda: _BrokenEngine())
    resp = health.ready()
    assert resp.status_code == 503
