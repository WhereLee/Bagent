"""API 路由清单与就绪探针的 hermetic 测试（不连 DB）。"""
from __future__ import annotations

from app.api.main import app
from app.api.routes import health


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
        "/agent",
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
