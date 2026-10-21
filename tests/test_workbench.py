"""P2-4：最小工作台页面契约。"""
from fastapi.testclient import TestClient

from app.api.main import app


def test_workbench_page_served():
    client = TestClient(app)
    res = client.get("/workbench")
    assert res.status_code == 200
    assert "Bagent Agent Workbench" in res.text
    assert "/agent/runs/stream" in res.text
    assert "/agent/runs/" in res.text and "/events" in res.text
