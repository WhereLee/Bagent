"""P3-5：工具完整结果留存与按需回查。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent.react import run_agent
from app.agent.state import InMemoryRunStore
from app.api.main import app
from app.api.routes import agent as agent_route
from app.tools import KbSearchArgs, ToolSpec


class LLM:
    def __init__(self):
        self.i = 0

    def complete(self, system, user):
        self.i += 1
        return ('{"tool":"long","args":{"query":"q"}}' if self.i == 1
                else '{"final":"ok"}')


def test_long_observation_kept_out_of_preview_but_available_on_detail(monkeypatch):
    store = InMemoryRunStore()
    long_text = "x" * 5000
    spec = ToolSpec(name="long", func=lambda query, top_k=5: long_text,
                    description="long", args_model=KbSearchArgs)
    result = run_agent("q", llm=LLM(), tools={"long": lambda **kwargs: long_text},
                       tool_specs={"long": spec}, store=store, max_steps=3)
    step = store.list_steps(result["run_id"])[0]
    assert len(step.observation) < 2000
    assert len(step.observation_full) > 5000

    monkeypatch.setattr(agent_route, "get_run_store", lambda: store)
    client = TestClient(app)
    res = client.get(f"/agent/runs/{result['run_id']}/steps/0")
    assert res.status_code == 200
    assert len(res.json()["observation_full"]) > 5000
    assert res.json()["has_full_observation"] is True
