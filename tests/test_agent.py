"""P1-C agent 循环与轨迹评测的单元测试（fake LLM + fake 工具，无网络/DB）。"""
from app.agent.react import parse_action, run_agent
from app.evaluation.trajectory import is_invalid_action, summarize_trajectories


class ScriptedLLM:
    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def complete(self, system, user):
        out = self.outs[min(self.i, len(self.outs) - 1)]
        self.i += 1
        return out


def _tools():
    return {"kb_search": lambda query, top_k=5: [{"source": "s", "content": query}],
            "kb_answer": lambda query: {"text": "42"}}


def test_parse_action():
    assert parse_action('{"tool":"kb_search","args":{"query":"x"}}') == {"tool": "kb_search", "args": {"query": "x"}}
    assert parse_action("nope") is None
    assert parse_action('{"final":"done"}')["final"] == "done"


def test_agent_converges_with_tool_then_final():
    llm = ScriptedLLM(['{"thought":"查","tool":"kb_search","args":{"query":"Q"}}',
                       '{"thought":"答","final":"答案是42"}'])
    r = run_agent("问题", llm=llm, tools=_tools(), max_steps=4)
    assert r["status"] == "final" and r["final"] == "答案是42"
    assert [s["action"] for s in r["steps"]] == ["kb_search", "final"]


def test_agent_respects_max_steps():
    llm = ScriptedLLM(['{"tool":"kb_search","args":{"query":"Q"}}'])   # 永远调工具，不收尾
    r = run_agent("问题", llm=llm, tools=_tools(), max_steps=3)
    assert r["status"] == "budget_exceeded" and len(r["steps"]) == 3


def test_invalid_then_unknown_still_runs():
    llm = ScriptedLLM(["garbage not json", '{"tool":"nope","args":{}}', '{"final":"ok"}'])
    r = run_agent("问题", llm=llm, tools=_tools(), max_steps=5)
    acts = [s["action"] for s in r["steps"]]
    assert "invalid" in acts and any(a.startswith("unknown:") for a in acts) and r["status"] == "final"


def test_hitl_awaiting_confirmation():
    llm = ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])
    r = run_agent("问题", llm=llm, tools={"research": lambda topic: {}}, max_steps=4)
    assert r["status"] == "awaiting_confirmation"


def test_hitl_confirmed_runs():
    llm = ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}', '{"final":"done"}'])
    r = run_agent("问题", llm=llm, tools={"research": lambda topic: {"markdown": "m"}},
                  max_steps=4, confirm=lambda t, a: True)
    assert r["status"] == "final"


def test_trajectory_metrics():
    trajs = [
        {"status": "final", "steps": [{"action": "kb_search"}, {"action": "final"}]},
        {"status": "budget_exceeded", "steps": [{"action": "invalid"}, {"action": "kb_search"}]},
    ]
    s = summarize_trajectories(trajs)
    assert s["n"] == 2 and s["converged_rate"] == 0.5
    assert abs(s["invalid_action_rate"] - 0.25) < 1e-9      # 4 步中 1 invalid
    assert s["avg_steps"] == 2.0
    assert is_invalid_action("unknown:x") and not is_invalid_action("kb_search")


def test_summarize_empty_safe():
    assert summarize_trajectories([])["n"] == 0
