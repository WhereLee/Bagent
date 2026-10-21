"""P0 agent 运行时与轨迹评测的单测（fake LLM + fake 工具，无网络/DB）。"""
from types import SimpleNamespace

from app.agent.react import parse_action, resume_run, run_agent, start_run
from app.agent.state import InMemoryRunStore, new_run
from app.evaluation.trajectory import is_invalid_action, summarize_trajectories
from app.generation.llm import LLMClient
from app.tools import TOOL_SPECS


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
    assert r["status"] == "completed" and r["final"] == "答案是42"
    assert [s["action"] for s in r["steps"]] == ["kb_search", "final"]


def test_agent_respects_max_steps():
    llm = ScriptedLLM(['{"tool":"kb_search","args":{"query":"Q"}}'])   # 永远调工具，不收尾
    r = run_agent("问题", llm=llm, tools=_tools(), max_steps=3)
    assert r["status"] == "budget_exceeded" and len(r["steps"]) == 3


def test_invalid_then_unknown_still_runs():
    llm = ScriptedLLM(["garbage not json", '{"tool":"nope","args":{}}', '{"final":"ok"}'])
    r = run_agent("问题", llm=llm, tools=_tools(), max_steps=5)
    acts = [s["action"] for s in r["steps"]]
    assert "invalid" in acts and any(a.startswith("unknown:") for a in acts) and r["status"] == "completed"
    unknown = next(s for s in r["steps"] if s["action"].startswith("unknown:"))
    assert '"error_type": "unknown"' in unknown["observation"]


def test_hitl_awaiting_confirmation():
    llm = ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])
    r = run_agent("问题", llm=llm, tools={"research": lambda topic: {}}, max_steps=4)
    assert r["status"] == "awaiting_approval"
    assert r["pending_action"]["tool"] == "research"


def test_hitl_confirmed_runs():
    llm = ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}', '{"final":"done"}'])
    r = run_agent("问题", llm=llm, tools={"research": lambda topic, **kwargs: {"markdown": "m"}},
                  max_steps=4, confirm=lambda t, a: True)
    assert r["status"] == "completed"


def test_agent_approve_resumes_and_completes():
    store = InMemoryRunStore()
    first = ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])
    r = run_agent("问题", llm=first, tools={"research": lambda topic, **kwargs: {"ok": True}},
                  max_steps=4, store=store)
    assert r["status"] == "awaiting_approval"
    run = store.get(r["run_id"])
    second = ScriptedLLM(['{"final":"done"}'])
    r2 = resume_run(
        r["run_id"], decision="approve", approval_token=run.approval_token,
        llm=second, tools={"research": lambda topic, **kwargs: {"ok": True}},
        tool_specs=TOOL_SPECS, store=store,
    )
    assert r2.status == "completed"
    assert [s.action for s in store.list_steps(r2.id)] == ["research", "final"]


def test_agent_reject_does_not_execute():
    store = InMemoryRunStore()
    called = {"n": 0}

    def research(topic, **kwargs):
        called["n"] += 1
        return {}

    llm = ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}'])
    r = run_agent("问题", llm=llm, tools={"research": research}, max_steps=4, store=store)
    run = store.get(r["run_id"])
    r2 = resume_run(
        r["run_id"], decision="reject", approval_token=run.approval_token,
        llm=ScriptedLLM(['{"final":"should not run"}']), tools={"research": research},
        tool_specs=TOOL_SPECS, store=store, reason="no",
    )
    assert r2.status == "rejected" and called["n"] == 0


def test_agent_cancel_running_run():
    store = InMemoryRunStore()
    run = new_run("问题")
    store.create(run)
    cancelled = resume_run(
        run.id, decision="cancel", approval_token=run.approval_token,
        llm=None, store=store,
    )
    assert cancelled.status == "cancelled"


def test_agent_aggregates_llm_usage_into_step_and_run():
    class UsageLLM:
        def complete(self, system, user):
            LLMClient._record_usage(SimpleNamespace(prompt_tokens=3, completion_tokens=4))
            return '{"final":"ok"}'

    r = run_agent("问题", llm=UsageLLM(), tools=_tools(), max_steps=2)
    assert r["token_input"] == 3 and r["token_output"] == 4
    assert r["steps"][-1]["token_input"] == 3
    assert r["steps"][-1]["token_output"] == 4


def test_agent_invalid_args_do_not_execute():
    called = {"n": 0}

    def kb_search(query, top_k=5):
        called["n"] += 1
        return []

    llm = ScriptedLLM(['{"tool":"kb_search","args":{}}', '{"final":"done"}'])
    r = run_agent("问题", llm=llm, tools={"kb_search": kb_search}, max_steps=4)
    assert r["status"] == "completed" and called["n"] == 0
    step = next(s for s in r["steps"] if s["action"] == "invalid_args:kb_search")
    assert '"error_type": "invalid_args"' in step["observation"]
    assert '"retryable": false' in step["observation"]


def test_cancel_during_llm_is_checked_before_tool():
    store = InMemoryRunStore()
    run = new_run("问题")
    store.create(run)
    called = {"n": 0}

    class CancelLLM:
        def complete(self, system, user):
            current = store.get(run.id)
            current.status = "cancelled"
            store.update(current)
            return '{"tool":"kb_search","args":{"query":"q"}}'

    def kb_search(query, top_k=5):
        called["n"] += 1
        return []

    result = start_run(
        "问题", run=run, llm=CancelLLM(), tools={"kb_search": kb_search},
        store=store, max_steps=3,
    )
    assert result.status == "cancelled" and called["n"] == 0
    assert store.list_steps(run.id) == []


def test_trajectory_metrics():
    trajs = [
        {"status": "final", "steps": [{"action": "kb_search"}, {"action": "final"}]},
        {"status": "budget_exceeded", "steps": [{"action": "invalid"}, {"action": "kb_search"}]},
    ]
    s = summarize_trajectories(trajs)
    assert s["n"] == 2 and s["converged_rate"] == 0.5
    assert s["completed_rate"] == 0.5
    assert abs(s["invalid_action_rate"] - 0.25) < 1e-9      # 4 步中 1 invalid
    assert s["avg_steps"] == 2.0
    assert is_invalid_action("unknown:x") and not is_invalid_action("kb_search")


def test_trajectory_awaiting_is_not_invalid():
    s = summarize_trajectories([{
        "status": "awaiting_approval",
        "steps": [{"action": "research", "latency_ms": 10, "token_input": 3, "token_output": 2}],
    }])
    assert s["awaiting_approval_rate"] == 1.0
    assert s["completed_rate"] == 0.0
    assert s["invalid_action_rate"] == 0.0
    assert s["total_tokens"] == 5


def test_trajectory_expected_tools():
    s = summarize_trajectories([{
        "status": "completed",
        "expected_tools": ["kb_search"],
        "steps": [{"action": "kb_search"}, {"action": "final"}],
    }])
    assert s["tool_sequence_match"] == 1.0


def test_summarize_empty_safe():
    assert summarize_trajectories([])["n"] == 0
