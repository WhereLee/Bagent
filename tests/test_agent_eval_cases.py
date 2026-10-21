"""P1：20—50 个 mock Agent 任务的 pass^k / 工具序列 / 禁止工具检查。"""
from __future__ import annotations

import time

from app.agent.react import run_agent
from app.evaluation.agent_cases import CASES
from app.evaluation.trajectory import pass_pow_k, summarize_trajectories
from app.tools import KbSearchArgs, TOOL_SPECS, ToolSpec


class CaseLLM:
    def __init__(self, case):
        self.case = case
        self.i = 0

    def _tool_json(self):
        return __import__("json").dumps(
            {"tool": self.case.tool, "args": self.case.args}, ensure_ascii=False
        )

    def complete(self, system, user):
        try:
            return self._complete(system, user)
        finally:
            self.i += 1

    def _complete(self, system, user):
        name = self.case.name
        if name == "budget":
            return self._tool_json()
        if name == "duplicate":
            return self._tool_json() if self.i < 2 else '{"final":"ok"}'
        if name == "invalid-args":
            return '{"tool":"kb_search","args":{}}' if self.i == 0 else '{"final":"ok"}'
        if name in ("research-await", "research-reject"):
            return self._tool_json()
        if self.case.tool is None:
            return '{"final":"ok"}'
        if self.i == 0:
            return self._tool_json()
        return '{"final":"ok"}'


def _tools():
    calls = {"n": 0}

    def flaky(query, top_k=5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("temporary")
        return [{"source": "flaky", "content": "ok"}]

    return {
        "kb_search": lambda query, top_k=5: [{"source": "kb", "content": "ok"}],
        "web_search": lambda query, max_results=5: [{"url": "https://x", "snippet": "ok"}],
        "kb_answer": lambda query, top_k=None: {"text": "ok", "sources": []},
        "memory_search": lambda query, k=3: [],
        "research": lambda topic, **kwargs: {"markdown": "ok"},
        "slow": lambda query, top_k=5: (time.sleep(0.2), [])[1],
        "flaky": flaky,
    }


def _specs():
    specs = dict(TOOL_SPECS)
    specs["slow"] = ToolSpec(
        name="slow", func=lambda query, top_k=5: [], description="slow",
        args_model=KbSearchArgs, timeout_s=0.01,
    )
    specs["flaky"] = ToolSpec(
        name="flaky", func=lambda query, top_k=5: [], description="flaky",
        args_model=KbSearchArgs, max_retries=1, retry_on=("unavailable",),
    )
    return specs


def test_mock_cases_pass_pow_k_and_policy():
    assert 20 <= len(CASES) <= 50
    groups: list[list[bool]] = []
    all_trajectories = []
    for case in CASES:
        outcomes: list[bool] = []
        for _ in range(3):
            confirm = (lambda *_: False) if case.name == "research-reject" else None
            result = run_agent(
                case.goal, llm=CaseLLM(case), tools=_tools(), tool_specs=_specs(),
                max_steps=case.max_steps, confirm=confirm,
            )
            result["expected_tools"] = case.expected_tools
            result["forbidden_tools"] = case.forbidden_tools
            summary = summarize_trajectories([result])
            policy_ok = summary["policy_violation_rate"] in (None, 0.0)
            low_conf_ok = (case.name != "low-confidence"
                           or (result.get("answer") or {}).get("low_confidence") is True)
            outcomes.append(
                result["status"] == case.expected_status
                and summary["tool_sequence_match"] == 1.0
                and policy_ok and low_conf_ok
            )
            all_trajectories.append(result)
        groups.append(outcomes)

    assert pass_pow_k(groups) == 1.0
    assert summarize_trajectories(all_trajectories)["forbidden_tool_calls"] == 0
