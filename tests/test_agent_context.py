"""P2-4：Agent 上下文压缩。"""
from __future__ import annotations

from app.agent.context import compact_history
from app.agent.state import AgentStep


def _steps(n: int):
    return [
        AgentStep(run_id="r", step_index=i, action=f"tool{i}",
                  observation=("x" * 500) + f" obs{i}")
        for i in range(n)
    ]


def test_compact_keeps_goal_and_recent_steps():
    history = compact_history("目标", _steps(8), keep_recent=3, obs_limit=600)
    assert history[0]["content"] == "目标"
    assert "较早步骤摘要" in history[1]["content"]
    assert "tool4" in history[1]["content"] and "tool7" not in history[1]["content"]
    joined = "\n".join(h["content"] for h in history)
    assert "tool7" in joined and "obs7" in joined
    assert max(len(h["content"]) for h in history) < 1000


def test_short_history_has_no_summary():
    history = compact_history("目标", _steps(2), keep_recent=3)
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant", "user"]
    assert all("较早步骤摘要" not in h["content"] for h in history)


def test_max_chars_never_drops_goal():
    history = compact_history("目标", _steps(8), keep_recent=3, max_chars=10)
    assert history[0]["content"] == "目标"
    assert len(history) == 1
