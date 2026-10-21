"""P2：有界上下文压缩。

不额外调用 LLM：把较早步骤压成动作/观察摘要，保留最近步骤原文。
好处是可预测、无额外成本；坏处是有损，所以原始 step 仍在 agent_steps 中可回查。
"""
from __future__ import annotations

import json

from app.agent.state import AgentStep

_SUMMARY_HEAD = "[较早步骤摘要；完整轨迹已持久化，可按 run_id 回查]"


def _one(step: AgentStep, *, obs_limit: int) -> dict:
    assistant = {"thought": step.thought, "tool": step.action, "args": step.args}
    out = {
        "role": "assistant",
        "content": json.dumps(assistant, ensure_ascii=False, default=str),
    }
    if step.observation is not None or step.error is not None:
        observation = (step.observation or step.error or "")[:obs_limit]
        out_obs = {
            "role": "user",
            "content": f"工具 {step.action} 结果：{observation}\n请给出下一步 JSON 动作或 final。",
        }
        return [out, out_obs]
    return [out]


def compact_history(goal: str, steps: list[AgentStep], *, max_chars: int = 12000,
                    keep_recent: int = 4, obs_limit: int = 240) -> list[dict]:
    """保留 goal + 最近步骤原文；较早步骤折叠为一行动作摘要。"""
    if not steps:
        return [{"role": "user", "content": goal}]

    recent = steps[-keep_recent:] if keep_recent > 0 else steps
    older = steps[:-keep_recent] if keep_recent > 0 else []
    history: list[dict] = [{"role": "user", "content": goal}]

    if older:
        lines = []
        for s in older:
            obs = (s.observation or s.error or "").replace("\n", " ")[:120]
            suffix = f" | {obs}" if obs else ""
            lines.append(f"- step {s.step_index}: {s.action}{suffix}")
        history.append({"role": "user", "content": _SUMMARY_HEAD + "\n" + "\n".join(lines)})

    for s in recent:
        history.extend(_one(s, obs_limit=obs_limit))

    # 总字符预算：从最早的非 goal 条目开始丢弃，goal 永不丢。
    while len(history) > 1 and sum(len(h["content"]) for h in history) > max_chars:
        history.pop(1)
    return history
