"""P1-C：显式有界 agent 循环（ReAct 式）+ 轨迹记录。

- 复用 app.tools.TOOLS 作动作空间；主 LLM 每步只产出一个动作 JSON，**自身绝不挂 web_search**
  （联网只经 web_search 工具走我们受控的检索路径）。
- 有界：max_steps 到即停；非法动作/未知工具计入 invalid；HITL：requires_confirm 的工具无放行则挂起。
"""
from __future__ import annotations

import json
import re
from typing import Callable

from app.generation.prompts import AGENT_SYSTEM
from app.observability.logging import get_logger, log_event
from app.tools import TOOLS, describe_tools

_log = get_logger("agent")
_JSON = re.compile(r"\{.*\}", re.S)


def parse_action(text: str) -> dict | None:
    """解析模型输出的单个动作 JSON；无法解析返回 None。"""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def run_agent(
    goal: str, *, llm, tools: dict | None = None, max_steps: int = 6,
    requires_confirm: set[str] | None = None, confirm: Callable[[str, dict], bool] | None = None,
) -> dict:
    """返回轨迹：steps[{thought,action,observation}]、final、status。"""
    tools = tools if tools is not None else TOOLS
    requires_confirm = requires_confirm or {"research"}
    menu = describe_tools() if tools is TOOLS else "\n".join(f"- {n}" for n in tools)
    history: list[dict] = [{"role": "user", "content": goal}]
    steps: list[dict] = []
    status = "budget_exceeded"

    for _ in range(max_steps):
        raw = llm.complete(system=AGENT_SYSTEM.format(tools=menu),
                           user="\n".join(_render(h) for h in history))
        act = parse_action(raw)
        if not act:
            steps.append({"thought": None, "action": "invalid", "observation": "无法解析动作"})
            history.append({"role": "user", "content": "上一步不是合法 JSON，请只输出一个 JSON 动作。"})
            continue
        if act.get("final") is not None:
            steps.append({"thought": act.get("thought"), "action": "final", "observation": ""})
            return {"goal": goal, "steps": steps, "final": str(act["final"]), "status": "final"}

        tool = act.get("tool")
        args = act.get("args") or {}
        if tool not in tools:
            steps.append({"thought": act.get("thought"), "action": f"unknown:{tool}", "observation": "未知工具"})
            history.append({"role": "user", "content": f"工具 {tool} 不存在。"})
            continue
        if tool in requires_confirm and not (confirm and confirm(tool, args)):
            steps.append({"thought": act.get("thought"), "action": f"awaiting_confirm:{tool}", "observation": "待人工确认"})
            return {"goal": goal, "steps": steps, "final": None, "status": "awaiting_confirmation"}

        try:
            obs = tools[tool](**args)
            obs_str = json.dumps(obs, ensure_ascii=False)[:1500]
            steps.append({"thought": act.get("thought"), "action": tool, "observation": obs_str})
        except Exception as e:  # noqa: BLE001
            obs_str = json.dumps({"error": str(e)[:200]}, ensure_ascii=False)
            steps.append({"thought": act.get("thought"), "action": f"error:{tool}", "observation": obs_str})
        history.append({"role": "user", "content": f"工具 {tool} 结果：{obs_str}\n请给出下一步 JSON 动作或 final。"})

    log_event(_log, "info", "agent_run", goal=goal[:60], steps=len(steps), status=status)
    return {"goal": goal, "steps": steps, "final": None, "status": status}


def _render(msg: dict) -> str:
    return f"{msg['role']}: {msg['content']}"
