"""显式有界 agent 循环。"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import AgentReq
from app.generation.llm import get_llm
from app.tools import build_tools

router = APIRouter()


@router.post("/agent")
def agent_run(req: AgentReq) -> dict:
    """显式有界 agent 循环：复用 app.tools 工具集（联网只经受控 web_search，主 LLM 不隐式联网）。"""
    from app.agent.react import run_agent
    from app.evaluation.trajectory import summarize_trajectories

    tools = build_tools(user_id=req.user_id)
    traj = run_agent(req.goal, llm=get_llm(), tools=tools, max_steps=req.max_steps)
    return {"result": traj, "metrics": summarize_trajectories([traj])}
