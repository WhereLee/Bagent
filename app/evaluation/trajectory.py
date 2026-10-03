"""agent 轨迹评测指标（纯函数，可单测）。

轨迹 = run_agent 的返回。测三件事：收敛率（拿到 final）、非法动作率（invalid/unknown/error）、
平均步数（成本/效率）——把"agent 走得对不对"变成有数字，而非只看最终答案。
"""
from __future__ import annotations

from app.evaluation.stats import bootstrap_ci, mean

_INVALID_PREFIXES = ("invalid", "unknown:", "error:", "awaiting_confirm:")


def is_invalid_action(action: str | None) -> bool:
    return action is None or any(str(action).startswith(p) for p in _INVALID_PREFIXES)


def summarize_trajectories(trajectories: list[dict]) -> dict:
    n = len(trajectories)
    if n == 0:
        return {"n": 0, "converged_rate": 0.0, "invalid_action_rate": 0.0, "avg_steps": 0.0,
                "converged_ci": (0.0, 0.0)}
    converged = [1.0 if t.get("status") == "final" else 0.0 for t in trajectories]
    step_flags, steps_counts = [], []
    for t in trajectories:
        steps = t.get("steps", [])
        steps_counts.append(len(steps))
        for s in steps:
            step_flags.append(1.0 if is_invalid_action(s.get("action")) else 0.0)
    lo, hi = bootstrap_ci(converged)
    return {
        "n": n,
        "converged_rate": mean(converged),
        "converged_ci": (lo, hi),
        "invalid_action_rate": mean(step_flags) if step_flags else 0.0,
        "avg_steps": mean([float(x) for x in steps_counts]),
    }
