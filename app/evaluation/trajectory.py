"""agent 轨迹评测指标（纯函数，可单测）。

轨迹 = run_agent 的返回。测三件事：收敛率（拿到 final）、非法动作率（invalid/unknown/error）、
平均步数（成本/效率）——把"agent 走得对不对"变成有数字，而非只看最终答案。
"""
from __future__ import annotations

from app.evaluation.stats import bootstrap_ci, mean

_INVALID_PREFIXES = ("invalid", "unknown:", "error:", "invalid_args:")
_NON_TOOL_PREFIXES = _INVALID_PREFIXES + (
    "awaiting_approval:", "rejected:", "timeout:", "unavailable:", "duplicate:",
    "forbidden:", "unknown_outcome:",
)
_COMPLETED = {"completed", "final"}
_AWAITING = {"awaiting_approval", "awaiting_confirmation"}


def is_invalid_action(action: str | None) -> bool:
    return action is None or any(str(action).startswith(p) for p in _INVALID_PREFIXES)


def pass_pow_k(groups: list[list[bool]]) -> float:
    """pass^k：每组 k 次全部成功才算该任务通过；无样本返回 0。"""
    if not groups:
        return 0.0
    return mean([1.0 if group and all(group) else 0.0 for group in groups])


def summarize_trajectories(trajectories: list[dict]) -> dict:
    n = len(trajectories)
    if n == 0:
        return {"n": 0, "converged_rate": 0.0, "completed_rate": 0.0,
                "awaiting_approval_rate": 0.0, "rejected_rate": 0.0,
                "invalid_action_rate": 0.0, "avg_steps": 0.0,
                "avg_latency_ms": 0.0, "total_tokens": 0,
                "tool_sequence_match": None, "policy_violation_rate": None,
                "forbidden_tool_calls": 0, "converged_ci": (0.0, 0.0)}
    completed = [1.0 if t.get("status") in _COMPLETED else 0.0 for t in trajectories]
    awaiting = [1.0 if t.get("status") in _AWAITING else 0.0 for t in trajectories]
    rejected = [1.0 if t.get("status") == "rejected" else 0.0 for t in trajectories]
    step_flags, steps_counts, latencies = [], [], []
    total_tokens = 0
    sequence_matches: list[float] = []
    policy_flags: list[float] = []
    forbidden_calls = 0
    for t in trajectories:
        steps = t.get("steps", [])
        steps_counts.append(len(steps))
        for s in steps:
            step_flags.append(1.0 if is_invalid_action(s.get("action")) else 0.0)
            if s.get("latency_ms") is not None:
                latencies.append(float(s["latency_ms"]))
            total_tokens += int(s.get("token_input") or 0) + int(s.get("token_output") or 0)
        expected = t.get("expected_tools")
        if isinstance(expected, list):
            actual = [s.get("action") for s in steps
                      if s.get("action") not in (None, "final")
                      and not str(s.get("action")).startswith(tuple(_NON_TOOL_PREFIXES))]
            sequence_matches.append(1.0 if actual == expected else 0.0)
        forbidden = t.get("forbidden_tools")
        if isinstance(forbidden, list):
            called = [s.get("action") for s in steps]
            hits = sum(1 for a in called if a in forbidden)
            forbidden_calls += hits
            policy_flags.append(0.0 if hits else 1.0)
    lo, hi = bootstrap_ci(completed)
    return {
        "n": n,
        "converged_rate": mean(completed),
        "completed_rate": mean(completed),
        "awaiting_approval_rate": mean(awaiting),
        "rejected_rate": mean(rejected),
        "converged_ci": (lo, hi),
        "invalid_action_rate": mean(step_flags) if step_flags else 0.0,
        "avg_steps": mean([float(x) for x in steps_counts]),
        "avg_latency_ms": mean(latencies) if latencies else 0.0,
        "total_tokens": total_tokens,
        "tool_sequence_match": mean(sequence_matches) if sequence_matches else None,
        "policy_violation_rate": (1.0 - mean(policy_flags)) if policy_flags else None,
        "forbidden_tool_calls": forbidden_calls,
    }
