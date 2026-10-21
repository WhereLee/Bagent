"""P3-6：可选的 Agent 真实运行评测（不进入 CI）。

默认使用真实 LLM + 真实工具；没有 API key 时请改用 pytest mock cases。
用法：
  python scripts/eval_agent_live.py --limit 5 --k 1
  python scripts/eval_agent_live.py --json
"""
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from app.agent.react import run_agent
from app.agent.state import InMemoryRunStore
from app.evaluation.agent_cases import CASES
from app.evaluation.trajectory import pass_pow_k, summarize_trajectories
from app.generation.llm import get_llm
from app.tools import TOOL_SPECS, build_tools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--include-approval", action="store_true")
    args = parser.parse_args()

    cases = [c for c in CASES if args.include_approval or not c.requires_approval][:args.limit]
    groups: list[list[bool]] = []
    trajectories = []
    for case in cases:
        outcomes = []
        for _ in range(max(1, args.k)):
            store = InMemoryRunStore()
            result = run_agent(
                case.goal, llm=get_llm(), tools=build_tools(),
                tool_specs=TOOL_SPECS, store=store, max_steps=case.max_steps,
                confirm=(lambda *_: True) if args.include_approval else None,
            )
            result["expected_tools"] = case.expected_tools
            result["forbidden_tools"] = case.forbidden_tools
            trajectories.append(result)
            outcomes.append(result["status"] == case.expected_status)
        groups.append(outcomes)

    summary = summarize_trajectories(trajectories)
    summary["pass_pow_k"] = pass_pow_k(groups)
    summary["cases"] = len(cases)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
