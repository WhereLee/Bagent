"""写回污染评测的核心判定（纯函数、可单测、离线确定）。

四类判定，把"同时给出对+错值"识别为**主动暴露冲突（好行为）**，与"采纳错误值（污染）"分开——
否则会把我们最想要的正确行为误算成失败。

- correct : 只给对值        - wrong(污染) : 只给错值
- conflict : 对错值都提到（标注了分歧）   - abstain : 都没提（未知/拒答）
"""
from __future__ import annotations

import re

from app.evaluation.stats import bootstrap_ci, mean

_OUTCOME_ORDER = ("correct", "wrong", "conflict", "abstain")


def _mentions(text: str, value: str) -> bool:
    """整词匹配（避免 900 命中 9000 之类的子串误判）。"""
    return re.search(rf"(?<!\d){re.escape(value)}(?!\d)", text or "") is not None


def classify_answer(text: str, correct_value: str, wrong_value: str) -> str:
    c = _mentions(text, correct_value)
    w = _mentions(text, wrong_value)
    if c and w:
        return "conflict"
    if w:
        return "wrong"
    if c:
        return "correct"
    return "abstain"


def outcome_across_repeats(outcomes: list[str]) -> str:
    """同一探针多次重复的聚合：任一 wrong 即判 wrong（对污染从严），
    否则有 conflict 记 conflict，有 correct 记 correct，全 abstain 记 abstain。"""
    s = set(outcomes)
    if "wrong" in s:
        return "wrong"
    if "conflict" in s:
        return "conflict"
    if "correct" in s:
        return "correct"
    return "abstain"


def outcome_report(outcomes: list[str]) -> dict:
    """对一组（每探针一个）结果标签算分布 + 污染率/暴露冲突率，污染率带 bootstrap 95%CI。"""
    n = len(outcomes)
    counts = {k: 0 for k in _OUTCOME_ORDER}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    contam_indicator = [1.0 if o == "wrong" else 0.0 for o in outcomes]
    lo, hi = bootstrap_ci(contam_indicator) if n else (0.0, 0.0)
    return {
        "n": n,
        "counts": counts,
        "rates": {k: (counts[k] / n if n else 0.0) for k in _OUTCOME_ORDER},
        "contamination_rate": mean(contam_indicator) if n else 0.0,
        "contamination_ci": (lo, hi),
        "surfaced_conflict_rate": counts["conflict"] / n if n else 0.0,
    }
