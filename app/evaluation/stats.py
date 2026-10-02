"""评估统计：bootstrap 置信区间 + 均值。让"指标"带不确定性，而非报单点。

纯函数、可复现（固定 seed）、可单测。
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """对样本均值做 bootstrap 重采样，返回 (下界, 上界)。

    样本量 0 -> (0,0)；1 -> (v,v)（无法估波动）。
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(
        mean([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(n_resamples)
    )
    lo_idx = int((1 - confidence) / 2 * n_resamples)
    hi_idx = int((1 + confidence) / 2 * n_resamples) - 1
    hi_idx = min(hi_idx, n_resamples - 1)
    return (means[lo_idx], means[hi_idx])


def summarize(values: Sequence[float], confidence: float = 0.95) -> dict:
    """指标列表的均值 + 标准差 + bootstrap 置信区间。"""
    lo, hi = bootstrap_ci(values, confidence=confidence)
    return {"mean": mean(values), "std": std(values), "ci_low": lo, "ci_high": hi, "n": len(values)}
