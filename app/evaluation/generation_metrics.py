"""生成质量指标：忠实度/幻觉率聚合、拒答准确率、答案相关性代理。

聚合与判定为纯逻辑（可单测）；faithfulness 的逐句判定在 faithfulness.py 由 LLM 完成，
本模块只消费其结果。answer_relevance_proxy 用词面重叠做廉价的信号（非最终指标）。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class GenSample:
    question: str
    expected_refusal: bool
    predicted_refusal: bool
    faithfulness: float | None  # None 表示不适用（拒答/未开启校验）


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def avg_faithfulness(samples: list[GenSample]) -> float:
    """仅统计非拒答样本的平均忠实度。"""
    vals = [s.faithfulness for s in samples if s.faithfulness is not None]
    return round(mean(vals), 4)


def hallucination_rate(samples: list[GenSample]) -> float:
    """平均幻觉率 = 1 - 平均忠实度（非拒答样本）。"""
    f = avg_faithfulness(samples)
    return round(1.0 - f, 4) if any(s.faithfulness is not None for s in samples) else 0.0


def refusal_accuracy(samples: list[GenSample]) -> float:
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s.expected_refusal == s.predicted_refusal)
    return round(correct / len(samples), 4)


def over_refusal_rate(samples: list[GenSample]) -> float:
    """该答却拒（漏答）的比例——拒答收紧不能把可答题也拒了。"""
    answerable = [s for s in samples if not s.expected_refusal]
    if not answerable:
        return 0.0
    wrong = sum(1 for s in answerable if s.predicted_refusal)
    return round(wrong / len(answerable), 4)
