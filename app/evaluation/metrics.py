"""检索质量指标（纯实现，二值相关）。

输入统一为：ranked_ids（按相关性降序的候选 id）、relevant_ids（该 query 的金标准 id 集合）。
指标：recall@k / precision@k / MRR / nDCG@k。
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / k


def mrr(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top = ranked_ids[:k]
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, rid in enumerate(top, start=1)
        if rid in relevant_ids
    )
    ideal_n = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    ranked_ids: list[int], relevant_ids: Iterable[int], k: int
) -> dict[str, float]:
    rel = set(relevant_ids)
    return {
        f"recall@{k}": recall_at_k(ranked_ids, rel, k),
        f"precision@{k}": precision_at_k(ranked_ids, rel, k),
        "mrr": mrr(ranked_ids, rel),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, rel, k),
    }
