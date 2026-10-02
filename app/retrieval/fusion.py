"""排名融合：Reciprocal Rank Fusion (RRF)。

RRF 只依赖各路结果的"排名"而非分数，天然规避了稠密相似度与 BM25 分数量纲不可比的问题。
score(d) = Σ_r  weight_r / (k + rank_r(d))，rank 从 1 开始，k 默认 60。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable, Sequence


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Hashable]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[Hashable, float]]:
    """融合多路"按相关性降序的 id 列表"，返回 [(id, fused_score), ...] 降序。"""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights 与 ranked_lists 长度不一致")

    scores: dict[Hashable, float] = defaultdict(float)
    for doc_ids, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(doc_ids, start=1):  # rank 从 1 开始
            scores[doc_id] += w / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
