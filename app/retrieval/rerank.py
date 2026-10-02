"""精排：Cross-Encoder rerank（BAAI/bge-reranker-base）。

粗召回(dense+lexical) 取 topN 后，用 Cross-Encoder 对每个 (query, doc) 联合打分，
重排得到 topK。Cross-Encoder 比 Bi-Encoder 准但慢（每对都要过一次模型），
因此只对少量候选精排——这正是"粗召回 + 精重排"的生产级链路。
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.config import get_settings


class Reranker:
    def __init__(self) -> None:
        # 命中本地缓存则离线，避免联网校验卡住（缓存位置由 HF_HOME 控制）
        os.environ.setdefault("HF_HOME", str(get_settings().models_dir))
        from sentence_transformers import CrossEncoder

        s = get_settings()
        self.model = CrossEncoder(s.reranker_model_name, device="cpu")

    def rerank(self, query: str, doc_texts: list[str]) -> list[float]:
        """返回与 doc_texts 等长的相关性分数（越大越相关）。"""
        if not doc_texts:
            return []
        pairs = [(query, d) for d in doc_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(x) for x in scores]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
