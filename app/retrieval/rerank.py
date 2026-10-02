"""精排：Cross-Encoder rerank（BAAI/bge-reranker-base），支持可插拔 int8 动态量化。

粗召回取 topN 后用 Cross-Encoder 对每个 (query, doc) 打分重排。CPU 上是吞吐瓶颈，
故提供 int8 动态量化（PyTorch `quantize_dynamic`，无需导出 ONNX、稳）：体积↓、每请求更快。
量化会否掉精度 → 用 eval_rerank_ab.py 在 DuRetrieval 上实测 Δ（别猜，测）。
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.config import get_settings


class Reranker:
    def __init__(self, int8: bool | None = None) -> None:
        os.environ.setdefault("HF_HOME", str(get_settings().models_dir))
        from sentence_transformers import CrossEncoder

        s = get_settings()
        self.model = CrossEncoder(s.reranker_model_name, device="cpu")
        self.quantized = False
        if s.rerank_threads and s.rerank_threads > 0:
            try:
                import torch
                torch.set_num_threads(int(s.rerank_threads))   # 防并发过订（压测踩过的坑）
            except Exception:  # noqa: BLE001
                pass
        if int8 if int8 is not None else s.reranker_int8:
            self.quantize_dynamic()

    def quantize_dynamic(self) -> None:
        """就地做 int8 动态量化（对 Linear 层）。幂等：已量化则跳过。"""
        if self.quantized:
            return
        import torch
        torch.ao.quantization.quantize_dynamic(
            self.model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
        self.quantized = True

    def rerank(self, query: str, doc_texts: list[str]) -> list[float]:
        """返回与 doc_texts 等长的相关性分数（越大越相关）。"""
        if not doc_texts:
            return []
        pairs = [(query, d) for d in doc_texts]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(x) for x in scores]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def reset_reranker() -> None:
    global _reranker
    _reranker = None
