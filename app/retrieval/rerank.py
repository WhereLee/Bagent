"""精排：Cross-Encoder rerank（BAAI/bge-reranker-base），支持可插拔 int8 动态量化。

粗召回取 topN 后用 Cross-Encoder 对每个 (query, doc) 打分重排。CPU 上是吞吐瓶颈，
故提供 int8 动态量化（PyTorch `quantize_dynamic`，无需导出 ONNX、稳）：体积↓、每请求更快。
量化会否掉精度 → 用 eval_rerank_ab.py 在 DuRetrieval 上实测 Δ（别猜，测）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from app.config import ROOT_DIR, get_settings


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


class OnnxReranker:
    """ONNX Runtime 后端的 reranker（用导出的 fp32 或 int8 .onnx）。与 Reranker 同接口。

    int8 小文件（~279MB）= 磁盘/内存↓且 CPU 更快；只喂导出时声明的 input_ids/attention_mask。
    """

    def __init__(self, model_path: str | Path, tokenizer_name: str | None = None,
                 max_length: int = 512) -> None:
        os.environ.setdefault("HF_HOME", str(get_settings().models_dir))
        import onnxruntime as ort
        from transformers import AutoTokenizer

        s = get_settings()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or s.reranker_model_name)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if s.rerank_threads and s.rerank_threads > 0:
            so.intra_op_num_threads = int(s.rerank_threads)
        self.sess = ort.InferenceSession(str(model_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.max_length = max_length
        self.quantized = True

    def rerank(self, query: str, doc_texts: list[str]) -> list[float]:
        if not doc_texts:
            return []
        enc = self.tokenizer([query] * len(doc_texts), doc_texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        feeds = {k: enc[k].numpy() for k in ("input_ids", "attention_mask") if k in self.input_names}
        logits = self.sess.run(None, feeds)[0]
        return [float(x) for x in logits.reshape(-1)]


def _build_reranker() -> Reranker | OnnxReranker:
    s = get_settings()
    if s.reranker_backend == "onnx":
        path = Path(s.reranker_onnx_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return OnnxReranker(path)
    return Reranker()


def get_reranker() -> Reranker | OnnxReranker:
    global _reranker
    if _reranker is None:
        _reranker = _build_reranker()
    return _reranker


def reset_reranker() -> None:
    global _reranker
    _reranker = None
