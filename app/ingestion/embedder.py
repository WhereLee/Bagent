"""稠密向量编码器：封装 sentence-transformers + BGE。

要点：
- 单例加载，避免每次请求重载模型（CPU 上加载很贵）。
- BGE 检索最佳实践：查询加指令前缀、文档不加；向量做 L2 归一化，
  配合库里的 cosine 距离。
- 模型权重从项目内 models/ 目录加载（HF_HOME 已在 config 中指向此处）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import get_settings

# BGE 中文检索官方推荐的查询指令
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        s = get_settings()
        self.model = SentenceTransformer(
            s.embedding_model_name,
            cache_folder=str(s.models_dir),
            device="cpu",
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = s.embedding_batch_size

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [BGE_QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
