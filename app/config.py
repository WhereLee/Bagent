"""集中式配置。

所有可调项从项目根目录的 .env 读取；代码里不写死任何密钥或路径。
额外职责：在导入 sentence-transformers 之前，把 HuggingFace 缓存目录
强制指向项目内的 models/，避免权重落到 C 盘（用户明确要求）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 = 本文件上溯两级（app/config.py -> 项目根）
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PostgreSQL / pgvector ---
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: str = ""
    pg_database: str = "bagent_rag"

    # --- LLM: Xiaomi MiMo ---
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"
    mimo_api_key: str = ""
    mimo_model: str = "mimo-v2.6-flash"
    mimo_thinking: str = "disabled"

    # --- Embedding ---
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_model_path: str = "./models"
    embedding_dim: int = 512
    embedding_batch_size: int = 64
    # 国内直连 huggingface.co 常失败，默认走镜像；可在 .env 覆盖
    hf_endpoint: str = "https://hf-mirror.com"

    # --- Retrieval ---
    retrieval_top_k: int = 5
    retrieval_candidate_n: int = 20
    # 检索模式：dense(纯向量) / hybrid(向量+BM25，RRF 融合)；hybrid 下再按需 rerank
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    # 父子块：命中子块后是否扩展为父块内容喂给 LLM
    retrieve_parent: bool = True

    # --- Chunking (parent-child / small-to-big) ---
    chunk_parent_tokens: int = 600
    chunk_child_tokens: int = 150
    chunk_child_overlap: int = 30

    # --- Rerank (Cross-Encoder) ---
    rerank_enabled: bool = True
    reranker_model_name: str = "BAAI/bge-reranker-base"

    # --- App ---
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy 2.x + psycopg3 的连接串。"""
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def models_dir(self) -> Path:
        p = (ROOT_DIR / self.embedding_model_path).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def apply_hf_cache(self) -> None:
        """把 HF 缓存写到项目目录；若模型已缓存则自动离线加载。

        国内直连 huggingface.co 校验会长时间卡住，模型一旦下载完成就强制
        离线，避免 ingest/query 进程卡在联网检查上。
        """
        os.environ.setdefault("HF_HOME", str(self.models_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(self.models_dir / "hub"))
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        # 模型快照已存在 -> 离线，避免网络校验卡住
        snapshot = self.models_dir / ("models--" + self.embedding_model_name.replace("/", "--"))
        if snapshot.exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.apply_hf_cache()
    return s
