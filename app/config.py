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

    # --- LLM: DeepSeek (OpenAI 兼容) ---
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-flash"

    # --- LLM provider 路由与韧性 ---
    llm_primary_provider: str = "deepseek"    # deepseek | mimo
    llm_fallback_provider: str = "mimo"        # 置空则不降级
    llm_timeout: float = 15.0
    llm_max_retries: int = 3                   # 每个 provider 的重试次数
    llm_retry_base_delay: float = 0.5          # 指数退避基准秒

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
    # 自适应精排：dense top1 分数高过该阈值则跳过 rerank（降延迟）；None 关闭
    rerank_skip_threshold: float | None = None
    reranker_int8: bool = False      # int8 动态量化（需先过 A/B 精度实测）
    rerank_threads: int = 0          # >0 时限制 torch 线程，防并发过订

    # --- Generation / Evidence chain (M3) ---
    force_citation: bool = True          # 是否强制引用（消融对照）
    faithfulness_threshold: float = 0.6  # 低于此值标记为低置信
    rewrite_enabled: bool = True         # 多轮查询改写
    faithfulness_enabled: bool = True    # 生成后做忠实度校验

    # --- Self-RAG / 知识冲突 (M8) ---
    self_rag_enabled: bool = False       # 默认关（不静默改变现有行为/成本）
    self_rag_max_iters: int = 2          # 最大重检轮数（防死循环）
    conflict_check_enabled: bool = True  # 随 self_rag 启用

    # --- App ---
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- Rate limit (M4) ---
    rate_limit_enabled: bool = True
    rate_limit_per_sec: float = 5.0   # 每 IP 每秒补充令牌数
    rate_limit_burst: int = 10        # 突发容量

    # --- Retrieval cache (M6) ---
    retrieval_cache_enabled: bool = True
    retrieval_cache_max: int = 256
    retrieval_cache_ttl: int = 300    # 秒；且写入/删除时主动失效

    # --- Security (M6) ---
    api_key: str = ""                 # 非空则启用 /query /search 等鉴权
    prompt_injection_guard: bool = True

    # --- Web search / 联网 (M9a) ---
    web_search_enabled: bool = False   # 默认关（不影响现有离线链路/测试）
    search_provider: str = "mock"      # mock | searxng
    searxng_base_url: str = "http://127.0.0.1:8080"
    searxng_engines: str = "baidu"     # 逗号分隔，默认只百度
    search_timeout: float = 8.0
    search_user_agent: str = "BagentRAG/0.1"
    web_search_max_results: int = 5
    web_fetch_fulltext: bool = True    # 用 trafilatura 抽正文
    web_content_max_chars: int = 4000
    search_min_interval: float = 1.0   # 上游节流（护百度限流）

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
