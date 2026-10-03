"""Prometheus 指标定义与记录辅助。

集中定义所有指标，业务模块通过这里的 helper 打点。
多进程部署（gunicorn 多 worker）需设置 PROMETHEUS_MULTIPROC_DIR 并使用
multiprocess 模式；当前单 worker 直接用默认 REGISTRY。
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# --- HTTP ---
HTTP_REQUESTS = Counter(
    "bagent_http_requests_total",
    "HTTP 请求总数",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "bagent_http_request_seconds",
    "HTTP 请求端到端时延(秒)",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# --- 阶段时延 ---
STAGE_LATENCY = Histogram(
    "bagent_stage_seconds",
    "各阶段时延(秒)：retrieval/llm_generate/rerank/faithfulness/rewrite",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# --- RAG 质量信号 ---
RAG_QUALITY = Counter(
    "bagent_rag_quality_total",
    "RAG 质量计数：refusal/low_confidence/grounded",
    ["outcome"],
)

# --- 在途请求 ---
INPROGRESS = Gauge(
    "bagent_http_inprogress",
    "正在处理的请求数",
)

# --- LLM 用量 ---
LLM_TOKENS = Counter(
    "bagent_llm_tokens_total",
    "LLM 调用消耗 token",
    ["type"],  # input / output
)

# --- LLM 韧性 ---
LLM_REQUESTS = Counter("bagent_llm_requests_total", "LLM 请求数", ["provider", "outcome"])
LLM_RETRIES = Counter("bagent_llm_retries_total", "LLM 重试次数", ["provider"])
LLM_FALLBACKS = Counter("bagent_llm_fallbacks_total", "LLM 降级到备 provider 次数")

# --- 限流 ---
RATE_LIMIT_REJECTED = Counter(
    "bagent_rate_limit_rejected_total",
    "被限流拒绝的请求数",
)

# --- 检索缓存 ---
RETRIEVAL_CACHE = Counter(
    "bagent_retrieval_cache_total",
    "检索缓存命中情况",
    ["result"],  # hit / miss
)

# --- 联网搜索 ---
WEB_SEARCH = Counter(
    "bagent_web_search_total",
    "联网搜索调用",
    ["provider", "outcome"],  # outcome: ok / empty / error
)

# --- 记忆写回 (M9b) ---
MEMORY_OPS = Counter(
    "bagent_memory_ops_total",
    "记忆操作",
    ["scope", "op"],  # op: add/update/noop/promote/invalidate
)


def render_metrics() -> tuple[bytes, str]:
    """返回 (/metrics 响应体, content-type)。"""
    return generate_latest(), CONTENT_TYPE_LATEST
