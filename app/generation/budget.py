"""成本预算与降级决策（纯逻辑，可单测）。

按 token 预算预估一次作答开销（系统提示 + top_k × 每块 + 生成），超预算则降级：
缩 top_k、跳 rerank（省 CPU）、关忠实度（省一次 LLM 调用）。预估用字符→token 粗算，
是保守上界，不追求精确（宁可少花不多花）。
"""
from __future__ import annotations

# 字符到 token 的保守系数（中文约 1 字≈1~1.5 token，取偏大以保守估成本）
CHARS_PER_TOKEN = 1.2
SYSTEM_TOKENS = 500          # 系统提示 + 引用格式说明等固定开销
OUTPUT_TOKENS = 400          # 预估生成输出


def estimate_tokens(top_k: int, chunk_chars: int) -> int:
    return int(SYSTEM_TOKENS + top_k * (chunk_chars / CHARS_PER_TOKEN) + OUTPUT_TOKENS)


def should_degrade(top_k: int, budget: int, avg_chunk_chars: int = 500) -> bool:
    """budget<=0 视为不限；否则预估超预算即降级。"""
    if budget <= 0:
        return False
    return estimate_tokens(top_k, avg_chunk_chars) > budget
