"""多轮查询改写：结合对话历史把追问改写为自包含的检索查询。

指代消解（"它的端口"->"XG-400 Pro 的端口"）在检索前完成，避免向量检索拿代词去匹配。
纯逻辑（历史序列化/结果规整）与 LLM 调用分离。
"""
from __future__ import annotations

from app.generation.prompts import REWRITE_SYSTEM, REWRITE_USER


def format_history(history: list[dict]) -> str:
    """history: [{"role":"user"|"assistant","content":str}, ...] -> 文本。"""
    lines = []
    for turn in history:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}: {turn.get('content','').strip()}")
    return "\n".join(lines)


def rewrite_query(question: str, history: list[dict], llm) -> str:
    """有历史才改写；无历史直接返回原问题（省一次 LLM 调用）。"""
    if not history:
        return question
    try:
        out = llm.complete(
            system=REWRITE_SYSTEM,
            user=REWRITE_USER.format(history=format_history(history), question=question),
        )
    except Exception:  # noqa: BLE001  provider 不可用时不改写，回退原问题
        return question
    rewritten = (out or "").strip()
    # 防御：模型输出为空时回退原问题
    return rewritten if rewritten else question
