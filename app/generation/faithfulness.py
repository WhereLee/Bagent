"""Faithfulness（忠实度）与幻觉率评估：LLM-as-judge 的原子断言蕴含判定。

方法（与 Ragas faithfulness 同源，但自实现、无重依赖）：
  答案 -> 按句拆成原子断言 -> 每句判断"能否被检索上下文支撑" -> faithfulness = 支持句/总句。
LLM 调用与纯逻辑分离：split_sentences / parse_verdict 可单测；judge 接受可注入的 llm。
"""
from __future__ import annotations

import re

from app.generation.citation import _CITE
from app.generation.prompts import FAITHFULNESS_SYSTEM, FAITHFULNESS_USER

# 中英文句子边界
_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")

REFUSAL_PHRASE = "根据现有资料无法回答"


def split_sentences(text: str) -> list[str]:
    """按句切分，去掉引用标记与空白，过滤无信息句。"""
    cleaned = _CITE.sub("", text)
    parts = [p.strip() for p in _SENT_SPLIT.split(cleaned)]
    return [p for p in parts if len(p) >= 2]


def parse_verdict(raw: str) -> bool:
    """把 judge 输出解析为 是否被支持。NOT_SUPPORTED 优先判否；无法判定保守判否。"""
    u = (raw or "").strip().upper().replace(" ", "_")
    if "NOT_SUPPORTED" in u:
        return False
    if "SUPPORTED" in u:
        return True
    return False


def judge_sentence(context: str, claim: str, llm) -> bool:
    verdict = llm.complete(system=FAITHFULNESS_SYSTEM, user=FAITHFULNESS_USER.format(context=context, claim=claim))
    return parse_verdict(verdict)


def assess_faithfulness(context: str, answer: str, llm, threshold: float = 0.6) -> dict:
    """返回 faithfulness 明细。空答案/拒答视为不适用（faithfulness=None）。"""
    if not answer or answer.strip() == REFUSAL_PHRASE or answer.startswith(REFUSAL_PHRASE):
        return {"faithfulness": None, "total_claims": 0, "supported": 0, "claims": [], "grounded": False, "is_refusal": True}

    claims = split_sentences(answer)
    verdicts = [judge_sentence(context, c, llm) for c in claims]
    supported = sum(1 for v in verdicts if v)
    total = len(claims)
    faith = (supported / total) if total else 0.0
    return {
        "faithfulness": round(faith, 4),
        "hallucination_rate": round(1 - faith, 4),
        "total_claims": total,
        "supported": supported,
        "claims": [{"claim": c, "supported": v} for c, v in zip(claims, verdicts)],
        "grounded": faith >= threshold,
        "is_refusal": False,
    }
