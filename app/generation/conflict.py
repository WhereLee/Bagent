"""知识冲突检测（M8 self-RAG 的一部分）。

给定问题与多条编号上下文，判断它们就该问题是否互相矛盾（同一对象同一属性给出不同值）。
LLM 判定 + 纯解析分离：parse_conflict_verdict 可单测。解析失败保守判"无冲突"并记录。
"""
from __future__ import annotations

import json
import re

from app.generation.prompts import CONFLICT_SYSTEM, CONFLICT_USER

_JSON = re.compile(r"\{.*\}", re.S)


def parse_conflict_verdict(raw: str) -> tuple[bool, str]:
    """把 judge 输出解析为 (是否冲突, 说明)。无法解析 -> (False, 原文)。"""
    if not raw:
        return False, ""
    m = _JSON.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            return bool(obj.get("conflict", False)), str(obj.get("explanation", ""))[:300]
        except json.JSONDecodeError:
            pass
    low = raw.lower()
    if '"conflict": true' in low.replace(" ", "") or low.strip().startswith("true"):
        return True, raw[:200]
    return False, raw[:200]


def detect_conflict(question: str, context: str, llm) -> dict:
    """返回 {conflict: bool, explanation: str}。空/单条上下文不判冲突。"""
    if not context or not context.strip():
        return {"conflict": False, "explanation": ""}
    raw = llm.complete(
        system=CONFLICT_SYSTEM,
        user=CONFLICT_USER.format(question=question, context=context),
    )
    conflict, expl = parse_conflict_verdict(raw)
    return {"conflict": conflict, "explanation": expl}
