"""M9b 记忆抽取：从对话里提炼值得长期记住的事实（LLM 调用 + 纯解析分离）。

parse_facts 可单测：稳健解析 JSON 数组、校验 scope、丢弃非法项。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.generation.prompts import MEMORY_EXTRACT_SYSTEM, MEMORY_EXTRACT_USER

_ARR = re.compile(r"\[.*\]", re.S)
_VALID_SCOPES = {"personal", "knowledge"}


@dataclass
class Fact:
    content: str
    scope: str        # personal | knowledge
    kind: str = "fact"


def parse_facts(raw: str) -> list[Fact]:
    if not raw:
        return []
    m = _ARR.search(raw)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[Fact] = []
    if not isinstance(data, list):
        return out
    for it in data:
        if not isinstance(it, dict):
            continue
        content = str(it.get("content", "")).strip()
        scope = str(it.get("scope", "knowledge")).strip().lower()
        if not content or scope not in _VALID_SCOPES:
            continue
        out.append(Fact(content=content, scope=scope, kind=str(it.get("kind", "fact"))))
    return out


def extract_facts(conversation: str, llm) -> list[Fact]:
    """从一轮对话文本抽取事实。conversation 由调用方拼好（问题+答案等）。"""
    if not conversation.strip():
        return []
    raw = llm.complete(system=MEMORY_EXTRACT_SYSTEM,
                       user=MEMORY_EXTRACT_USER.format(conversation=conversation[:4000]))
    return parse_facts(raw)
