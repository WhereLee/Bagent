"""引用标记的解析与校验（纯逻辑，不依赖 LLM）。

约定：答案用 [n] 引用第 n 条资料（1-based）。校验其是否落在实际检索到的资料数内，
剔除无效引用，避免"凭空引用不存在的来源"。
"""
from __future__ import annotations

import re

_CITE = re.compile(r"\[(\d+)\]")


def extract_citations(text: str) -> list[int]:
    """按出现顺序返回所有引用编号（可能含越界的）。"""
    return [int(m) for m in _CITE.findall(text)]


def split_valid_invalid(text: str, num_sources: int) -> tuple[list[int], list[int]]:
    """返回 (有效引用, 无效引用)。有效 = 1..num_sources。"""
    valid: list[int] = []
    invalid: list[int] = []
    for n in extract_citations(text):
        if 1 <= n <= num_sources:
            valid.append(n)
        else:
            invalid.append(n)
    return valid, invalid


def strip_invalid_citations(text: str, num_sources: int) -> str:
    """移除越界的 [n]，保留有效引用。"""
    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= num_sources else ""

    return _CITE.sub(_repl, text)


def has_citation(text: str, num_sources: int) -> bool:
    valid, _ = split_valid_invalid(text, num_sources)
    return len(valid) > 0
