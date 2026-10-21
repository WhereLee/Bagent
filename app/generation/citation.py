"""引用标记的解析与校验（纯逻辑，不依赖 LLM）。

约定：答案用 [n] 引用第 n 条资料（1-based）。校验其是否落在实际检索到的资料数内，
剔除无效引用，避免"凭空引用不存在的来源"。
"""
from __future__ import annotations

import re

_CITE = re.compile(r"\[(\d+)\]")
# 流式过滤用：末尾可能是尚未闭合的引用（"[" 或 "[12"）
_PENDING_CITE = re.compile(r"\[\d*$")


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


class CitationStreamFilter:
    """流式引用过滤器：跨 chunk 缓冲未闭合的 [n]，只放行有效引用。

    例：模型分块输出 "答案[" + "9] 和 [1" + "]"，会在完整拿到 [9] 后剔除，
    保留 [1]，避免把越界引用发给客户端。
    """

    def __init__(self, num_sources: int) -> None:
        self.num_sources = num_sources
        self._buf = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        buf = self._buf + chunk
        self._buf = ""
        last_open = buf.rfind("[")
        last_close = buf.rfind("]")
        head = buf
        if last_open > last_close:
            tail = buf[last_open:]
            if _PENDING_CITE.fullmatch(tail):
                head, self._buf = buf[:last_open], tail
        return strip_invalid_citations(head, self.num_sources)

    def flush(self) -> str:
        out = strip_invalid_citations(self._buf, self.num_sources)
        self._buf = ""
        return out
