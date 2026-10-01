"""文本切块。

M1 提供递归字符/段落切分（paragraph -> line -> 按 token 强制切）+ 重叠窗口。
策略以函数形式暴露，M2 会新增语义切分与父子块，接口不变。

设计约束：单个块（不含 overlap）的 token 数不超过 chunk_tokens。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkResult:
    index: int
    text: str
    token_count: int


def _char_weight(ch: str) -> float:
    """单字符 token 权重：CJK 记 1，其它记 0.25（英文约 4 字符/词）。"""
    return 1.0 if "\u4e00" <= ch <= "\u9fff" else 0.25


def _approx_tokens(text: str) -> int:
    """粗略 token 估算（CJK 逐字 + 非 CJK 按 4 字符一词），仅用于稳定切分粒度。"""
    if not text:
        return 0
    total = sum(_char_weight(ch) for ch in text)
    return int(total) + 1


def _split_by_sep(text: str, sep: str) -> list[str]:
    return [p for p in text.split(sep) if p.strip()]


def _force_split(text: str, max_tokens: int) -> list[str]:
    """按 token 预算逐字符强制切，保证每片 <= max_tokens。"""
    out: list[str] = []
    buf: list[str] = []
    tokens = 0.0
    for ch in text:
        w = _char_weight(ch)
        if tokens + w > max_tokens and buf:
            out.append("".join(buf))
            buf = [ch]
            tokens = w
        else:
            buf.append(ch)
            tokens += w
    if buf:
        out.append("".join(buf))
    return out


def _merge(items: list[str], max_tokens: int) -> list[str]:
    """贪心合并相邻片段，尽量填满但不超预算。"""
    merged: list[str] = []
    cur = ""
    for it in items:
        cand = it if not cur else f"{cur}\n{it}"
        if _approx_tokens(cand) <= max_tokens or not cur:
            cur = cand
        else:
            merged.append(cur)
            cur = it
    if cur:
        merged.append(cur)
    return merged


def _apply_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """把前一块的尾部拼到当前块前面，形成重叠（会略微增大块体积）。"""
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-int(overlap_tokens * 2):]  # 近似：按字符粗略取前块尾部
        out.append((tail + "\n" + cur).strip())
    return out


def chunk_text(
    text: str,
    chunk_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[ChunkResult]:
    """递归切分：优先段落，再句子/行，最后按 token 强制切。"""
    text = text.strip()
    if not text:
        return []

    pieces: list[str] = []
    for para in _split_by_sep(text, "\n\n") or [text]:
        if _approx_tokens(para) <= chunk_tokens:
            pieces.append(para)
            continue
        for line in _split_by_sep(para, "\n") or [para]:
            if _approx_tokens(line) <= chunk_tokens:
                pieces.append(line)
            else:
                pieces.extend(_force_split(line, chunk_tokens))

    blocks = _apply_overlap(_merge(pieces, chunk_tokens), overlap_tokens)

    return [
        ChunkResult(index=i, text=b.strip(), token_count=_approx_tokens(b))
        for i, b in enumerate(blocks)
        if b.strip()
    ]
