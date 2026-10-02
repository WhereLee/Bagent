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


@dataclass
class ChildChunk:
    index: int          # 在父块内的序号
    text: str
    token_count: int


@dataclass
class ParentChunk:
    text: str
    token_count: int
    children: list[ChildChunk]


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


def _split_units(text: str) -> list[str]:
    """把一段文本拆成自然单元（段落->行）。"""
    for sep in ("\n\n", "\n"):
        parts = _split_by_sep(text, sep)
        if len(parts) > 1:
            return parts
    return [text]


def _pack(children_src: list[str], budget: int, overlap: int) -> list[str]:
    """把片段打包成 <=budget 的块（超长单元先强制切），再施加 overlap。"""
    expanded: list[str] = []
    for u in children_src:
        if _approx_tokens(u) <= budget:
            expanded.append(u)
        else:
            expanded.extend(_force_split(u, budget))
    merged = _merge(expanded, budget)
    return _apply_overlap(merged, overlap)


def chunk_parent_child(
    text: str,
    parent_tokens: int = 600,
    child_tokens: int = 150,
    child_overlap: int = 30,
) -> list[ParentChunk]:
    """父子块(small-to-big)：父块供生成用（上下文大），子块供检索用（匹配准）。

    保证 child_tokens <= parent_tokens；子块在父块范围内切分。
    """
    text = text.strip()
    if not text:
        return []
    if child_tokens > parent_tokens:
        child_tokens = parent_tokens

    parents: list[ParentChunk] = []
    # 1) 先切父块（不重叠，保证可完整回给 LLM）
    parent_src = _pack(_split_units(text), parent_tokens, overlap=0)
    for p_block in parent_src:
        # 2) 父块内切子块（带 overlap）
        child_blocks = _pack(_split_units(p_block), child_tokens, child_overlap)
        if not child_blocks:
            child_blocks = [p_block]
        children = [
            ChildChunk(index=i, text=c.strip(), token_count=_approx_tokens(c))
            for i, c in enumerate(child_blocks)
            if c.strip()
        ]
        parents.append(
            ParentChunk(text=p_block.strip(), token_count=_approx_tokens(p_block), children=children)
        )
    return parents
