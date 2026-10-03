"""Self-RAG 反思重检（M8）。

retrieve -> 充分性自检(LLM) -> 不足则改写 query 重检 -> 至多 max_iters 轮。
纯解析 parse_sufficiency 可单测；LLM 与检索通过参数注入，便于测试。
"""
from __future__ import annotations

from app.generation.prompts import (
    REFORMULATE_SYSTEM,
    REFORMULATE_USER,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
)
from app.observability.logging import get_logger, log_event
from app.retrieval.retriever import retrieve

_log = get_logger("selfrag")


def parse_sufficiency(raw: str) -> bool:
    """INSUFFICIENT 优先判否；含 SUFFICIENT 判是；否则保守判不足。"""
    u = (raw or "").strip().upper()
    if "INSUFFICIENT" in u:
        return False
    if "SUFFICIENT" in u:
        return True
    return False


def _context_of(chunks) -> str:
    return "\n\n---\n\n".join(f"[{i}] {c.context}" for i, c in enumerate(chunks, start=1))


def retrieve_with_reflection(
    query: str,
    *,
    top_k: int | None = None,
    max_iters: int = 2,
    llm,
    mode: str | None = None,
    rerank: bool | None = None,
    tenant: str | None = None,
) -> tuple[list, dict]:
    """返回 (合并去重后的候选块, meta)。meta: iterations/queries/sufficient。"""
    queries = [query]
    chunks = retrieve(query, top_k=top_k, mode=mode, rerank=rerank, tenant=tenant)
    sufficient = False

    for _ in range(max(0, max_iters - 1)):
        ctx = _context_of(chunks)
        v = llm.complete(system=SUFFICIENCY_SYSTEM, user=SUFFICIENCY_USER.format(context=ctx, question=query))
        sufficient = parse_sufficiency(v)
        if sufficient:
            break
        newq = (llm.complete(
            system=REFORMULATE_SYSTEM,
            user=REFORMULATE_USER.format(question=query, context=ctx),
        ) or "").strip()
        if not newq or newq in queries:
            break  # 没改进就停，避免空转
        queries.append(newq)
        extra = retrieve(newq, top_k=top_k, mode=mode, rerank=rerank, tenant=tenant)
        # 合并去重（按 chunk_id），保留已排好的顺序
        seen = {c.chunk_id for c in chunks}
        for c in extra:
            if c.chunk_id not in seen:
                chunks.append(c)
                seen.add(c.chunk_id)

    meta = {"iterations": len(queries), "queries": queries, "sufficient": sufficient}
    log_event(_log, "info", "selfrag_reflection", iterations=meta["iterations"], sufficient=sufficient)
    return chunks, meta
