"""研究证据采集：多源(KB ⊕ 联网)汇聚 + 有界反思重检，喂给逐节成文。

复用 self-RAG 的充分性判定与改写；联网作为又一路证据源(trust=draft)。
纯编排，可注入 retrieve/web_search 便于测试。
"""
from __future__ import annotations

from app.generation.prompts import (
    REFORMULATE_SYSTEM,
    REFORMULATE_USER,
    SUFFICIENCY_SYSTEM,
    SUFFICIENCY_USER,
)
from app.retrieval.selfrag import parse_sufficiency
from app.retrieval.store import RetrievedChunk


def _dedup_merge(existing: list[RetrievedChunk], new: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen = {c.content.strip() for c in existing}
    out = list(existing)
    for c in new:
        k = c.content.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(f"[{i}] {c.context}" for i, c in enumerate(chunks, 1))


def gather_evidence(
    query: str, *, use_kb: bool = True, use_web: bool = False,
    top_k: int | None = None, max_iters: int = 2, llm,
    kb_retrieve=None, web_retrieve=None, tenant: str | None = None,
) -> tuple[list[RetrievedChunk], dict]:
    """返回 (去重后的证据块, meta)。meta: iterations/queries/sufficient/sources。"""
    if kb_retrieve is None:
        from app.retrieval.retriever import retrieve as kb_retrieve  # 延迟导入避免环
    if web_retrieve is None and use_web:
        from app.search.web_search import web_search as web_retrieve

    chunks: list[RetrievedChunk] = []
    queries = [query]
    q = query
    sufficient = False
    for it in range(max(1, max_iters)):
        cur: list[RetrievedChunk] = []
        if use_kb:
            cur = _dedup_merge(cur, kb_retrieve(q, top_k=top_k, tenant=tenant))
        if use_web and web_retrieve is not None:
            cur = _dedup_merge(cur, web_retrieve(q))
        chunks = _dedup_merge(chunks, cur)
        if not chunks:
            break
        sufficient = parse_sufficiency(
            llm.complete(system=SUFFICIENCY_SYSTEM, user=SUFFICIENCY_USER.format(context=_context(chunks), question=query))
        )
        if sufficient or it == max_iters - 1:
            break
        newq = (llm.complete(system=REFORMULATE_SYSTEM,
                             user=REFORMULATE_USER.format(question=query, context=_context(chunks))) or "").strip()
        if not newq or newq in queries:
            break
        q = newq
        queries.append(newq)

    meta = {"iterations": len(queries), "queries": queries, "sufficient": sufficient,
            "sources": (["kb"] if use_kb else []) + (["web"] if use_web else [])}
    return chunks, meta
