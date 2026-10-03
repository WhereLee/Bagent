"""共享工具注册表：把 Bagent 的能力包成"工具"，供 MCP server 与 agent 循环**复用同一套实现**。

约定：入参/出参都是 JSON 友好的纯数据；每个工具是薄封装，业务逻辑仍在各自模块。
联网只经 `web_search`（内部走 SearXNG/受控 MiMo 降级）——**主生成 LLM 永不挂 web_search**。
"""
from __future__ import annotations

from dataclasses import asdict


def kb_search(query: str, top_k: int = 5) -> list[dict]:
    from app.retrieval.retriever import retrieve
    return [asdict(c) for c in retrieve(query, top_k=top_k)]


def kb_answer(query: str, top_k: int | None = None, user_id: str | None = None) -> dict:
    from app.generation.generator import answer_query
    return asdict(answer_query(query, top_k=top_k, user_id=user_id))


def web_search(query: str, max_results: int = 5) -> list[dict]:
    from app.search.web_search import web_search as _ws
    return [asdict(c) for c in _ws(query, max_results=max_results)]


def research(topic: str, use_kb: bool = True, use_web: bool = False, max_sections: int = 4) -> dict:
    from app.generation.llm import get_llm
    from app.research import agent
    doc = agent.research(topic, llm=get_llm(), use_kb=use_kb, use_web=use_web, max_sections=max_sections)
    return {"markdown": doc.to_markdown(), "doc": doc.to_dict()}


def memory_search(query: str, user_id: str | None = None, k: int = 3) -> list[dict]:
    from app.db.session import get_session
    from app.ingestion.embedder import get_embedder
    from app.memory.store import search_memories
    session = get_session()
    try:
        vec = get_embedder().encode_query(query)
        mems = search_memories(session, vec, owner_user_id=user_id, min_trust="verified", k=k)
        return [{"content": m.content, "scope": m.scope, "trust": m.trust, "source": m.source_ref} for m in mems]
    finally:
        session.close()


# name -> callable；供 MCP/agent 统一登记与调用
TOOLS = {
    "kb_search": kb_search,
    "kb_answer": kb_answer,
    "web_search": web_search,
    "research": research,
    "memory_search": memory_search,
}
