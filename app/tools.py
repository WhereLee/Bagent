"""共享工具注册表：把 Bagent 的能力包成"工具"，供 MCP server 与 agent 循环**复用同一套实现**。

约定：入参/出参都是 JSON 友好的纯数据；每个工具是薄封装，业务逻辑仍在各自模块。
联网只经 `web_search`（内部走 SearXNG/受控 MiMo 降级）——**主生成 LLM 永不挂 web_search**。

HTTP agent 用 `build_tools(user_id=...)` 把用户上下文绑在闭包里，不暴露成 LLM 可填参数。
"""
from __future__ import annotations

from dataclasses import asdict


def _kb_search_impl(query: str, top_k: int, *, compact: bool) -> list[dict]:
    from app.retrieval.retriever import retrieve

    chunks = retrieve(query, top_k=top_k)
    if compact:
        return [{"source": c.source, "score": round(c.score, 4), "snippet": c.content[:200]}
                for c in chunks]
    return [asdict(c) for c in chunks]


def kb_search(query: str, top_k: int = 5) -> list[dict]:
    """检索知识库，返回与问题相关的来源片段。"""
    return _kb_search_impl(query, top_k, compact=False)


def _kb_answer_impl(query: str, top_k: int | None, *, user_id: str | None) -> dict:
    from app.generation.generator import answer_query

    return asdict(answer_query(query, top_k=top_k, user_id=user_id))


def kb_answer(query: str, top_k: int | None = None, user_id: str | None = None) -> dict:
    """基于知识库直接生成带引用的答案。"""
    return _kb_answer_impl(query, top_k, user_id=user_id)


def _web_search_impl(query: str, max_results: int, *, compact: bool) -> list[dict]:
    from app.search.web_search import web_search as _ws

    chunks = _ws(query, max_results=max_results)
    if compact:
        return [{"url": c.metadata.get("url") or c.source, "snippet": c.content[:200]}
                for c in chunks]
    return [asdict(c) for c in chunks]


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """联网检索（受控 provider），返回网页来源片段。"""
    return _web_search_impl(query, max_results, compact=False)


def _research_impl(topic: str, use_kb: bool, use_web: bool, max_sections: int) -> dict:
    from app.generation.llm import get_llm
    from app.research import agent

    doc = agent.research(topic, llm=get_llm(), use_kb=use_kb, use_web=use_web,
                         max_sections=max_sections)
    return {"markdown": doc.to_markdown(), "doc": doc.to_dict()}


def research(topic: str, use_kb: bool = True, use_web: bool = False, max_sections: int = 4) -> dict:
    """生成一份带引用的研究文档（需要人工确认）。"""
    return _research_impl(topic, use_kb, use_web, max_sections)


def _memory_search_impl(query: str, k: int, *, user_id: str | None) -> list[dict]:
    from app.db.session import get_session
    from app.ingestion.embedder import get_embedder
    from app.memory.store import search_memories

    session = get_session()
    try:
        vec = get_embedder().encode_query(query)
        mems = search_memories(session, vec, owner_user_id=user_id,
                               min_trust="verified", k=k)
        return [{"content": m.content, "scope": m.scope, "trust": m.trust,
                 "source": m.source_ref} for m in mems]
    finally:
        session.close()


def memory_search(query: str, user_id: str | None = None, k: int = 3) -> list[dict]:
    """检索已验证的记忆（个人偏好/共享知识）。"""
    return _memory_search_impl(query, k, user_id=user_id)


# name -> callable；供 MCP/agent 统一登记与调用
TOOLS = {
    "kb_search": kb_search,
    "kb_answer": kb_answer,
    "web_search": web_search,
    "research": research,
    "memory_search": memory_search,
}


def build_tools(*, user_id: str | None = None, compact: bool = False) -> dict:
    """生成绑定请求上下文的工具集，供 HTTP agent 使用。

    user_id 只在闭包里传递，不暴露成工具参数，避免模型自行切换用户。
    compact=True 时 kb_search/web_search 只返回短 snippet（/query agent 用，避免 observation 超长）。
    """
    def _kb_search(query: str, top_k: int = 5) -> list[dict]:
        """检索知识库，返回与问题相关的来源片段。"""
        return _kb_search_impl(query, top_k, compact=compact)

    def _kb_answer(query: str, top_k: int | None = None) -> dict:
        """基于知识库直接生成带引用的答案。"""
        return _kb_answer_impl(query, top_k, user_id=user_id)

    def _web_search(query: str, max_results: int = 5) -> list[dict]:
        """联网检索（受控 provider），返回网页来源片段。"""
        return _web_search_impl(query, max_results, compact=compact)

    def _research(topic: str, use_kb: bool = True, use_web: bool = False,
                  max_sections: int = 4) -> dict:
        """生成一份带引用的研究文档（需要人工确认）。"""
        return _research_impl(topic, use_kb, use_web, max_sections)

    def _memory_search(query: str, k: int = 3) -> list[dict]:
        """检索已验证的记忆（个人偏好/共享知识）。"""
        return _memory_search_impl(query, k, user_id=user_id)

    return {
        "kb_search": _kb_search,
        "kb_answer": _kb_answer,
        "web_search": _web_search,
        "research": _research,
        "memory_search": _memory_search,
    }


def describe_tools(tools: dict | None = None) -> str:
    """给 agent 的工具菜单（名/参数/用途），与传入工具集同源，避免两套描述漂移。"""
    import inspect

    selected = TOOLS if tools is None else tools
    lines = []
    for name, fn in selected.items():
        sig = ", ".join(f"{p.name}:{p.annotation.__name__ if hasattr(p.annotation, '__name__') else 'any'}"
                        f"{'' if p.default is inspect._empty else '=' + repr(p.default)}"
                        for p in inspect.signature(fn).parameters.values())
        desc = (fn.__doc__ or "").strip().splitlines()[0] if (fn.__doc__ or "") else name
        lines.append(f"- {name}({sig}): {desc}")
    return "\n".join(lines)
