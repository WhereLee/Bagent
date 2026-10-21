"""共享工具注册表：把 Bagent 的能力包成"工具"，供 MCP server 与 agent 循环**复用同一套实现**。

约定：入参/出参都是 JSON 友好的纯数据；每个工具是薄封装，业务逻辑仍在各自模块。
联网只经 `web_search`（内部走 SearXNG/受控 MiMo 降级）——**主生成 LLM 永不挂 web_search**。

HTTP agent 用 `build_tools(user_id=...)` 把用户上下文绑在闭包里，不暴露成 LLM 可填参数。
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError


class KbSearchArgs(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)


class KbAnswerArgs(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1)


class ResearchArgs(BaseModel):
    topic: str = Field(min_length=1)
    use_kb: bool = True
    use_web: bool = False
    max_sections: int = Field(default=4, ge=1)


class MemorySearchArgs(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=3, ge=1)


class PublishResearchArgs(BaseModel):
    session_id: str = Field(min_length=1)
    event_id: int | None = Field(default=None, ge=1)
    event_name: str | None = None
    event_type: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    """工具的 LLM 契约：描述、参数模型、是否副作用、是否需要人工审批。"""
    name: str
    func: Callable[..., Any]
    description: str
    args_model: type[BaseModel]
    side_effect: bool = False
    requires_approval: bool = False
    required_scope: str | None = None
    idempotency_key_field: str | None = None
    # P1 执行层：未显式设置时按工具类型解析；不写死业务阈值。
    timeout_s: float | None = None
    max_calls: int | None = None
    max_retries: int = 0
    retry_on: tuple[str, ...] = ()
    idempotent: bool = True


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
    from app.research import agent, session as research_session

    doc = agent.research(topic, llm=get_llm(), use_kb=use_kb, use_web=use_web,
                         max_sections=max_sections)
    sid = research_session.create(doc)
    return {"session_id": sid, "markdown": doc.to_markdown(), "doc": doc.to_dict()}


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


def _publish_research_impl(session_id: str, event_id: int | None = None,
                           event_name: str | None = None,
                           event_type: str | None = None) -> dict:
    from app.db.session import get_session
    from app.ingestion.embedder import get_embedder
    from app.memory.store import create_event
    from app.research.publish import publish
    from app.research.session import get as get_research

    doc = get_research(session_id)
    if doc is None:
        raise ValueError("research session not found")
    session = get_session()
    try:
        eid = event_id
        if not eid and event_name:
            ev = create_event(session, name=event_name, type=event_type)
            eid = ev.id
        added = publish(doc, session=session, embedder=get_embedder(), event_id=eid)
        session.commit()
        return {"session_id": session_id, "published": added, "event_id": eid,
                "trust": "draft"}
    finally:
        session.close()


def publish_research(session_id: str, event_id: int | None = None,
                     event_name: str | None = None,
                     event_type: str | None = None) -> dict:
    """把一份 research session 的论断发布为 knowledge 记忆（draft）。"""
    return _publish_research_impl(session_id, event_id, event_name, event_type)


# name -> callable；供 MCP/agent 统一登记与调用
TOOLS = {
    "kb_search": kb_search,
    "kb_answer": kb_answer,
    "web_search": web_search,
    "research": research,
    "memory_search": memory_search,
    "publish_research": publish_research,
}


# 静态工具契约；HTTP agent 的 user_id 仍由 build_tools 闭包绑定，不暴露给 LLM。
TOOL_SPECS: dict[str, ToolSpec] = {
    "kb_search": ToolSpec(
        name="kb_search", func=kb_search,
        description="检索知识库，返回与问题相关的来源片段。",
        args_model=KbSearchArgs, required_scope="agent:read",
        max_retries=1, retry_on=("timeout", "unavailable"),
    ),
    "kb_answer": ToolSpec(
        name="kb_answer", func=kb_answer,
        description="基于知识库直接生成带引用的答案。",
        args_model=KbAnswerArgs, required_scope="agent:read",
    ),
    "web_search": ToolSpec(
        name="web_search", func=web_search,
        description="联网检索（受控 provider），返回网页来源片段。",
        args_model=WebSearchArgs, required_scope="agent:read",
        max_retries=1, retry_on=("timeout", "unavailable"),
    ),
    "research": ToolSpec(
        name="research", func=research,
        description="生成一份带引用的研究文档（需要人工确认）。",
        args_model=ResearchArgs, required_scope="agent:research",
        requires_approval=True,
    ),
    "memory_search": ToolSpec(
        name="memory_search", func=memory_search,
        description="检索已验证的记忆（个人偏好/共享知识）。",
        args_model=MemorySearchArgs, required_scope="agent:memory",
    ),
    "publish_research": ToolSpec(
        name="publish_research", func=publish_research,
        description="把研究文档发布为 knowledge 记忆（draft，需人工确认）。",
        args_model=PublishResearchArgs, required_scope="agent:research",
        side_effect=True, requires_approval=True, idempotent=True,
    ),
}

DEFAULT_REQUIRES_APPROVAL = {name for name, spec in TOOL_SPECS.items() if spec.requires_approval}


def validate_tool_args(name: str, args: dict, specs: dict[str, ToolSpec] | None = None
                       ) -> tuple[dict | None, list[dict] | None]:
    """按 ToolSpec 校验参数；返回 (clean_args, errors)，不执行工具。"""
    spec = (specs if specs is not None else TOOL_SPECS).get(name)
    if spec is None:
        return args, None
    try:
        model = spec.args_model.model_validate(args or {})
        return model.model_dump(exclude_none=True), None
    except ValidationError as e:
        return None, e.errors(include_url=False)


def build_tools(*, user_id: str | None = None, compact: bool = False,
                scopes: set[str] | frozenset[str] | None = None) -> dict:
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

    def _publish_research(session_id: str, event_id: int | None = None,
                          event_name: str | None = None,
                          event_type: str | None = None) -> dict:
        """把研究文档发布为 knowledge 记忆（draft）。"""
        return _publish_research_impl(session_id, event_id, event_name, event_type)

    result = {
        "kb_search": _kb_search,
        "kb_answer": _kb_answer,
        "web_search": _web_search,
        "research": _research,
        "memory_search": _memory_search,
        "publish_research": _publish_research,
    }
    if scopes is None:
        return result
    scope_set = set(scopes)
    return {
        name: fn for name, fn in result.items()
        if _scope_allowed(TOOL_SPECS.get(name), scope_set)
    }


def _scope_allowed(spec: ToolSpec | None, scopes: set[str]) -> bool:
    if spec is None or spec.required_scope is None:
        return True
    return "*" in scopes or spec.required_scope in scopes


def describe_tools(tools: dict | None = None, specs: dict[str, ToolSpec] | None = None) -> str:
    """给 agent 的工具菜单（名/参数/用途），优先使用 ToolSpec 避免描述漂移。"""
    selected = TOOLS if tools is None else tools
    selected_specs = specs if specs is not None else TOOL_SPECS
    lines = []
    for name, fn in selected.items():
        spec = selected_specs.get(name)
        if spec is not None:
            params = ", ".join(
                f"{fname}:{_type_name(field.annotation)}"
                f"{'' if field.is_required() else '=' + repr(field.default)}"
                for fname, field in spec.args_model.model_fields.items()
            )
            desc = spec.description
        else:
            params = ", ".join(
                f"{p.name}:{_type_name(p.annotation)}"
                f"{'' if p.default is inspect._empty else '=' + repr(p.default)}"
                for p in inspect.signature(fn).parameters.values()
            )
            desc = (fn.__doc__ or "").strip().splitlines()[0] if (fn.__doc__ or "") else name
        lines.append(f"- {name}({params}): {desc}")
    return "\n".join(lines)


def _type_name(annotation: Any) -> str:
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")
