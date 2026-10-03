"""Bagent RAG —— FastAPI 服务入口。"""
from __future__ import annotations

from dataclasses import asdict

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import Response

from app.config import get_settings
from app.db.session import get_engine, init_schema
from app.generation.generator import answer_query
from app.generation.llm import get_llm
from app.ingestion.embedder import get_embedder
from app.ingestion.indexer import ingest_file
from app.db.session import get_session
from app.tenant import TenantError, resolve_tenant
from app.memory.store import invalidate as mem_invalidate
from app.memory.store import list_memories, set_trust
from app.memory.writeback import run_writeback
from app.observability.logging import configure_logging, get_logger
from app.observability.metrics import render_metrics
from app.observability.middleware import ObservabilityMiddleware
from app.ratelimit import RateLimiter
from app.retrieval.retriever import retrieve
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.store import delete_document
from app.research import agent as research_agent
from app.research import session as research_session

@asynccontextmanager
async def _lifespan(_app):
    """启动时：结构化日志 + 幂等建表 + 预热连接池（取代已弃用的 on_event）。"""
    configure_logging(get_settings().log_level)
    init_schema()
    get_engine()
    yield


app = FastAPI(title="Bagent RAG", version="0.1.0", lifespan=_lifespan)

_s = get_settings()
if _s.rate_limit_enabled:
    _limiter: RateLimiter | None = RateLimiter(_s.rate_limit_per_sec, _s.rate_limit_burst)
else:
    _limiter = None
app.add_middleware(ObservabilityMiddleware, limiter=_limiter, api_key=(_s.api_key or None),
                   trust_proxy=_s.trust_proxy_headers)


class IngestReq(BaseModel):
    path: str


class DeleteReq(BaseModel):
    source: str


class QueryReq(BaseModel):
    query: str
    top_k: int | None = None
    self_rag: bool | None = None   # 覆盖默认：是否启用 self-RAG/冲突检测
    user_id: str | None = None     # 传入则启用个人记忆/写回


class ChatReq(BaseModel):
    query: str
    history: list[dict] = []  # [{"role":"user"|"assistant","content":str}]
    top_k: int | None = None
    self_rag: bool | None = None
    user_id: str | None = None


class SearchReq(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health() -> dict:
    s = get_settings()
    info = {
        "status": "ok",
        "llm_provider": s.llm_primary_provider,
        "llm_fallback": s.llm_fallback_provider or None,
        "embedding": s.embedding_model_name,
        "retrieval_mode": s.retrieval_mode,
        "auth": bool(s.api_key),
    }
    try:
        info["llm_model"] = get_llm().chain[0].model
    except Exception:  # noqa: BLE001  provider 未配置时不影响健康检查
        info["llm_model"] = None
    return info


@app.get("/metrics")
def metrics() -> Response:
    body, ctype = render_metrics()
    return Response(content=body, media_type=ctype)


@app.post("/ingest")
def ingest(req: IngestReq, request: Request) -> dict:
    tenant = _tenant_of(request)
    # 路径限制在白名单目录下（防用任意 .txt/.md/.pdf 读取服务器文件并回显）
    from pathlib import Path
    from app.config import ROOT_DIR
    root = (ROOT_DIR / get_settings().ingest_allowed_root).resolve()
    try:
        target = Path(req.path).resolve()
        target.relative_to(root)
    except Exception:
        raise HTTPException(status_code=400, detail=f"ingest path 必须位于 {root} 下")
    try:
        result = ingest_file(target, tenant=tenant)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")
    return asdict(result)


@app.post("/search")
def search(req: SearchReq, request: Request) -> dict:
    """检索预览：走与 /query 相同的 retrieve()（hybrid+RRF+rerank+父扩展），按租户过滤。"""
    hits = retrieve(req.query, top_k=req.top_k, tenant=_tenant_of(request))
    return {
        "query": req.query,
        "hits": [asdict(h) for h in hits],
    }


@app.post("/delete")
def delete_doc(req: DeleteReq, request: Request) -> dict:
    tenant = _tenant_of(request)
    session = get_session()
    try:
        ok = delete_document(session, req.source, tenant=tenant)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    if ok:
        get_retrieval_cache().clear()
    return {"source": req.source, "deleted": ok}


@app.post("/query")
def query(req: QueryReq, request: Request, background: BackgroundTasks) -> dict:
    tenant = _tenant_of(request)
    answer = answer_query(req.query, top_k=req.top_k, self_rag=req.self_rag,
                          user_id=req.user_id, tenant=tenant)
    _schedule_writeback(background, req.user_id, req.query, answer.text, tenant)
    return asdict(answer)


@app.post("/chat")
def chat(req: ChatReq, request: Request, background: BackgroundTasks) -> dict:
    """多轮问答：带上历史，服务端做查询改写后检索生成。"""
    tenant = _tenant_of(request)
    answer = answer_query(req.query, top_k=req.top_k, history=req.history,
                          self_rag=req.self_rag, user_id=req.user_id, tenant=tenant)
    _schedule_writeback(background, req.user_id, req.query, answer.text, tenant)
    return asdict(answer)


_log_api = get_logger("api")


def _schedule_writeback(background: BackgroundTasks, user_id, query, answer_text, tenant=None) -> None:
    if not (get_settings().memory_enabled and user_id):
        return
    background.add_task(_do_writeback, user_id, query, answer_text, tenant)


def _do_writeback(user_id: str, query: str, answer_text: str, tenant: str | None = None) -> None:
    """sleep-time：回答之后后台写回，失败不影响已返回的响应。"""
    s = get_settings()
    session = get_session()
    try:
        conv = f"用户：{query}\n助手：{answer_text}"
        run_writeback(session=session, conversation=conv, user_id=user_id,
                      llm=get_llm(), embedder=get_embedder(),
                      dedup_sim=s.memory_dedup_sim, promote_threshold=s.memory_promote_support,
                      tenant=tenant)
    except Exception as e:  # noqa: BLE001
        session.rollback()
        _log_api.warning("writeback_failed", err=str(e)[:160])
    finally:
        session.close()


class MemoryPromoteReq(BaseModel):
    id: int
    trust: str  # draft|verified|curated


class MemoryInvalidateReq(BaseModel):
    id: int


@app.get("/memory")
def memory_list(request: Request, scope: str | None = None, user_id: str | None = None, limit: int = 50) -> dict:
    tenant = _tenant_of(request)
    session = get_session()
    try:
        rows = list_memories(session, scope=scope, owner_user_id=user_id, tenant=tenant, limit=limit)
        return {"count": len(rows), "memories": [
            {"id": m.id, "scope": m.scope, "owner": m.owner_user_id, "kind": m.kind,
             "content": m.content, "trust": m.trust, "support": m.support,
             "valid_at": m.valid_at.isoformat() if m.valid_at else None,
             "invalid_at": m.invalid_at.isoformat() if m.invalid_at else None}
            for m in rows]}
    finally:
        session.close()


@app.post("/memory/promote")
def memory_promote(req: MemoryPromoteReq, request: Request) -> dict:
    if req.trust not in ("draft", "verified", "curated"):
        raise HTTPException(status_code=400, detail="bad trust")
    tenant = _tenant_of(request)
    session = get_session()
    try:
        ok = set_trust(session, req.id, req.trust, tenant=tenant)
        session.commit()
    finally:
        session.close()
    if not ok:
        raise HTTPException(status_code=403, detail="memory not found or not in your tenant")
    return {"id": req.id, "trust": req.trust, "ok": ok}


@app.post("/memory/invalidate")
def memory_invalidate(req: MemoryInvalidateReq, request: Request) -> dict:
    tenant = _tenant_of(request)
    session = get_session()
    try:
        ok = mem_invalidate(session, req.id, tenant=tenant)
        session.commit()
    finally:
        session.close()
    if not ok:
        raise HTTPException(status_code=403, detail="memory not found or not in your tenant")
    return {"id": req.id, "invalidated": ok}


class ResearchReq(BaseModel):
    topic: str
    use_kb: bool = True
    use_web: bool | None = None   # 默认取 config.web_search_enabled
    max_sections: int = 4
    user_id: str | None = None


class RefineReq(BaseModel):
    index: int
    use_kb: bool = True
    use_web: bool | None = None


def _web_flag(use_web) -> bool:
    return get_settings().web_search_enabled if use_web is None else bool(use_web)


def _tenant_of(request: Request) -> str | None:
    """从签名令牌解析当前租户；未启用强制时返回 None（单租户，不过滤）。"""
    try:
        return resolve_tenant(request.headers.get("X-Tenant"))
    except TenantError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e


@app.post("/research")
def research(req: ResearchReq, request: Request) -> dict:
    tenant = _tenant_of(request)
    doc = research_agent.research(req.topic, llm=get_llm(), use_kb=req.use_kb,
                                  use_web=_web_flag(req.use_web), max_sections=req.max_sections,
                                  tenant=tenant)
    sid = research_session.create(doc, tenant=tenant)
    return {"session_id": sid, "doc": doc.to_dict(), "markdown": doc.to_markdown()}


@app.get("/research/{sid}")
def research_get(sid: str, request: Request) -> dict:
    doc = research_session.get(sid, tenant=_tenant_of(request))
    if doc is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": sid, "doc": doc.to_dict(), "markdown": doc.to_markdown()}


@app.post("/research/{sid}/refine")
def research_refine(sid: str, req: RefineReq, request: Request) -> dict:
    doc = research_session.refine_section(sid, req.index, llm=get_llm(),
                                          use_kb=req.use_kb, use_web=_web_flag(req.use_web),
                                          tenant=_tenant_of(request))
    if doc is None:
        raise HTTPException(status_code=404, detail="session/section not found")
    return {"session_id": sid, "doc": doc.to_dict(), "markdown": doc.to_markdown()}
