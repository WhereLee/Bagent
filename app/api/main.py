"""Bagent RAG —— FastAPI 服务入口。"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response

from app.config import get_settings
from app.db.session import get_engine, init_schema
from app.generation.generator import answer_query
from app.generation.llm import get_llm
from app.ingestion.indexer import ingest_file
from app.db.session import get_session
from app.observability.logging import configure_logging
from app.observability.metrics import render_metrics
from app.observability.middleware import ObservabilityMiddleware
from app.ratelimit import RateLimiter
from app.retrieval.retriever import retrieve
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.store import delete_document

app = FastAPI(title="Bagent RAG", version="0.1.0")

_s = get_settings()
if _s.rate_limit_enabled:
    _limiter: RateLimiter | None = RateLimiter(_s.rate_limit_per_sec, _s.rate_limit_burst)
else:
    _limiter = None
app.add_middleware(ObservabilityMiddleware, limiter=_limiter, api_key=(_s.api_key or None))


@app.on_event("startup")
def _startup() -> None:
    configure_logging(get_settings().log_level)
    init_schema()          # 幂等建表
    get_engine()           # 预热连接池


class IngestReq(BaseModel):
    path: str


class DeleteReq(BaseModel):
    source: str


class QueryReq(BaseModel):
    query: str
    top_k: int | None = None


class ChatReq(BaseModel):
    query: str
    history: list[dict] = []  # [{"role":"user"|"assistant","content":str}]
    top_k: int | None = None


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
def ingest(req: IngestReq) -> dict:
    try:
        result = ingest_file(req.path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")
    return asdict(result)


@app.post("/search")
def search(req: SearchReq) -> dict:
    """检索预览：走与 /query 相同的 retrieve()（hybrid+RRF+rerank+父扩展），保证口径一致。"""
    hits = retrieve(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "hits": [asdict(h) for h in hits],
    }


@app.post("/delete")
def delete_doc(req: DeleteReq) -> dict:
    session = get_session()
    try:
        ok = delete_document(session, req.source)
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
def query(req: QueryReq) -> dict:
    answer = answer_query(req.query, top_k=req.top_k)
    return asdict(answer)


@app.post("/chat")
def chat(req: ChatReq) -> dict:
    """多轮问答：带上历史，服务端做查询改写后检索生成。"""
    answer = answer_query(req.query, top_k=req.top_k, history=req.history)
    return asdict(answer)
