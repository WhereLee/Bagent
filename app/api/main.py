"""Bagent RAG —— FastAPI 服务入口。"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.db.session import get_engine, init_schema
from app.generation.generator import answer_query
from app.ingestion.indexer import ingest_file
from app.db.session import get_session
from app.ingestion.embedder import get_embedder
from app.retrieval.store import vector_search

app = FastAPI(title="Bagent RAG", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    init_schema()          # 幂等建表
    get_engine()           # 预热连接池


class IngestReq(BaseModel):
    path: str


class QueryReq(BaseModel):
    query: str
    top_k: int | None = None


class SearchReq(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "model": s.mimo_model, "embedding": s.embedding_model_name}


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
    embedder = get_embedder()
    qvec = embedder.encode_query(req.query)
    session = get_session()
    try:
        hits = vector_search(session, qvec, top_k=req.top_k)
    finally:
        session.close()
    return {
        "query": req.query,
        "hits": [asdict(h) for h in hits],
    }


@app.post("/query")
def query(req: QueryReq) -> dict:
    answer = answer_query(req.query, top_k=req.top_k)
    return asdict(answer)
