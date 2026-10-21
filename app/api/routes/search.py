"""检索预览。"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from app.api.schemas import SearchReq
from app.retrieval.retriever import retrieve

router = APIRouter()


@router.post("/search")
def search(req: SearchReq) -> dict:
    """检索预览：走与 /query 相同的 retrieve()（hybrid+RRF+rerank+父扩展）。"""
    hits = retrieve(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "hits": [asdict(h) for h in hits],
    }
