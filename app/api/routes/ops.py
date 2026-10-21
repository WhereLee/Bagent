"""运维可见性：文档清单与租户级统计（只读）。"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from sqlalchemy import Text, func, select

from app.api.deps import tenant_of
from app.db.models import Chunk, Document, Memory
from app.db.session import get_session

router = APIRouter()


@router.get("/documents")
def documents(request: Request, limit: int = Query(50, ge=1, le=200),
              offset: int = Query(0, ge=0)) -> dict:
    """列出活跃文档及其父/子块数量（按 tenant 过滤）。"""
    tenant = tenant_of(request)
    session = get_session()
    try:
        child = Chunk.metadata_["role"].astext == "child"
        parent = Chunk.metadata_["role"].astext == "parent"
        q = (
            select(
                Document.id, Document.source, Document.media_type, Document.updated_at,
                func.count(Chunk.id).label("n_chunks"),
                func.count(Chunk.id).filter(child).label("n_children"),
                func.count(Chunk.id).filter(parent).label("n_parents"),
            )
            .join(Chunk, Chunk.document_id == Document.id)
            .where(Document.is_deleted == False)  # noqa: E712
            .group_by(Document.id, Document.source, Document.media_type, Document.updated_at)
            .order_by(Document.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        if tenant is not None:
            q = q.where(func.cast(Document.metadata_["tenant"], Text) == tenant)
        rows = session.execute(q).mappings().all()
        return {"count": len(rows), "documents": [
            {
                "id": r["id"],
                "source": r["source"],
                "media_type": r["media_type"],
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                "n_chunks": int(r["n_chunks"]),
                "n_children": int(r["n_children"]),
                "n_parents": int(r["n_parents"]),
            }
            for r in rows
        ]}
    finally:
        session.close()


@router.get("/stats")
def stats(request: Request) -> dict:
    """租户级文档/块/记忆计数与 trust 分布（events 无 tenant 列，不纳入）。"""
    tenant = tenant_of(request)
    session = get_session()
    try:
        doc_q = select(func.count()).select_from(Document).where(Document.is_deleted == False)  # noqa: E712
        chunk_q = (
            select(func.count()).select_from(Chunk)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.is_deleted == False)  # noqa: E712
        )
        mem_q = select(func.count()).select_from(Memory).where(Memory.is_deleted == False)  # noqa: E712
        trust_q = (
            select(Memory.trust, func.count())
            .where(Memory.is_deleted == False)  # noqa: E712
            .group_by(Memory.trust)
        )
        if tenant is not None:
            doc_filter = func.cast(Document.metadata_["tenant"], Text) == tenant
            doc_q = doc_q.where(doc_filter)
            chunk_q = chunk_q.where(doc_filter)
            mem_q = mem_q.where(Memory.tenant_id == tenant)
            trust_q = trust_q.where(Memory.tenant_id == tenant)

        documents_n = session.scalar(doc_q) or 0
        chunks_n = session.scalar(chunk_q) or 0
        memories_n = session.scalar(mem_q) or 0
        trust_rows = session.execute(trust_q).all()
        return {
            "documents": int(documents_n),
            "chunks": int(chunks_n),
            "memories": int(memories_n),
            "memories_by_trust": {str(r[0]): int(r[1]) for r in trust_rows},
        }
    finally:
        session.close()
