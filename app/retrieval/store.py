"""chunks 表的读写与向量检索（pgvector）。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document


def compute_doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    source: str
    content: str
    score: float  # cosine 相似度（越大越相关）
    metadata: dict


def upsert_document(
    session: Session,
    source: str,
    doc_text: str,
    media_type: str,
    chunks: list[tuple[int, str, int, np.ndarray]],  # (index, text, token_count, embedding)
) -> tuple[int, int, bool]:
    """写入文档及其块。若同 hash 已存在则跳过（M1 的增量语义）。

    返回 (document_id, 写入块数, 是否新建)。
    """
    doc_hash = compute_doc_hash(doc_text)
    existing = session.scalar(select(Document).where(Document.doc_hash == doc_hash))
    if existing:
        return existing.id, 0, False

    doc = Document(
        source=source,
        doc_hash=doc_hash,
        media_type=media_type,
        metadata_={},
    )
    session.add(doc)
    session.flush()  # 取得 doc.id

    for idx, content, token_count, emb in chunks:
        session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=idx,
                content=content,
                token_count=token_count,
                embedding=emb.tolist(),
                metadata_={},
            )
        )
    return doc.id, len(chunks), True


def vector_search(session: Session, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
    """基于 cosine 距离的 TopK 检索。

    用 `<=>`（cosine distance）算子，score = 1 - distance。
    """
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec) + "]"
    sql = text(
        """
        SELECT c.id, c.document_id, d.source, c.content, c.metadata,
               1 - (c.embedding <=> CAST(:q AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.is_deleted = FALSE
        ORDER BY c.embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )
    rows = session.execute(sql, {"q": vec_literal, "k": top_k}).mappings().all()
    return [
        RetrievedChunk(
            chunk_id=r["id"],
            document_id=r["document_id"],
            source=r["source"],
            content=r["content"],
            score=float(r["score"]),
            metadata=r["metadata"] or {},
        )
        for r in rows
    ]
