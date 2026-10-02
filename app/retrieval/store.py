"""chunks 表的读写与检索（pgvector + 父子块）。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.ingestion.chunking import ParentChunk


def compute_doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RetrievedChunk:
    chunk_id: int                 # 命中的（子）块 id
    document_id: int
    source: str
    content: str                  # 子块原文（用于 rerank 比对）
    score: float
    metadata: dict
    parent_id: int | None = None
    context: str = ""             # 喂给 LLM 的文本：父扩展后为父块，否则=子块

    def __post_init__(self) -> None:
        if not self.context:
            self.context = self.content


def _existing_doc(session: Session, doc_hash: str) -> Document | None:
    return session.scalar(select(Document).where(Document.doc_hash == doc_hash))


def insert_parent_child(
    session: Session,
    source: str,
    doc_text: str,
    media_type: str,
    parents: list[ParentChunk],
    child_vectors: np.ndarray,
) -> tuple[int, int, bool]:
    """写入父子块。child_vectors 与"按父块顺序展平的所有子块"一一对应。

    父块 embedding=NULL（仅供上下文），子块 embedding 有值（供检索）。
    返回 (document_id, 子块数, 是否新建)。
    """
    doc_hash = compute_doc_hash(doc_text)
    existing = _existing_doc(session, doc_hash)
    if existing:
        return existing.id, 0, False

    doc = Document(source=source, doc_hash=doc_hash, media_type=media_type, metadata_={})
    session.add(doc)
    session.flush()

    vi = 0
    n_children = 0
    for parent in parents:
        parent_row = Chunk(
            document_id=doc.id,
            parent_id=None,
            chunk_index=0,
            content=parent.text,
            token_count=parent.token_count,
            embedding=None,
            metadata_={"role": "parent"},
        )
        session.add(parent_row)
        session.flush()  # 取 parent_row.id

        for child in parent.children:
            session.add(
                Chunk(
                    document_id=doc.id,
                    parent_id=parent_row.id,
                    chunk_index=child.index,
                    content=child.text,
                    token_count=child.token_count,
                    embedding=child_vectors[vi].tolist() if child_vectors.size else None,
                    metadata_={"role": "child"},
                )
            )
            vi += 1
            n_children += 1

    return doc.id, n_children, True


def vector_search(session: Session, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
    """稠密 TopK（只检索有向量的子块）。cosine 距离，score = 1 - distance。"""
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec) + "]"
    sql = text(
        """
        SELECT c.id, c.document_id, d.source, c.content, c.metadata, c.parent_id,
               1 - (c.embedding <=> CAST(:q AS vector)) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.is_deleted = FALSE AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )
    rows = session.execute(sql, {"q": vec_literal, "k": top_k}).mappings().all()
    return [
        RetrievedChunk(
            chunk_id=r["id"], document_id=r["document_id"], source=r["source"],
            content=r["content"], score=float(r["score"]), metadata=r["metadata"] or {},
            parent_id=r["parent_id"],
        )
        for r in rows
    ]


def load_children(session: Session, child_ids: list[int]) -> dict[int, RetrievedChunk]:
    """按 id 批量取子块（保序信息交给调用方）。"""
    if not child_ids:
        return {}
    rows = session.execute(
        select(Chunk, Document.source)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(child_ids))
    ).all()
    out: dict[int, RetrievedChunk] = {}
    for chunk, source in rows:
        out[chunk.id] = RetrievedChunk(
            chunk_id=chunk.id, document_id=chunk.document_id, source=source,
            content=chunk.content, score=0.0, metadata=chunk.metadata_ or {},
            parent_id=chunk.parent_id,
        )
    return out


def expand_to_parents(session: Session, chunks: list[RetrievedChunk]) -> None:
    """把命中子块的 context 扩展为其父块内容（small-to-big）。就地修改。"""
    parent_ids = [c.parent_id for c in chunks if c.parent_id]
    if not parent_ids:
        return
    parents = {
        p.id: p.content
        for p in session.scalars(select(Chunk).where(Chunk.id.in_(parent_ids))).all()
    }
    for c in chunks:
        if c.parent_id and c.parent_id in parents:
            c.context = parents[c.parent_id]
