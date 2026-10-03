"""memories 表读写与检索（pgvector）。与 chunks 分开：这里是"记忆"，含 scope/trust/双时态。"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.db.models import Memory
from app.memory.lifecycle import TRUST_RANK


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def _vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def add_memory(
    session: Session, *, scope: str, content: str, embedding,
    owner_user_id: str | None = None, kind: str = "fact",
    source_type: str = "research", source_ref: str | None = None,
    trust: str = "draft", confidence: float = 0.5,
    valid_at: datetime | None = None, invalid_at: datetime | None = None,
) -> tuple[Memory, bool]:
    """写入一条记忆；按 content_hash 幂等去重。返回 (行, 是否新)。embedding 为 list/np。"""
    h = content_hash(content)
    exist = session.scalar(select(Memory).where(Memory.content_hash == h, Memory.is_deleted == False))  # noqa: E712
    if exist is not None:
        return exist, False
    emb = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
    row = Memory(
        scope=scope, owner_user_id=owner_user_id, kind=kind, content=content,
        content_hash=h, source_type=source_type, source_ref=source_ref,
        trust=trust, confidence=confidence, support=1, embedding=emb,
        valid_at=valid_at, invalid_at=invalid_at,
    )
    session.add(row)
    session.flush()
    return row, True


def update_content(session: Session, mem_id: int, content: str, embedding, trust: str | None = None) -> None:
    row = session.get(Memory, mem_id)
    if row is None:
        return
    row.content = content
    row.content_hash = content_hash(content)
    row.embedding = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
    if trust is not None:
        row.trust = trust
    row.support = int((row.support or 1))
    row.updated_at = func.now()


def bump_support(session: Session, mem_id: int) -> Memory | None:
    """已有相似记忆再次被佐证：佐证数+1（多源佐证促升的依据）。"""
    row = session.get(Memory, mem_id)
    if row is None:
        return None
    row.support = (row.support or 1) + 1
    row.updated_at = func.now()
    return row


def find_similar(session: Session, vec, *, scope: str, owner_user_id: str | None, k: int = 5) -> list[tuple[int, float, str]]:
    sql = text(
        """
        SELECT id, 1 - (embedding <=> CAST(:q AS vector)) AS sim, content
        FROM memories
        WHERE is_deleted = FALSE AND scope = :scope AND embedding IS NOT NULL
          AND (owner_user_id IS NOT DISTINCT FROM :owner)
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )
    rows = session.execute(sql, {"q": _vec_literal(vec), "scope": scope,
                                 "owner": owner_user_id, "k": k}).mappings().all()
    return [(r["id"], float(r["sim"]), r["content"]) for r in rows]


def search_memories(session: Session, vec, *, scope: str | None = None,
                    owner_user_id: str | None = None, min_trust: str = "verified",
                    k: int = 3, now: datetime | None = None) -> list[Memory]:
    """按向量取记忆，且只返回：未删 + trust>=min_trust + 此刻有效。"""
    now = now or datetime.now(timezone.utc)
    allowed = [t for t, r in TRUST_RANK.items() if r >= TRUST_RANK.get(min_trust, 1)]
    q = select(Memory).where(
        Memory.is_deleted == False,  # noqa: E712
        Memory.embedding.isnot(None),
        Memory.trust.in_(allowed),
        (Memory.valid_at.is_(None)) | (Memory.valid_at <= now),
        (Memory.invalid_at.is_(None)) | (Memory.invalid_at > now),
    )
    if scope is not None:
        q = q.where(Memory.scope == scope)
    if owner_user_id is not None:
        q = q.where(Memory.owner_user_id == owner_user_id)
    q = q.order_by(Memory.embedding.cosine_distance(vec)).limit(k)
    return list(session.scalars(q).all())


def set_trust(session: Session, mem_id: int, trust: str) -> bool:
    row = session.get(Memory, mem_id)
    if row is None:
        return False
    row.trust = trust
    row.updated_at = func.now()
    return True


def invalidate(session: Session, mem_id: int, invalid_at: datetime | None = None) -> bool:
    """标记失效（不删，保留历史）。默认此刻失效。"""
    row = session.get(Memory, mem_id)
    if row is None:
        return False
    row.invalid_at = invalid_at or datetime.now(timezone.utc)
    row.updated_at = func.now()
    return True


def list_memories(session: Session, *, scope: str | None = None,
                  owner_user_id: str | None = None, limit: int = 50) -> list[Memory]:
    q = select(Memory).where(Memory.is_deleted == False)  # noqa: E712
    if scope:
        q = q.where(Memory.scope == scope)
    if owner_user_id:
        q = q.where(Memory.owner_user_id == owner_user_id)
    q = q.order_by(Memory.updated_at.desc()).limit(limit)
    return list(session.scalars(q).all())
