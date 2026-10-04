"""memories 表读写与检索（pgvector）。与 chunks 分开：这里是"记忆"，含 scope/trust/双时态。"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.db.models import Event, Memory
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
    tenant_id: str | None = None, level: str = "fact", tags: dict | None = None,
    event_id: int | None = None,
) -> tuple[Memory, bool]:
    """写入一条记忆；按 (tenant, scope, owner, content_hash) 分域幂等去重。"""
    h = content_hash(content)
    q = select(Memory).where(Memory.content_hash == h, Memory.scope == scope,
                             Memory.is_deleted == False)  # noqa: E712
    q = q.where(Memory.owner_user_id == owner_user_id if owner_user_id is not None
                else Memory.owner_user_id.is_(None))
    q = q.where(Memory.tenant_id == tenant_id if tenant_id is not None
                else Memory.tenant_id.is_(None))
    exist = session.scalar(q)
    if exist is not None:
        return exist, False
    emb = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
    row = Memory(
        scope=scope, owner_user_id=owner_user_id, kind=kind, content=content,
        content_hash=h, source_type=source_type, source_ref=source_ref,
        trust=trust, confidence=confidence, support=1, embedding=emb,
        valid_at=valid_at, invalid_at=invalid_at, tenant_id=tenant_id,
        level=level, tags=tags or {}, event_id=event_id,
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


def find_similar(session: Session, vec, *, scope: str, owner_user_id: str | None, k: int = 5, tenant: str | None = None) -> list[tuple[int, float, str]]:
    sql = text(
        """
        SELECT id, 1 - (embedding <=> CAST(:q AS vector)) AS sim, content
        FROM memories
        WHERE is_deleted = FALSE AND scope = :scope AND embedding IS NOT NULL
          AND (owner_user_id IS NOT DISTINCT FROM :owner)
          AND (CAST(:tenant AS text) IS NULL OR tenant_id = :tenant)
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )
    rows = session.execute(sql, {"q": _vec_literal(vec), "scope": scope,
                                 "owner": owner_user_id, "tenant": tenant, "k": k}).mappings().all()
    return [(r["id"], float(r["sim"]), r["content"]) for r in rows]


def search_memories(session: Session, vec, *, scope: str | None = None,
                    owner_user_id: str | None = None, tenant: str | None = None,
                    level: str | None = None, any_tags: dict | None = None,
                    min_trust: str = "verified",
                    k: int = 3, now: datetime | None = None) -> list[Memory]:
    """按向量取记忆：未删 + trust>=min_trust + 此刻有效 + 本租户 (+可选 level/标签过滤)。"""
    now = now or datetime.now(timezone.utc)
    allowed = [t for t, r in TRUST_RANK.items() if r >= TRUST_RANK.get(min_trust, 1)]
    q = select(Memory).where(
        Memory.is_deleted == False,  # noqa: E712
        Memory.embedding.isnot(None),
        Memory.trust.in_(allowed),
        (Memory.valid_at.is_(None)) | (Memory.valid_at <= now),
        (Memory.invalid_at.is_(None)) | (Memory.invalid_at > now),
    )
    if tenant is not None:
        q = q.where(Memory.tenant_id == tenant)
    if level is not None:
        q = q.where(Memory.level == level)
    if any_tags:
        q = q.where(Memory.tags.contains(any_tags))   # JSONB @> 包含
    if scope is not None:
        q = q.where(Memory.scope == scope)
    if owner_user_id is not None:
        q = q.where(Memory.owner_user_id == owner_user_id)
    q = q.order_by(Memory.embedding.cosine_distance(vec)).limit(k)
    return list(session.scalars(q).all())


def set_trust(session: Session, mem_id: int, trust: str, tenant: str | None = None) -> bool:
    row = session.get(Memory, mem_id)
    if row is None:
        return False
    if tenant is not None and row.tenant_id != tenant:
        return False                              # 跨租户不得改他人记忆(fail-closed)
    row.trust = trust
    row.updated_at = func.now()
    return True


def invalidate(session: Session, mem_id: int, invalid_at: datetime | None = None,
               tenant: str | None = None) -> bool:
    """标记失效（不删，保留历史）。默认此刻失效；tenant 非空则只能失效本租户的。"""
    row = session.get(Memory, mem_id)
    if row is None:
        return False
    if tenant is not None and row.tenant_id != tenant:
        return False
    row.invalid_at = invalid_at or datetime.now(timezone.utc)
    row.updated_at = func.now()
    return True


def list_memories(session: Session, *, scope: str | None = None,
                  owner_user_id: str | None = None, tenant: str | None = None,
                  event_id: int | None = None, limit: int = 50) -> list[Memory]:
    q = select(Memory).where(Memory.is_deleted == False)  # noqa: E712
    if tenant is not None:
        q = q.where(Memory.tenant_id == tenant)
    if event_id is not None:
        q = q.where(Memory.event_id == event_id)
    if scope:
        q = q.where(Memory.scope == scope)
    if owner_user_id:
        q = q.where(Memory.owner_user_id == owner_user_id)
    q = q.order_by(Memory.updated_at.desc()).limit(limit)
    return list(session.scalars(q).all())


def create_event(session: Session, *, name: str, type: str | None = None,
                 tags: dict | None = None, summary: str | None = None,
                 created_by: str | None = None) -> Event:
    ev = Event(name=name, type=type, tags=tags or {}, summary=summary, created_by=created_by)
    session.add(ev)
    session.flush()
    return ev


def get_event(session: Session, event_id: int) -> Event | None:
    ev = session.get(Event, event_id)
    return None if (ev is None or ev.is_deleted) else ev


def event_playbook(session: Session, event_id: int) -> dict:
    """整份事件经验档案：{event, playbook:[...], facts:[...]}。"""
    ev = get_event(session, event_id)
    mems = list_memories(session, event_id=event_id)
    return {
        "event": None if ev is None else {"id": ev.id, "name": ev.name, "type": ev.type, "summary": ev.summary},
        "playbook": [m.content for m in mems if m.level == "playbook"],
        "facts": [{"content": m.content, "source": m.source_ref, "trust": m.trust} for m in mems if m.level != "playbook"],
    }
