"""记忆查询、促升、失效与结果回标。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import MemoryFeedbackReq, MemoryInvalidateReq, MemoryPromoteReq
from app.db.session import get_session
from app.memory.store import (
    invalidate as mem_invalidate,
    list_memories,
    record_feedback,
    set_trust,
)

router = APIRouter()


@router.get("/memory")
def memory_list(scope: str | None = None, user_id: str | None = None,
                limit: int = 50) -> dict:
    session = get_session()
    try:
        rows = list_memories(session, scope=scope, owner_user_id=user_id, limit=limit)
        return {"count": len(rows), "memories": [
            {"id": m.id, "scope": m.scope, "owner": m.owner_user_id, "kind": m.kind,
             "content": m.content, "trust": m.trust, "support": m.support,
             "valid_at": m.valid_at.isoformat() if m.valid_at else None,
             "invalid_at": m.invalid_at.isoformat() if m.invalid_at else None}
            for m in rows]}
    finally:
        session.close()


@router.post("/memory/promote")
def memory_promote(req: MemoryPromoteReq) -> dict:
    if req.trust not in ("draft", "verified", "curated"):
        raise HTTPException(status_code=400, detail="bad trust")
    session = get_session()
    try:
        ok = set_trust(session, req.id, req.trust)
        session.commit()
    finally:
        session.close()
    if not ok:
        raise HTTPException(status_code=403, detail="memory not found")
    return {"id": req.id, "trust": req.trust, "ok": ok}


@router.post("/memory/invalidate")
def memory_invalidate(req: MemoryInvalidateReq) -> dict:
    session = get_session()
    try:
        ok = mem_invalidate(session, req.id)
        session.commit()
    finally:
        session.close()
    if not ok:
        raise HTTPException(status_code=403, detail="memory not found")
    return {"id": req.id, "invalidated": ok}


@router.post("/memory/feedback")
def memory_feedback(req: MemoryFeedbackReq) -> dict:
    """P3-C 结果回标：用一条经验后的成败驱动促升/作废（弥补"多源一致≠真相"）。"""
    session = get_session()
    try:
        action = record_feedback(session, req.id, ok=req.ok)
    finally:
        session.close()
    if action is None:
        raise HTTPException(status_code=403, detail="memory not found")
    return {"id": req.id, "ok": req.ok, "action": action}
