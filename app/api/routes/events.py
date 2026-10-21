"""事件/复盘归档。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import tenant_of
from app.api.schemas import EventReq
from app.db.session import get_session
from app.memory.store import create_event, event_playbook

router = APIRouter()


@router.post("/events")
def events_create(req: EventReq, request: Request) -> dict:
    tenant_of(request)   # 校验租户令牌；事件元数据本身保持全局（无 tenant 列）
    session = get_session()
    try:
        ev = create_event(session, name=req.name, type=req.type, created_by=req.created_by)
        session.commit()
        return {"id": ev.id, "name": ev.name, "type": ev.type}
    finally:
        session.close()


@router.get("/events/{event_id}/playbook")
def events_playbook(event_id: int, request: Request) -> dict:
    tenant = tenant_of(request)
    session = get_session()
    try:
        return event_playbook(session, event_id, tenant=tenant)
    finally:
        session.close()
