"""研究型 agent 的创建、取回、精研与发布。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import tenant_of, web_flag
from app.api.schemas import RefineReq, ResearchReq
from app.db.session import get_session
from app.generation.llm import get_llm
from app.ingestion.embedder import get_embedder
from app.memory.store import create_event
from app.research import agent as research_agent
from app.research import session as research_session
from app.research.publish import publish as research_publish

router = APIRouter()


@router.post("/research")
def research(req: ResearchReq, request: Request) -> dict:
    tenant = tenant_of(request)
    doc = research_agent.research(req.topic, llm=get_llm(), use_kb=req.use_kb,
                                  use_web=web_flag(req.use_web), max_sections=req.max_sections,
                                  tenant=tenant)
    sid = research_session.create(doc, tenant=tenant)
    return {"session_id": sid, "doc": doc.to_dict(), "markdown": doc.to_markdown()}


@router.get("/research/{sid}")
def research_get(sid: str, request: Request) -> dict:
    doc = research_session.get(sid, tenant=tenant_of(request))
    if doc is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": sid, "doc": doc.to_dict(), "markdown": doc.to_markdown()}


@router.post("/research/{sid}/refine")
def research_refine(sid: str, req: RefineReq, request: Request) -> dict:
    doc = research_session.refine_section(sid, req.index, llm=get_llm(),
                                          use_kb=req.use_kb, use_web=web_flag(req.use_web),
                                          tenant=tenant_of(request))
    if doc is None:
        raise HTTPException(status_code=404, detail="session/section not found")
    return {"session_id": sid, "doc": doc.to_dict(), "markdown": doc.to_markdown()}


@router.post("/research/{sid}/publish")
def research_publish_endpoint(sid: str, request: Request, body: dict | None = None) -> dict:
    """把研究文档的论断沉淀为 knowledge 记忆（draft）；可归属到一个事件(不存在则新建)。"""
    tenant = tenant_of(request)
    doc = research_session.get(sid, tenant=tenant)
    if doc is None:
        raise HTTPException(status_code=404, detail="session not found")
    body = body or {}
    session = get_session()
    try:
        event_id = body.get("event_id")
        if not event_id and body.get("event_name"):
            ev = create_event(session, name=body["event_name"], type=body.get("event_type"),
                              created_by=body.get("created_by"))
            event_id = ev.id
        added = research_publish(doc, session=session, embedder=get_embedder(),
                                 tenant=tenant, event_id=event_id)
        session.commit()
    finally:
        session.close()
    return {"session_id": sid, "published": added, "event_id": event_id, "trust": "draft"}
