"""P2-2 集成：真实 Postgres 上 research publish 写入 knowledge memory。"""
from __future__ import annotations

import pytest
from sqlalchemy import delete, select

from app.db.models import Memory
from app.db.session import get_session, init_schema
from app.research.doc import Claim, Reference, ResearchDoc
from app.research import session as research_session
from app import tools

pytestmark = pytest.mark.integration


class _Embedder:
    def encode_documents(self, docs):
        return [[0.0] * 512 for _ in docs]


def _cleanup(prefix: str) -> None:
    session = get_session()
    try:
        session.execute(delete(Memory).where(Memory.content.like(f"{prefix}%")))
        session.commit()
    finally:
        session.close()


def test_publish_research_writes_draft_memory(monkeypatch):
    init_schema()
    prefix = "P2-publish-test"
    _cleanup(prefix)
    doc = ResearchDoc(
        topic=prefix,
        references=[Reference(id=1, source="p2-source", trust="curated")],
        claims=[Claim(text="建议先验证再发布", reference_ids=[1])],
    )
    sid = research_session.create(doc)
    monkeypatch.setattr("app.ingestion.embedder.get_embedder", lambda: _Embedder())
    try:
        out = tools.publish_research(sid)
        assert out["published"] == 1 and out["trust"] == "draft"
        session = get_session()
        try:
            row = session.scalar(select(Memory).where(Memory.content.like(f"{prefix}%")))
            assert row is not None and row.scope == "knowledge" and row.trust == "draft"
        finally:
            session.close()
    finally:
        _cleanup(prefix)
