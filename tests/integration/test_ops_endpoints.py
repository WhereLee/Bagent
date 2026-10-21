"""运维接口集成测试：真 pgvector 上验证 /ready、/documents、/stats。"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.main import app
from app.db.models import Chunk, Document
from app.db.session import get_session, init_schema

pytestmark = pytest.mark.integration


def test_ops_endpoints_against_real_db():
    init_schema()
    run = uuid.uuid4().hex[:8]
    source = f"{run}|ops.md"
    session = get_session()
    doc_id = None
    try:
        doc = Document(source=source, doc_hash=uuid.uuid4().hex,
                       media_type="md", metadata_={})
        session.add(doc)
        session.flush()
        doc_id = doc.id
        session.add_all([
            Chunk(document_id=doc.id, parent_id=None, chunk_index=0,
                  content="父块", metadata_={"role": "parent"}),
            Chunk(document_id=doc.id, parent_id=None, chunk_index=1,
                  content="子块", metadata_={"role": "child"}),
        ])
        session.commit()
    finally:
        session.close()

    try:
        with TestClient(app) as client:   # with：触发 lifespan（init_schema + 连接池预热）
            ready = client.get("/ready")
            assert ready.status_code == 200 and ready.json()["status"] == "ready"

            docs = client.get("/documents", params={"limit": 200})
            assert docs.status_code == 200
            row = next(d for d in docs.json()["documents"] if d["source"] == source)
            assert row["n_chunks"] == 2 and row["n_children"] == 1 and row["n_parents"] == 1

            stats = client.get("/stats")
            assert stats.status_code == 200
            body = stats.json()
            assert body["documents"] >= 1 and body["chunks"] >= 2
            assert isinstance(body["memories_by_trust"], dict)
    finally:
        if doc_id is not None:
            cleanup = get_session()
            try:
                cleanup.execute(delete(Chunk).where(Chunk.document_id == doc_id))
                cleanup.execute(delete(Document).where(Document.id == doc_id))
                cleanup.commit()
            finally:
                cleanup.close()
