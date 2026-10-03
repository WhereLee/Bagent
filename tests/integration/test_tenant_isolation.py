"""⑥ 跨租户隔离的集成测试（真 pgvector + bge）。A 检索绝不该召回 B 的文档。"""
from __future__ import annotations

import uuid

import pytest

from app.db.session import get_session, init_schema
from app.ingestion.chunking import chunk_parent_child
from app.ingestion.embedder import get_embedder
from app.retrieval.retriever import retrieve
from app.retrieval.store import delete_document, index_document, vector_search

pytestmark = pytest.mark.integration

DOC_A = "泽西电源管理器的默认管理端口是 6601。"
DOC_B = "火星考勤系统的默认打卡方式是人脸识别。"


@pytest.fixture()
def tenants():
    init_schema()
    ta, tb = "TA-" + uuid.uuid4().hex[:6], "TB-" + uuid.uuid4().hex[:6]
    emb = get_embedder()
    s = get_session()
    srcs = {}
    try:
        for t, text in ((ta, DOC_A), (tb, DOC_B)):
            src = f"{t}|dev.md"
            srcs[t] = src
            parents = chunk_parent_child(text, parent_tokens=200, child_tokens=100, child_overlap=0)
            ct = [c.text for p in parents for c in p.children]
            index_document(s, src, text, "md", parents, emb.encode_documents(ct), tenant_id=t)
        s.commit()
    finally:
        s.close()
    yield ta, tb, srcs
    s = get_session()
    try:
        for src in srcs.values():
            delete_document(s, src)
        s.commit()
    finally:
        s.close()


def test_vector_search_is_tenant_scoped(tenants):
    ta, tb, _ = tenants
    emb = get_embedder()
    q = emb.encode_query("默认管理端口 6601")
    s = get_session()
    try:
        a_only = vector_search(s, q, 5, tenant=ta)
        b_only = vector_search(s, q, 5, tenant=tb)
    finally:
        s.close()
    assert any(c.content == DOC_A for c in a_only)
    assert all(ta in c.source for c in a_only)                  # 只有 A 的文档
    assert all(c.content != DOC_A for c in b_only)              # B 看不到 A 的内容


def test_hybrid_retrieve_scoped(tenants):
    ta, tb, _ = tenants
    hits = retrieve("默认管理端口 6601", mode="hybrid", rerank=False, top_k=3, tenant=ta)
    assert hits and all(ta in h.source for h in hits)
    hits_b = retrieve("默认管理端口 6601", mode="hybrid", rerank=False, top_k=3, tenant=tb)
    assert all(tb in h.source for h in hits_b)                  # 若命中则必属 B，绝不泄漏 A


def test_cross_tenant_delete_denied(tenants):
    ta, tb, srcs = tenants
    s = get_session()
    try:
        # 用 B 的租户去删 A 的文档 → 删不动（fail-closed）
        assert delete_document(s, srcs[ta], tenant=tb) is False
        s.commit()
    finally:
        s.close()
