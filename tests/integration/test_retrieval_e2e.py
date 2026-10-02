"""端到端集成测试：真实 PostgreSQL/pgvector + bge embedding。

覆盖纯逻辑单测够不到的关键路径：
- 父子块入库后，dense 与 hybrid(BM25+RRF) 都能召回正确来源；
- 增量删除（#3 修过的 bug 类）后，dense 与 lexical 两路都不再召回。
CI 的 integration job 用 pgvector service container 跑；本地也可对任意配好 PG_* 的库跑。
"""
from __future__ import annotations

import uuid

import pytest

from app.config import get_settings
from app.db.session import get_session, init_schema
from app.ingestion.chunking import chunk_parent_child
from app.ingestion.embedder import get_embedder
from app.retrieval.retriever import retrieve
from app.retrieval.store import delete_document, index_document, vector_search

pytestmark = pytest.mark.integration

# 三个互不混淆的"产品事实"，含稀有 token 以便检索判定
DOCS = {
    "alpha_zebra_router.md": "Zebra 路由器型号 A-100 的默认管理端口是 8071，整机质保两年。",
    "beta_koala_switch.md": "Koala 交换机型号 B-200 的默认管理端口是 9082，支持 fortyGig 上行。",
    "gamma_yak_gateway.md": "Yak 网关型号 C-300 的默认管理端口是 7093，采用 solar 供电。",
}
QUERIES = {
    "alpha_zebra_router.md": "Zebra 路由器 A-100 的默认管理端口",
    "beta_koala_switch.md": "Koala 交换机 B-200 默认管理端口",
    "gamma_yak_gateway.md": "Yak 网关 C-300 的默认管理端口",
}


@pytest.fixture(scope="module")
def corpus():
    """把测试语料写入独立 source（带 run 前缀），结束清理。"""
    init_schema()
    s = get_settings()
    embedder = get_embedder()
    run = uuid.uuid4().hex[:8]
    tagged = {f"{run}|{name}": text for name, text in DOCS.items()}

    session = get_session()
    try:
        for source, text in tagged.items():
            parents = chunk_parent_child(text, parent_tokens=200, child_tokens=100, child_overlap=0)
            child_texts = [c.text for p in parents for c in p.children]
            vecs = embedder.encode_documents(child_texts)
            index_document(session, source, text, "md", parents, vecs)
        session.commit()
    finally:
        session.close()
    yield run, tagged
    # 清理
    from sqlalchemy import delete as sa_delete
    from app.db.models import Chunk, Document
    session = get_session()
    try:
        ids = [d.id for d in session.query(Document).filter(Document.source.like(f"{run}|%"))]
        if ids:
            session.execute(sa_delete(Chunk).where(Chunk.document_id.in_(ids)))
            session.execute(sa_delete(Document).where(Document.id.in_(ids)))
        session.commit()
    finally:
        session.close()


def _sources(chunks):
    return {c.source.split("|", 1)[1] for c in chunks}


def test_dense_retrieves_correct_doc(corpus):
    run, _ = corpus
    embedder = get_embedder()
    session = get_session()
    try:
        for name, q in QUERIES.items():
            qv = embedder.encode_query(q)
            hits = vector_search(session, qv, top_k=2)
            assert name in _sources(hits), (name, _sources(hits))
    finally:
        session.close()


def test_hybrid_retrieves_correct_doc(corpus):
    # 语料小且 token 稀有，用原始 query 走完整 hybrid 链路即可判定
    for name, q in QUERIES.items():
        hits = retrieve(q, mode="hybrid", rerank=False, top_k=2)
        assert name in _sources(hits), (name, _sources(hits))


def test_delete_removes_from_dense_and_lexical(corpus):
    run, tagged = corpus
    target = f"{run}|gamma_yak_gateway.md"
    session = get_session()
    try:
        ok = delete_document(session, target)
        session.commit()
    finally:
        session.close()
    assert ok
    # 词法路：BM25 索引应因 generation 变化重建，不再含被删块
    from app.retrieval.lexical import get_lexical_retriever
    lex = get_lexical_retriever()
    lex_ids = [cid for cid, _ in lex.search(QUERIES["gamma_yak_gateway.md"], 10)]
    sess = get_session()
    try:
        from sqlalchemy import select
        from app.db.models import Chunk, Document
        rows = sess.execute(
            select(Chunk.id, Document.source).join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(lex_ids or [0]))
        ).all()
        assert not any(r[1].endswith("gamma_yak_gateway.md") for r in rows)
        # 稠密路
        qv = get_embedder().encode_query(QUERIES["gamma_yak_gateway.md"])
        dense = vector_search(sess, qv, top_k=2)
    finally:
        sess.close()
    assert "gamma_yak_gateway.md" not in _sources(dense)
