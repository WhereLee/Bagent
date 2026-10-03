"""索引 pipeline：解析 -> 父子块切分 -> 子块向量化 -> 入库。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.db.session import get_session
from app.ingestion.chunking import chunk_parent_child
from app.ingestion.embedder import get_embedder
from app.ingestion.parser import parse_file
from app.observability.logging import get_logger, log_event
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.store import index_document

_log = get_logger("ingest")


@dataclass
class IngestResult:
    source: str
    document_id: int | None
    n_children: int
    n_parents: int
    action: str  # created / updated / skipped


def ingest_file(path: str | Path, tenant: str | None = None) -> IngestResult:
    source = str(path)
    text, media_type = parse_file(path)
    if not text.strip():
        return IngestResult(source, None, 0, 0, "empty")

    s = get_settings()
    parents = chunk_parent_child(
        text,
        parent_tokens=s.chunk_parent_tokens,
        child_tokens=s.chunk_child_tokens,
        child_overlap=s.chunk_child_overlap,
    )

    # 展平所有子块并批量向量化（子块才需要向量）
    child_texts = [c.text for p in parents for c in p.children]
    vectors = (
        get_embedder().encode_documents(child_texts)
        if child_texts
        else np.empty((0, s.embedding_dim), dtype=np.float32)
    )

    session = get_session()
    try:
        doc_id, n_children, action = index_document(
            session, source, text, media_type, parents, vectors, tenant_id=tenant
        )
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    if action in ("created", "updated"):
        get_retrieval_cache().clear()  # 语料变更 -> 失效检索缓存

    return IngestResult(source, doc_id, n_children, len(parents), action)


def ingest_paths(paths) -> list[IngestResult]:
    """批量入库并逐条结构化日志。"""
    results = []
    for p in paths:
        r = ingest_file(p)
        log_event(_log, "info", "ingest", source=r.source, action=r.action,
                  parents=r.n_parents, children=r.n_children, document_id=r.document_id)
        results.append(r)
    return results
