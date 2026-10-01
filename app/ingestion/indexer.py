"""索引 pipeline：把一篇文档走完 解析 -> 切块 -> 向量化 -> 入库。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.db.session import get_session
from app.ingestion.chunking import chunk_text
from app.ingestion.embedder import get_embedder
from app.ingestion.parser import parse_file
from app.retrieval.store import upsert_document


@dataclass
class IngestResult:
    source: str
    document_id: int | None
    n_chunks: int
    created: bool


def ingest_file(path: str | Path) -> IngestResult:
    source = str(path)
    text, media_type = parse_file(path)
    if not text.strip():
        return IngestResult(source=source, document_id=None, n_chunks=0, created=False)

    chunks = chunk_text(text)
    embedder = get_embedder()
    vectors = embedder.encode_documents([c.text for c in chunks])

    rows = [
        (c.index, c.text, c.token_count, vectors[i])
        for i, c in enumerate(chunks)
    ]

    session = get_session()
    try:
        doc_id, n, created = upsert_document(session, source, text, media_type, rows)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return IngestResult(source=source, document_id=doc_id, n_chunks=n, created=created)
