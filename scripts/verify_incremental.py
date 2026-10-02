"""#3 增量正确性验证：更新/删除后，dense 与 lexical 两路都不应再召回旧内容。

集成脚本（需 DB + 模型），不进 CI。步骤：
 1) 摄取两篇手册
 2) 删除 gateway_pro_manual
 3) 检索 "XG-400 Pro 端口" —— 断言结果里没有来自被删文档的块（dense + lexical 都不召回）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR  # noqa: E402
from app.db.session import get_session, init_schema  # noqa: E402
from app.ingestion.indexer import ingest_file  # noqa: E402
from app.retrieval.lexical import get_lexical_retriever  # noqa: E402
from app.retrieval.retriever import retrieve  # noqa: E402
from app.retrieval.store import delete_document  # noqa: E402

PRO = str(ROOT_DIR / "data" / "corpus" / "gateway_pro_manual.md")


def sources_of(q: str) -> set[str]:
    return {Path(c.source).name for c in retrieve(q, top_k=5, mode="hybrid", rerank=False)}


def lexical_sources(q: str) -> set[str]:
    """直接看词法索引返回的 chunk 属于哪些文档。"""
    ids = [cid for cid, _ in get_lexical_retriever().search(q, 10)]
    if not ids:
        return set()
    from sqlalchemy import select
    from app.db.models import Chunk, Document
    s = get_session()
    try:
        rows = s.execute(
            select(Chunk.id, Document.source).join(Document, Document.id == Chunk.document_id).where(Chunk.id.in_(ids))
        ).all()
        return {Path(r[1]).name for r in rows}
    finally:
        s.close()


def main() -> None:
    init_schema()
    for f in ["gateway_manual.md", "gateway_pro_manual.md"]:
        ingest_file(ROOT_DIR / "data" / "corpus" / f)

    print("删除前 dense sources:", sources_of("XG-400 Pro 的默认管理端口"))
    print("删除前 lexical sources:", lexical_sources("XG-400 Pro 端口"))

    session = get_session()
    try:
        ok = delete_document(session, PRO)
        session.commit()
    finally:
        session.close()
    print("delete_document ->", ok)

    dense_after = sources_of("XG-400 Pro 的默认管理端口")
    lex_after = lexical_sources("XG-400 Pro 端口")
    print("删除后 dense sources:", dense_after)
    print("删除后 lexical sources:", lex_after)

    assert "gateway_pro_manual.md" not in dense_after, "dense 仍召回已删文档！"
    assert "gateway_pro_manual.md" not in lex_after, "lexical 仍召回已删文档！"
    print("\nPASS: 删除后 dense 与 lexical 均不再召回该文档。")


if __name__ == "__main__":
    main()
