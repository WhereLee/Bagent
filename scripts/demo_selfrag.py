"""self-RAG/冲突处理 端到端演示：对同一事实存在矛盾文档时，系统应检出冲突并降级。

写入临时矛盾文档 -> 开 self_rag 查询 -> 断言/展示 conflict 被标记 -> 清理。集成脚本。
用法： .venv\\Scripts\\python.exe scripts\\demo_selfrag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR  # noqa: E402
from app.db.session import get_session  # noqa: E402
from app.generation.generator import answer_query  # noqa: E402
from app.ingestion.indexer import ingest_file  # noqa: E402
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.store import delete_document  # noqa: E402

_TMP = ROOT_DIR / "data" / "corpus"
A = _TMP / "_demo_conflict_a.md"
B = _TMP / "_demo_conflict_b.md"


def main() -> None:
    A.write_text("星链 XG-777 边缘网关出厂默认管理端口为 8443（HTTPS），登录用户 admin。\n", encoding="utf-8")
    B.write_text("星链 XG-777 边缘网关出厂默认管理端口为 9000（HTTP），登录用户 root。\n", encoding="utf-8")
    try:
        ingest_file(A)
        ingest_file(B)
        get_retrieval_cache().clear()

        ans = answer_query("XG-777 边缘网关的默认管理端口是多少？", top_k=5, self_rag=True)
        print("conflict_detected:", ans.conflict)
        print("conflict_note:", ans.conflict_note)
        print("iterations:", ans.iterations, "evidence_sufficient:", ans.evidence_sufficient)
        print("low_confidence:", ans.low_confidence, "grounded:", ans.grounded)
        print("answer:", ans.text[:160])
        assert ans.conflict, "未检出矛盾（检索未同时召回两份或 judge 判否）——需查"
        print("\nPASS: 冲突被检出并降级（未偷选一个答案）。")
    finally:
        for doc in (A, B):
            if doc.exists():
                s = get_session()
                try:
                    delete_document(s, str(doc))
                    s.commit()
                finally:
                    s.close()
                doc.unlink()
        get_retrieval_cache().clear()


if __name__ == "__main__":
    main()
