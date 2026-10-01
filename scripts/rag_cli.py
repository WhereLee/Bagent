"""命令行 RAG 工具：入库 / 检索 / 问答（不依赖 HTTP 服务）。

用法：
  .venv\\Scripts\\python.exe scripts\\rag_cli.py ingest data\\corpus
  .venv\\Scripts\\python.exe scripts\\rag_cli.py ingest path\\to\\file.md
  .venv\\Scripts\\python.exe scripts\\rag_cli.py search "问题" -k 5
  .venv\\Scripts\\python.exe scripts\\rag_cli.py query  "问题"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.generator import answer_query  # noqa: E402
from app.ingestion.indexer import ingest_file  # noqa: E402

SUPPORTED = {".pdf", ".docx", ".html", ".htm", ".md", ".markdown", ".txt"}


def _iter_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return [p for p in sorted(target.rglob("*")) if p.suffix.lower() in SUPPORTED]


def cmd_ingest(target: str) -> None:
    files = _iter_files(Path(target))
    if not files:
        print(f"未发现可入库文件: {target}")
        return
    for f in files:
        r = ingest_file(f)
        state = "新建" if r.created else "已存在(跳过)"
        print(f"[ingest] {f} -> doc_id={r.document_id} chunks={r.n_chunks} {state}")


def cmd_query(text: str, top_k: int | None) -> None:
    ans = answer_query(text, top_k=top_k)
    print("\n=== 回答 ===")
    print(ans.text)
    print("\n=== 来源 ===")
    for s in ans.sources:
        print(f"  [{s['id']}] score={s['score']} {s['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bagent RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("target")

    p_q = sub.add_parser("query")
    p_q.add_argument("text")
    p_q.add_argument("-k", "--top-k", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args.target)
    elif args.cmd == "query":
        cmd_query(args.text, args.top_k)


if __name__ == "__main__":
    main()
