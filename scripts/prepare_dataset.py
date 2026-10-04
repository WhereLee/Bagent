"""从 C-MTEB/DuRetrieval 采样，构建带真 qrels 的文档级检索评测集并入库（独立 bench 库）。

通用检索基准、文档级相关性、可复现（固定 seed + 本地缓存原始数据，避免重复流扫）。
集成脚本，不进 CI。
用法： .venv\\Scripts\\python.exe scripts\\prepare_dataset.py [--queries 80] [--passages 1200]
强制重下： 设环境变量 DU_FORCE_REFRESH=1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# 必须在导入 app.config 之前设定：连独立 bench 库 + 走镜像
os.environ.setdefault("PG_DATABASE", "bagent_bench")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from app.config import ROOT_DIR, get_settings  # noqa: E402
from app.db.session import get_session, init_schema  # noqa: E402
from app.ingestion.chunking import chunk_parent_child  # noqa: E402
from app.ingestion.embedder import get_embedder  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402

SEED = 42
_CACHE = ROOT_DIR / "data" / "benchmark" / "_du_cache.json"


def _fetch_raw(n_passages: int) -> tuple[dict, dict, dict]:
    """流式取原始数据：前 N 段落作语料；queries 全量；qrels 仅保留 gold 落在语料内的查询。"""
    passages: dict[str, str] = {}
    for row in load_dataset("C-MTEB/DuRetrieval", streaming=True)["corpus"]:
        passages[row["id"]] = row["text"]
        if len(passages) >= n_passages:
            break
    pid_set = set(passages)

    q_txt: dict[str, str] = {}
    for row in load_dataset("C-MTEB/DuRetrieval", streaming=True)["queries"]:
        q_txt[row["id"]] = row["text"]

    qrels: dict[str, list[str]] = {}
    for row in load_dataset("C-MTEB/DuRetrieval-qrels", streaming=True)["dev"]:
        if str(row.get("score", "0")) not in ("0", "", "None") and row["pid"] in pid_set and row["qid"] in q_txt:
            qrels.setdefault(row["qid"], []).append(row["pid"])

    return passages, q_txt, qrels


def _load_raw(n_passages: int) -> tuple[dict, dict, dict]:
    if _CACHE.exists() and os.environ.get("DU_FORCE_REFRESH") != "1":
        blob = json.loads(_CACHE.read_text(encoding="utf-8"))
        return blob["passages"], blob["q_txt"], blob["qrels"]
    passages, q_txt, qrels = _fetch_raw(n_passages)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps({"passages": passages, "q_txt": q_txt, "qrels": qrels}, ensure_ascii=False), encoding="utf-8")
    return passages, q_txt, qrels


def build(queries: int, passages_n: int, fresh: bool = False, golden_only: bool = False) -> None:
    configure_logging(get_settings().log_level)
    init_schema()
    if fresh and not golden_only:
        s0 = get_session()
        try:
            from sqlalchemy import text as _text
            s0.execute(_text("DELETE FROM chunks"))
            s0.execute(_text("DELETE FROM documents"))
            s0.commit()
        finally:
            s0.close()
    passages, q_txt, qrels = _load_raw(passages_n)

    rng = random.Random(SEED)
    eligible = list(qrels.keys())
    rng.shuffle(eligible)
    chosen_q = eligible[:queries]

    n_child = 0
    if not golden_only:
        # 入库（每段落一个文档，source=pid）
        embedder = get_embedder()
        s = get_settings()
        session = get_session()
        try:
            for pid, text in passages.items():
                if not text.strip():
                    continue
                parents = chunk_parent_child(text, parent_tokens=s.chunk_parent_tokens,
                                            child_tokens=s.chunk_child_tokens, child_overlap=s.chunk_child_overlap)
                child_texts = [c.text for p in parents for c in p.children]
                if not child_texts:
                    parents = chunk_parent_child(text, parent_tokens=s.chunk_parent_tokens, child_tokens=300, child_overlap=0)
                    child_texts = [c.text for p in parents for c in p.children]
                    if not child_texts:
                        continue
                vectors = embedder.encode_documents(child_texts)
                _index_multi(session, pid, text, vectors, parents)
                n_child += len(child_texts)
                session.commit()  # 每文档提交，缩短事务
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    out = ROOT_DIR / "data" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with (out / "duretrieval.jsonl").open("w", encoding="utf-8") as f:
        for qid in chosen_q:
            rel = list(dict.fromkeys(qrels[qid]))
            f.write(json.dumps({"question": q_txt[qid], "relevant_sources": rel}, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"语料段落={len(passages)} 子块≈{n_child} golden查询={n_written}" + ("  [golden-only]" if golden_only else ""), flush=True)


def _index_multi(session, pid: str, text: str, vectors, parents) -> None:
    """把一段落的父子块写入（source=pid，doc-level 检索单元）。"""
    from sqlalchemy import select

    from app.db.models import Chunk, Document
    from app.retrieval.store import compute_doc_hash

    doc_hash = compute_doc_hash(text)
    doc = session.scalar(
        select(Document).where(Document.source == pid, Document.is_deleted == False)  # noqa: E712
    )
    if doc is not None and doc.doc_hash == doc_hash:
        return
    if doc is None:
        doc = Document(source=pid, doc_hash=doc_hash, media_type="passage", metadata_={})
        session.add(doc)
        session.flush()
    vi = 0
    for parent in parents:
        prow = Chunk(document_id=doc.id, parent_id=None, chunk_index=0, content=parent.text,
                     token_count=parent.token_count, embedding=None, metadata_={"role": "parent"})
        session.add(prow)
        session.flush()
        for child in parent.children:
            session.add(Chunk(document_id=doc.id, parent_id=prow.id, chunk_index=child.index,
                              content=child.text, token_count=child.token_count,
                              embedding=vectors[vi].tolist(), metadata_={"role": "child"}))
            vi += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=80)
    ap.add_argument("--passages", type=int, default=1200)
    ap.add_argument("--fresh", action="store_true", help="先清空 bench 库再重建")
    ap.add_argument("--golden-only", action="store_true", help="不重嵌/不入库，仅从本地缓存扩写 duretrieval.jsonl")
    a = ap.parse_args()
    build(a.queries, a.passages, a.fresh, a.golden_only)
