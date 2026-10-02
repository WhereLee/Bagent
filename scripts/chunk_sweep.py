"""#1 chunk_size 扫参：在不同子块尺寸下重灌语料，用 DuRetrieval 真 qrels 测检索质量。

每个尺寸点：清空 bench 库 -> 按该 child_tokens 重新摄取 -> 文档级 recall@10/MRR/nDCG。
集成脚本（需 bench 库 + 模型），较慢（每点重嵌入），不进 CI。
用法： .venv\\Scripts\\python.exe scripts\\chunk_sweep.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PG_DATABASE", "bagent_bench")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

import prepare_dataset as pd  # noqa: E402  (import 时设定 bench 环境)
from app.config import ROOT_DIR, get_settings  # noqa: E402
from app.db.session import get_session  # noqa: E402
from app.evaluation.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.retrieval.retriever import retrieve  # noqa: E402

SIZES = [80, 150, 300]
K = 10
RETRIEVE_CFG = [("dense", dict(mode="dense")), ("hybrid", dict(mode="hybrid"))]


def truncate() -> None:
    s = get_session()
    try:
        s.execute(text("DELETE FROM chunks"))
        s.execute(text("DELETE FROM documents"))
        s.commit()
    finally:
        s.close()


def load_golden():
    p = ROOT_DIR / "data" / "benchmark" / "duretrieval.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ranked_sources(query, fetch=30, **kw):
    chunks = retrieve(query, top_k=fetch, **kw)
    seen, out = set(), []
    for c in chunks:
        if c.source not in seen:
            seen.add(c.source); out.append(c.source)
    return out


def main() -> None:
    configure_logging()
    rows = []
    for child in SIZES:
        os.environ["CHUNK_CHILD_TOKENS"] = str(child)
        os.environ["CHUNK_PARENT_TOKENS"] = str(max(600, child * 2))
        get_settings.cache_clear()
        truncate()
        pd.build(queries=80, passages_n=1200)  # 首次会拉取并缓存原始数据，后续点复用缓存
        golden = load_golden()
        for name, kw in RETRIEVE_CFG:
            acc = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
            for item in golden:
                ranked = ranked_sources(item["question"], **kw)
                rel = set(item["relevant_sources"])
                acc["recall"] += recall_at_k(ranked, rel, K)
                acc["mrr"] += mrr(ranked, rel)
                acc["ndcg"] += ndcg_at_k(ranked, rel, K)
            n = len(golden) or 1
            rows.append((child, name, acc["recall"] / n, acc["mrr"] / n, acc["ndcg"] / n))

    print(f"\n# Chunk-size sweep (DuRetrieval, doc-level K={K})\n")
    print(f"{'child_tok':<12}{'config':<10}{'recall@10':>12}{'mrr':>10}{'ndcg@10':>12}")
    print("-" * 46)
    for child, name, r, m, nd in rows:
        print(f"{child:<12}{name:<10}{r:>12.3f}{m:>10.3f}{nd:>12.3f}")


if __name__ == "__main__":
    main()
