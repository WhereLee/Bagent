"""在 DuRetrieval 真 qrels 上做文档级检索评测（并顺带 #2 的 candidate 扫参）。

文档级相关性：召回块的 source(pid) 是否 ∈ 该 query 的相关 passage 集合——与切块尺寸解耦。
集成脚本（需 bench 库 + 模型），不进 CI。
用法： .venv\\Scripts\\python.exe scripts\\eval_benchmark.py [-k 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PG_DATABASE", "bagent_bench")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR, get_settings  # noqa: E402
from app.evaluation.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from app.evaluation.stats import summarize  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.retrieval.retriever import retrieve  # noqa: E402

# 消融矩阵：dense / hybrid / hybrid+rerank(不同 candidate_n，体现 #2 扫参)
CONFIGS = [
    ("dense", dict(mode="dense", rerank=False)),
    ("hybrid", dict(mode="hybrid", rerank=False)),
    ("hybrid+rerank c20", dict(mode="hybrid", rerank=True, candidate_n=20)),
    ("hybrid+rerank c50", dict(mode="hybrid", rerank=True, candidate_n=50)),
]


def ranked_sources(query: str, fetch: int, **kw) -> list[str]:
    """检索并折叠为去重后的 source 序列（文档级排名）。"""
    chunks = retrieve(query, top_k=fetch, **kw)
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        if c.source not in seen:
            seen.add(c.source)
            out.append(c.source)
    return out


def main(k: int, limit: int, no_rerank: bool = False) -> None:
    configure_logging(get_settings().log_level)
    golden = [
        json.loads(l)
        for l in (ROOT_DIR / "data" / "benchmark" / "duretrieval.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][:limit]
    if not golden:
        print("基准 golden 为空，先跑 prepare_dataset.py")
        return

    configs = [c for c in CONFIGS if not (no_rerank and "rerank" in c[0])]
    fetch = max(k * 3, 30)  # 多取再折叠，保证有足够去重文档数
    print(f"\n# DuRetrieval eval (doc-level, K={k}), {len(golden)} queries — 均值±bootstrap95%CI\n", flush=True)
    cols = [f"recall@{k}", "mrr", f"ndcg@{k}"]
    print(f"{'config':<20}" + "".join(f"{c:>22}" for c in cols), flush=True)
    print("-" * (20 + 22 * len(cols)), flush=True)

    for name, kw in configs:
        vals = {c: [] for c in cols}
        for item in golden:
            ranked = ranked_sources(item["question"], fetch, **kw)
            rel = set(item["relevant_sources"])
            vals[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
            vals["mrr"].append(mrr(ranked, rel))
            vals[f"ndcg@{k}"].append(ndcg_at_k(ranked, rel, k))
        row = f"{name:<20}"
        for c in cols:
            sm = summarize(vals[c])
            row += f"{sm['mean']:>9.3f}±{(sm['ci_high']-sm['ci_low'])/2:<12.3f}"
        print(row, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=60, help="参与评测的查询数")
    ap.add_argument("--no-rerank", action="store_true", help="只跑 dense/hybrid（快）")
    args = ap.parse_args()
    main(args.k, args.limit, args.no_rerank)
