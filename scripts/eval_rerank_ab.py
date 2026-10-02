"""Reranker 量化 A/B：在 DuRetrieval 真 qrels 上比 fp32 vs int8 的检索质量。

固定 candidate_n=20、只切换 reranker 是否 int8 → 差异纯来自量化。关缓存避免两趟串味。
用法： .venv\\Scripts\\python.exe scripts\\eval_rerank_ab.py [--limit 60] [-k 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PG_DATABASE", "bagent_bench")
os.environ.setdefault("RETRIEVAL_CACHE_ENABLED", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR  # noqa: E402
from app.evaluation.metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402
from app.evaluation.stats import summarize  # noqa: E402
from app.retrieval import retriever  # noqa: E402
from app.retrieval.rerank import OnnxReranker, Reranker  # noqa: E402
from app.config import ROOT_DIR as _ROOT  # noqa: E402


def ranked_sources(query, fetch, r, cand):
    chunks = retriever.retrieve(query, top_k=fetch, mode="hybrid", rerank=True, candidate_n=cand)
    seen, out = set(), []
    for c in chunks:
        if c.source not in seen:
            seen.add(c.source); out.append(c.source)
    return out


def run_pass(golden, k, r, cand, fetch):
    vals = {f"recall@{k}": [], "mrr": [], f"ndcg@{k}": []}
    for item in golden:
        ranked = ranked_sources(item["question"], fetch, r, cand)
        rel = set(item["relevant_sources"])
        vals[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
        vals["mrr"].append(mrr(ranked, rel))
        vals[f"ndcg@{k}"].append(ndcg_at_k(ranked, rel, k))
    return {m: summarize(v) for m, v in vals.items()}


def main(k: int, limit: int, cand: int) -> None:
    p = ROOT_DIR / "data" / "benchmark" / "duretrieval.jsonl"
    golden = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    fetch = max(k * 3, 30)

    r = Reranker(int8=False)
    retriever.get_reranker = lambda: r          # 强制用同一实例（可就地量化切换）

    p_fp32 = run_pass(golden, k, r, cand, fetch)
    r.quantize_dynamic()                         # 就地 torch int8
    p_tint8 = run_pass(golden, k, r, cand, fetch)

    onnx = OnnxReranker(_ROOT / "models" / "reranker-onnx" / "model_int8.onnx")
    retriever.get_reranker = lambda: onnx
    p_onnx = run_pass(golden, k, onnx, cand, fetch)

    print(f"\n# Reranker 量化 A/B (DuRetrieval, cand={cand}, K={k}, n={len(golden)}) — 均值±95%CI\n")
    print(f"{'metric':<12}{'st-fp32':>20}{'st-int8':>20}{'onnx-int8':>20}")
    print("-" * 72)
    for m in [f"recall@{k}", "mrr", f"ndcg@{k}"]:
        a, b, c2 = p_fp32[m], p_tint8[m], p_onnx[m]
        cell = lambda s: f"{s['mean']:.3f}±{(s['ci_high']-s['ci_low'])/2:.3f}"
        print(f"{m:<12}{cell(a):>20}{cell(b):>20}{cell(c2):>20}")
    print("\n判读：与 st-fp32 同行比，Δ 落在 CI 内或 nDCG 掉 <~1% → 该后端可采纳。")
    print("onnx-int8 额外收益：磁盘 279MB（vs 1.1G）、CPU 推理更快。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--cand", type=int, default=20)
    a = ap.parse_args()
    main(a.k, a.limit, a.cand)
