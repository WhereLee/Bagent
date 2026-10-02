"""检索消融评测：在 golden set 上对比不同检索配置，产出量化指标。

用法： .venv\\Scripts\\python.exe scripts\\eval_retrieval.py [-k 5]
需要先 ingest 语料并下载 reranker 模型。指标为检索层（recall/precision/MRR/nDCG）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR, get_settings  # noqa: E402
from app.db.session import get_session  # noqa: E402
from app.evaluation.dataset import label_relevance, load_golden  # noqa: E402
from app.evaluation.metrics import evaluate  # noqa: E402
from app.retrieval.retriever import retrieve  # noqa: E402

CONFIGS = [
    ("dense", dict(mode="dense", rerank=False)),
    ("hybrid", dict(mode="hybrid", rerank=False)),
    ("dense+rerank", dict(mode="dense", rerank=True)),
    ("hybrid+rerank", dict(mode="hybrid", rerank=True)),
]


def run(k: int) -> None:
    s = get_settings()
    golden = label_relevance(get_session(), load_golden(ROOT_DIR / "data" / "golden" / "golden.jsonl"))
    if not golden:
        print("golden set 为空")
        return

    metric_keys = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]
    print(f"\n# Retrieval ablation (K={k}), {len(golden)} queries\n")
    header = f"{'config':<14}" + "".join(f"{m:>14}" for m in metric_keys)
    print(header)
    print("-" * len(header))

    for name, kwargs in CONFIGS:
        acc = {m: 0.0 for m in metric_keys}
        for item in golden:
            ranked = [c.chunk_id for c in retrieve(item.question, top_k=k, **kwargs)]
            scores = evaluate(ranked, item.relevant_ids, k)
            for m in metric_keys:
                acc[m] += scores[m]
        row = f"{name:<14}" + "".join(f"{acc[m]/len(golden):>14.3f}" for m in metric_keys)
        print(row)

    # 标注覆盖检查：提醒哪些 query 没在库中找到相关块
    missing = [i.question for i in golden if not i.relevant_ids]
    if missing:
        print(f"\n[warn] {len(missing)} 条 query 未匹配到任何相关子块（片段与切块不一致，需检查）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=get_settings().retrieval_top_k)
    run(ap.parse_args().k)
