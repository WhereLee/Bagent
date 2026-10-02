"""量化 bge-reranker-base 并本机对比：int8 动态量化 vs fp32 的 体积/延迟/打分一致性。

用法： .venv\\Scripts\\python.exe scripts\\quantize_reranker.py
不做检索质量 A/B（那在 eval_rerank_ab.py）；这里只看"量化省了多少、分数序变没变"。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PG_DATABASE", "bagent_bench")
os.environ.setdefault("OMP_NUM_THREADS", "1")   # 单线程测，避免线程数干扰延迟对比
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from app.retrieval.rerank import Reranker  # noqa: E402

# 一组 (query, [候选...]) 样例
PAIRS = [
    ("秦惠文王 采用谁的连横之策", ["张仪以连横游说六国", "苏秦合纵抗秦", "李冰修都江堰", "商鞅变法"]),
    ("XG-200 默认管理端口", ["默认登录端口为8443(HTTPS)", "整机质保3年", "支持PoE受电", "工作温度-20到60"]),
    ("网关固件升级注意", ["升级过程请勿断电否则回滚", "默认IP为192.168.1.1", "支持TLS1.3"]),
    ("交换机千兆电口数量", ["提供24个千兆电口与4个万兆上行", "满载功耗约60W不支持PoE", "支持VLAN与LACP"]),
]


def param_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def score_all(r: Reranker):
    out = []
    for q, docs in PAIRS:
        out.append(r.rerank(q, docs))
    return out


def timed(r: Reranker, reps: int = 5):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        score_all(r)
        best = min(best, time.perf_counter() - t)
    return best


def spearman(a: list[list[float]], b: list[list[float]]) -> float:
    """逐 query 内排名的 Spearman 一致性，再平均。"""
    import itertools

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rk = [0] * len(xs)
        for r, i in enumerate(order):
            rk[i] = r
        return rk

    sims = []
    for qa, qb in zip(a, b):
        ra, rb = ranks(qa), ranks(qb)
        n = len(ra)
        if n < 2:
            continue
        d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
        sims.append(1 - 6 * d2 / (n * (n * n - 1)))
    return sum(sims) / len(sims) if sims else 1.0


def main() -> None:
    torch.set_num_threads(1)
    fp = Reranker(int8=False)
    bytes_fp32 = param_bytes(fp.model.model)
    scores_fp32 = score_all(fp)
    t_fp32 = timed(fp)

    fp.quantize_dynamic()
    bytes_int8 = param_bytes(fp.model.model)
    scores_int8 = score_all(fp)
    t_int8 = timed(fp)

    rho = spearman(scores_fp32, scores_int8)
    print("\n# bge-reranker-base: fp32 vs int8 动态量化（本机实测）\n")
    print(f"参数量(fp32)      : {bytes_fp32/1e6:.1f} MB")
    print(f"参数量(int8)      : {bytes_int8/1e6:.1f} MB   (↓ {(1-bytes_int8/bytes_fp32)*100:.1f}%)")
    print(f"打分耗时 fp32      : {t_fp32*1000:.0f} ms / 批")
    print(f"打分耗时 int8      : {t_int8*1000:.0f} ms / 批   (×{t_fp32/t_int8:.2f} 快)")
    print(f"候选内排名一致性 rho: {rho:.4f}  (1.0=完全一致)")
    print("\n说明：体积/提速是本机单线程实测；排名一致性看'该排的还排不排得对'。")
    print("真正的检索质量 Δ 用 eval_rerank_ab.py 在 DuRetrieval 上跑（recall/MRR/nDCG ± CI）。")


if __name__ == "__main__":
    main()
