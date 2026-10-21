r"""BM25 微基准：倒排索引 vs 旧全量扫描（纯内存，无需 DB/模型）。

用法：
    .\.venv\Scripts\python.exe scripts\bench_bm25.py --docs 10000 --queries 200

对比项：
- 建索引耗时
- 相同查询集的总耗时 / 单查询均耗时
- top-K 文档 id 是否与全量扫描一致（排序正确性）

这是合成数据基准，只用于验证"只对命中文档打分"的收益量级；
真实语料上的召回质量仍以 scripts/eval_retrieval.py 为准。
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.lexical import BM25Index  # noqa: E402


class BruteForceBM25:
    """旧实现的参照版：每次查询遍历全库文档。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[int] = []
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}
        self.avgdl = 0.0
        self.n = 0

    def build(self, docs: list[tuple[int, list[str]]]) -> None:
        self.doc_ids = [d[0] for d in docs]
        self.doc_tf = [Counter(d[1]) for d in docs]
        self.doc_len = [len(d[1]) for d in docs]
        df: dict[str, int] = {}
        for tf in self.doc_tf:
            for term in tf:
                df[term] = df.get(term, 0) + 1
        self.df = df
        self.n = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        if self.n == 0 or not query_tokens or top_k <= 0:
            return []
        scores = [0.0] * self.n
        for term in set(query_tokens):
            idf = self._idf(term)
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / (f + norm)
        order = sorted(range(self.n), key=lambda i: scores[i], reverse=True)
        out: list[tuple[int, float]] = []
        for i in order:
            if scores[i] <= 0 or len(out) >= top_k:
                break
            out.append((self.doc_ids[i], scores[i]))
        return out


def _make_docs(n_docs: int, vocab_size: int, seed: int = 42) -> list[tuple[int, list[str]]]:
    rnd = random.Random(seed)
    vocab = [f"t{i}" for i in range(vocab_size)]
    docs: list[tuple[int, list[str]]] = []
    for doc_id in range(n_docs):
        # 每篇 40~80 个词，偏向少量高频词 + 长尾词，接近真实语料的倾斜分布
        length = rnd.randint(40, 80)
        toks = [rnd.choice(vocab) for _ in range(length)]
        docs.append((doc_id, toks))
    return docs


def _make_queries(n_queries: int, vocab_size: int, seed: int = 7) -> list[list[str]]:
    rnd = random.Random(seed)
    vocab = [f"t{i}" for i in range(vocab_size)]
    return [[rnd.choice(vocab) for _ in range(rnd.randint(2, 5))] for _ in range(n_queries)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=10000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--vocab", type=int, default=2000)
    args = ap.parse_args()

    docs = _make_docs(args.docs, args.vocab)
    queries = _make_queries(args.queries, args.vocab)

    t0 = time.perf_counter()
    brute = BruteForceBM25()
    brute.build(docs)
    t_brute_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    inv = BM25Index()
    inv.build(docs)
    t_inv_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    brute_out = [brute.search(q, args.top_k) for q in queries]
    t_brute_search = time.perf_counter() - t0

    t0 = time.perf_counter()
    inv_out = [inv.search(q, args.top_k) for q in queries]
    t_inv_search = time.perf_counter() - t0

    mismatches = sum(
        1
        for b, i in zip(brute_out, inv_out)
        if [doc_id for doc_id, _ in b] != [doc_id for doc_id, _ in i]
    )
    speedup = (t_brute_search / t_inv_search) if t_inv_search else float("inf")

    print(f"docs={args.docs} queries={args.queries} top_k={args.top_k} vocab={args.vocab}")
    print(f"build  brute={t_brute_build:.3f}s  inverted={t_inv_build:.3f}s")
    print(f"search brute={t_brute_search:.3f}s  inverted={t_inv_search:.3f}s  speedup={speedup:.2f}x")
    print(f"top-k mismatches: {mismatches}/{len(queries)}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
