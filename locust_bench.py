"""用公开检索基准压测 /search（默认 DuRetrieval 600 条真实 query）。

为什么不复用 locustfile.py：那里的 4 条固定 query 会全部命中检索缓存，
50 并发曾测出 34 req/s 的假象；换成分布分散的真实 query 后，同一份服务
只有约 1.4 req/s，才能反映 CPU rerank 的真实成本。

用法：
    locust -f locust_bench.py --host http://127.0.0.1:8000 \
        --headless -u 10 -r 2 -t 60s

查询集默认读 data/benchmark/duretrieval.jsonl（由 scripts/prepare_dataset.py
生成，每行 {"question": ..., "relevant_sources": [...]}）。可用环境变量
LOCUST_BENCH_FILE 指向其他同结构 jsonl。

注意：命中检索缓存会显著抬高吞吐。要测冷查询，可把
RETRIEVAL_CACHE_ENABLED 设为 false 后重启服务，或保证查询集足够大。
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

_DEFAULT = Path(__file__).resolve().parent / "data" / "benchmark" / "duretrieval.jsonl"
_PATH = Path(os.environ.get("LOCUST_BENCH_FILE", str(_DEFAULT)))
QUERIES = [
    json.loads(line)["question"]
    for line in _PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not QUERIES:
    raise RuntimeError(f"查询集为空：{_PATH}（先跑 scripts/prepare_dataset.py）")


class BenchUser(HttpUser):
    wait_time = between(0.2, 0.8)

    @task
    def search(self) -> None:
        self.client.post(
            "/search",
            json={"query": random.choice(QUERIES), "top_k": 5},
            name="/search",
        )
