"""Locust 压测脚本（集成工具，不进 CI）。

Web 模式：  locust -f locustfile.py --host http://127.0.0.1:8000
无头压测：  locust -f locustfile.py --host http://127.0.0.1:8000 \
              --headless -u 50 -r 10 -t 60s
产出：RPS、p50/p95/p99 时延、失败率。

说明：/query 会走 LLM（慢、花钱），权重设低；/search 只走检索，是压测检索/服务吞吐的主力。
"""
from __future__ import annotations

import os

from locust import HttpUser, between, task

QUERIES = [
    "XG-200 的默认管理端口是多少？",
    "XG-400 Pro 支持哪些 PoE 标准？",
    "SW-2400 有多少个千兆电口？",
    "设备的日志默认保留多久？",
]


class RagUser(HttpUser):
    host = os.environ.get("LOCUST_HOST", "http://127.0.0.1:8000")
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        import random

        self.q = random.choice(QUERIES)

    @task(5)
    def search(self) -> None:
        self.client.post("/search", json={"query": self.q, "top_k": 5}, name="/search")

    @task(1)
    def query(self) -> None:
        # /query 走 MiMo（LLM 绑定、有成本、延迟不代表本系统算力）；默认关闭，仅测检索层吞吐
        if os.environ.get("LOCUST_INCLUDE_QUERY") != "1":
            return
        self.client.post("/query", json={"query": self.q}, name="/query", timeout=60)

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="/health")
