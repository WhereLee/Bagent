"""限流冒烟：预热一次后并发打 /search，统计状态码分布。"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = "http://127.0.0.1:8010"


def call(path: str, payload: dict, timeout: float = 60) -> int:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    # 1) 预热（触发模型加载，只发 1 次）
    print("warmup status:", call("/search", {"query": "端口", "top_k": 3}))

    # 2) 并发突发 30 次
    with ThreadPoolExecutor(max_workers=30) as ex:
        codes = list(ex.map(lambda _: call("/search", {"query": "质保几年", "top_k": 3}), range(30)))
    from collections import Counter

    dist = Counter(codes)
    print("burst status distribution:", dict(dist))
    print("429 被限流次数:", dist.get(429, 0))


if __name__ == "__main__":
    main()
