"""检索结果缓存：LRU + TTL。

热点查询直接跳过 embed + 向量检索 + rerank（rerank 在 CPU 上是吞吐瓶颈）。
失效：语料写入/删除时显式 `clear()`；跨进程/多 worker 时各进程独立缓存（TTL 兜底），
分布式一致性需换成 Redis（接口 `get/put/clear` 不变）。
时钟可注入，便于单测。
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


class RetrievalCache:
    def __init__(self, max_size: int, ttl: float, clock=time.monotonic) -> None:
        self.max_size = int(max_size)
        self.ttl = float(ttl)
        self._clock = clock
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expire, value = item
            if self._clock() >= expire:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)  # LRU
            self.hits += 1
            return value

    def put(self, key: Hashable, value: Any) -> None:
        if self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (self._clock() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)  # 淘汰最久未用

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


_cache: RetrievalCache | None = None


def get_retrieval_cache() -> RetrievalCache:
    global _cache
    if _cache is None:
        from app.config import get_settings
        s = get_settings()
        _cache = RetrievalCache(s.retrieval_cache_max, s.retrieval_cache_ttl)
    return _cache


def reset_retrieval_cache() -> None:
    """测试用：丢弃缓存实例。"""
    global _cache
    _cache = None
