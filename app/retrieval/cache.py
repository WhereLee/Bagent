"""检索结果缓存：LRU + TTL。

热点查询直接跳过 embed + 向量检索 + rerank（rerank 在 CPU 上是吞吐瓶颈）。
失效：语料写入/删除时显式 `clear()`；跨进程/多 worker 时各进程独立缓存（TTL 兜底），
分布式一致性需换成 Redis（接口 `get/put/clear` 不变）。
时钟可注入，便于单测。
"""
from __future__ import annotations

import hashlib
import json
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


class RedisCache:
    """后端为 Redis 的检索缓存（跨进程/多 worker 共享，避免缓存戳化）。

    值约定为 JSON 可列（调用方存 list[dict]）；接口与 RetrievalCache 一致。client 可注入以便单测。"""
    _PREFIX = "bagent:rc:"

    def __init__(self, client, ttl: float) -> None:
        self.r = client
        self.ttl = float(ttl)
        self.hits = 0
        self.misses = 0

    def _k(self, key: Hashable) -> str:
        return self._PREFIX + hashlib.sha1(repr(key).encode()).hexdigest()

    def get(self, key):
        raw = self.r.get(self._k(key))
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def put(self, key, value) -> None:
        if self.ttl <= 0:
            return
        self.r.setex(self._k(key), int(self.ttl), json.dumps(value, ensure_ascii=False))

    def clear(self) -> None:
        # 按前缀扫描删除（不用 FLUSHDB，避免误伤同实例其他键）
        for k in self.r.scan_iter(match=self._PREFIX + "*", count=200):
            self.r.delete(k)

    def __len__(self) -> int:
        return sum(1 for _ in self.r.scan_iter(match=self._PREFIX + "*", count=1000))


_cache: "RetrievalCache | RedisCache | None" = None


def get_retrieval_cache():
    global _cache
    if _cache is None:
        from app.config import get_settings
        s = get_settings()
        if s.redis_url:
            import redis
            _cache = RedisCache(redis.Redis.from_url(s.redis_url, decode_responses=True),
                                ttl=s.retrieval_cache_ttl)
        else:
            _cache = RetrievalCache(s.retrieval_cache_max, s.retrieval_cache_ttl)
    return _cache


def reset_retrieval_cache() -> None:
    """测试用：丢弃缓存实例。"""
    global _cache
    _cache = None
