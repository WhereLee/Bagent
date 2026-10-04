"""Redis 缓存后端测试：注入 fake client，验接口与 JSON 往返、TTL 传递、clear 前缀隔离。"""
from app.retrieval.cache import RedisCache


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttls = {}

    def get(self, k):
        return self.store.get(k)

    def setex(self, k, ttl, v):
        self.store[k] = v
        self.ttls[k] = ttl

    def delete(self, k):
        self.store.pop(k, None)

    def scan_iter(self, match=None, count=None):
        prefix = (match or "*").rstrip("*")
        return iter([k for k in list(self.store) if k.startswith(prefix)])


def test_redis_cache_roundtrip_json():
    r = FakeRedis()
    c = RedisCache(r, ttl=60)
    key = ("q", "hybrid", True, 20, 5, 3, None)
    assert c.get(key) is None and c.misses == 1
    c.put(key, [{"chunk_id": 1, "content": "你好", "score": 0.9}])
    got = c.get(key)
    assert got == [{"chunk_id": 1, "content": "你好", "score": 0.9}] and c.hits == 1
    # TTL 透传
    assert r.ttls[c._k(key)] == 60


def test_redis_cache_clear_scoped_prefix():
    r = FakeRedis()
    r.store["other:key"] = "x"
    c = RedisCache(r, ttl=10)
    c.put(("a",), [{"v": 1}])
    assert len(c) == 1
    c.clear()
    assert len(c) == 0
    assert "other:key" in r.store          # 不误伤非本前缀键


def test_redis_cache_ttl_zero_noop():
    r = FakeRedis()
    c = RedisCache(r, ttl=0)
    c.put(("a",), [{"v": 1}])
    assert c.get(("a",)) is None
