"""限流：令牌桶（Token Bucket）。

设计：算法是标准令牌桶；后端用**进程内存**存桶。多实例横向扩展时应换成
Redis（Lua 原子扣减）——`RateLimiter` 的接口 `allow(key)` 不变，只替换存储。
这是"载体可换、算法不变"的可扩展设计，非因项目小而简化。
时钟可注入，便于单测。
"""
from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, rate: float, capacity: float, clock=time.monotonic) -> None:
        self.rate = float(rate)          # 每秒补充令牌数
        self.capacity = float(capacity)  # 桶容量（突发上限）
        self.tokens = float(capacity)
        self._clock = clock
        self.ts = clock()
        self.lock = threading.Lock()

    def allow(self) -> tuple[bool, float]:
        """返回 (是否放行, 建议重试秒数)。放行时重试秒数为 0。"""
        with self.lock:
            now = self._clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
            self.ts = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            # 需要等多久才有 1 个令牌
            wait = (1.0 - self.tokens) / self.rate
            return False, wait


class RateLimiter:
    """按 key（如客户端 IP）维护独立令牌桶。"""

    def __init__(self, rate: float, burst: int, clock=time.monotonic) -> None:
        if rate <= 0 or burst <= 0:
            raise ValueError("rate 与 burst 必须为正")
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str) -> TokenBucket:
        b = self._buckets.get(key)
        if b is None:
            with self._lock:
                b = self._buckets.get(key)
                if b is None:
                    b = TokenBucket(self.rate, self.burst, clock=self._clock)
                    self._buckets[key] = b
        return b

    def allow(self, key: str) -> tuple[bool, float]:
        return self._bucket(key).allow()


class RedisRateLimiter:
    """跨进程共享的固定窗口限流（多 worker 下全局生效）。

    与内存令牌桶语义近似但不完全等价（窗口边界会放过 1~2x 突发）；单实例仍用内存
    令牌桶更平滑。接口 `allow(key)->(bool, retry_sec)` 与 RateLimiter 一致。client 可注入以便单测。
    """
    _PREFIX = "bagent:rl:"

    def __init__(self, client, rate: float, burst: int, clock=time.time) -> None:
        self.r = client
        self.rate = float(rate)
        self.burst = int(burst)
        self._clock = clock

    def allow(self, key: str) -> tuple[bool, float]:
        win = int(self._clock())               # 1 秒窗口
        limit = int(self.rate) + self.burst    # 该窗口允许量
        rk = f"{self._PREFIX}{key}:{win}"
        count = self.r.incr(rk)
        if count == 1:
            self.r.expire(rk, 2)
        if count <= limit:
            return True, 0.0
        return False, float(win + 1 - self._clock())


def get_rate_limiter():
    from app.config import get_settings
    s = get_settings()
    if s.redis_url:
        import redis
        return RedisRateLimiter(redis.Redis.from_url(s.redis_url, decode_responses=True),
                                s.rate_limit_per_sec, s.rate_limit_burst)
    return RateLimiter(s.rate_limit_per_sec, s.rate_limit_burst)
