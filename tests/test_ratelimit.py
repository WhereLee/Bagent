"""限流令牌桶单元测试（注入假时钟，无时间依赖）。"""
from app.ratelimit import RateLimiter, TokenBucket


class Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, d: float) -> None:
        self.t += d


def test_burst_allows_then_denies():
    c = Clock(0.0)
    b = TokenBucket(rate=1.0, capacity=3, clock=c)
    assert [b.allow()[0] for _ in range(3)] == [True, True, True]
    allowed, wait = b.allow()
    assert allowed is False
    assert wait > 0


def test_refills_over_time():
    c = Clock(0.0)
    b = TokenBucket(rate=2.0, capacity=1, clock=c)
    assert b.allow()[0] is True          # 用掉初始 1 个令牌
    assert b.allow()[0] is False         # 桶空
    c.advance(1.0)                       # 过 1 秒补充 2 个（受容量限制为 1）
    assert b.allow()[0] is True


def test_capacity_caps_tokens():
    c = Clock(0.0)
    b = TokenBucket(rate=100.0, capacity=2, clock=c)
    c.advance(10.0)  # 理论上补 1000，但被容量限制为 2
    assert b.allow()[0] is True
    assert b.allow()[0] is True
    assert b.allow()[0] is False


def test_limiter_keys_are_isolated():
    c = Clock(0.0)
    lim = RateLimiter(rate=1.0, burst=1, clock=c)
    assert lim.allow("ip-a")[0] is True
    assert lim.allow("ip-a")[0] is False   # a 用尽
    assert lim.allow("ip-b")[0] is True    # b 独立


def test_invalid_params_raise():
    import pytest

    with pytest.raises(ValueError):
        RateLimiter(rate=0, burst=5)
    with pytest.raises(ValueError):
        RateLimiter(rate=5, burst=0)
