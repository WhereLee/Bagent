# 让 pytest 把仓库根加入 sys.path，使测试能 import app.*
# （内容留空，仅作为 rootdir 锚点）

import pytest


@pytest.fixture(autouse=True)
def _reset_inmemory_rate_limiter():
    """单测共享 app 的进程内令牌桶；每个测试前清空，避免请求数累积成 429。"""
    from app.api.main import app

    limiter = getattr(app.state, "rate_limiter", None)
    buckets = getattr(limiter, "_buckets", None)
    if buckets is not None:
        with limiter._lock:
            buckets.clear()
    yield
