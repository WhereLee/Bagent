"""成本预算/降级纯逻辑测试（离线）。"""
from app.generation.budget import estimate_tokens, should_degrade


def test_no_budget_never_degrades():
    assert should_degrade(20, 0) is False
    assert should_degrade(20, -1) is False


def test_over_budget_degrades():
    # 小预算 + 大 top_k → 超预算降级
    assert should_degrade(20, 1000) is True
    # 大预算 → 不降级
    assert should_degrade(5, 100000) is False


def test_estimate_grows_with_top_k_and_context():
    assert estimate_tokens(10, 500) > estimate_tokens(5, 500)
    assert estimate_tokens(5, 1000) > estimate_tokens(5, 500)
