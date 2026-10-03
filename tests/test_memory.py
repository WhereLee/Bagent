"""M9b 记忆：生命周期纯规则 + 抽取解析 的单元测试。"""
from datetime import datetime, timedelta, timezone

import pytest

from app.memory.extract import parse_facts
from app.memory.lifecycle import (
    decide_op,
    is_active,
    promote_trust,
    trust_ok,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_trust_gate():
    assert trust_ok("verified", "verified") is True
    assert trust_ok("curated", "verified") is True
    assert trust_ok("draft", "verified") is False      # draft 不作答
    assert trust_ok("draft", "draft") is True


def test_is_active_window():
    assert is_active(None, None, NOW) is True
    assert is_active(NOW - timedelta(1), None, NOW) is True          # 已到生效
    assert is_active(NOW + timedelta(1), None, NOW) is False         # 未来才生效
    assert is_active(None, NOW + timedelta(1), NOW) is True          # 未失效
    assert is_active(None, NOW - timedelta(1), NOW) is False         # 已过期


def test_promote_trust():
    assert promote_trust(1, 2) == "draft"
    assert promote_trust(2, 2) == "verified"
    assert promote_trust(9, 2, "curated") == "curated"    # 不回退


def test_decide_op():
    assert decide_op([], 0.92) == ("ADD", None)
    assert decide_op([(5, 0.98)], 0.92) == ("UPDATE", 5)
    assert decide_op([(5, 0.7)], 0.92) == ("ADD", None)          # 不够像 -> 新增
    assert decide_op([(5, 0.98)], 0.92, exact_hash_match=True) == ("NOOP", None)


@pytest.mark.parametrize("raw,expect", [
    ('[{"content":"喜欢靠窗","scope":"personal","kind":"preference"}]', 1),
    ('[]', 0),
    ('not json', 0),
    ('[{"content":"","scope":"personal"}]', 0),                       # 空 content 丢弃
    ('[{"content":"x","scope":"bogus"}]', 0),                         # 非法 scope 丢弃
    ('前言 [{"content":"秦惠文王用张仪","scope":"knowledge"}] 后语', 1),  # 混排文本能抠出数组
])
def test_parse_facts(raw, expect):
    assert len(parse_facts(raw)) == expect
