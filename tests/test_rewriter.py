"""多轮查询改写的单元测试（假 LLM，无网络）。"""
from app.generation.rewriter import format_history, rewrite_query


class Echo:
    def __init__(self, out):
        self._out = out

    def complete(self, system, user):
        return self._out


def test_format_history_orders_and_labels():
    hist = [{"role": "user", "content": "XG-200端口?"}, {"role": "assistant", "content": "8443"}]
    txt = format_history(hist)
    assert "用户: XG-200端口?" in txt
    assert "助手: 8443" in txt


def test_no_history_returns_original():
    assert rewrite_query("它多少钱", [], Echo("改写")) == "它多少钱"


def test_rewrite_uses_llm():
    hist = [{"role": "user", "content": "XG-400 Pro"}]
    assert rewrite_query("它的端口呢", hist, Echo("XG-400 Pro 的端口")) == "XG-400 Pro 的端口"


def test_empty_llm_output_falls_back_to_original():
    hist = [{"role": "user", "content": "x"}]
    assert rewrite_query("原始问题", hist, Echo("   ")) == "原始问题"
