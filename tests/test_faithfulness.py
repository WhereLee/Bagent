"""Faithfulness 纯逻辑单元测试（拆句/判定/聚合），用假 LLM 避免网络。"""
from app.generation.faithfulness import (
    REFUSAL_PHRASE,
    assess_faithfulness,
    parse_verdict,
    split_sentences,
)


class FakeJudge:
    """按关键词返回判定，模拟 MiMo judge。"""

    def __init__(self, unsupported_keywords=()):
        self.unsupported = unsupported_keywords

    def complete(self, system, user):
        claim = user.split("【断言】")[-1]
        return "NOT_SUPPORTED" if any(k in claim for k in self.unsupported) else "SUPPORTED"


def test_split_sentences_strips_citations_and_splits():
    sents = split_sentences("端口是8443[1]。质保三年[2][3]。支持PoE")
    assert sents == ["端口是8443", "质保三年", "支持PoE"]


def test_parse_verdict_not_supported_takes_precedence():
    assert parse_verdict("NOT_SUPPORTED") is False
    assert parse_verdict("SUPPORTED") is True
    assert parse_verdict("  supported. ") is True
    assert parse_verdict("") is False      # 保守判否
    assert parse_verdict("maybe") is False


def test_assess_all_supported():
    rep = assess_faithfulness("上下文", "端口8443。质保3年。", FakeJudge())
    assert rep["faithfulness"] == 1.0
    assert rep["hallucination_rate"] == 0.0
    assert rep["grounded"] is True
    assert rep["total_claims"] == 2


def test_assess_partial_unsupported():
    rep = assess_faithfulness("上下文", "端口8443。天空是绿色的。", FakeJudge(unsupported_keywords=("天空",)))
    assert rep["total_claims"] == 2
    assert rep["supported"] == 1
    assert rep["faithfulness"] == 0.5
    assert rep["grounded"] is False  # 0.5 < 0.6 阈值


def test_assess_refusal_is_not_scored():
    rep = assess_faithfulness("上下文", REFUSAL_PHRASE, FakeJudge())
    assert rep["is_refusal"] is True
    assert rep["faithfulness"] is None
    assert rep["grounded"] is False
