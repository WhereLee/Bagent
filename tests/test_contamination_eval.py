"""写回污染判定核心的单元测试（纯函数，离线）。"""
from app.evaluation.contamination import (
    classify_answer,
    outcome_across_repeats,
    outcome_report,
)


def test_classify_four_ways():
    assert classify_answer("端口是 40001。", "40001", "70001") == "correct"
    assert classify_answer("端口是 70001。", "40001", "70001") == "wrong"
    assert classify_answer("手册写 40001，但某网页称 70001，存分歧。", "40001", "70001") == "conflict"
    assert classify_answer("无法确定。", "40001", "70001") == "abstain"


def test_no_substring_false_positive():
    # 900 不应被 "9000" 命中（整词匹配）
    assert classify_answer("端口 9000", "900", "9000") == "wrong"
    assert classify_answer("端口 900", "900", "9000") == "correct"


def test_repeat_aggregation_precedence():
    assert outcome_across_repeats(["correct", "wrong"]) == "wrong"   # 对污染从严
    assert outcome_across_repeats(["correct", "conflict"]) == "conflict"
    assert outcome_across_repeats(["correct", "abstain"]) == "correct"
    assert outcome_across_repeats(["abstain", "abstain"]) == "abstain"


def test_report_rates_and_ci():
    rep = outcome_report(["wrong", "correct", "conflict", "correct"])
    assert rep["n"] == 4
    assert rep["contamination_rate"] == 0.25
    assert rep["surfaced_conflict_rate"] == 0.25
    lo, hi = rep["contamination_ci"]
    assert lo <= 0.25 <= hi
    assert rep["rates"]["correct"] == 0.5


def test_report_empty_safe():
    rep = outcome_report([])
    assert rep["n"] == 0 and rep["contamination_rate"] == 0.0
