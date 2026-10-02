"""生成指标聚合的单元测试（纯逻辑）。"""
from app.evaluation.generation_metrics import (
    GenSample,
    avg_faithfulness,
    hallucination_rate,
    over_refusal_rate,
    refusal_accuracy,
)


def _s(q, exp_ref, pred_ref, f):
    return GenSample(question=q, expected_refusal=exp_ref, predicted_refusal=pred_ref, faithfulness=f)


def test_avg_faithfulness_ignores_refusals():
    samples = [_s("a", False, False, 1.0), _s("b", False, False, 0.5), _s("c", True, True, None)]
    assert avg_faithfulness(samples) == 0.75


def test_hallucination_rate_is_complement():
    samples = [_s("a", False, False, 0.6), _s("b", False, False, 0.8)]
    assert hallucination_rate(samples) == 0.3  # 1 - 0.7


def test_refusal_accuracy_counts_correct_predicted_refusals():
    samples = [
        _s("可答", False, False, 0.9),   # 正确（没误拒）
        _s("误拒", False, True, None),   # 错误
        _s("正确拒答", True, True, None),  # 正确
        _s("乱答不该拒", True, False, 0.2),  # 错误
    ]
    assert refusal_accuracy(samples) == 0.5


def test_over_refusal_rate_only_over_answerable():
    samples = [_s("a", False, True, None), _s("b", False, False, 0.9), _s("c", True, True, None)]
    # 可答题 2 条，其中 1 条被误拒
    assert over_refusal_rate(samples) == 0.5


def test_empty_samples():
    assert avg_faithfulness([]) == 0.0
    assert refusal_accuracy([]) == 0.0
