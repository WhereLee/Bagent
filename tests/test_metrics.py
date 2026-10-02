"""检索指标的单元测试（含手算期望值）。"""
from app.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def test_recall_precision_perfect_ranking():
    ranked = [1, 2, 3, 4]
    rel = {1, 2}
    assert recall_at_k(ranked, rel, k=4) == 1.0
    assert precision_at_k(ranked, rel, k=4) == 0.5


def test_recall_partial():
    ranked = [1, 9, 9, 9]
    rel = {1, 2}
    assert recall_at_k(ranked, rel, k=4) == 0.5  # 只命中 1


def test_mrr_first_relevant_at_position_3():
    ranked = [7, 8, 5, 6]
    rel = {5}
    assert abs(mrr(ranked, rel) - 1 / 3) < 1e-9


def test_mrr_no_relevant_is_zero():
    assert mrr([1, 2, 3], {9}) == 0.0


def test_ndcg_full_when_relevant_at_top():
    ranked = [1, 2, 3]
    rel = {1, 2}
    assert abs(ndcg_at_k(ranked, rel, k=3) - 1.0) < 1e-9


def test_ndcg_penalizes_lower_rank():
    ranked_good = [1, 2, 3]
    ranked_bad = [3, 2, 1]
    rel = {1}
    assert ndcg_at_k(ranked_good, rel, k=3) > ndcg_at_k(ranked_bad, rel, k=3)


def test_empty_relevance_returns_zero():
    assert recall_at_k([1, 2], set(), k=2) == 0.0
