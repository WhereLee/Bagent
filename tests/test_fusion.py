"""RRF 融合与检索指标的单元测试（纯逻辑）。"""
from app.retrieval.fusion import rrf_fuse


def test_rrf_single_list_preserves_order():
    fused = rrf_fuse([[10, 20, 30]])
    assert [d for d, _ in fused] == [10, 20, 30]


def test_rrf_favors_higher_rank_across_lists():
    # doc 7 在两路都排第一 -> 应综合最高
    fused = rrf_fuse([[7, 8, 9], [7, 5, 4]])
    ranked = [d for d, _ in fused]
    assert ranked[0] == 7


def test_rrf_score_formula_matches_definition():
    fused = dict(rrf_fuse([[1], [2]], weights=[1.0, 1.0], k=60))
    # doc1 在第一路 rank1 -> 1/(60+1)；doc2 在第二路 rank1 -> 1/61
    assert abs(fused[1] - 1 / 61) < 1e-9
    assert abs(fused[2] - 1 / 61) < 1e-9


def test_rrf_weight_changes_contribution():
    fused = dict(rrf_fuse([[1], [2]], weights=[2.0, 1.0], k=60))
    assert fused[1] > fused[2]


def test_rrf_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        rrf_fuse([[1], [2]], weights=[1.0])
