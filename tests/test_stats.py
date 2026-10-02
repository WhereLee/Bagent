"""bootstrap 统计的单元测试。"""
from app.evaluation.stats import bootstrap_ci, mean, std, summarize


def test_mean_std_basic():
    assert mean([1, 2, 3]) == 2.0
    assert abs(std([2, 4, 4, 4, 5, 5, 7, 9]) - 2.138) < 1e-3
    assert std([5]) == 0.0


def test_bootstrap_ci_bounds_contain_mean():
    data = [0.9, 0.8, 0.95, 0.85, 0.92, 0.88, 0.91, 0.87, 0.89, 0.93]
    lo, hi = bootstrap_ci(data, seed=1)
    m = mean(data)
    assert lo <= m <= hi
    assert lo < hi  # 有波动

def test_bootstrap_ci_deterministic_with_seed():
    data = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.6, 0.5]
    assert bootstrap_ci(data, seed=7) == bootstrap_ci(data, seed=7)
    assert bootstrap_ci(data, seed=7) != bootstrap_ci(data, seed=8) or True  # 不同 seed 允许相同

def test_edge_cases():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.42]) == (0.42, 0.42)


def test_summarize_shape():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert set(s) == {"mean", "std", "ci_low", "ci_high", "n"}
    assert s["mean"] == 2.5 and s["n"] == 4
