"""父子块切分的单元测试。"""
from app.ingestion.chunking import chunk_parent_child


def _sample(n_sections=6, per_section=30):
    return "\n\n".join(f"第{i}节内容。" + "详细说明句子。" * per_section for i in range(n_sections))


def test_each_parent_has_at_least_one_child():
    parents = chunk_parent_child(_sample(), parent_tokens=200, child_tokens=60, child_overlap=10)
    assert parents
    assert all(len(p.children) >= 1 for p in parents)


def test_child_token_within_budget_plus_overlap():
    pt, ct, ov = 200, 60, 10
    parents = chunk_parent_child(_sample(), parent_tokens=pt, child_tokens=ct, child_overlap=ov)
    # overlap 会把前块尾部拼上，允许 2*ov 的额外字符余量 + 估算误差
    bound = ct + 2 * ov + 3
    for p in parents:
        for c in p.children:
            assert c.token_count <= bound


def test_parent_within_budget():
    parents = chunk_parent_child(_sample(), parent_tokens=180, child_tokens=60, child_overlap=0)
    for p in parents:
        assert p.token_count <= 180 + 3


def test_empty_text():
    assert chunk_parent_child("   ") == []


def test_more_children_than_parents():
    parents = chunk_parent_child(_sample(n_sections=6, per_section=60), parent_tokens=400, child_tokens=80, child_overlap=0)
    total_children = sum(len(p.children) for p in parents)
    assert total_children >= len(parents)
