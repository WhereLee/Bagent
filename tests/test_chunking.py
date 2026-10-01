"""chunking 单元测试（纯逻辑，无需外部服务）。"""
from app.ingestion.chunking import _approx_tokens, _force_split, chunk_text


def test_empty_returns_nothing():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_indices_are_contiguous():
    text = "\n\n".join(f"这是第 {i} 段，内容若干。" * 5 for i in range(8))
    chunks = chunk_text(text, chunk_tokens=40, overlap_tokens=0)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_respects_token_budget_without_overlap():
    # overlap=0 时，任何块的 token 数不应超过预算（含少量估算误差 +2）
    text = "星链网关支持多种接口。" * 300
    budget = 80
    chunks = chunk_text(text, chunk_tokens=budget, overlap_tokens=0)
    assert chunks, "应至少切出一个块"
    assert all(c.token_count <= budget + 2 for c in chunks)


def test_force_split_guarantees_bound():
    parts = _force_split("一" * 100, max_tokens=10)
    assert all(len(p) <= 10 for p in parts)
    assert sum(len(p) for p in parts) == 100  # 不丢字符


def test_overlap_prepends_previous_tail():
    text = "\n\n".join(f"段落{i}：" + "内容" * 30 for i in range(4))
    no_ov = chunk_text(text, chunk_tokens=40, overlap_tokens=0)
    with_ov = chunk_text(text, chunk_tokens=40, overlap_tokens=10)
    # 第一块不受 overlap 影响
    assert no_ov[0].text == with_ov[0].text
    # 后续块因重叠而变长或相等，且块数一致
    assert len(no_ov) == len(with_ov)
    assert all(w.token_count >= n.token_count for n, w in zip(no_ov, with_ov))


def test_token_counting_cjk_vs_latin():
    assert _approx_tokens("中文字") == 4      # 3 个 CJK + 1
    assert _approx_tokens("") == 0
    # 4 个拉丁字符约等于 1 token
    assert _approx_tokens("abcd") == 2        # int(4*0.25)+1 = 2
