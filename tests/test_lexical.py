"""BM25 词法检索的单元测试（jieba 分词，无需 DB/网络）。"""
import math
import random
from collections import Counter

from app.retrieval.lexical import BM25Index, tokenize


def test_tokenize_filters_punctuation_and_blanks():
    toks = tokenize("你好，世界。 abc 123")
    assert "你好" in toks and "世界" in toks
    assert "，" not in toks and "。" not in toks
    assert all(t.strip() for t in toks)
    # 拉丁词转小写
    assert "abc" in toks


def test_bm25_ranks_doc_with_matching_terms_first():
    docs = [
        (1, tokenize("星链网关支持千兆以太网口和光口")),
        (2, tokenize("电源适配器质保一年")),
        (3, tokenize("网关固件升级说明文档")),
    ]
    idx = BM25Index()
    idx.build(docs)
    result = idx.search(tokenize("网关 以太网"), top_k=3)
    ids = [d for d, _ in result]
    assert ids[0] == 1  # 同时含"网关"与"以太网"


def test_bm25_excludes_non_matching_docs():
    idx = BM25Index()
    idx.build([(1, tokenize("apple pie")), (2, tokenize("banana bread"))])
    result = idx.search(tokenize("apple"), top_k=5)
    assert [d for d, _ in result] == [1]


def test_bm25_empty_index_or_query():
    idx = BM25Index()
    assert idx.search(tokenize("任意"), top_k=3) == []
    idx.build([(1, tokenize("hello"))])
    assert idx.search([], top_k=3) == []


def test_bm25_tenant_filter_keeps_only_target_tenant():
    idx = BM25Index()
    idx.build([(1, ["网关"], "t1"), (2, ["网关"], "t2"), (3, ["网关"])])
    assert [d for d, _ in idx.search(["网关"], top_k=5, tenant="t1")] == [1]
    assert [d for d, _ in idx.search(["网关"], top_k=5, tenant="t2")] == [2]
    assert {d for d, _ in idx.search(["网关"], top_k=5)} == {1, 2, 3}


def _brute_force(docs, query_tokens, top_k, k1=1.5, b=0.75):
    """测试内参照实现：与旧版一致的逐查询词全库扫描。"""
    n = len(docs)
    doc_ids = [d[0] for d in docs]
    doc_tf = [Counter(d[1]) for d in docs]
    doc_len = [len(d[1]) for d in docs]
    avgdl = (sum(doc_len) / n) if n else 0.0
    df: dict[str, int] = {}
    for tf in doc_tf:
        for term in tf:
            df[term] = df.get(term, 0) + 1
    scores = [0.0] * n
    for term in set(query_tokens):
        idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
        for i, tf in enumerate(doc_tf):
            f = tf.get(term, 0)
            if not f:
                continue
            norm = k1 * (1 - b + b * doc_len[i] / avgdl)
            scores[i] += idf * (f * (k1 + 1)) / (f + norm)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        if scores[i] <= 0 or len(out) >= top_k:
            break
        out.append((doc_ids[i], scores[i]))
    return out


def test_bm25_inverted_index_matches_bruteforce():
    rnd = random.Random(2026)
    vocab = [f"w{i}" for i in range(80)]
    docs = []
    for doc_id in range(120):
        toks = [rnd.choice(vocab) for _ in range(rnd.randint(5, 20))]
        docs.append((doc_id, toks))
    idx = BM25Index()
    idx.build(docs)
    for _ in range(30):
        query = [rnd.choice(vocab) for _ in range(rnd.randint(1, 4))]
        expected = _brute_force(docs, query, top_k=10)
        actual = idx.search(query, top_k=10)
        assert [d for d, _ in actual] == [d for d, _ in expected]
        for (_, a), (_, e) in zip(actual, expected):
            assert abs(a - e) < 1e-9
