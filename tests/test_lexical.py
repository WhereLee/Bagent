"""BM25 词法检索的单元测试（jieba 分词，无需 DB/网络）。"""
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
