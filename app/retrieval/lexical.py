"""词法（稀疏）检索腿：真实 Okapi BM25 + jieba 中文分词。

架构约束（务必知悉）：
- 本实现的 BM25 是"真算法"（k1/b/IDF 齐全），但**索引载体是进程内存**——
  语料从 DB 载入后建内存索引。原因：目标环境 Windows + PostgreSQL 18 无法便捷安装
  pg_search(Rust, BM25) 扩展。
- 一致性：用 (chunk 数, max(chunk id)) 作为 generation 标记，检测到变化即重建。
- 规模化迁移路径：语料增大到内存索引不合适时，切换到 pg_search / Elasticsearch 做词法腿，
  本模块的对外接口 search(query, top_k) 保持不变即可替换。
"""
from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from threading import Lock

import jieba
from sqlalchemy import func, select

from app.db.models import Chunk, Document
from app.db.session import get_session
from app.observability.logging import get_logger, log_event

_log = get_logger("lexical")

# 常见中文标点/空白，分词后过滤
_STOP_CHARS = set("，。、；：？！“”‘’（）《》〈〉【】…—－-_,.;:?!'\"()<>[]{}\n\r\t /\\")


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 过滤标点/空白/单字符噪声，转小写。"""
    toks: list[str] = []
    for raw in jieba.lcut(text):
        t = raw.strip().lower()
        if not t or t in _STOP_CHARS:
            continue
        if all(c in _STOP_CHARS for c in t):
            continue
        toks.append(t)
    return toks


class BM25Index:
    """纯内存 BM25 倒排索引：build 一次，query 多次。

    postings 让查询只遍历"命中查询词"的文档，而不是每次都对全库打分；
    索引仍在进程内存中，规模上限与迁移路径见模块文档。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[int] = []
        self.doc_tenant: list[str | None] = []
        # term -> [(doc_idx, tf), ...]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}
        self.avgdl: float = 0.0
        self.n: int = 0

    def build(self, docs: list[tuple]) -> None:
        # 兼容 (id, toks) 与 (id, toks, tenant)；tenant 缺省为 None
        doc_ids = [d[0] for d in docs]
        doc_tenant = [d[2] if len(d) > 2 else None for d in docs]
        doc_len = [len(d[1]) for d in docs]

        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d[1]).items():
                postings[term].append((i, tf))

        self.doc_ids = doc_ids
        self.doc_tenant = doc_tenant
        self.doc_len = doc_len
        self.postings = dict(postings)
        self.df = {term: len(items) for term, items in self.postings.items()}
        self.n = len(docs)
        self.avgdl = (sum(doc_len) / self.n) if self.n else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # BM25+ 风格保证 IDF 恒正
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int, tenant: str | None = None) -> list[tuple[int, float]]:
        if self.n == 0 or not query_tokens or top_k <= 0:
            return []
        scores: dict[int, float] = {}
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for i, f in postings:
                if tenant is not None and self.doc_tenant[i] != tenant:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1)) / (f + norm)
        if not scores:
            return []
        # 分数降序；同分时 doc_idx 小者优先，与旧版全量稳定排序一致
        best = heapq.nlargest(top_k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [(self.doc_ids[i], score) for i, score in best]


class LexicalRetriever:
    """带 generation 检测的 BM25 检索器（进程内单例）。"""

    def __init__(self) -> None:
        self._index = BM25Index()
        self._generation = (-1, -1, -1.0)
        self._lock = Lock()

    def _current_generation(self, session):
        """活跃(未软删)子块的版本指纹：数量 + max(id) + max(updated_at epoch)。

        软删除会改变活跃子块数与 documents.updated_at，从而触发重建。
        """
        row = session.execute(
            select(
                func.count(Chunk.id),
                func.coalesce(func.max(Chunk.id), 0),
                func.coalesce(func.max(func.extract("epoch", Document.updated_at)), 0.0),
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.embedding.isnot(None), Document.is_deleted == False)  # noqa: E712
        ).one()
        return int(row[0]), int(row[1]), float(row[2])

    def _ensure_index(self, session) -> None:
        gen = self._current_generation(session)
        if gen == self._generation and self._index.n:
            return
        with self._lock:
            if self._current_generation(session) == self._generation and self._index.n:
                return
            # 只把未软删文档的子块纳入 BM25 索引（修复：旧版会召回已删文档）
            rows = session.execute(
                select(Chunk.id, Chunk.content, Document.metadata_)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.embedding.isnot(None), Document.is_deleted == False)  # noqa: E712
            ).all()
            docs = [(r.id, tokenize(r.content), (r.metadata_ or {}).get("tenant")) for r in rows]
            # 先构建新索引再一次性替换，避免并发搜索看到半构建状态
            new_index = BM25Index()
            new_index.build(docs)
            self._index = new_index
            self._generation = gen
            log_event(_log, "info", "bm25_rebuild", n_docs=self._index.n, generation=str(gen))

    def search(self, query: str, top_k: int, tenant: str | None = None) -> list[tuple[int, float]]:
        session = get_session()
        try:
            self._ensure_index(session)
            index = self._index   # 取本地引用：即使并发重建，本次查询也用完整快照
            return index.search(tokenize(query), top_k, tenant)
        finally:
            session.close()


_lexical: LexicalRetriever | None = None


def get_lexical_retriever() -> LexicalRetriever:
    global _lexical
    if _lexical is None:
        _lexical = LexicalRetriever()
    return _lexical
