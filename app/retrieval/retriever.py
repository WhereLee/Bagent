"""检索编排：dense / hybrid(dense+BM25, RRF) / (+rerank) / 父子扩展。

同一套接口通过 mode/rerank 参数切换，便于对 golden set 做消融对比。
"""
from __future__ import annotations

from app.config import get_settings
from app.db.session import get_session
from app.ingestion.embedder import get_embedder
from app.observability.logging import get_logger, log_event
from app.observability.metrics import RETRIEVAL_CACHE, STAGE_LATENCY
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.fusion import rrf_fuse
from app.retrieval.lexical import get_lexical_retriever
from app.retrieval.rerank import get_reranker
from app.retrieval.store import (
    RetrievedChunk,
    expand_to_parents,
    load_children,
    vector_search,
)

_log = get_logger("retrieval")


def retrieve(
    query: str,
    top_k: int | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    candidate_n: int | None = None,
) -> list[RetrievedChunk]:
    s = get_settings()
    mode = mode or s.retrieval_mode
    do_rerank = s.rerank_enabled if rerank is None else rerank
    top_k = top_k or s.retrieval_top_k
    cand_n = candidate_n or s.retrieval_candidate_n

    # 缓存：命中则跳过 embed+检索+rerank（key 含全部影响结果的参数）
    cache = get_retrieval_cache() if s.retrieval_cache_enabled else None
    key: tuple | None = None
    if cache is not None:
        key = (query, mode, do_rerank, cand_n, top_k, s.retrieve_parent)
        cached = cache.get(key)
        if cached is not None:
            RETRIEVAL_CACHE.labels(result="hit").inc()
            return cached
        RETRIEVAL_CACHE.labels(result="miss").inc()

    qvec = get_embedder().encode_query(query)

    session = get_session()
    with STAGE_LATENCY.labels(stage="retrieval").time():
        try:
            dense_hits = vector_search(session, qvec, cand_n)
            dense_ids = [h.chunk_id for h in dense_hits]

            if mode == "hybrid":
                lexical_ids = [cid for cid, _ in get_lexical_retriever().search(query, cand_n)]
                fused = rrf_fuse([dense_ids, lexical_ids], k=s.rrf_k)
                ordered_ids = [cid for cid, _ in fused]
            else:  # dense
                ordered_ids = dense_ids

            pool = load_children(session, ordered_ids)
            candidates = [pool[cid] for cid in ordered_ids if cid in pool]

            # 自适应早退：dense 首分已足够高（置信充分）则跳过 rerank 降延迟
            if (do_rerank and s.rerank_skip_threshold is not None
                    and dense_hits and dense_hits[0].score >= s.rerank_skip_threshold):
                do_rerank = False
                log_event(_log, "info", "rerank_skipped", top1=round(dense_hits[0].score, 4),
                          threshold=s.rerank_skip_threshold)

            if do_rerank and candidates:
                # 只对粗召回的前 candidate_n 个精排（Cross-Encoder 贵，限制精排面）
                to_rerank = candidates[:cand_n]
                with STAGE_LATENCY.labels(stage="rerank").time():
                    scores = get_reranker().rerank(query, [c.content for c in to_rerank])
                for c, sc in zip(to_rerank, scores):
                    c.score = sc
                to_rerank.sort(key=lambda c: c.score, reverse=True)
                candidates = to_rerank

            top = candidates[:top_k]
            if s.retrieve_parent:
                expand_to_parents(session, top)
            log_event(
                _log, "info", "retrieval",
                mode=mode, rerank=do_rerank, candidate_n=cand_n, top_k=top_k,
                n_hits=len(top), top_score=round(top[0].score, 4) if top else None,
            )
            if cache is not None and key is not None:
                cache.put(key, top)
            return top
        finally:
            session.close()
