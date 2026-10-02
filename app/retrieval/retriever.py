"""检索编排：dense / hybrid(dense+BM25, RRF) / (+rerank) / 父子扩展。

同一套接口通过 mode/rerank 参数切换，便于对 golden set 做消融对比。
"""
from __future__ import annotations

from app.config import get_settings
from app.db.session import get_session
from app.ingestion.embedder import get_embedder
from app.observability.metrics import STAGE_LATENCY
from app.retrieval.fusion import rrf_fuse
from app.retrieval.lexical import get_lexical_retriever
from app.retrieval.rerank import get_reranker
from app.retrieval.store import (
    RetrievedChunk,
    expand_to_parents,
    load_children,
    vector_search,
)


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

            if do_rerank and candidates:
                with STAGE_LATENCY.labels(stage="rerank").time():
                    scores = get_reranker().rerank(query, [c.content for c in candidates])
                for c, sc in zip(candidates, scores):
                    c.score = sc
                candidates.sort(key=lambda c: c.score, reverse=True)

            top = candidates[:top_k]
            if s.retrieve_parent:
                expand_to_parents(session, top)
            return top
        finally:
            session.close()
