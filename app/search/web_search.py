"""联网检索编排：SearchProvider -> SSRF 过滤 -> (可选)全文抽取 -> 安全清洗 -> RetrievedChunk。

产出的是**本次查询可用的证据块**（trust=draft, 带 url/时间），供 self-RAG 多源佐证；
持久化写回是 M9b 的事，这里不落库。含最小间隔节流，护上游（百度会限流/封 IP）。
"""
from __future__ import annotations

import threading
import time

from app.config import get_settings
from app.observability.logging import get_logger, log_event
from app.observability.metrics import WEB_SEARCH
from app.retrieval.store import RetrievedChunk
from app.search.clean import clean_web_text, is_safe_url
from app.search.extract import extract
from app.search.factory import get_search_provider

_log = get_logger("search")

_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    min_interval = get_settings().search_min_interval
    with _lock:
        wait = _last_call + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def web_search(query: str, max_results: int | None = None) -> list[RetrievedChunk]:
    s = get_settings()
    if not s.web_search_enabled:
        return []
    provider = get_search_provider()
    k = max_results or s.web_search_max_results
    _throttle()
    try:
        results = provider.search(query, max_results=k)
    except Exception as e:  # noqa: BLE001
        WEB_SEARCH.labels(provider=provider.name, outcome="error").inc()
        log_event(_log, "warning", "web_search_error", err=str(e)[:160])
        return []

    chunks: list[RetrievedChunk] = []
    for i, r in enumerate(results):
        if not is_safe_url(r.url):
            log_event(_log, "debug", "blocked_unsafe_url", url=r.url[:120])
            continue
        text, published = r.snippet, r.published
        if s.web_fetch_fulltext:
            ex = extract(r.url, timeout=s.search_timeout, user_agent=s.search_user_agent)
            if ex and ex.text:
                text, published = ex.text, (ex.published or r.published)
        safe = clean_web_text(text, s.web_content_max_chars)
        if not safe:
            continue
        chunks.append(RetrievedChunk(
            chunk_id=-(i + 1),                 # 负 id，与 DB 块区分，仅供本轮上下文
            document_id=0, source=r.url, content=safe, score=0.0,
            metadata={"source_type": "web", "url": r.url, "published": published,
                      "trust": "draft", "engine": r.engine, "title": r.title},
            context=safe,
        ))
    WEB_SEARCH.labels(provider=provider.name, outcome="ok" if chunks else "empty").inc()
    log_event(_log, "info", "web_search", provider=provider.name, q=query[:60], n_results=len(results), n_kept=len(chunks))
    return chunks
