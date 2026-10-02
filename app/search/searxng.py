"""SearxNGProvider：调用自架 SearXNG 的 JSON API（默认百度系引擎）。

前置：SearXNG 实例已启用 `search.formats:[html,json]` 且开放 baidu 引擎（见 deploy/searxng）。
只做"取 SERP"，不抓全文（全文在 web_search 里由 trafilatura 处理）。网络失败返回空列表，交由上层降级。
"""
from __future__ import annotations

from app.observability.logging import get_logger, log_event
from app.search.base import SearchResult

_log = get_logger("search.searxng")


class SearxNGProvider:
    name = "searxng"

    def __init__(self, base_url: str, *, engines: str = "baidu", timeout: float = 8.0,
                 user_agent: str = "BagentRAG/0.1", client=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.engines = engines
        self.timeout = timeout
        self.user_agent = user_agent
        self._client = client  # 允许注入（测试用）；否则懒建

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        import httpx

        client = self._client or httpx.Client(timeout=self.timeout, headers={"User-Agent": self.user_agent})
        params = {
            "q": query,
            "format": "json",
            "engines": self.engines,
            "safesearch": "2",
            "language": "zh-CN",
        }
        try:
            resp = client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001  上游不稳，降级为空
            log_event(_log, "warning", "searxng_error", err=str(e)[:160])
            return []
        return self._parse(data, max_results)

    @staticmethod
    def _parse(data: dict, max_results: int) -> list[SearchResult]:
        out: list[SearchResult] = []
        for r in (data or {}).get("results", [])[:max_results]:
            url = r.get("url") or r.get("parsed_url")
            if not url:
                continue
            out.append(SearchResult(
                title=r.get("title", ""),
                url=url,
                snippet=r.get("content", "") or "",
                published=r.get("publishedDate") or r.get("date"),
                engine=r.get("engine", "searxng"),
            ))
        return out
