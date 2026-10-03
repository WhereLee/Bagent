"""MiMoWebProvider：把小米 MiMo「联网服务插件」当作一路受控检索源。

关键约束（成本护栏）：**只用于"取网页证据"，绝不在主生成 LLM 上挂 web_search**——按量模型联网会
几乎必然 cache-miss 导致成本暴涨。这里是一次**独立的、按次计费的检索调用**，由配置开关/降级触发，
取回 url_citation（+正文链接兜底）归一为 SearchResult，喂我们既有的 SSRF/清洗/多源管线。
"""
from __future__ import annotations

import re

from app.observability.logging import get_logger, log_event
from app.search.base import SearchResult

_log = get_logger("search.mimo")
_URL = re.compile(r"https?://[^\s)\"'<>，。]+")


class MiMoWebProvider:
    name = "mimo"

    def __init__(self, base_url: str, api_key: str, model: str, *,
                 timeout: float = 30.0, max_keyword: int = 3, client=None) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_keyword = max_keyword
        self._client = client  # 注入以便 hermetic 测试

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        import httpx

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是检索器。只用联网工具找到的网页作答，逐条给出标题、URL、摘要；不要臆测。"},
                {"role": "user", "content": query},
            ],
            "tools": [{"type": "web_search", "force_search": True, "max_keyword": self.max_keyword}],
            "temperature": 0.2, "stream": False,
        }
        client = self._client or httpx.Client(timeout=self.timeout, headers={"api-key": self.api_key})
        try:
            resp = client.post(self.url, json=body)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001  上游失败→降级为空
            log_event(_log, "warning", "mimo_web_error", err=str(e)[:160])
            return []
        return self._parse(data, max_results)

    @staticmethod
    def _parse(data: dict, max_results: int) -> list[SearchResult]:
        try:
            msg = (data.get("choices") or [{}])[0].get("message", {})
        except (IndexError, AttributeError):
            return []
        out: list[SearchResult] = []
        seen: set[str] = set()

        def add(title: str, url: str, snippet: str) -> None:
            if url and url not in seen and url.startswith("http"):
                seen.add(url)
                out.append(SearchResult(title=title or url[:60], url=url, snippet=snippet, engine="mimo-web"))

        for a in (msg.get("annotations") or []):
            if not isinstance(a, dict):
                continue
            c = a.get("url_citation") or a        # 兼容嵌套与平铺（MiMo 实为平铺）
            snippet = c.get("summary") or c.get("snippet") or c.get("content") or ""   # 真实字段是 summary
            add(str(c.get("title", "")), str(c.get("url", "")), str(snippet))

        if not out:                                # 无结构化标注→从正文兜底抽链接
            content = msg.get("content") or ""
            for line in content.splitlines():
                m = _URL.search(line)
                if m:
                    add(line[:60], m.group(0), line.strip())
        return out[:max_results]
