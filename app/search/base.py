"""联网搜索的抽象层：SearchResult 与 SearchProvider 协议。

设计：把"哪家搜索 API"隔离在实现外（SearXNG/百度托管/mock 都实现同一接口），
上层只认 SearchResult -> 经清洗/抽取后归一为 RetrievedChunk 进多源佐证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    published: str | None = None   # 引擎给出的时间（若有），ISO 或原文
    engine: str = ""               # 来源引擎/供应商标记
    extra: dict = field(default_factory=dict)


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """返回按相关性排序的原始搜索结果（标题/URL/摘要/时间）。不做全文抓取。"""
        ...
