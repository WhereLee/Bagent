"""MockProvider：确定性假搜索结果，用于离线开发与 hermetic 测试（不联网）。"""
from __future__ import annotations

from app.search.base import SearchResult

_SAMPLE = {
    "秦惠文王": [
        SearchResult("秦惠文王_百度百科", "https://baike.baidu.com/item/秦惠文王",
                     "秦惠文王嬴驷，秦国第二位王，任用张仪连横，灭巴蜀。", "2024-01-01", "mock"),
        SearchResult("惠文王与张仪", "https://example.org/huiwen-zhangyi",
                     "张仪以连横之说游说六国，惠文王采纳，奠定秦并巴蜀之基。", "2023-06-01", "mock"),
    ],
}


class MockProvider:
    name = "mock"

    def __init__(self, fixed: dict[str, list[SearchResult]] | None = None) -> None:
        self._fixed = fixed or _SAMPLE

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        for key, res in self._fixed.items():
            if key in query or query in key:
                return res[:max_results]
        return [SearchResult(f"mock result {i}", f"https://example.com/{i}", f"snippet {i}", engine="mock")
                for i in range(min(max_results, 3))]
