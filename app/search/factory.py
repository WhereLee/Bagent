"""按配置返回 SearchProvider 单例（searxng | mock）。将来可加 baidu_api 等。"""
from __future__ import annotations

from app.config import get_settings
from app.search.base import SearchProvider

_provider: SearchProvider | None = None


def get_search_provider() -> SearchProvider:
    global _provider
    if _provider is not None:
        return _provider
    s = get_settings()
    if s.search_provider == "searxng":
        from app.search.searxng import SearxNGProvider
        _provider = SearxNGProvider(
            s.searxng_base_url, engines=s.searxng_engines,
            timeout=s.search_timeout, user_agent=s.search_user_agent,
        )
    else:
        from app.search.mock import MockProvider
        _provider = MockProvider()
    return _provider


def reset_search_provider() -> None:
    """测试用：丢弃单例。"""
    global _provider
    _provider = None
