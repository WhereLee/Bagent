"""按配置返回 SearchProvider（searxng | mimo | mock）+ 可选降级源。"""
from __future__ import annotations

from app.config import get_settings
from app.search.base import SearchProvider

_providers: dict[str, SearchProvider] = {}


def build(name: str) -> SearchProvider | None:
    """按名构造 provider（带缓存）。未知名返回 None。"""
    if not name:
        return None
    if name in _providers:
        return _providers[name]
    s = get_settings()
    p: SearchProvider | None
    if name == "searxng":
        from app.search.searxng import SearxNGProvider
        p = SearxNGProvider(s.searxng_base_url, engines=s.searxng_engines,
                            timeout=s.search_timeout, user_agent=s.search_user_agent)
    elif name == "mimo":
        from app.search.mimo import MiMoWebProvider
        p = MiMoWebProvider(s.mimo_base_url, s.mimo_api_key, s.mimo_model,
                            timeout=s.search_timeout, max_keyword=s.mimo_max_keyword)
    elif name == "mock":
        from app.search.mock import MockProvider
        p = MockProvider()
    else:
        return None
    _providers[name] = p
    return p


def get_search_provider() -> SearchProvider:
    return build(get_settings().search_provider) or build("mock")  # type: ignore[return-value]


def get_fallback_provider() -> SearchProvider | None:
    fb = get_settings().search_fallback
    return build(fb) if fb and fb != get_settings().search_provider else None


def reset_search_provider() -> None:
    """测试用：丢弃所有缓存的 provider。"""
    _providers.clear()
