"""路由共享依赖：联网开关。"""
from __future__ import annotations

from app.config import get_settings


def web_flag(use_web) -> bool:
    return get_settings().web_search_enabled if use_web is None else bool(use_web)
