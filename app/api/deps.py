"""路由共享依赖：租户解析与联网开关。"""
from __future__ import annotations

from fastapi import HTTPException, Request

from app.config import get_settings
from app.tenant import TenantError, resolve_tenant


def tenant_of(request: Request) -> str | None:
    """从签名令牌解析当前租户；未启用强制时返回 None（单租户，不过滤）。"""
    try:
        return resolve_tenant(request.headers.get("X-Tenant"))
    except TenantError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e


def web_flag(use_web) -> bool:
    return get_settings().web_search_enabled if use_web is None else bool(use_web)
