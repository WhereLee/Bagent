"""租户签名令牌（数据面强制的信任根）。

Java 管理面签发 `tenant|<hmac>` 令牌；Bagent 用共享密钥验签后推导过滤器。
**绝不信任裸传的 X-Tenant-Id**——必须验签，且 fail-closed（验不过就拒绝，不放行全量）。
"""
from __future__ import annotations

import hashlib
import hmac

from app.config import get_settings


def sign_tenant(tenant: str, secret: str) -> str:
    sig = hmac.new(secret.encode("utf-8"), tenant.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{tenant}|{sig}"


def verify_tenant(token: str | None, secret: str) -> str | None:
    """校验令牌，返回 tenant_id；无效/缺失返回 None（调用方据此 fail-closed）。"""
    if not token or not secret or "|" not in token:
        return None
    tenant, _, sig = token.rpartition("|")
    if not tenant:
        return None
    expected = hmac.new(secret.encode("utf-8"), tenant.encode("utf-8"), hashlib.sha256).hexdigest()
    return tenant if hmac.compare_digest(sig, expected) else None


def resolve_tenant(header_value: str | None) -> str | None:
    """按当前配置解析请求租户：
    - 未启用强制（单租户/开发）→ None（不过滤，兼容既有行为）。
    - 启用 → 必须验签通过，否则抛 NotAuthenticated（由调用端点转 403）。
    """
    s = get_settings()
    if not s.tenant_enforcement_enabled:
        return None
    tenant = verify_tenant(header_value, s.tenant_secret)
    if tenant is None:
        raise TenantError("missing or invalid tenant token")
    return tenant


class TenantError(Exception):
    """租户鉴权失败（端点转 403，fail-closed）。"""
