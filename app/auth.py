"""P1：最小 JWT 身份层（默认关闭，保持本地兼容）。

开启后从 Bearer token 取 sub/scope；body 里的 user_id 不再可信。
这只解决身份与 scope，不等于完整 RBAC/IdP；生产应接 OIDC/SSO。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException

from app.config import get_settings

_jwk_client = None
_jwk_url = ""


@dataclass(frozen=True)
class Principal:
    sub: str
    scopes: frozenset[str]

    def has_scope(self, scope: str | None) -> bool:
        return scope is None or "*" in self.scopes or scope in self.scopes


def _scopes_of(payload: dict[str, Any]) -> frozenset[str]:
    raw = payload.get("scope", payload.get("scopes", []))
    if isinstance(raw, str):
        values = raw.split()
    elif isinstance(raw, list):
        values = [str(x) for x in raw]
    else:
        values = []
    return frozenset(v for v in values if v)


def _decode(token: str, settings) -> dict:
    import jwt

    if settings.oidc_jwks_url:
        global _jwk_client, _jwk_url
        if _jwk_client is None or _jwk_url != settings.oidc_jwks_url:
            _jwk_client = jwt.PyJWKClient(settings.oidc_jwks_url)
            _jwk_url = settings.oidc_jwks_url
        signing_key = _jwk_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token, signing_key.key, algorithms=["RS256", "ES256"],
            audience=settings.jwt_audience or None,
            issuer=settings.jwt_issuer or settings.oidc_issuer or None,
            options={"require": ["exp", "sub"]},
        )
    return jwt.decode(
        token, settings.jwt_secret, algorithms=[settings.jwt_alg],
        audience=settings.jwt_audience or None,
        issuer=settings.jwt_issuer or None,
        options={"require": ["exp", "sub"]},
    )


def get_principal(authorization: str | None = Header(default=None)) -> Principal | None:
    """AUTH_ENABLED=false 返回 None；开启后无有效 Bearer 直接 401。"""
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    if not settings.jwt_secret and not settings.oidc_jwks_url:
        raise HTTPException(status_code=500, detail="auth_not_configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401, detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = _decode(token, settings)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=401, detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    sub = str(payload.get("sub") or "").strip()
    if not sub:
        raise HTTPException(status_code=401, detail="invalid bearer subject")
    return Principal(sub=sub, scopes=_scopes_of(payload))
