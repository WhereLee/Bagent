"""P1 JWT identity：签名/过期/issuer/audience/scope。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException

from app import auth

SECRET = "s" * 32


def _settings(**kwargs):
    base = dict(auth_enabled=True, jwt_secret=SECRET, jwt_alg="HS256",
                jwt_issuer="issuer", jwt_audience="audience")
    base.update(kwargs)
    return SimpleNamespace(**base)


def _token(**overrides):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "u1",
        "scope": "agent:read agent:research",
        "iss": "issuer",
        "aud": "audience",
        "exp": now + timedelta(minutes=5),
    }
    payload.update(overrides)
    return jwt.encode(payload, SECRET, algorithm="HS256")


def test_auth_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _settings(auth_enabled=False))
    assert auth.get_principal(authorization=None) is None


def test_valid_token_principal(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    principal = auth.get_principal(authorization=f"Bearer {_token()}")
    assert principal.sub == "u1"
    assert principal.has_scope("agent:read")
    assert not principal.has_scope("agent:memory")


def test_missing_and_invalid_token(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    with pytest.raises(HTTPException) as missing:
        auth.get_principal(authorization=None)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        auth.get_principal(authorization="Bearer nope")
    assert invalid.value.status_code == 401


def test_expired_and_wrong_audience(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _settings())
    expired = _token(exp=datetime.now(timezone.utc) - timedelta(seconds=1))
    with pytest.raises(HTTPException) as err:
        auth.get_principal(authorization=f"Bearer {expired}")
    assert err.value.status_code == 401
    wrong_aud = jwt.encode({
        "sub": "u1", "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        "iss": "issuer", "aud": "other", "scope": "agent:read",
    }, SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as err2:
        auth.get_principal(authorization=f"Bearer {wrong_aud}")
    assert err2.value.status_code == 401
