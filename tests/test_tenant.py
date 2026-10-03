"""⑥ 多租户数据面强制的单元测试（hermetic）。"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import get_settings
from app import tenant as tn


def test_sign_verify_roundtrip():
    tok = tn.sign_tenant("acme", "secret")
    assert tn.verify_tenant(tok, "secret") == "acme"


def test_verify_rejects_tamper_and_bad_secret():
    tok = tn.sign_tenant("acme", "secret")
    assert tn.verify_tenant(tok, "other") is None          # 密钥不符
    assert tn.verify_tenant("acme|deadbeef", "secret") is None  # 签名伪造
    assert tn.verify_tenant("acme", "secret") is None       # 无签名
    assert tn.verify_tenant(None, "secret") is None


def test_resolve_off_returns_none(monkeypatch):
    s = get_settings().model_copy(update={"tenant_enforcement_enabled": False})
    monkeypatch.setattr(tn, "get_settings", lambda: s)
    assert tn.resolve_tenant(None) is None                  # 未启用→不过滤（兼容单租户）


def test_resolve_fail_closed(monkeypatch):
    s = get_settings().model_copy(update={"tenant_enforcement_enabled": True, "tenant_secret": "k"})
    monkeypatch.setattr(tn, "get_settings", lambda: s)
    with pytest.raises(tn.TenantError):                     # 启用但无令牌 → 拒
        tn.resolve_tenant(None)
    assert tn.resolve_tenant(tn.sign_tenant("t1", "k")) == "t1"


def test_api_query_rejects_missing_tenant(monkeypatch):
    s = get_settings().model_copy(update={"tenant_enforcement_enabled": True, "tenant_secret": "k"})
    monkeypatch.setattr(tn, "get_settings", lambda: s)
    client = TestClient(app)
    r = client.post("/query", json={"query": "x"})          # 无 X-Tenant
    assert r.status_code == 403                             # fail-closed，且不触库/LLM
    r2 = client.get("/health")                              # 健康检查豁免
    assert r2.status_code == 200
