"""联网搜索层单元测试：SearXNG 解析 + 假 httpx client + web_search 编排（全 mock，不联网）。"""
from types import SimpleNamespace

from app.search.base import SearchResult
from app.search.mock import MockProvider
from app.search import web_search as ws
from app.search.searxng import SearxNGProvider


# ---- SearXNG 解析 ----
def test_searxng_parse_maps_fields():
    data = {"results": [
        {"title": "T1", "url": "https://a.com/1", "content": "c1", "publishedDate": "2024-01-01", "engine": "baidu"},
        {"title": "no-url"},  # 无 url 跳过
    ]}
    res = SearxNGProvider._parse(data, max_results=5)
    assert len(res) == 1
    assert res[0].url == "https://a.com/1" and res[0].engine == "baidu"


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None):
        self.calls.append((url, params))
        return _FakeResp(self.payload)


def test_searxng_search_uses_injected_client_and_json_params():
    client = _FakeClient({"results": [{"title": "t", "url": "https://x/1", "content": "s"}]})
    p = SearxNGProvider("http://searx:8080", engines="baidu", client=client)
    out = p.search("北京", max_results=3)
    assert out and out[0].url == "https://x/1"
    url, params = client.calls[0]
    assert url == "http://searx:8080/search"
    assert params["format"] == "json" and params["engines"] == "baidu"


def test_searxng_error_degrades_empty():
    class Boom:
        def get(self, *a, **k):
            raise RuntimeError("down")
    p = SearxNGProvider("http://searx", client=Boom())
    assert p.search("q") == []


# ---- web_search 编排 ----
def _fake_settings():
    return SimpleNamespace(
        web_search_enabled=True, web_search_max_results=5, web_fetch_fulltext=False,
        web_content_max_chars=4000, search_min_interval=0.0, search_timeout=8.0, search_user_agent="t",
    )


def test_web_search_pipeline_with_mock(monkeypatch):
    monkeypatch.setattr(ws, "get_settings", _fake_settings)
    monkeypatch.setattr(ws, "get_search_provider", lambda: MockProvider())
    monkeypatch.setattr(ws, "is_safe_url", lambda u, **k: True)  # DNS/SSRF 由 test_web_clean 覆盖

    chunks = ws.web_search("秦惠文王")
    assert chunks, "应产出证据块"
    c = chunks[0]
    assert c.metadata["source_type"] == "web"
    assert c.metadata["trust"] == "draft"
    assert c.chunk_id < 0                      # 与 DB 块区分的负 id
    assert c.context == c.content              # 喂给 LLM 的即清洗后文本


def test_web_search_disabled_returns_empty(monkeypatch):
    s = _fake_settings()
    s.web_search_enabled = False
    monkeypatch.setattr(ws, "get_settings", lambda: s)
    assert ws.web_search("任意") == []
