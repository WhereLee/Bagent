"""共享工具注册表与 MCP server 的单元测试（hermetic，不触库/网络）。"""
import pytest

from app import tools
from app.retrieval.store import RetrievedChunk


def test_tools_registry_shape():
    assert set(tools.TOOLS) == {"kb_search", "kb_answer", "web_search", "research", "memory_search"}
    assert all(callable(f) for f in tools.TOOLS.values())


def test_web_search_tool_returns_json_safe(monkeypatch):
    from app.search import web_search as ws
    monkeypatch.setattr(ws, "web_search",
                        lambda q, max_results=None: [RetrievedChunk(
                            chunk_id=-1, document_id=0, source="https://x/1", content="c",
                            score=0.0, metadata={}, context="c")])
    out = tools.web_search("q")
    assert isinstance(out, list) and out[0]["source"] == "https://x/1"


def test_mcp_server_binds_all_tools():
    pytest.importorskip("mcp")
    from app.mcp_server import build_server
    import asyncio

    server = build_server()
    listed = asyncio.run(server.list_tools())
    names = {t.name for t in listed}
    assert set(tools.TOOLS) <= names


def test_build_tools_binds_tenant_and_user(monkeypatch):
    seen = {}

    def fake_retrieve(query, top_k=None, tenant=None, **kwargs):
        seen["tenant"] = tenant
        return []

    def fake_answer(query, top_k=None, user_id=None, tenant=None, **kwargs):
        from app.generation.generator import Answer

        seen["user_id"] = user_id
        seen["answer_tenant"] = tenant
        return Answer(query=query, text="ok", sources=[])

    monkeypatch.setattr("app.retrieval.retriever.retrieve", fake_retrieve)
    monkeypatch.setattr("app.generation.generator.answer_query", fake_answer)

    bound = tools.build_tools(tenant="T1", user_id="U1", compact=True)
    bound["kb_search"]("q", top_k=3)
    assert seen["tenant"] == "T1"

    assert bound["kb_answer"]("q")["text"] == "ok"
    assert seen["answer_tenant"] == "T1" and seen["user_id"] == "U1"

    desc = tools.describe_tools(bound)
    assert "kb_search" in desc and "memory_search" in desc
