"""P2-3：MCP gateway HTTP 入口。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import mcp_gateway as route
from app.mcp_gateway import MCPRemoteTool


class FakeGateway:
    def list_tools(self):
        return [MCPRemoteTool("fs", "read_file", "read")]

    def call_tool(self, ref, arguments):
        if ref != "mcp__fs__read_file":
            raise PermissionError("blocked")
        return {"ref": ref, "arguments": arguments}


def test_mcp_gateway_tools_and_call():
    route.reset_gateway_for_tests(FakeGateway())
    try:
        client = TestClient(app)
        tools = client.get("/mcp/tools")
        assert tools.status_code == 200
        assert tools.json()["tools"][0]["name"] == "mcp__fs__read_file"

        called = client.post("/mcp/call", json={
            "tool": "mcp__fs__read_file", "arguments": {"path": "a"},
        })
        assert called.status_code == 200
        assert called.json()["result"]["arguments"] == {"path": "a"}

        blocked = client.post("/mcp/call", json={
            "tool": "mcp__db__delete_row", "arguments": {},
        })
        assert blocked.status_code == 403
    finally:
        route.reset_gateway_for_tests(None)


def test_mcp_gateway_unconfigured_503():
    route.reset_gateway_for_tests(None)
    client = TestClient(app)
    assert client.get("/mcp/tools").status_code == 503
