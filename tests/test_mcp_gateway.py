"""P2-3：MCP gateway 命名空间、allowlist、审计与 ToolSpec 转换。"""
from __future__ import annotations

from app.mcp_gateway import MCPRemoteTool, MCPToolGateway, parse_ref


class FakeClient:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            MCPRemoteTool("fs", "read_file", "read"),
            MCPRemoteTool("db", "delete_row", "destructive"),
        ]

    def call_tool(self, server, tool, arguments):
        self.calls.append((server, tool, arguments))
        return {"server": server, "tool": tool, "arguments": arguments}


def test_gateway_namespaces_and_allowlists():
    client = FakeClient()
    gateway = MCPToolGateway(client, allowlist={"mcp__fs__read_file"})
    tools = gateway.list_tools()
    assert [t.ref for t in tools] == ["mcp__fs__read_file"]
    assert gateway.call_tool("mcp__fs__read_file", {"path": "a"})["tool"] == "read_file"
    assert client.calls == [("fs", "read_file", {"path": "a"})]


def test_gateway_blocks_unlisted_tool():
    gateway = MCPToolGateway(FakeClient(), allowlist={"mcp__fs__read_file"})
    try:
        gateway.call_tool("mcp__db__delete_row", {})
    except PermissionError as e:
        assert "not allowlisted" in str(e)
    else:
        raise AssertionError("unlisted tool must be blocked")


def test_gateway_tool_specs_are_approval_gated():
    gateway = MCPToolGateway(FakeClient(), allowlist={"mcp__fs__read_file"})
    specs = gateway.as_tool_specs()
    spec = specs["mcp__fs__read_file"]
    assert spec.side_effect is True
    assert spec.requires_approval is True
    assert spec.required_scope == "agent:mcp"


def test_parse_ref():
    assert parse_ref("mcp__fs__read_file") == ("fs", "read_file")
