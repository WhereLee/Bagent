"""P2-3：MCP gateway 命名空间、allowlist、审计与 ToolSpec 转换。"""
from __future__ import annotations

import json

from app.mcp_gateway import MCPRemoteTool, MCPToolGateway, gateway_from_env, parse_ref


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


def test_schema_pin_blocks_changed_schema():
    class C:
        def list_tools(self):
            return [MCPRemoteTool("fs", "read_file", "read",
                                  input_schema={"type": "object", "properties": {"path": {"type": "string"}}})]

        def call_tool(self, server, tool, arguments):
            return {}

    tool = C().list_tools()[0]
    gateway = MCPToolGateway(C(), allowlist={tool.ref},
                             schema_pins={tool.ref: tool.schema_hash})
    assert gateway.list_tools()
    changed = MCPToolGateway(C(), allowlist={tool.ref},
                             schema_pins={tool.ref: "different"})
    assert changed.list_tools() == []
    missing = MCPToolGateway(C(), allowlist={tool.ref}, schema_pins={"x": "y"})
    assert missing.list_tools() == []


def test_gateway_env_reads_token_env_without_exposing_it(monkeypatch):
    monkeypatch.setenv("MCP_GATEWAY_SERVERS", json.dumps([{
        "name": "fs", "transport": "stdio", "command": "python",
        "args": ["server.py"], "token_env": "FS_MCP_TOKEN",
    }]))
    monkeypatch.setenv("MCP_GATEWAY_ALLOWLIST", "mcp__fs__read_file")
    gw = gateway_from_env()
    assert gw is not None
    assert gw.client.servers[0].token_env == "FS_MCP_TOKEN"
    assert "secret" not in str(gw.schema_pins)


def test_parse_ref():
    assert parse_ref("mcp__fs__read_file") == ("fs", "read_file")
