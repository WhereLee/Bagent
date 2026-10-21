"""P2：MCP client / tool gateway 的治理层。

- 统一命名空间：mcp__<server>__<tool>，避免多服务器工具重名。
- allowlist：默认由环境显式配置；未配置时不隐式暴露外部工具。
- 外部工具默认按副作用 + 人工审批处理；annotations 只作为提示，不作为授权。
- 审计只记录 server/tool/status/latency，不落参数和返回正文。

mcp SDK 不装进主 web venv（见 requirements-mcp.txt）；MCPClient 内部懒加载。
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from app.observability.logging import get_logger, log_event
from app.observability.metrics import MCP_GATEWAY_CALLS
from app.tools import ToolSpec

_log = get_logger("mcp.gateway")
_PREFIX = "mcp__"


class GatewayArgs(BaseModel):
    """动态 MCP 工具参数；具体 schema 由上游工具描述承担。"""
    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class MCPRemoteTool:
    server: str
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{_PREFIX}{self.server}__{self.name}"


class MCPClientProtocol(Protocol):
    def list_tools(self) -> list[MCPRemoteTool]: ...
    def call_tool(self, server: str, tool: str, arguments: dict) -> Any: ...


def parse_ref(ref: str) -> tuple[str, str]:
    if not ref.startswith(_PREFIX):
        raise ValueError(f"invalid MCP tool ref: {ref}")
    rest = ref[len(_PREFIX):]
    if "__" not in rest:
        raise ValueError(f"invalid MCP tool ref: {ref}")
    server, tool = rest.split("__", 1)
    if not server or not tool:
        raise ValueError(f"invalid MCP tool ref: {ref}")
    return server, tool


class MCPToolGateway:
    def __init__(self, client: MCPClientProtocol, *, allowlist: set[str] | None = None,
                 scope: str = "agent:mcp", approval: bool = True) -> None:
        self.client = client
        self.allowlist = set(allowlist) if allowlist is not None else None
        self.scope = scope
        self.approval = approval
        self._tools: dict[str, MCPRemoteTool] | None = None

    def _allowed(self, tool: MCPRemoteTool) -> bool:
        if self.allowlist is None:
            return False
        return tool.ref in self.allowlist or f"{_PREFIX}{tool.server}__*" in self.allowlist

    def refresh(self) -> list[MCPRemoteTool]:
        tools = [t for t in self.client.list_tools() if self._allowed(t)]
        self._tools = {t.ref: t for t in tools}
        return tools

    def list_tools(self) -> list[MCPRemoteTool]:
        if self._tools is None:
            self.refresh()
        return list((self._tools or {}).values())

    def call_tool(self, ref: str, arguments: dict) -> Any:
        if self._tools is None:
            self.refresh()
        tool = (self._tools or {}).get(ref)
        if tool is None:
            MCP_GATEWAY_CALLS.labels(server="unknown", tool=ref, status="blocked").inc()
            log_event(_log, "warning", "mcp_tool_blocked", tool_ref=ref)
            raise PermissionError(f"MCP tool not allowlisted: {ref}")
        try:
            result = self.client.call_tool(tool.server, tool.name, arguments)
            MCP_GATEWAY_CALLS.labels(server=tool.server, tool=tool.name, status="ok").inc()
            log_event(_log, "info", "mcp_tool_called", server=tool.server, tool=tool.name)
            return result
        except Exception:
            MCP_GATEWAY_CALLS.labels(server=tool.server, tool=tool.name, status="error").inc()
            log_event(_log, "error", "mcp_tool_failed", server=tool.server, tool=tool.name)
            raise

    def as_tool_specs(self) -> dict[str, ToolSpec]:
        specs: dict[str, ToolSpec] = {}
        for tool in self.list_tools():
            def call(_tool=tool, **kwargs):
                return self.call_tool(_tool.ref, kwargs)

            specs[tool.ref] = ToolSpec(
                name=tool.ref, func=call,
                description=tool.description or f"MCP tool {tool.ref}",
                args_model=GatewayArgs, side_effect=True,
                requires_approval=self.approval, required_scope=self.scope,
                idempotent=False,
            )
        return specs


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"          # stdio | streamable_http
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


class MCPClient:
    """最小 MCP client；每次 list/call 建短会话，避免跨请求持有连接。"""

    def __init__(self, servers: list[MCPServerConfig]) -> None:
        self.servers = servers

    def _server(self, name: str) -> MCPServerConfig:
        for server in self.servers:
            if server.name == name:
                return server
        raise KeyError(f"unknown MCP server: {name}")

    async def _with_session(self, config: MCPServerConfig, fn):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if config.transport == "stdio":
            params = StdioServerParameters(command=config.command, args=list(config.args),
                                           env=config.env or None)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await fn(session)
        if config.transport == "streamable_http":
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(config.url, headers=config.headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await fn(session)
        raise ValueError(f"unsupported MCP transport: {config.transport}")

    def _run(self, coro):
        return asyncio.run(coro)

    def list_tools(self) -> list[MCPRemoteTool]:
        result: list[MCPRemoteTool] = []
        for config in self.servers:
            async def read(session, _config=config):
                listed = await session.list_tools()
                out = []
                for t in listed.tools:
                    annotations = t.annotations
                    if hasattr(annotations, "model_dump"):
                        annotations = annotations.model_dump()
                    out.append(MCPRemoteTool(
                        server=_config.name, name=t.name,
                        description=t.description or "",
                        input_schema=t.inputSchema or {},
                        annotations=annotations or {},
                    ))
                return out

            result.extend(self._run(self._with_session(config, read)))
        return result

    def call_tool(self, server: str, tool: str, arguments: dict) -> Any:
        config = self._server(server)

        async def call(session):
            result = await session.call_tool(tool, arguments)
            return result.model_dump() if hasattr(result, "model_dump") else result

        return self._run(self._with_session(config, call))


def gateway_from_env() -> MCPToolGateway | None:
    """从 MCP_GATEWAY_SERVERS / MCP_GATEWAY_ALLOWLIST 构造；未配置返回 None。"""
    raw = os.getenv("MCP_GATEWAY_SERVERS", "").strip()
    if not raw:
        return None
    servers = []
    for item in json.loads(raw):
        servers.append(MCPServerConfig(
            name=item["name"], transport=item.get("transport", "stdio"),
            command=item.get("command", ""), args=tuple(item.get("args", [])),
            url=item.get("url", ""), headers=item.get("headers", {}),
            env=item.get("env", {}),
        ))
    allow = {x.strip() for x in os.getenv("MCP_GATEWAY_ALLOWLIST", "").split(",") if x.strip()}
    return MCPToolGateway(MCPClient(servers), allowlist=allow or None)
