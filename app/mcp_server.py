"""MCP server：把 app.tools 的能力作为 MCP 工具暴露（Claude / Cursor 等可接入）。

运行：`python -m app.mcp_server`（stdio）。依赖 mcp（见 requirements-dev）。
业务实现全在 app.tools，MCP 与未来 agent 循环共用同一套工具。
"""
from __future__ import annotations

from app.tools import TOOLS


def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Bagent")
    for name, fn in TOOLS.items():
        desc = (fn.__doc__ or name).strip().splitlines()[0]
        mcp.tool(name=name, description=desc)(fn)
    return mcp


if __name__ == "__main__":
    build_server().run()
