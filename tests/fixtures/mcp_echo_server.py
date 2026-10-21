"""独立 MCP stdio smoke server（需要 requirements-mcp.txt 环境）。"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bagent-echo")


@mcp.tool()
def echo(text: str) -> str:
    """Echo text back."""
    return text


if __name__ == "__main__":
    mcp.run()
