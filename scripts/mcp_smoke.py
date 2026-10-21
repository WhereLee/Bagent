"""P3-2：MCP stdio list/call 冒烟（独立 mcp 环境运行）。

用法（在 requirements-mcp.txt 的 venv 中）：
  python scripts/mcp_smoke.py
"""
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import sys
from pathlib import Path

from app.mcp_gateway import MCPClient, MCPServerConfig, MCPToolGateway


def main() -> None:
    server = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mcp_echo_server.py"
    client = MCPClient([MCPServerConfig(
        name="echo", transport="stdio", command=sys.executable, args=(str(server),),
    )])
    gateway = MCPToolGateway(client, allowlist={"mcp__echo__echo"})
    tools = gateway.list_tools()
    print(json.dumps([
        {"name": t.ref, "schema_hash": t.schema_hash} for t in tools
    ], ensure_ascii=False))
    if not tools:
        raise SystemExit("no allowlisted tools")
    result = gateway.call_tool("mcp__echo__echo", {"text": "hello"})
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
