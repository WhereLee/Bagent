"""P2：MCP gateway HTTP 入口。

主应用默认不装 mcp SDK；未配置 MCP_GATEWAY_SERVERS 时端点返回 503。
工具发现与调用都经过 allowlist 和 agent:mcp scope 校验。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import MCPCallReq
from app.auth import Principal, get_principal
from app.mcp_gateway import gateway_from_env

router = APIRouter()
_gateway = None


def get_gateway():
    global _gateway
    if _gateway is None:
        _gateway = gateway_from_env()
    if _gateway is None:
        raise HTTPException(status_code=503, detail="MCP gateway not configured")
    return _gateway


def reset_gateway_for_tests(gateway=None) -> None:
    global _gateway
    _gateway = gateway


def _require_scope(principal: Principal | None) -> None:
    if principal is not None and not principal.has_scope("agent:mcp"):
        raise HTTPException(status_code=403, detail="agent:mcp scope required")


@router.get("/mcp/tools")
def mcp_tools(principal: Principal | None = Depends(get_principal)) -> dict:
    _require_scope(principal)
    tools = get_gateway().list_tools()
    return {"tools": [
        {"name": t.ref, "description": t.description, "input_schema": t.input_schema,
         "server": t.server}
        for t in tools
    ]}


@router.post("/mcp/call")
def mcp_call(req: MCPCallReq, principal: Principal | None = Depends(get_principal)) -> dict:
    _require_scope(principal)
    try:
        result = get_gateway().call_tool(req.tool, req.arguments)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"mcp upstream failed: {e}"[:200]) from e
    return {"tool": req.tool, "result": result}
