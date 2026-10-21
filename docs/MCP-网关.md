# MCP 网关

## 1. 定位

Bagent 主应用不安装 MCP SDK；`app/mcp_gateway.py` 只提供治理与协议适配，实际连接走独立 `requirements-mcp.txt` 环境。

网关做四件事：

1. 命名空间：`mcp__<server>__<tool>`
2. deny-by-default allowlist
3. schema pin（工具 schema 变化默认阻断）
4. 审计与指标，不记录参数/返回正文

## 2. 配置

```dotenv
MCP_GATEWAY_SERVERS=[{"name":"fs","transport":"stdio","command":"python","args":["server.py"],"token_env":"FS_MCP_TOKEN"}]
MCP_GATEWAY_ALLOWLIST=mcp__fs__read_file
MCP_GATEWAY_SCHEMA_PINS={"mcp__fs__read_file":"<schema-hash>"}
MCP_GATEWAY_TIMEOUT_S=30
```

`token_env` 是环境变量名，token 只在连接上游时注入；不写日志、不出现在错误消息。

未配置 `MCP_GATEWAY_SERVERS` 时，`/mcp/tools` 与 `/mcp/call` 返回 503。

## 3. HTTP 入口

```http
GET /mcp/tools
POST /mcp/call {"tool":"mcp__fs__read_file","arguments":{"path":"a.txt"}}
```

- 需要 `agent:mcp` scope（AUTH 开启时）。
- 未 allowlist 的工具返回 403。
- schema pin 不匹配的工具不会出现在 tools 列表。
- 外部工具在 Agent 中默认 `side_effect=True`、`requires_approval=True`、`idempotent=False`。

## 4. 冒烟

在独立 MCP 环境：

```bash
python scripts/mcp_smoke.py
```

输出工具名、schema hash 和一次 `echo` 调用结果。

## 5. 安全边界

- Tool annotations 只是提示，不作为授权。
- 生产接入远程 MCP 时，应使用 OAuth 2.1、PKCE、resource metadata 和 audience 校验；当前受控 bearer 注入是过渡方案。
- 不把 token 透传给下游业务 API。
