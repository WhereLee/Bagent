"""Bagent RAG —— FastAPI 服务入口（应用装配 + 生命周期）。

路由按领域拆分在 app/api/routes/，这里只负责：
- lifespan（结构化日志、幂等建表、连接池预热）
- 中间件（request_id / 指标 / 鉴权 / 限流）
- 把各路由挂到同一个 app 上（对外 URL 不变）
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import agent, events, health, ingest, mcp_gateway, memory, ops, query, research, search, workbench
from app.config import get_settings
from app.db.session import get_engine, init_schema
from app.observability.logging import configure_logging
from app.observability.middleware import ObservabilityMiddleware
from app.ratelimit import get_rate_limiter


@asynccontextmanager
async def _lifespan(_app):
    """启动时：结构化日志 + 幂等建表 + 预热连接池（取代已弃用的 on_event）。"""
    configure_logging(get_settings().log_level)
    init_schema()
    get_engine()
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="Bagent RAG", version="0.1.0", lifespan=_lifespan)

    s = get_settings()
    limiter = get_rate_limiter() if s.rate_limit_enabled else None
    application.state.rate_limiter = limiter   # 测试/运维可检查；不改限流语义
    application.add_middleware(
        ObservabilityMiddleware,
        limiter=limiter,
        api_key=(s.api_key or None),
        trust_proxy=s.trust_proxy_headers,
    )

    for module in (health, ops, ingest, search, query, memory, research, events, agent,
                   mcp_gateway, workbench):
        application.include_router(module.router)
    return application


app = create_app()
