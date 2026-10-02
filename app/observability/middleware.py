"""HTTP 中间件：request_id 贯穿 + 时延/计数指标 + 限流。"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.observability.logging import get_logger, request_id_var
from app.observability.metrics import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    INPROGRESS,
    RATE_LIMIT_REJECTED,
)
from app.ratelimit import RateLimiter
from app.security import constant_time_equal

log = get_logger("http")

EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}


def client_ip(request: Request) -> str:
    """优先取反向代理透传的 X-Forwarded-For 第一段。"""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter, api_key: str | None = None) -> None:
        super().__init__(app)
        self.limiter = limiter
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request_id_var.set(rid)

        path = request.url.path
        method = request.method

        # 鉴权：配置了 API Key 则非豁免端点必须携带匹配的 X-API-Key
        if self.api_key and path not in EXEMPT_PATHS:
            given = request.headers.get("x-api-key", "")
            if not constant_time_equal(given, self.api_key):
                resp = JSONResponse({"error": "unauthorized"}, status_code=401)
                resp.headers["X-Request-ID"] = rid
                return resp

        if self.limiter is not None and path not in EXEMPT_PATHS:
            allowed, retry_after = self.limiter.allow(client_ip(request))
            if not allowed:
                RATE_LIMIT_REJECTED.inc()
                resp = JSONResponse(
                    {"error": "rate_limited", "retry_after": round(retry_after, 2)},
                    status_code=429,
                )
                resp.headers["Retry-After"] = str(max(1, int(retry_after + 1)))
                resp.headers["X-Request-ID"] = rid
                return resp

        INPROGRESS.inc()
        start = time.perf_counter()
        status = 0
        try:
            response: Response = await call_next(request)
            status = getattr(response, "status_code", 0)
        finally:
            INPROGRESS.dec()
            dur = time.perf_counter() - start
            HTTP_LATENCY.labels(method=method, path=path).observe(dur)
            HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
            log.info("request", extra={"fields": {"method": method, "path": path, "status": status, "duration_ms": round(dur * 1000, 1)}})

        response.headers["X-Request-ID"] = rid
        return response
