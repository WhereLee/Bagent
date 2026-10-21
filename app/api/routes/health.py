"""健康检查、就绪探针与 Prometheus 指标。"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text
from starlette.responses import Response

from app.config import get_settings
from app.db.session import get_engine
from app.generation.llm import get_llm
from app.observability.metrics import render_metrics

router = APIRouter()


@router.get("/health")
def health() -> dict:
    s = get_settings()
    info = {
        "status": "ok",
        "llm_provider": s.llm_primary_provider,
        "llm_fallback": s.llm_fallback_provider or None,
        "embedding": s.embedding_model_name,
        "retrieval_mode": s.retrieval_mode,
        "auth": bool(s.api_key),
    }
    try:
        info["llm_model"] = get_llm().chain[0].model
    except Exception:  # noqa: BLE001  provider 未配置时不影响健康检查
        info["llm_model"] = None
    return info


@router.get("/ready")
def ready():
    """就绪探针：DB 可连接即 200；用于容器/负载均衡 readiness（不是 liveness）。"""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "not_ready", "error": str(e)[:200]}, status_code=503)
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    body, ctype = render_metrics()
    return Response(content=body, media_type=ctype)
