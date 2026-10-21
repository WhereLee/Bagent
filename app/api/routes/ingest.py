"""文档入库与删除。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import tenant_of
from app.api.schemas import DeleteReq, IngestReq
from app.config import ROOT_DIR, get_settings
from app.db.session import get_session
from app.ingestion.indexer import ingest_file
from app.retrieval.cache import get_retrieval_cache
from app.retrieval.store import delete_document

router = APIRouter()


@router.post("/ingest")
def ingest(req: IngestReq, request: Request) -> dict:
    tenant = tenant_of(request)
    # 路径限制在白名单目录下（防用任意 .txt/.md/.pdf 读取服务器文件并回显）
    root = (ROOT_DIR / get_settings().ingest_allowed_root).resolve()
    try:
        target = Path(req.path).resolve()
        target.relative_to(root)
    except Exception:
        raise HTTPException(status_code=400, detail=f"ingest path 必须位于 {root} 下")
    try:
        result = ingest_file(target, tenant=tenant)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"入库失败: {e}")
    return asdict(result)


@router.post("/delete")
def delete_doc(req: DeleteReq, request: Request) -> dict:
    tenant = tenant_of(request)
    session = get_session()
    try:
        ok = delete_document(session, req.source, tenant=tenant)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    if ok:
        get_retrieval_cache().clear()
    return {"source": req.source, "deleted": ok}
