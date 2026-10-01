"""数据库会话与初始化。"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import ROOT_DIR, get_settings

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().sqlalchemy_url, pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    return Session(get_engine())


def init_schema() -> None:
    """执行 schema.sql（幂等）。"""
    sql = (ROOT_DIR / "app" / "db" / "schema.sql").read_text(encoding="utf-8")
    engine = get_engine()
    # schema.sql 是纯 DDL，按分号切分逐条执行以规避扩展语句的单句限制
    with engine.begin() as conn:
        for stmt in filter(None, (s.strip() for s in sql.split(";"))):
            conn.execute(text(stmt))
