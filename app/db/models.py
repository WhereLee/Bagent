"""SQLAlchemy ORM 映射（与 app/db/schema.sql 对应）。"""
from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    doc_hash: Mapped[str] = mapped_column(Text, unique=True)
    media_type: Mapped[str | None] = mapped_column(String)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("chunks.id"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(512))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Memory(Base):
    """M9b 记忆：scope=personal(私有)/knowledge(共享)，双时态 + trust 生命周期。"""
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    scope: Mapped[str] = mapped_column(String)
    owner_user_id: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String, default="fact")
    level: Mapped[str] = mapped_column(String, default="fact")   # fact | playbook
    tags: Mapped[dict] = mapped_column(JSONB, default=dict)      # type/location/season/...
    event_id: Mapped[int | None] = mapped_column(BigInteger)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String, default="research")
    source_ref: Mapped[str | None] = mapped_column(Text)
    trust: Mapped[str] = mapped_column(String, default="draft")
    confidence: Mapped[float] = mapped_column(default=0.5)
    support: Mapped[int] = mapped_column(Integer, default=1)
    use_success: Mapped[int] = mapped_column(Integer, default=0)   # P3-C 结果回标
    use_fail: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(512))
    valid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    invalid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Event(Base):
    """P3-B 事件/复盘：一次活动的经验档案容器（关联 memories.event_id）。"""
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    type: Mapped[str | None] = mapped_column(String)
    tags: Mapped[dict] = mapped_column(JSONB, default=dict)
    summary: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentRun(Base):
    """P0 Agent 运行时：一次 run 的状态与汇总（技术表，不是业务实体）。"""
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    goal: Mapped[str] = mapped_column(Text)
    user_id: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="running")
    max_steps: Mapped[int] = mapped_column(Integer, default=6)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    pending_action: Mapped[dict | None] = mapped_column(JSONB)
    final: Mapped[str | None] = mapped_column(Text)
    answer: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    approval_token: Mapped[str] = mapped_column(Text)
    scopes: Mapped[list | None] = mapped_column(JSONB)
    lease_owner: Mapped[str | None] = mapped_column(Text)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    token_input: Mapped[int] = mapped_column(Integer, default=0)
    token_output: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ToolOperation(Base):
    """P3：副作用工具操作账本（幂等键 + 未知态对账）。"""
    __tablename__ = "tool_operations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    operation_id: Mapped[str] = mapped_column(Text)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    step_index: Mapped[int] = mapped_column(Integer)
    tool: Mapped[str] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(Text, unique=True)
    args_hash: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String, default="started")
    external_ref: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentStep(Base):
    """P0 Agent 运行时：逐步骤轨迹（工具/参数/观察/延迟/token）。"""
    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    step_index: Mapped[int] = mapped_column(Integer)
    thought: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    args: Mapped[dict] = mapped_column(JSONB, default=dict)
    observation: Mapped[str | None] = mapped_column(Text)
    observation_full: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    token_input: Mapped[int] = mapped_column(Integer, default=0)
    token_output: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
