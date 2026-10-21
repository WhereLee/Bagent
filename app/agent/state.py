"""P0 Agent 运行时状态与存储。

- AgentRun/AgentStep 是技术运行时数据，不是业务实体。
- RunStore 抽象出内存/Postgres 两种实现：单测用内存，HTTP 用 Postgres。
- 每次 store 操作使用短会话，不跨 LLM 调用持有数据库 Session。
"""
from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from sqlalchemy import select

from app.db.models import AgentRun as AgentRunRow
from app.db.models import AgentStep as AgentStepRow
from app.db.session import get_session


class RunStatus(str, Enum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


TERMINAL_STATUSES = frozenset({
    RunStatus.COMPLETED.value,
    RunStatus.REJECTED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
    RunStatus.BUDGET_EXCEEDED.value,
})


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def new_run(goal: str, *, max_steps: int = 6, user_id: str | None = None) -> "AgentRun":
    return AgentRun(
        id=uuid.uuid4().hex,
        goal=goal,
        user_id=user_id,
        max_steps=max_steps,
        approval_token=secrets.token_hex(16),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@dataclass
class AgentRun:
    id: str
    goal: str
    status: str = RunStatus.RUNNING.value
    approval_token: str = ""
    user_id: str | None = None
    max_steps: int = 6
    step_count: int = 0
    pending_action: dict | None = None
    final: str | None = None
    answer: dict | None = None
    error: str | None = None
    token_input: int = 0
    token_output: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class AgentStep:
    run_id: str
    step_index: int
    action: str
    thought: str | None = None
    args: dict = field(default_factory=dict)
    observation: str | None = None
    error: str | None = None
    latency_ms: int | None = None
    token_input: int = 0
    token_output: int = 0
    id: int | None = None
    created_at: datetime | None = None


class RunStore(Protocol):
    def create(self, run: AgentRun) -> AgentRun: ...
    def get(self, run_id: str) -> AgentRun | None: ...
    def update(self, run: AgentRun) -> None: ...
    def add_step(self, step: AgentStep) -> None: ...
    def list_steps(self, run_id: str) -> list[AgentStep]: ...


class InMemoryRunStore:
    """单测/无 DB 场景使用；进程重启即丢，不用于生产。"""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._steps: dict[str, list[AgentStep]] = {}

    def create(self, run: AgentRun) -> AgentRun:
        if run.id in self._runs:
            raise ValueError(f"run already exists: {run.id}")
        self._runs[run.id] = run
        self._steps.setdefault(run.id, [])
        return run

    def get(self, run_id: str) -> AgentRun | None:
        return self._runs.get(run_id)

    def update(self, run: AgentRun) -> None:
        if run.id not in self._runs:
            raise KeyError(f"run not found: {run.id}")
        self._runs[run.id] = run
        run.updated_at = datetime.now(timezone.utc)

    def add_step(self, step: AgentStep) -> None:
        if step.run_id not in self._runs:
            raise KeyError(f"run not found: {step.run_id}")
        self._steps.setdefault(step.run_id, []).append(step)

    def list_steps(self, run_id: str) -> list[AgentStep]:
        return sorted(self._steps.get(run_id, []), key=lambda s: s.step_index)

    def clear(self) -> None:
        self._runs.clear()
        self._steps.clear()


class PostgresRunStore:
    """生产实现：每次操作短会话，commit 后关闭。"""

    @staticmethod
    def _to_run(row: AgentRunRow) -> AgentRun:
        return AgentRun(
            id=row.id, goal=row.goal, status=row.status, approval_token=row.approval_token,
            user_id=row.user_id, max_steps=row.max_steps, step_count=row.step_count,
            pending_action=row.pending_action, final=row.final, answer=row.answer,
            error=row.error, token_input=row.token_input, token_output=row.token_output,
            created_at=row.created_at, updated_at=row.updated_at,
        )

    @staticmethod
    def _to_step(row: AgentStepRow) -> AgentStep:
        return AgentStep(
            id=row.id, run_id=row.run_id, step_index=row.step_index, thought=row.thought,
            action=row.action, args=row.args or {}, observation=row.observation,
            error=row.error, latency_ms=row.latency_ms, token_input=row.token_input,
            token_output=row.token_output, created_at=row.created_at,
        )

    def create(self, run: AgentRun) -> AgentRun:
        session = get_session()
        try:
            row = AgentRunRow(
                id=run.id, goal=run.goal, user_id=run.user_id, status=run.status,
                max_steps=run.max_steps, step_count=run.step_count,
                pending_action=run.pending_action, final=run.final, answer=run.answer,
                error=run.error, approval_token=run.approval_token,
                token_input=run.token_input, token_output=run.token_output,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_run(row)
        finally:
            session.close()

    def get(self, run_id: str) -> AgentRun | None:
        session = get_session()
        try:
            row = session.get(AgentRunRow, run_id)
            return self._to_run(row) if row is not None else None
        finally:
            session.close()

    def update(self, run: AgentRun) -> None:
        session = get_session()
        try:
            row = session.get(AgentRunRow, run.id)
            if row is None:
                raise KeyError(f"run not found: {run.id}")
            for name in ("goal", "user_id", "status", "max_steps", "step_count",
                         "pending_action", "final", "answer", "error", "approval_token",
                         "token_input", "token_output"):
                setattr(row, name, getattr(run, name))
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
            run.updated_at = row.updated_at
        finally:
            session.close()

    def add_step(self, step: AgentStep) -> None:
        session = get_session()
        try:
            row = AgentStepRow(
                run_id=step.run_id, step_index=step.step_index, thought=step.thought,
                action=step.action, args=step.args or {}, observation=step.observation,
                error=step.error, latency_ms=step.latency_ms,
                token_input=step.token_input, token_output=step.token_output,
            )
            session.add(row)
            session.commit()
            step.id = row.id
            step.created_at = row.created_at
        finally:
            session.close()

    def list_steps(self, run_id: str) -> list[AgentStep]:
        session = get_session()
        try:
            rows = session.scalars(
                select(AgentStepRow).where(AgentStepRow.run_id == run_id)
                .order_by(AgentStepRow.step_index)
            ).all()
            return [self._to_step(r) for r in rows]
        finally:
            session.close()


def run_to_dict(run: AgentRun, *, include_token: bool = False) -> dict:
    payload = {
        "run_id": run.id,
        "goal": run.goal,
        "user_id": run.user_id,
        "status": run.status,
        "max_steps": run.max_steps,
        "step_count": run.step_count,
        "pending_action": run.pending_action,
        "final": run.final,
        "answer": run.answer,
        "error": run.error,
        "token_input": run.token_input,
        "token_output": run.token_output,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
    }
    if include_token:
        payload["approval_token"] = run.approval_token
    return payload


def step_to_dict(step: AgentStep) -> dict:
    return {
        "step_index": step.step_index,
        "thought": step.thought,
        "action": step.action,
        "args": step.args,
        "observation": step.observation,
        "error": step.error,
        "latency_ms": step.latency_ms,
        "token_input": step.token_input,
        "token_output": step.token_output,
        "created_at": step.created_at.isoformat() if step.created_at else None,
    }
