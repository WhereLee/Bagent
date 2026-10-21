"""P0 Agent 运行时状态与存储。

- AgentRun/AgentStep 是技术运行时数据，不是业务实体。
- RunStore 抽象出内存/Postgres 两种实现：单测用内存，HTTP 用 Postgres。
- 每次 store 操作使用短会话，不跨 LLM 调用持有数据库 Session。
"""
from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from app.db.models import AgentRun as AgentRunRow
from app.db.models import AgentStep as AgentStepRow
from app.db.models import ToolOperation as ToolOperationRow
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


def new_run(goal: str, *, max_steps: int = 6, user_id: str | None = None,
            scopes: list[str] | None = None) -> "AgentRun":
    return AgentRun(
        id=uuid.uuid4().hex,
        goal=goal,
        user_id=user_id,
        scopes=list(scopes) if scopes is not None else None,
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
    scopes: list[str] | None = None
    max_steps: int = 6
    step_count: int = 0
    lease_owner: str | None = None
    lease_until: datetime | None = None
    attempt: int = 0
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
    observation_full: str | None = None
    error: str | None = None
    latency_ms: int | None = None
    token_input: int = 0
    token_output: int = 0
    id: int | None = None
    created_at: datetime | None = None


@dataclass
class ToolOperation:
    operation_id: str
    run_id: str
    step_index: int
    tool: str
    idempotency_key: str
    args_hash: str
    state: str = "started"
    external_ref: str | None = None
    result: dict | None = None
    error: str | None = None
    attempts: int = 1
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    is_new: bool = False


class RunStore(Protocol):
    def create(self, run: AgentRun) -> AgentRun: ...
    def get(self, run_id: str) -> AgentRun | None: ...
    def update(self, run: AgentRun) -> None: ...
    def add_step(self, step: AgentStep) -> None: ...
    def list_steps(self, run_id: str) -> list[AgentStep]: ...
    def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> bool: ...
    def renew_lease(self, run_id: str, owner: str, ttl_s: float) -> bool: ...
    def release_lease(self, run_id: str, owner: str) -> None: ...
    def list_recoverable(self, *, limit: int = 10) -> list[AgentRun]: ...
    def begin_operation(self, op: ToolOperation) -> ToolOperation: ...
    def mark_operation(self, key: str, state: str, *, result: dict | None = None,
                       error: str | None = None, external_ref: str | None = None) -> None: ...
    def get_operation(self, key: str) -> ToolOperation | None: ...
    def list_operations(self, run_id: str) -> list[ToolOperation]: ...
    def list_runs(self, *, user_id: str | None = None, limit: int = 20,
                  status: str | None = None) -> list[AgentRun]: ...


class InMemoryRunStore:
    """单测/无 DB 场景使用；进程重启即丢，不用于生产。"""

    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._steps: dict[str, list[AgentStep]] = {}
        self._ops: dict[str, ToolOperation] = {}

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

    def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> bool:
        run = self._runs.get(run_id)
        now = datetime.now(timezone.utc)
        if run is None or run.status not in (RunStatus.RUNNING.value,
                                             RunStatus.AWAITING_APPROVAL.value):
            return False
        if run.lease_until is not None and run.lease_until > now and run.lease_owner != owner:
            return False
        run.lease_owner = owner
        run.lease_until = now + timedelta(seconds=ttl_s)
        run.attempt += 1
        run.updated_at = now
        return True

    def renew_lease(self, run_id: str, owner: str, ttl_s: float) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.status != RunStatus.RUNNING.value or run.lease_owner != owner:
            return False
        run.lease_until = datetime.now(timezone.utc) + timedelta(seconds=ttl_s)
        return True

    def release_lease(self, run_id: str, owner: str) -> None:
        run = self._runs.get(run_id)
        if run is not None and run.lease_owner == owner:
            run.lease_owner = None
            run.lease_until = None

    def list_recoverable(self, *, limit: int = 10) -> list[AgentRun]:
        now = datetime.now(timezone.utc)
        rows = [r for r in self._runs.values()
                if r.status == RunStatus.RUNNING.value
                and (r.lease_until is None or r.lease_until <= now)]
        rows.sort(key=lambda r: r.updated_at or r.created_at or now)
        return rows[:limit]

    def begin_operation(self, op: ToolOperation) -> ToolOperation:
        existing = self._ops.get(op.idempotency_key)
        if existing is not None:
            if existing.state == "failed":
                existing.state = "started"
                existing.attempts += 1
                existing.updated_at = datetime.now(timezone.utc)
                existing.is_new = True
                return existing
            existing.is_new = False
            return existing
        op.created_at = datetime.now(timezone.utc)
        op.updated_at = op.created_at
        op.is_new = True
        self._ops[op.idempotency_key] = op
        return op

    def mark_operation(self, key: str, state: str, *, result: dict | None = None,
                       error: str | None = None, external_ref: str | None = None) -> None:
        op = self._ops.get(key)
        if op is None:
            raise KeyError(f"operation not found: {key}")
        op.state = state
        op.result = result if result is not None else op.result
        op.error = error
        op.external_ref = external_ref if external_ref is not None else op.external_ref
        op.updated_at = datetime.now(timezone.utc)

    def get_operation(self, key: str) -> ToolOperation | None:
        return self._ops.get(key)

    def list_operations(self, run_id: str) -> list[ToolOperation]:
        return sorted((o for o in self._ops.values() if o.run_id == run_id),
                      key=lambda o: (o.step_index, o.id or 0))

    def list_runs(self, *, user_id: str | None = None, limit: int = 20,
                  status: str | None = None) -> list[AgentRun]:
        rows = [r for r in self._runs.values()
                if (user_id is None or r.user_id == user_id)
                and (status is None or r.status == status)]
        rows.sort(key=lambda r: r.updated_at or r.created_at or datetime.min.replace(tzinfo=timezone.utc),
                  reverse=True)
        return rows[:limit]

    def clear(self) -> None:
        self._runs.clear()
        self._steps.clear()
        self._ops.clear()


class PostgresRunStore:
    """生产实现：每次操作短会话，commit 后关闭。"""

    @staticmethod
    def _to_run(row: AgentRunRow) -> AgentRun:
        return AgentRun(
            id=row.id, goal=row.goal, status=row.status, approval_token=row.approval_token,
            user_id=row.user_id, scopes=row.scopes, max_steps=row.max_steps,
            step_count=row.step_count, pending_action=row.pending_action,
            final=row.final, answer=row.answer, error=row.error,
            lease_owner=row.lease_owner, lease_until=row.lease_until, attempt=row.attempt or 0,
            token_input=row.token_input, token_output=row.token_output,
            created_at=row.created_at, updated_at=row.updated_at,
        )

    @staticmethod
    def _to_step(row: AgentStepRow) -> AgentStep:
        return AgentStep(
            id=row.id, run_id=row.run_id, step_index=row.step_index, thought=row.thought,
            action=row.action, args=row.args or {}, observation=row.observation,
            observation_full=row.observation_full, error=row.error,
            latency_ms=row.latency_ms, token_input=row.token_input,
            token_output=row.token_output, created_at=row.created_at,
        )

    def create(self, run: AgentRun) -> AgentRun:
        session = get_session()
        try:
            row = AgentRunRow(
                id=run.id, goal=run.goal, user_id=run.user_id, scopes=run.scopes,
                status=run.status, max_steps=run.max_steps, step_count=run.step_count,
                pending_action=run.pending_action, final=run.final, answer=run.answer,
                error=run.error, approval_token=run.approval_token,
                lease_owner=run.lease_owner, lease_until=run.lease_until, attempt=run.attempt,
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
            for name in ("goal", "user_id", "scopes", "status", "max_steps", "step_count",
                         "pending_action", "final", "answer", "error", "approval_token",
                         "lease_owner", "lease_until", "attempt",
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
                observation_full=step.observation_full, error=step.error,
                latency_ms=step.latency_ms,
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

    def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> bool:
        session = get_session()
        now = datetime.now(timezone.utc)
        until = now + timedelta(seconds=ttl_s)
        try:
            result = session.execute(
                update(AgentRunRow).where(
                    AgentRunRow.id == run_id,
                    AgentRunRow.status.in_([RunStatus.RUNNING.value,
                                            RunStatus.AWAITING_APPROVAL.value]),
                    or_(AgentRunRow.lease_until.is_(None), AgentRunRow.lease_until < now,
                        AgentRunRow.lease_owner == owner),
                ).values(lease_owner=owner, lease_until=until,
                         attempt=AgentRunRow.attempt + 1, updated_at=now)
            )
            session.commit()
            return result.rowcount == 1
        finally:
            session.close()

    def renew_lease(self, run_id: str, owner: str, ttl_s: float) -> bool:
        session = get_session()
        now = datetime.now(timezone.utc)
        try:
            result = session.execute(
                update(AgentRunRow).where(
                    AgentRunRow.id == run_id,
                    AgentRunRow.status == RunStatus.RUNNING.value,
                    AgentRunRow.lease_owner == owner,
                ).values(lease_until=now + timedelta(seconds=ttl_s), updated_at=now)
            )
            session.commit()
            return result.rowcount == 1
        finally:
            session.close()

    def release_lease(self, run_id: str, owner: str) -> None:
        session = get_session()
        try:
            session.execute(
                update(AgentRunRow).where(AgentRunRow.id == run_id,
                                          AgentRunRow.lease_owner == owner)
                .values(lease_owner=None, lease_until=None)
            )
            session.commit()
        finally:
            session.close()

    def list_recoverable(self, *, limit: int = 10) -> list[AgentRun]:
        session = get_session()
        now = datetime.now(timezone.utc)
        try:
            rows = session.scalars(
                select(AgentRunRow).where(
                    AgentRunRow.status == RunStatus.RUNNING.value,
                    or_(AgentRunRow.lease_until.is_(None), AgentRunRow.lease_until < now),
                ).order_by(AgentRunRow.updated_at).limit(limit)
            ).all()
            return [self._to_run(r) for r in rows]
        finally:
            session.close()


    @staticmethod
    def _to_operation(row: ToolOperationRow) -> ToolOperation:
        return ToolOperation(
            id=row.id, operation_id=row.operation_id, run_id=row.run_id,
            step_index=row.step_index, tool=row.tool,
            idempotency_key=row.idempotency_key, args_hash=row.args_hash,
            state=row.state, external_ref=row.external_ref, result=row.result,
            error=row.error, attempts=row.attempts,
            created_at=row.created_at, updated_at=row.updated_at,
        )

    def begin_operation(self, op: ToolOperation) -> ToolOperation:
        session = get_session()
        try:
            existing = session.scalar(
                select(ToolOperationRow).where(
                    ToolOperationRow.idempotency_key == op.idempotency_key)
            )
            if existing is not None:
                if existing.state == "failed":
                    existing.state = "started"
                    existing.attempts = (existing.attempts or 1) + 1
                    existing.updated_at = datetime.now(timezone.utc)
                    session.commit()
                    out = self._to_operation(existing)
                    out.is_new = True
                    return out
                return self._to_operation(existing)
            row = ToolOperationRow(
                operation_id=op.operation_id, run_id=op.run_id,
                step_index=op.step_index, tool=op.tool,
                idempotency_key=op.idempotency_key, args_hash=op.args_hash,
                state=op.state, attempts=op.attempts,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            out = self._to_operation(row)
            out.is_new = True
            return out
        except IntegrityError:
            session.rollback()
            row = session.scalar(
                select(ToolOperationRow).where(
                    ToolOperationRow.idempotency_key == op.idempotency_key)
            )
            if row is None:
                raise
            return self._to_operation(row)
        finally:
            session.close()

    def mark_operation(self, key: str, state: str, *, result: dict | None = None,
                       error: str | None = None, external_ref: str | None = None) -> None:
        session = get_session()
        try:
            row = session.scalar(
                select(ToolOperationRow).where(ToolOperationRow.idempotency_key == key)
            )
            if row is None:
                raise KeyError(f"operation not found: {key}")
            row.state = state
            if result is not None:
                row.result = result
            row.error = error
            if external_ref is not None:
                row.external_ref = external_ref
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
        finally:
            session.close()

    def get_operation(self, key: str) -> ToolOperation | None:
        session = get_session()
        try:
            row = session.scalar(
                select(ToolOperationRow).where(ToolOperationRow.idempotency_key == key)
            )
            return self._to_operation(row) if row is not None else None
        finally:
            session.close()

    def list_runs(self, *, user_id: str | None = None, limit: int = 20,
                  status: str | None = None) -> list[AgentRun]:
        session = get_session()
        try:
            q = select(AgentRunRow)
            if user_id is not None:
                q = q.where(AgentRunRow.user_id == user_id)
            if status is not None:
                q = q.where(AgentRunRow.status == status)
            rows = session.scalars(q.order_by(AgentRunRow.updated_at.desc()).limit(limit)).all()
            return [self._to_run(r) for r in rows]
        finally:
            session.close()

    def list_operations(self, run_id: str) -> list[ToolOperation]:
        session = get_session()
        try:
            rows = session.scalars(
                select(ToolOperationRow).where(ToolOperationRow.run_id == run_id)
                .order_by(ToolOperationRow.step_index, ToolOperationRow.id)
            ).all()
            return [self._to_operation(r) for r in rows]
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
        "attempt": run.attempt,
        "lease_until": run.lease_until.isoformat() if run.lease_until else None,
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


def operation_to_dict(op: ToolOperation) -> dict:
    return {
        "operation_id": op.operation_id, "run_id": op.run_id,
        "step_index": op.step_index, "tool": op.tool,
        "idempotency_key": op.idempotency_key, "args_hash": op.args_hash,
        "state": op.state, "external_ref": op.external_ref,
        "result": op.result, "error": op.error, "attempts": op.attempts,
        "created_at": op.created_at.isoformat() if op.created_at else None,
        "updated_at": op.updated_at.isoformat() if op.updated_at else None,
    }


def step_to_dict(step: AgentStep) -> dict:
    return {
        "step_index": step.step_index,
        "thought": step.thought,
        "action": step.action,
        "args": step.args,
        "observation": step.observation,
        "has_full_observation": bool(step.observation_full),
        "error": step.error,
        "latency_ms": step.latency_ms,
        "token_input": step.token_input,
        "token_output": step.token_output,
        "created_at": step.created_at.isoformat() if step.created_at else None,
    }
