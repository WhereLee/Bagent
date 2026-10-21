"""P1：Agent 事件流（AG-UI 语义子集）。

事件只携带 run/step/status/tool/耗时/token 等有界字段，不携带 prompt、args、
observation 或最终答案正文。event_id 由 step_index 派生，便于从 DB 重放。
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Callable

EventSink = Callable[[dict], None]
_sink_var: contextvars.ContextVar[EventSink | None] = contextvars.ContextVar("agent_event_sink", default=None)


@contextmanager
def use_event_sink(sink: EventSink | None):
    token = _sink_var.set(sink)
    try:
        yield
    finally:
        _sink_var.reset(token)


def emit_event(event: dict) -> None:
    sink = _sink_var.get()
    if sink is not None:
        sink(event)


def status_event_type(status: str) -> str:
    return {
        "completed": "run_completed",
        "failed": "run_failed",
        "cancelled": "run_cancelled",
        "awaiting_approval": "approval_required",
        "budget_exceeded": "run_budget_exceeded",
    }.get(status, "run_status")


def _status_payload(run) -> dict:
    return {
        "run_id": run.id,
        "status": run.status,
        "step_count": run.step_count,
        "error": run.error or None,
    }


def events_from_run(run, steps: list) -> list[dict]:
    """从持久化 run/step 重建事件；event_id 与运行时实时流一致。"""
    events = [{
        "event_id": 0, "event": "run_started",
        "run_id": run.id, "step_index": None,
        "status": "running", "data": {"run_id": run.id},
    }]
    for i, step in enumerate(steps):
        idx = step.step_index
        events.append({
            "event_id": 2 * i + 1, "event": "step_started",
            "run_id": run.id, "step_index": idx, "status": "started",
            "data": {"step_index": idx},
        })
        events.append({
            "event_id": 2 * i + 2, "event": "tool_result",
            "run_id": run.id, "step_index": idx,
            "status": "error" if step.error else "ok",
            "data": {
                "action": step.action,
                "latency_ms": step.latency_ms,
                "token_input": step.token_input,
                "token_output": step.token_output,
            },
        })
    pending = run.pending_action or {}
    events.append({
        "event_id": 2 * len(steps) + 2,
        "event": status_event_type(run.status),
        "run_id": run.id,
        "step_index": pending.get("step_index"),
        "status": run.status,
        "data": {**_status_payload(run), "tool": pending.get("tool")},
    })
    return events


def replay_after(events: list[dict], last_event_id: int | None) -> list[dict]:
    if last_event_id is None:
        return events
    return [e for e in events if int(e["event_id"]) > last_event_id]
