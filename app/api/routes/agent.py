"""Agent 运行入口：新 run/HITL 端点 + 旧 /agent 兼容入口。"""
from __future__ import annotations

import json
from queue import Empty, Queue
from threading import Thread

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.events import events_from_run, replay_after
from app.agent.react import resume_run, start_run
from app.agent.state import (
    TERMINAL_STATUSES,
    PostgresRunStore,
    RunStore,
    new_run,
    run_to_dict,
    step_to_dict,
)
from app.api.schemas import AgentReq, AgentRunReq, ApproveReq, CancelReq, RejectReq
from app.auth import Principal, get_principal
from app.config import get_settings
from app.evaluation.trajectory import summarize_trajectories
from app.generation.llm import get_llm
from app.tools import DEFAULT_REQUIRES_APPROVAL, TOOL_SPECS, build_tools

router = APIRouter()


def get_run_store() -> RunStore:
    """测试可 monkeypatch 为 InMemoryRunStore。"""
    return PostgresRunStore()


def _tools(user_id: str | None = None, scopes=None):
    return build_tools(user_id=user_id, scopes=scopes)


def _effective_user(body_user_id: str | None, principal: Principal | None) -> str | None:
    return principal.sub if principal is not None else body_user_id


def _check_owner(run, principal: Principal | None) -> None:
    if principal is not None and run.user_id != principal.sub:
        raise HTTPException(status_code=403, detail="run owner mismatch")


def _serialize(run, store: RunStore, *, include_token: bool = False) -> dict:
    payload = run_to_dict(run, include_token=include_token)
    payload["steps"] = [step_to_dict(s) for s in store.list_steps(run.id)]
    payload["metrics"] = summarize_trajectories([payload])
    return payload


def _resume(run_id: str, *, decision: str, approval_token: str,
            step_index: int | None = None, reason: str | None = None,
            principal: Principal | None = None):
    store = get_run_store()
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    _check_owner(run, principal)
    if not approval_token or approval_token != run.approval_token:
        raise HTTPException(status_code=403, detail="invalid approval token")
    if decision == "cancel":
        if run.status in TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail=f"run already terminal: {run.status}")
    elif decision in ("approve", "reject"):
        if run.status != "awaiting_approval":
            raise HTTPException(status_code=409, detail=f"run is not awaiting approval: {run.status}")
        pending_step = (run.pending_action or {}).get("step_index")
        if pending_step is None:
            raise HTTPException(status_code=409, detail="pending action is missing")
        if step_index is not None and int(step_index) != int(pending_step):
            raise HTTPException(status_code=409, detail="step_index does not match pending action")
    try:
        return resume_run(
            run_id, decision=decision, approval_token=approval_token,
            step_index=step_index, reason=reason,
            llm=get_llm() if decision == "approve" else None,
            tools=_tools(run.user_id) if decision == "approve" else None,
            tool_specs=TOOL_SPECS, store=store,
            requires_confirm=DEFAULT_REQUIRES_APPROVAL,
            finalize=True, check_faithfulness=get_settings().faithfulness_enabled,
            scopes=principal.scopes if principal is not None else None,
        ), store
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.post("/agent/runs")
def agent_run(req: AgentRunReq, principal: Principal | None = Depends(get_principal)) -> dict:
    """创建并同步执行一次 agent run；命中审批时返回 awaiting_approval。"""
    store = get_run_store()
    user_id = _effective_user(req.user_id, principal)
    run = start_run(
        req.goal, llm=get_llm(), tools=_tools(user_id, principal.scopes if principal else None),
        max_steps=req.max_steps, tool_specs=TOOL_SPECS, store=store, user_id=user_id,
        requires_confirm=DEFAULT_REQUIRES_APPROVAL,
        finalize=True, check_faithfulness=get_settings().faithfulness_enabled,
        scopes=principal.scopes if principal is not None else None,
    )
    return _serialize(run, store, include_token=True)


_SSE_HEARTBEAT_S = 15.0


def _sse_frame(event: dict) -> str:
    event_id = event.get("event_id")
    id_line = f"id: {event_id}\n" if event_id is not None else ""
    return (f"{id_line}event: {event['event']}\n"
            f"data: {json.dumps(event, ensure_ascii=False)}\n\n")


@router.post("/agent/runs/stream")
def agent_run_stream(req: AgentRunReq, principal: Principal | None = Depends(get_principal)):
    """创建并流式运行；断开连接不取消 run，之后可用 events 端点重放。"""
    store = get_run_store()
    user_id = _effective_user(req.user_id, principal)
    run = store.create(new_run(req.goal, max_steps=req.max_steps, user_id=user_id))
    queue: Queue = Queue()

    def sink(event: dict) -> None:
        queue.put(event)

    def worker() -> None:
        try:
            start_run(
                req.goal, llm=get_llm(),
                tools=_tools(user_id, principal.scopes if principal else None),
                max_steps=req.max_steps, tool_specs=TOOL_SPECS, store=store,
                user_id=user_id, requires_confirm=DEFAULT_REQUIRES_APPROVAL,
                finalize=True, check_faithfulness=get_settings().faithfulness_enabled,
                run=run, event_sink=sink,
                scopes=principal.scopes if principal is not None else None,
            )
        except Exception as e:  # noqa: BLE001
            error = f"agent stream failed: {e}"[:200]
            run.status = "failed"
            run.error = error
            store.update(run)
            queue.put({
                "event_id": 2 * run.step_count + 2, "event": "run_failed",
                "run_id": run.id, "step_index": None, "status": "failed",
                "data": {"run_id": run.id, "status": "failed", "error": error},
            })
        finally:
            queue.put(None)

    Thread(target=worker, name=f"agent-stream-{run.id}", daemon=True).start()

    def gen():
        while True:
            try:
                event = queue.get(timeout=_SSE_HEARTBEAT_S)
            except Empty:
                yield _sse_frame({
                    "event": "heartbeat", "run_id": run.id,
                    "status": run.status, "data": {"run_id": run.id},
                })
                continue
            if event is None:
                break
            yield _sse_frame(event)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
            "X-Run-ID": run.id, "X-Approval-Token": run.approval_token,
        },
    )


@router.get("/agent/runs/{run_id}/events")
def agent_events(run_id: str, last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
                 principal: Principal | None = Depends(get_principal)):
    """按 Last-Event-ID 重放事件；客户端断开不会影响 run。"""
    store = get_run_store()
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    _check_owner(run, principal)
    events = replay_after(events_from_run(run, store.list_steps(run.id)), last_event_id)

    def gen():
        for event in events:
            yield _sse_frame(event)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agent/runs/{run_id}")
def agent_get(run_id: str, principal: Principal | None = Depends(get_principal)) -> dict:
    store = get_run_store()
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    _check_owner(run, principal)
    return _serialize(run, store)


@router.post("/agent/runs/{run_id}/approve")
def agent_approve(run_id: str, req: ApproveReq,
                  principal: Principal | None = Depends(get_principal)) -> dict:
    run, store = _resume(
        run_id, decision="approve", approval_token=req.approval_token,
        step_index=req.step_index, principal=principal,
    )
    return _serialize(run, store)


@router.post("/agent/runs/{run_id}/reject")
def agent_reject(run_id: str, req: RejectReq,
                 principal: Principal | None = Depends(get_principal)) -> dict:
    run, store = _resume(
        run_id, decision="reject", approval_token=req.approval_token,
        step_index=req.step_index, reason=req.reason, principal=principal,
    )
    return _serialize(run, store)


@router.post("/agent/runs/{run_id}/cancel")
def agent_cancel(run_id: str, req: CancelReq,
                 principal: Principal | None = Depends(get_principal)) -> dict:
    run, store = _resume(
        run_id, decision="cancel", approval_token=req.approval_token,
        principal=principal,
    )
    return _serialize(run, store)


@router.post("/agent")
def agent_run_legacy(req: AgentReq, principal: Principal | None = Depends(get_principal)) -> dict:
    """兼容旧入口：保留 {result, metrics} 结构，并附加 run_id。"""
    store = get_run_store()
    user_id = _effective_user(req.user_id, principal)
    run = start_run(
        req.goal, llm=get_llm(), tools=_tools(user_id, principal.scopes if principal else None),
        max_steps=req.max_steps, tool_specs=TOOL_SPECS, store=store, user_id=user_id,
        requires_confirm=DEFAULT_REQUIRES_APPROVAL,
        finalize=True, check_faithfulness=get_settings().faithfulness_enabled,
        scopes=principal.scopes if principal is not None else None,
    )
    result = _serialize(run, store, include_token=run.status == "awaiting_approval")
    return {"result": result, "metrics": result["metrics"]}
