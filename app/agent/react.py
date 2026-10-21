"""P0：可持久化、可审批、可恢复的有界 ReAct Agent 运行时。

兼容旧 `run_agent()` 返回值；新增 store/run/approval_token 支持 HITL。
主 LLM 每步只产出一个动作 JSON；联网只经受控 web_search，主 LLM 不隐式联网。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Callable

from app.agent.context import compact_history
from app.agent.events import EventSink, emit_event, status_event_type, use_event_sink
from app.agent.finalize import finalize_agent_answer
from app.agent.state import (
    TERMINAL_STATUSES,
    ToolOperation,
    AgentRun,
    AgentStep,
    InMemoryRunStore,
    RunStatus,
    RunStore,
    new_run,
    step_to_dict,
)
from app.config import get_settings
from app.generation.generator import Answer
from app.generation.prompts import AGENT_SYSTEM
from app.observability.logging import agent_context, get_logger, log_event
from app.observability.metrics import (
    AGENT_APPROVALS,
    AGENT_RUNS_TOTAL,
    AGENT_STEP_LATENCY,
    AGENT_STEPS_TOTAL,
    AGENT_TOOL_RETRIES,
    AGENT_OPERATIONS_TOTAL,
)
from app.observability.tracing import trace_span
from app.observability.usage import collect_usage
from app.tools import (
    DEFAULT_REQUIRES_APPROVAL,
    TOOL_SPECS,
    TOOLS,
    ToolSpec,
    describe_tools,
    validate_tool_args,
)

_log = get_logger("agent")
_JSON = re.compile(r"\{.*\}", re.S)
_TOOL_EXECUTOR = ThreadPoolExecutor(thread_name_prefix="bagent-tool")


def _lease_owner(prefix: str = "api") -> str:
    return f"{prefix}:{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex[:8]}"


def _set_local_lease(run: AgentRun, owner: str, ttl: float) -> None:
    run.lease_owner = owner
    run.lease_until = datetime.now(timezone.utc) + timedelta(seconds=ttl)
    run.attempt += 1


class ToolExecutionError(Exception):
    def __init__(self, error_type: str, message: str, *, retryable: bool = False,
                 hint: str = "", details=None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.hint = hint
        self.details = details


def parse_action(text: str) -> dict | None:
    """解析模型输出的单个动作 JSON；无法解析返回 None。"""
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def run_agent(
    goal: str, *, llm, tools: dict | None = None, max_steps: int = 6,
    requires_confirm: set[str] | None = None,
    confirm: Callable[[str, dict], bool] | None = None,
    tool_specs: dict[str, ToolSpec] | None = None,
    store: RunStore | None = None,
    user_id: str | None = None,
    finalize: bool = True,
    check_faithfulness: bool = False,
    event_sink: EventSink | None = None,
    scopes: set[str] | frozenset[str] | None = None,
) -> dict:
    """兼容入口：无 store 时用内存 store，返回旧轨迹结构（status 已归一）。"""
    store = store if store is not None else InMemoryRunStore()
    run = new_run(goal, max_steps=max_steps, user_id=user_id)
    run = store.create(run)
    with use_event_sink(event_sink):
        _emit_run_started(run)
        with trace_span("invoke_agent", attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "bagent-agent",
            "bagent.run_id": run.id,
        }) as span:
            run = _drive(
                run, store=store, llm=llm, tools=tools, requires_confirm=requires_confirm,
                confirm=confirm, tool_specs=tool_specs, finalize=finalize,
                check_faithfulness=check_faithfulness, scopes=scopes,
            )
            span.set_attribute("bagent.status", run.status)
    return _result(run, store)


def start_run(
    goal: str, *, llm, tools: dict | None = None, max_steps: int = 6,
    requires_confirm: set[str] | None = None,
    confirm: Callable[[str, dict], bool] | None = None,
    tool_specs: dict[str, ToolSpec] | None = None,
    store: RunStore | None = None,
    user_id: str | None = None,
    finalize: bool = True,
    check_faithfulness: bool = False,
    run: AgentRun | None = None,
    event_sink: EventSink | None = None,
    scopes: set[str] | frozenset[str] | None = None,
    lease_owner: str | None = None,
    lease_ttl_s: float | None = None,
) -> AgentRun:
    store = store if store is not None else InMemoryRunStore()
    if run is None:
        run = store.create(new_run(goal, max_steps=max_steps, user_id=user_id,
                                   scopes=list(scopes) if scopes is not None else None))
    elif run.scopes is None and scopes is not None:
        run.scopes = list(scopes)
    owner = lease_owner or _lease_owner("api")
    ttl = lease_ttl_s if lease_ttl_s is not None else get_settings().agent_lease_ttl_s
    if not store.acquire_lease(run.id, owner, ttl):
        raise RuntimeError(f"run lease unavailable: {run.id}")
    _set_local_lease(run, owner, ttl)
    try:
        with use_event_sink(event_sink):
            _emit_run_started(run)
            with trace_span("invoke_agent", attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "bagent-agent",
                "bagent.run_id": run.id,
            }) as span:
                run = _drive(
                    run, store=store, llm=llm, tools=tools, requires_confirm=requires_confirm,
                    confirm=confirm, tool_specs=tool_specs, finalize=finalize,
                    check_faithfulness=check_faithfulness, scopes=scopes,
                    lease_owner=owner, lease_ttl_s=ttl,
                )
                span.set_attribute("bagent.status", run.status)
                return run
    finally:
        store.release_lease(run.id, owner)
        run.lease_owner = None
        run.lease_until = None


def resume_run(
    run_id: str, *, decision: str, approval_token: str,
    llm, store: RunStore, tools: dict | None = None,
    tool_specs: dict[str, ToolSpec] | None = None,
    requires_confirm: set[str] | None = None,
    confirm: Callable[[str, dict], bool] | None = None,
    step_index: int | None = None,
    reason: str | None = None,
    finalize: bool = True,
    check_faithfulness: bool = False,
    scopes: set[str] | frozenset[str] | None = None,
    lease_owner: str | None = None,
    lease_ttl_s: float | None = None,
) -> AgentRun:
    """恢复 awaiting_approval 的 run：approve/reject/cancel。"""
    run = store.get(run_id)
    if run is None:
        raise KeyError(f"run not found: {run_id}")
    if not approval_token or approval_token != run.approval_token:
        raise PermissionError("invalid approval token")

    if decision == "cancel":
        if run.status in TERMINAL_STATUSES:
            raise ValueError(f"run already terminal: {run.status}")
        run.status = RunStatus.CANCELLED.value
        run.pending_action = None
        _set_status(store, run, RunStatus.CANCELLED.value)
        AGENT_APPROVALS.labels(decision="cancel").inc()
        return run

    if decision not in ("approve", "reject"):
        raise ValueError(f"unknown decision: {decision}")
    if run.status != RunStatus.AWAITING_APPROVAL.value:
        raise ValueError(f"run is not awaiting approval: {run.status}")
    owner = lease_owner or _lease_owner("resume")
    ttl = lease_ttl_s if lease_ttl_s is not None else get_settings().agent_lease_ttl_s
    if not store.acquire_lease(run.id, owner, ttl):
        raise RuntimeError(f"run lease unavailable: {run.id}")
    _set_local_lease(run, owner, ttl)
    pending = run.pending_action or {}
    pending_step = pending.get("step_index")
    if step_index is not None and int(step_index) != int(pending_step):
        raise ValueError("step_index does not match pending action")

    if decision == "reject":
        run.status = RunStatus.REJECTED.value
        run.error = reason or "人工拒绝"
        run.pending_action = None
        _set_status(store, run, RunStatus.REJECTED.value)
        AGENT_APPROVALS.labels(decision="reject").inc()
        return run

    # approve：执行 pending action，然后继续循环
    AGENT_APPROVALS.labels(decision="approve").inc()
    tool = str(pending.get("tool") or "")
    args = pending.get("args") or {}
    thought = pending.get("thought")
    run.status = RunStatus.RUNNING.value
    store.update(run)
    if tool not in (tools or TOOLS):
        _append_step(store, run, AgentStep(
            run_id=run.id, step_index=run.step_count, thought=thought,
            action=f"unknown:{tool}", args=args,
            observation=json.dumps({
                "error_type": "unknown", "message": "未知工具",
                "retryable": False, "hint": "使用可用工具或直接回答",
            }, ensure_ascii=False)[:1500],
        ))
    else:
        _execute_tool(
            run, store, tools=tools or TOOLS, tool=tool, args=args,
            thought=thought, step_index=run.step_count,
            specs=tool_specs if tool_specs is not None else TOOL_SPECS,
            scopes=scopes,
        )
    run.pending_action = None
    store.update(run)
    with trace_span("invoke_agent", attributes={
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "bagent-agent",
        "bagent.run_id": run.id, "bagent.resume": True,
    }) as span:
        run = _drive(
            run, store=store, llm=llm, tools=tools, requires_confirm=requires_confirm,
            confirm=confirm, tool_specs=tool_specs, finalize=finalize,
            check_faithfulness=check_faithfulness, scopes=scopes,
            lease_owner=owner, lease_ttl_s=ttl,
        )
        span.set_attribute("bagent.status", run.status)
        store.release_lease(run.id, owner)
        run.lease_owner = None
        run.lease_until = None
        return run


def continue_run(
    run_id: str, *, llm, store: RunStore, tools: dict | None = None,
    tool_specs: dict[str, ToolSpec] | None = None,
    requires_confirm: set[str] | None = None,
    confirm: Callable[[str, dict], bool] | None = None,
    finalize: bool = True,
    check_faithfulness: bool = False,
    scopes: set[str] | frozenset[str] | None = None,
    lease_owner: str | None = None,
    lease_ttl_s: float | None = None,
) -> AgentRun:
    """恢复一个仍为 running 的 run；供崩溃恢复 worker 使用。"""
    run = store.get(run_id)
    if run is None or run.status != RunStatus.RUNNING.value:
        return run
    owner = lease_owner or _lease_owner("worker")
    ttl = lease_ttl_s if lease_ttl_s is not None else get_settings().agent_lease_ttl_s
    if not store.acquire_lease(run.id, owner, ttl):
        return run
    _set_local_lease(run, owner, ttl)
    active_scopes = scopes if scopes is not None else run.scopes
    try:
        if run.pending_action:
            pending = run.pending_action
            tool = str(pending.get("tool") or "")
            args = pending.get("args") or {}
            thought = pending.get("thought")
            if tool not in (tools or TOOLS):
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=run.step_count, thought=thought,
                    action=f"unknown:{tool}", args=args,
                    observation=json.dumps({
                        "error_type": "unknown", "message": "未知工具",
                        "retryable": False, "hint": "使用可用工具或直接回答",
                    }, ensure_ascii=False)[:1500],
                ))
            else:
                _execute_tool(
                    run, store, tools=tools or TOOLS, tool=tool, args=args,
                    thought=thought, step_index=run.step_count,
                    specs=tool_specs if tool_specs is not None else TOOL_SPECS,
                    scopes=active_scopes,
                )
            run.pending_action = None
            store.update(run)
        with trace_span("invoke_agent", attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": "bagent-agent",
            "bagent.run_id": run.id, "bagent.recover": True,
        }) as span:
            run = _drive(
                run, store=store, llm=llm, tools=tools,
                requires_confirm=requires_confirm, confirm=confirm,
                tool_specs=tool_specs, finalize=finalize,
                check_faithfulness=check_faithfulness,
                scopes=active_scopes,
                lease_owner=owner, lease_ttl_s=ttl,
            )
            span.set_attribute("bagent.status", run.status)
            return run
    finally:
        store.release_lease(run.id, owner)
        run.lease_owner = None
        run.lease_until = None


def _drive(
    run: AgentRun, *, store: RunStore, llm, tools: dict | None,
    requires_confirm: set[str] | None,
    confirm: Callable[[str, dict], bool] | None,
    tool_specs: dict[str, ToolSpec] | None,
    finalize: bool, check_faithfulness: bool,
    scopes: set[str] | frozenset[str] | None = None,
    lease_owner: str | None = None,
    lease_ttl_s: float | None = None,
) -> AgentRun:
    tools = tools if tools is not None else TOOLS
    specs = tool_specs if tool_specs is not None else TOOL_SPECS
    requires_confirm = DEFAULT_REQUIRES_APPROVAL if requires_confirm is None else requires_confirm
    menu = describe_tools(tools, specs=specs)

    while run.step_count < run.max_steps:
        latest = store.get(run.id)
        if latest is None:
            raise KeyError(f"run not found: {run.id}")
        run = latest
        if run.status != RunStatus.RUNNING.value:
            return run
        if lease_owner is not None:
            ttl = lease_ttl_s if lease_ttl_s is not None else get_settings().agent_lease_ttl_s
            if not store.renew_lease(run.id, lease_owner, ttl):
                return run

        steps = store.list_steps(run.id)
        settings = get_settings()
        history = compact_history(
            run.goal, steps,
            max_chars=settings.agent_context_max_chars,
            keep_recent=settings.agent_keep_recent_steps,
            obs_limit=settings.agent_compact_observation_chars,
        )
        step_index = run.step_count
        emit_event({
            "event_id": 2 * step_index + 1, "event": "step_started",
            "run_id": run.id, "step_index": step_index, "status": "started",
            "data": {"step_index": step_index},
        })
        started = time.perf_counter()
        with agent_context(run.id, step_index):
            try:
                s = get_settings()
                model = (s.deepseek_model if s.llm_primary_provider == "deepseek"
                         else s.mimo_model)
                with trace_span("chat", attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": s.llm_primary_provider,
                    "gen_ai.request.model": model,
                    "bagent.run_id": run.id, "bagent.step_id": step_index,
                }) as chat_span:
                    with collect_usage() as usage:
                        raw = llm.complete(
                            system=AGENT_SYSTEM.format(tools=menu),
                            user="\n".join(_render(h) for h in history),
                        )
                    chat_span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                    chat_span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            except Exception as e:  # noqa: BLE001
                error = f"LLM 调用失败: {e}"[:200]
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, action="error:llm",
                    observation=json.dumps({"error": error}, ensure_ascii=False),
                    error=error,
                ))
                _set_status(store, run, RunStatus.FAILED.value, error=error)
                return run

            latency_ms = int((time.perf_counter() - started) * 1000)
            act = parse_action(raw)
            if not act:
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, action="invalid",
                    observation="无法解析动作",
                    latency_ms=latency_ms,
                    token_input=usage.prompt_tokens, token_output=usage.completion_tokens,
                ))
                continue

            if act.get("final") is not None:
                final_text = str(act["final"])
                answer = None
                if finalize:
                    try:
                        answer = finalize_agent_answer(
                            run.goal, steps, final_text,
                            check_faithfulness=check_faithfulness,
                        )
                    except Exception as e:  # noqa: BLE001
                        log_event(_log, "warning", "agent_finalize_failed", err=str(e)[:160])
                        answer = Answer(query=run.goal, text=final_text, sources=[],
                                        grounded=False, low_confidence=True)
                run.final = answer.text if answer is not None else final_text
                run.answer = asdict(answer) if answer is not None else None
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, thought=act.get("thought"),
                    action="final", observation="",
                    latency_ms=latency_ms,
                    token_input=usage.prompt_tokens, token_output=usage.completion_tokens,
                ))
                _set_status(store, run, RunStatus.COMPLETED.value)
                return run

            tool = act.get("tool")
            args = act.get("args") or {}
            if not isinstance(tool, str) or not tool or not isinstance(args, dict):
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, thought=act.get("thought"),
                    action="invalid", args=args if isinstance(args, dict) else {},
                    observation="动作缺少合法 tool/args",
                    latency_ms=latency_ms,
                    token_input=usage.prompt_tokens, token_output=usage.completion_tokens,
                ))
                continue
            if tool not in tools:
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, thought=act.get("thought"),
                    action=f"unknown:{tool}", args=args,
                    observation=json.dumps({
                        "error_type": "unknown", "message": "未知工具",
                        "retryable": False, "hint": "使用可用工具或直接回答",
                    }, ensure_ascii=False)[:1500],
                    latency_ms=latency_ms,
                    token_input=usage.prompt_tokens, token_output=usage.completion_tokens,
                ))
                continue

            clean_args, errors = validate_tool_args(tool, args, specs)
            if errors:
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, thought=act.get("thought"),
                    action=f"invalid_args:{tool}", args=args,
                    observation=json.dumps({
                        "error_type": "invalid_args", "message": "参数校验失败",
                        "retryable": False, "hint": "修正参数后再试",
                        "details": errors,
                    }, ensure_ascii=False)[:1500],
                    latency_ms=latency_ms,
                    token_input=usage.prompt_tokens, token_output=usage.completion_tokens,
                ))
                continue

            if _has_duplicate(store, run, tool, clean_args):
                _append_duplicate(store, run, tool=tool, args=clean_args,
                                  thought=act.get("thought"), step_index=step_index,
                                  token_input=usage.prompt_tokens,
                                  token_output=usage.completion_tokens)
                continue

            current = store.get(run.id)
            if current is None or current.status != RunStatus.RUNNING.value:
                return current or run
            run = current

            spec = specs.get(tool)
            if not _scope_allowed(spec, scopes):
                _append_forbidden(store, run, tool=tool, args=clean_args,
                                  thought=act.get("thought"), step_index=step_index,
                                  token_input=usage.prompt_tokens,
                                  token_output=usage.completion_tokens)
                continue
            needs_approval = (spec.requires_approval if spec is not None
                              else tool in requires_confirm)
            if needs_approval:
                if confirm is not None:
                    if not confirm(tool, clean_args):
                        _append_step(store, run, AgentStep(
                            run_id=run.id, step_index=step_index, thought=act.get("thought"),
                            action=f"rejected:{tool}", args=clean_args,
                            observation=json.dumps({
                                "error_type": "rejected", "message": "人工拒绝",
                                "retryable": False, "hint": "换用其他方式或联系人工",
                            }, ensure_ascii=False)[:1500],
                            latency_ms=latency_ms,
                            token_input=usage.prompt_tokens, token_output=usage.completion_tokens,
                        ))
                        _set_status(store, run, RunStatus.REJECTED.value, error="人工拒绝")
                        return run
                else:
                    # pending_action 不占 step_index；只汇总本次 LLM 用量，不写 step 行。
                    # 审批通过后才在同一个 step_index 写工具结果，避免唯一键冲突。
                    run.pending_action = {
                        "step_index": step_index,
                        "thought": act.get("thought"),
                        "tool": tool,
                        "args": clean_args,
                    }
                    run.token_input += usage.prompt_tokens
                    run.token_output += usage.completion_tokens
                    log_event(
                        _log, "info", "agent_approval_requested",
                        run_id=run.id, tool=tool, step_index=step_index,
                    )
                    _set_status(store, run, RunStatus.AWAITING_APPROVAL.value)
                    return run

            _execute_tool(
                run, store, tools=tools, tool=tool, args=clean_args,
                thought=act.get("thought"), step_index=step_index, specs=specs,
                scopes=scopes,
                token_input=usage.prompt_tokens, token_output=usage.completion_tokens,
            )

    _set_status(store, run, RunStatus.BUDGET_EXCEEDED.value)
    return run


def _execute_tool(
    run: AgentRun, store: RunStore, *, tools: dict, tool: str, args: dict,
    thought: str | None, step_index: int,
    specs: dict[str, ToolSpec] | None = None,
    scopes: set[str] | frozenset[str] | None = None,
    token_input: int = 0, token_output: int = 0,
) -> None:
    spec = (specs if specs is not None else TOOL_SPECS).get(tool)
    if not _scope_allowed(spec, scopes):
        _append_forbidden(store, run, tool=tool, args=args, thought=thought,
                          step_index=step_index, token_input=token_input,
                          token_output=token_output)
        return
    if _has_duplicate(store, run, tool, args):
        _append_duplicate(store, run, tool=tool, args=args, thought=thought,
                          step_index=step_index, token_input=token_input,
                          token_output=token_output)
        return

    op_key = None
    if spec is not None and spec.side_effect:
        args_hash = hashlib.sha256(_canonical_args(args).encode("utf-8")).hexdigest()[:32]
        op_key = f"{run.id}:{step_index}:{tool}:{args_hash[:16]}"
        try:
            op = store.begin_operation(ToolOperation(
                operation_id=uuid.uuid4().hex, run_id=run.id, step_index=step_index,
                tool=tool, idempotency_key=op_key, args_hash=args_hash,
            ))
        except Exception as e:  # noqa: BLE001
            _append_operation_error(store, run, tool=tool, args=args, thought=thought,
                                    step_index=step_index, error_type="unavailable",
                                    message=f"操作账本不可用: {e}",
                                    token_input=token_input, token_output=token_output)
            return
        if not op.is_new:
            if op.state == "succeeded":
                AGENT_OPERATIONS_TOTAL.labels(state="replayed").inc()
                observation_full = json.dumps(op.result or {}, ensure_ascii=False, default=str)
                observation = observation_full[:1500]
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, thought=thought,
                    action=tool, args=args, observation=observation,
                    observation_full=observation_full, latency_ms=0, token_input=token_input, token_output=token_output,
                ))
                return
            if op.state in ("started", "unknown"):
                AGENT_OPERATIONS_TOTAL.labels(state="unknown").inc()
                _append_operation_error(
                    store, run, tool=tool, args=args, thought=thought,
                    step_index=step_index, error_type="unknown_outcome",
                    message="该副作用操作已有未完成记录，禁止自动重试",
                    hint="先查询外部状态或人工接管",
                    token_input=token_input, token_output=token_output,
                )
                return

    call_args = dict(args)
    if spec is not None and spec.side_effect and spec.idempotent and spec.idempotency_key_field:
        field = spec.idempotency_key_field
        if not call_args.get(field):
            digest = hashlib.sha256(_canonical_args(args).encode("utf-8")).hexdigest()[:16]
            call_args[field] = f"{run.id}:{step_index}:{tool}:{digest}"

    timeout_s = _resolve_timeout(tool, spec)
    attempts = 1
    if spec is not None and spec.idempotent:
        attempts += max(0, int(spec.max_retries or 0))
    started = time.perf_counter()

    with trace_span(
        f"execute_tool {tool}",
        attributes={"gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool, "bagent.run_id": run.id,
                    "bagent.step_id": step_index},
    ) as span:
        for attempt in range(attempts):
            try:
                obs = _invoke_tool(tool, tools[tool], call_args, timeout_s)
                observation_full = json.dumps(obs, ensure_ascii=False, default=str)
                observation = observation_full[:1500]
                latency_ms = int((time.perf_counter() - started) * 1000)
                if op_key is not None:
                    result = obs if isinstance(obs, dict) else {"value": obs}
                    try:
                        store.mark_operation(op_key, "succeeded", result=result)
                        AGENT_OPERATIONS_TOTAL.labels(state="succeeded").inc()
                    except Exception:  # noqa: BLE001
                        pass
                span.set_attribute("bagent.tool.status", "ok")
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, thought=thought,
                    action=tool, args=args, observation=observation,
                    observation_full=observation_full, latency_ms=latency_ms,
                    token_input=token_input, token_output=token_output,
                ))
                return
            except Exception as e:  # noqa: BLE001
                err = _classify_tool_error(e)
                if (err.error_type == "timeout" and spec is not None
                        and spec.side_effect and not spec.idempotent):
                    err = ToolExecutionError(
                        "unknown_outcome", f"工具 {tool} 超时且可能有副作用",
                        retryable=False, hint="先查询外部状态或转人工，不要重试",
                    )
                if _should_retry(spec, err) and attempt < attempts - 1:
                    AGENT_STEPS_TOTAL.labels(tool=tool, status="retry").inc()
                    AGENT_TOOL_RETRIES.labels(tool=tool).inc()
                    span.set_attribute("bagent.tool.retry_count", attempt + 1)
                    time.sleep(get_settings().llm_retry_base_delay * (2 ** attempt)
                               + random.uniform(0, 0.2))
                    continue
                if op_key is not None:
                    op_state = ("unknown" if spec is not None and spec.side_effect
                                else "failed")
                    try:
                        store.mark_operation(op_key, op_state, error=str(err)[:300])
                        AGENT_OPERATIONS_TOTAL.labels(state=op_state).inc()
                    except Exception:  # noqa: BLE001
                        pass
                payload = {"error_type": err.error_type, "message": str(err)[:300],
                           "retryable": err.retryable, "hint": err.hint,
                           "details": err.details}
                observation_full = json.dumps(payload, ensure_ascii=False, default=str)
                observation = observation_full[:1500]
                latency_ms = int((time.perf_counter() - started) * 1000)
                span.set_attribute("bagent.tool.status", err.error_type)
                _append_step(store, run, AgentStep(
                    run_id=run.id, step_index=step_index, thought=thought,
                    action=f"{err.error_type}:{tool}", args=args,
                    observation=observation, observation_full=observation_full,
                    error=err.error_type,
                    latency_ms=latency_ms,
                    token_input=token_input, token_output=token_output,
                ))
                return


def _invoke_tool(tool: str, fn: Callable, args: dict, timeout_s: float | None):
    # Python 线程不能强杀；P1 只对只读工具启用超时，底层 HTTP 仍需自身 timeout。
    if not timeout_s:
        return fn(**args)
    future = _TOOL_EXECUTOR.submit(fn, **args)
    try:
        return future.result(timeout=timeout_s)
    except FuturesTimeoutError as e:
        raise ToolExecutionError(
            "timeout", f"工具 {tool} 执行超时", retryable=True,
            hint="缩小查询范围或稍后重试",
        ) from e


def _classify_tool_error(err: Exception) -> ToolExecutionError:
    if isinstance(err, ToolExecutionError):
        return err
    name = type(err).__name__.lower()
    if isinstance(err, (TimeoutError, ConnectionError, OSError)) or "timeout" in name or "connect" in name:
        return ToolExecutionError(
            "unavailable", str(err)[:200], retryable=True,
            hint="下游暂不可用，稍后重试或缩小范围",
        )
    return ToolExecutionError(
        "unknown", str(err)[:200], retryable=False,
        hint="检查参数或联系管理员",
    )


def _should_retry(spec: ToolSpec | None, err: ToolExecutionError) -> bool:
    return bool(spec and err.retryable and spec.idempotent
                and err.error_type in spec.retry_on)


def _resolve_timeout(tool: str, spec: ToolSpec | None) -> float | None:
    if spec is not None and spec.timeout_s is not None:
        return spec.timeout_s
    s = get_settings()
    if tool in ("kb_search", "web_search"):
        return s.search_timeout
    if tool == "kb_answer":
        return s.llm_timeout
    return None


def _scope_allowed(spec: ToolSpec | None, scopes: set[str] | frozenset[str] | None) -> bool:
    if scopes is None or spec is None or spec.required_scope is None:
        return True
    return "*" in scopes or spec.required_scope in scopes


def _append_operation_error(
    store: RunStore, run: AgentRun, *, tool: str, args: dict, thought: str | None,
    step_index: int, error_type: str, message: str, hint: str = "",
    token_input: int, token_output: int,
) -> None:
    observation_full = json.dumps({
        "error_type": error_type, "message": message,
        "retryable": False, "hint": hint,
    }, ensure_ascii=False)
    observation = observation_full[:1500]
    _append_step(store, run, AgentStep(
        run_id=run.id, step_index=step_index, thought=thought,
        action=f"{error_type}:{tool}", args=args, observation=observation,
        observation_full=observation_full, error=error_type, latency_ms=0,
        token_input=token_input, token_output=token_output,
    ))


def _append_forbidden(
    store: RunStore, run: AgentRun, *, tool: str, args: dict, thought: str | None,
    step_index: int, token_input: int, token_output: int,
) -> None:
    observation = json.dumps({
        "error_type": "forbidden",
        "message": f"当前身份没有工具 {tool} 的权限",
        "retryable": False,
        "hint": "换用有权限的工具或联系管理员",
    }, ensure_ascii=False)[:1500]
    _append_step(store, run, AgentStep(
        run_id=run.id, step_index=step_index, thought=thought,
        action=f"forbidden:{tool}", args=args, observation=observation,
        error="forbidden", latency_ms=0,
        token_input=token_input, token_output=token_output,
    ))


def _canonical_args(args: dict) -> str:
    return json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)


def _has_duplicate(store: RunStore, run: AgentRun, tool: str, args: dict) -> bool:
    key = _canonical_args(args)
    return any(s.action == tool and _canonical_args(s.args) == key
               for s in store.list_steps(run.id))


def _append_duplicate(
    store: RunStore, run: AgentRun, *, tool: str, args: dict, thought: str | None,
    step_index: int, token_input: int, token_output: int,
) -> None:
    observation = json.dumps({
        "error_type": "duplicate",
        "message": f"工具 {tool} 以相同参数重复调用",
        "retryable": False,
        "hint": "换查询词、换工具或直接回答",
    }, ensure_ascii=False)[:1500]
    _append_step(store, run, AgentStep(
        run_id=run.id, step_index=step_index, thought=thought,
        action=f"duplicate:{tool}", args=args, observation=observation,
        error="duplicate", latency_ms=0,
        token_input=token_input, token_output=token_output,
    ))


def _emit_run_started(run: AgentRun) -> None:
    emit_event({
        "event_id": 0, "event": "run_started", "run_id": run.id,
        "step_index": None, "status": run.status,
        "data": {"run_id": run.id},
    })


def _append_step(store: RunStore, run: AgentRun, step: AgentStep) -> None:
    store.add_step(step)
    run.step_count += 1
    run.token_input += step.token_input
    run.token_output += step.token_output
    store.update(run)
    tool_label = _tool_label(step.action)
    step_status = "error" if step.error or step.action.startswith(
        ("invalid", "unknown:", "error:", "invalid_args:", "rejected:",
         "timeout:", "unavailable:", "duplicate:", "forbidden:", "unknown_outcome:")
    ) else "ok"
    AGENT_STEPS_TOTAL.labels(tool=tool_label, status=step_status).inc()
    if step.latency_ms is not None:
        AGENT_STEP_LATENCY.labels(tool=tool_label).observe(step.latency_ms / 1000.0)
    log_event(
        _log, "info", "agent_step",
        run_id=run.id, step_id=step.step_index,
        tool=tool_label, status=step_status, latency_ms=step.latency_ms,
        token_input=step.token_input, token_output=step.token_output,
    )
    emit_event({
        "event_id": 2 * step.step_index + 2, "event": "tool_result",
        "run_id": run.id, "step_index": step.step_index, "status": step_status,
        "data": {"action": step.action, "latency_ms": step.latency_ms,
                 "token_input": step.token_input, "token_output": step.token_output},
    })


def _set_status(store: RunStore, run: AgentRun, status: str, *, error: str | None = None) -> None:
    run.status = status
    if error is not None:
        run.error = error
    store.update(run)
    if status in TERMINAL_STATUSES:
        AGENT_RUNS_TOTAL.labels(status=status).inc()
    log_event(
        _log, "info", "agent_run",
        run_id=run.id, status=status, step_count=run.step_count,
        token_input=run.token_input, token_output=run.token_output,
        error=run.error,
    )
    pending = run.pending_action or {}
    emit_event({
        "event_id": 2 * run.step_count + 2,
        "event": status_event_type(status), "run_id": run.id,
        "step_index": pending.get("step_index"), "status": status,
        "data": {"run_id": run.id, "status": status, "step_count": run.step_count,
                 "tool": pending.get("tool"), "error": run.error or None},
    })


def _result(run: AgentRun, store: RunStore) -> dict:
    return {
        "goal": run.goal,
        "steps": [step_to_dict(s) for s in store.list_steps(run.id)],
        "final": run.final,
        "status": run.status,
        "run_id": run.id,
        "answer": run.answer,
        "pending_action": run.pending_action,
        "token_input": run.token_input,
        "token_output": run.token_output,
    }


def _history(goal: str, steps: list[AgentStep]) -> list[dict]:
    history: list[dict] = [{"role": "user", "content": goal}]
    for s in steps:
        if s.action == "final":
            continue
        action = s.action
        if action.startswith(("invalid", "unknown:", "error:")):
            assistant = {"thought": s.thought, "error": s.observation or s.error}
        elif action.startswith(("awaiting_approval:", "rejected:")):
            assistant = {"thought": s.thought, "tool": action, "args": s.args}
        else:
            assistant = {"thought": s.thought, "tool": action, "args": s.args}
        history.append({"role": "assistant",
                        "content": json.dumps(assistant, ensure_ascii=False)})
        if s.observation is not None or s.error is not None:
            history.append({
                "role": "user",
                "content": f"工具 {action} 结果：{s.observation or s.error}\n请给出下一步 JSON 动作或 final。",
            })
    return history


def _render(msg: dict) -> str:
    return f"{msg['role']}: {msg['content']}"


def _tool_label(action: str) -> str:
    if action.startswith("unknown:"):
        return "unknown"
    if action.startswith("error:"):
        return "error"
    if action.startswith("invalid_args:"):
        return "invalid_args"
    if action.startswith("timeout:"):
        return "timeout"
    if action.startswith("unavailable:"):
        return "unavailable"
    if action.startswith("duplicate:"):
        return "duplicate"
    if action.startswith("forbidden:"):
        return "forbidden"
    if action.startswith("unknown_outcome:"):
        return "unknown_outcome"
    if action.startswith("awaiting_approval:"):
        return "awaiting_approval"
    if action.startswith("rejected:"):
        return "rejected"
    return action or "unknown"
