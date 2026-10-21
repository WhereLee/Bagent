"""P2：Agent 崩溃恢复 worker。

扫描 lease 过期的 running run，重新抢租约并从已持久化 step 继续执行。
它是可选进程，不改变 API 的同步入口；生产可用 systemd/docker compose 单独跑。
"""
from __future__ import annotations

import time
from typing import Callable

from app.agent.react import continue_run
from app.agent.state import PostgresRunStore, RunStore, is_terminal
from app.config import get_settings
from app.db.session import init_schema
from app.generation.llm import get_llm
from app.observability.logging import get_logger, log_event
from app.tools import DEFAULT_REQUIRES_APPROVAL, TOOL_SPECS, build_tools

_log = get_logger("agent.worker")


def recover_once(
    *,
    store: RunStore | None = None,
    llm_factory: Callable = get_llm,
    tools_factory: Callable = build_tools,
    limit: int | None = None,
) -> list[str]:
    """恢复至多 limit 个过期 run，返回实际恢复的 run_id。"""
    settings = get_settings()
    store = store if store is not None else PostgresRunStore()
    recovered: list[str] = []
    for run in store.list_recoverable(limit=limit or settings.agent_recovery_limit):
        try:
            tools = tools_factory(user_id=run.user_id, scopes=run.scopes)
            result = continue_run(
                run.id, llm=llm_factory(), tools=tools,
                tool_specs=TOOL_SPECS, store=store,
                requires_confirm=DEFAULT_REQUIRES_APPROVAL,
                finalize=True, check_faithfulness=settings.faithfulness_enabled,
                scopes=run.scopes,
            )
            if result is not None and is_terminal(result.status):
                recovered.append(run.id)
                log_event(_log, "info", "agent_recovered", run_id=run.id,
                          status=result.status, attempt=result.attempt)
        except Exception as e:  # noqa: BLE001
            log_event(_log, "error", "agent_recovery_failed", run_id=run.id,
                      err=str(e)[:200])
            current = store.get(run.id)
            if current is not None and current.status == "running":
                current.status = "failed"
                current.error = f"recovery failed: {e}"[:200]
                store.update(current)
    return recovered


def run_forever(*, once: bool = False) -> None:
    settings = get_settings()
    init_schema()
    if once:
        recover_once()
        return
    log_event(_log, "info", "agent_worker_started",
              poll_s=settings.agent_worker_poll_s)
    while True:
        recover_once()
        time.sleep(settings.agent_worker_poll_s)
