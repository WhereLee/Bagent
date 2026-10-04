"""M9b 异步写回（sleep-time）：从对话抽取事实，按去重/佐证规则写入记忆。

设计要点：
- 不阻塞回答：由 FastAPI BackgroundTasks 调用，开自己的 session。
- personal 偏好由用户亲述 → 直接 verified（用户是自己偏好的权威，且私有低污染）。
- knowledge/research 结论 → 先 draft；再次被佐证(bump support)达阈值促升 verified。
- llm/embedder 注入，便于测试与复用。
"""
from __future__ import annotations

from sqlalchemy import select

from app.observability.logging import get_logger, log_event
from app.observability.metrics import MEMORY_OPS
from app.memory.extract import extract_facts
from app.memory.lifecycle import decide_op, promote_trust
from app.memory.store import (
    add_memory,
    bump_support,
    content_hash,
    find_similar,
    update_content,
)
from app.db.models import Memory

_log = get_logger("memory.writeback")


def _exists_hash(session, content: str) -> bool:
    return session.scalar(
        select(Memory).where(Memory.content_hash == content_hash(content),
                             Memory.is_deleted == False)  # noqa: E712
    ) is not None


def run_writeback(
    *, session, conversation: str, user_id: str | None, llm, embedder,
    dedup_sim: float, promote_threshold: int, source_ref: str | None = None,
    tenant: str | None = None,
) -> dict:
    facts = extract_facts(conversation, llm)
    summary = {"add": 0, "update": 0, "noop": 0, "promote": 0, "skip": 0}
    for f in facts:
        owner = user_id if f.scope == "personal" else None
        if f.scope == "personal" and not owner:
            summary["skip"] += 1        # 无用户身份不写私有记忆
            continue
        vec = embedder.encode_documents([f.content])[0]
        similar = find_similar(session, vec, scope=f.scope, owner_user_id=owner, tenant=tenant, k=3)
        op, tid = decide_op(similar, dedup_sim, exact_hash_match=_exists_hash(session, f.content))
        MEMORY_OPS.labels(scope=f.scope, op=op.lower()).inc()

        if op == "NOOP":
            summary["noop"] += 1
        elif op == "UPDATE" and tid is not None:
            row = bump_support(session, tid)          # 视为再次佐证
            if row is not None:
                new_trust = promote_trust(row.support, promote_threshold, row.trust)
                if new_trust != row.trust:
                    row.trust = new_trust
                    summary["promote"] += 1
                    MEMORY_OPS.labels(scope=f.scope, op="promote").inc()
                elif row.trust == "draft" and f.kind == "claim":
                    update_content(session, tid, f.content, vec)  # 内容有实质更新才改写
            summary["update"] += 1
        else:  # ADD
            trust = "verified" if f.scope == "personal" else "draft"
            add_memory(session, scope=f.scope, content=f.content, embedding=vec,
                       owner_user_id=owner, kind=f.kind, source_type="research",
                       source_ref=source_ref, trust=trust, tenant_id=tenant,
                       level=f.level, tags=f.tags)
            summary["add"] += 1

    session.commit()
    log_event(_log, "info", "memory_writeback", facts=len(facts), **summary)
    return summary
