"""P0：把 agent 轨迹中的证据归一化，并复用 RAG 的最终答案闸门。

只处理 kb_search/web_search 的 observation；没有证据时保留原文但标记低置信，
不新增“必须拒答”的业务规则。
"""
from __future__ import annotations

import json
from typing import Any

from app.generation.generator import Answer, PreparedAnswer, finalize_answer
from app.generation.prompts import SYSTEM_CITED
from app.retrieval.store import RetrievedChunk


def _field(step: Any, name: str, default=None):
    if isinstance(step, dict):
        return step.get(name, default)
    return getattr(step, name, default)


def _items_from_observation(observation: str | None) -> list[dict]:
    if not observation:
        return []
    try:
        value = json.loads(observation)
    except (TypeError, ValueError):
        return []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def collect_evidence(steps: list[Any]) -> list[RetrievedChunk]:
    """按步骤顺序收集 kb/web 证据，按 source 去重（保留第一次出现的内容）。"""
    evidence: list[RetrievedChunk] = []
    seen: set[str] = set()

    def add(item: dict, fallback_index: int) -> None:
        source = str(item.get("source") or item.get("url") or "").strip()
        content = item.get("content") or item.get("snippet") or item.get("context")
        if not source or not content:
            return
        if source in seen:
            return
        seen.add(source)
        metadata = dict(item.get("metadata") or {})
        metadata.setdefault("source_type", "web" if item.get("url") else "kb")
        evidence.append(RetrievedChunk(
            chunk_id=int(item.get("chunk_id") or fallback_index),
            document_id=int(item.get("document_id") or 0),
            source=source,
            content=str(content),
            score=float(item.get("score") or 0.0),
            metadata=metadata,
            context=str(item.get("context") or content),
        ))

    for idx, step in enumerate(steps, start=1):
        action = str(_field(step, "action", ""))
        if action not in ("kb_search", "web_search"):
            continue
        for item in _items_from_observation(_field(step, "observation")):
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("sources"), list):
                for source_item in item["sources"]:
                    if isinstance(source_item, dict):
                        add(source_item, idx)
            else:
                add(item, idx)
    return evidence


def finalize_agent_answer(
    goal: str,
    steps: list[Any],
    final_text: str | None,
    *,
    check_faithfulness: bool = False,
) -> Answer:
    """生成带来源的 Answer；无证据时保留原文并标记 low_confidence。"""
    evidence = collect_evidence(steps)
    text = final_text or ""
    if not evidence:
        return Answer(query=goal, text=text, sources=[], grounded=False,
                      low_confidence=True, is_refusal=False)

    blocks = [f"[{i}] (来源: {c.source})\n{c.context}" for i, c in enumerate(evidence, start=1)]
    context = "\n\n---\n\n".join(blocks)
    prep = PreparedAnswer(
        query=goal,
        rewritten_query=goal,
        chunks=evidence,
        retrieval_context=context,
        context=context,
        system=SYSTEM_CITED,
        degraded=False,
        self_rag_applied=False,
        iterations=1,
        evidence_sufficient=None,
        check_faithfulness=check_faithfulness,
        refusal=None,
    )
    if not text.strip():
        return Answer(query=goal, text=text, sources=[], grounded=False,
                      low_confidence=True, is_refusal=False)
    return finalize_answer(prep, text)
