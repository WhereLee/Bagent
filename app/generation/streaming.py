"""SSE 流式作答：prepare -> token 流 -> finalize。

事件顺序：start -> meta -> token* -> done；异常时在 token 之后插入 error，仍以 done 收尾。
done 携带完整 Answer 字段（含 sources/faithfulness/invalid_citations/conflict_note），
客户端可据此更新引用与置信提示。
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict

from app.generation.citation import CitationStreamFilter
from app.generation.generator import (
    Answer,
    PreparedAnswer,
    _sources,
    build_user_prompt,
    finalize_answer,
    prepare_answer,
)
from app.generation.llm import LLMUnavailable, get_llm
from app.observability.logging import get_logger, log_event
from app.observability.metrics import STAGE_LATENCY

_log = get_logger("rag_stream")

_UNAVAILABLE = "服务暂时不可用（模型后端均不可用），请稍后重试。"
_PREPARE_FAILED = "服务暂时不可用（检索/生成准备失败），请稍后重试。"
_EMPTY = "模型未返回内容，请稍后重试。"


def _safe_finalize(prep: PreparedAnswer, raw: str, sent: str) -> Answer:
    """流式收尾失败不能把连接打断：退回保守 Answer（低置信、保留已发送文本）。"""
    try:
        return finalize_answer(prep, raw, streamed_text=sent, include_conflict_prefix=False)
    except Exception as e:  # noqa: BLE001
        log_event(_log, "error", "stream_finalize_failed", err=str(e)[:180])
        ans = Answer(
            query=prep.query, rewritten_query=prep.rewritten_query, text=sent,
            sources=_sources(prep.chunks), low_confidence=True,
            self_rag_applied=prep.self_rag_applied, iterations=prep.iterations,
            evidence_sufficient=prep.evidence_sufficient, degraded=prep.degraded,
        )
        return ans


def stream_answer(
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
    force_citation: bool | None = None,
    check_faithfulness: bool | None = None,
    self_rag: bool | None = None,
    user_id: str | None = None,
    use_memory: bool | None = None,
    memory_min_trust: str | None = None,
) -> Iterator[tuple[str, dict]]:
    yield "start", {"query": query}

    try:
        prep = prepare_answer(
            query, top_k=top_k, history=history, force_citation=force_citation,
            check_faithfulness=check_faithfulness, self_rag=self_rag, user_id=user_id,
            use_memory=use_memory, memory_min_trust=memory_min_trust,
        )
    except Exception as e:  # noqa: BLE001
        log_event(_log, "error", "stream_prepare_failed", err=str(e)[:180])
        ans = Answer(query=query, text=_PREPARE_FAILED, sources=[], low_confidence=True)
        yield "error", {"message": "prepare_failed"}
        yield "done", asdict(ans)
        return

    if prep.refusal is not None:
        refusal = prep.refusal
        yield "meta", {"query": refusal.query, "rewritten_query": refusal.rewritten_query,
                       "sources": refusal.sources, "degraded": refusal.degraded}
        yield "token", {"text": refusal.text}
        yield "done", asdict(refusal)
        return

    yield "meta", {"query": prep.query, "rewritten_query": prep.rewritten_query,
                   "sources": _sources(prep.chunks), "degraded": prep.degraded}

    filt = CitationStreamFilter(len(prep.chunks))
    raw_parts: list[str] = []
    sent_parts: list[str] = []
    try:
        llm = get_llm()
        with STAGE_LATENCY.labels(stage="llm_stream").time():
            for delta in llm.stream(system=prep.system, user=build_user_prompt(prep)):
                raw_parts.append(delta)
                out = filt.feed(delta)
                if out:
                    sent_parts.append(out)
                    yield "token", {"text": out}
            tail = filt.flush()
            if tail:
                sent_parts.append(tail)
                yield "token", {"text": tail}
    except LLMUnavailable as e:
        log_event(_log, "error", "stream_llm_unavailable", err=str(e)[:180])
        raw = "".join(raw_parts)
        sent = "".join(sent_parts)
        if raw:
            ans = _safe_finalize(prep, raw, sent)
            ans.low_confidence = True
        else:
            ans = Answer(query=prep.query, text=_UNAVAILABLE,
                         sources=_sources(prep.chunks), low_confidence=True)
        yield "error", {"message": "llm_unavailable"}
        yield "done", asdict(ans)
        return
    except Exception as e:  # noqa: BLE001
        log_event(_log, "error", "stream_failed", err=str(e)[:180])
        raw = "".join(raw_parts)
        sent = "".join(sent_parts)
        if raw:
            ans = _safe_finalize(prep, raw, sent)
            ans.low_confidence = True
        else:
            ans = Answer(query=prep.query, text=_UNAVAILABLE,
                         sources=_sources(prep.chunks), low_confidence=True)
        yield "error", {"message": "stream_failed"}
        yield "done", asdict(ans)
        return

    raw = "".join(raw_parts)
    sent = "".join(sent_parts)
    if not raw:
        ans = Answer(query=prep.query, text=_EMPTY,
                     sources=_sources(prep.chunks), low_confidence=True)
    else:
        ans = _safe_finalize(prep, raw, sent)
    yield "done", asdict(ans)
