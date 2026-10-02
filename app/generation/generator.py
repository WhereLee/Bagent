"""RAG 查询 pipeline（M3）：改写 -> 检索 -> 生成 -> 引用校验 -> 忠实度/置信标记。"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.generation.citation import split_valid_invalid, strip_invalid_citations
from app.generation.faithfulness import REFUSAL_PHRASE, assess_faithfulness
from app.generation.llm import LLMUnavailable, get_llm
from app.generation.prompts import SYSTEM_BASE, SYSTEM_CITED
from app.generation.rewriter import rewrite_query
from app.generation.conflict import detect_conflict
from app.observability.logging import get_logger, log_event
from app.observability.metrics import RAG_QUALITY, STAGE_LATENCY
from app.retrieval.retriever import retrieve
from app.retrieval.selfrag import retrieve_with_reflection
from app.retrieval.store import RetrievedChunk
from app.security import detect_injection, sanitize_query

_log = get_logger("rag")

REFUSAL_MSG = REFUSAL_PHRASE + "（知识库中未检索到相关内容）。"


@dataclass
class Answer:
    query: str
    text: str
    sources: list[dict]
    rewritten_query: str | None = None
    faithfulness: float | None = None
    hallucination_rate: float | None = None
    grounded: bool = False
    is_refusal: bool = False
    low_confidence: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    # M8 self-RAG / 冲突
    self_rag_applied: bool = False
    iterations: int = 1
    evidence_sufficient: bool | None = None
    conflict: bool = False
    conflict_note: str = ""


def _build_context(chunks: list[RetrievedChunk], guard: bool = False) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        ctx = sanitize_query(c.context) if guard else c.context
        blocks.append(f"[{i}] (来源: {c.source})\n{ctx}")
    return "\n\n---\n\n".join(blocks)


def answer_query(
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
    force_citation: bool | None = None,
    check_faithfulness: bool | None = None,
    self_rag: bool | None = None,
) -> Answer:
    s = get_settings()
    force_citation = s.force_citation if force_citation is None else force_citation
    check_faith = s.faithfulness_enabled if check_faithfulness is None else check_faithfulness
    use_self_rag = s.self_rag_enabled if self_rag is None else self_rag
    guard = s.prompt_injection_guard
    if guard and detect_injection(query):
        log_event(_log, "warning", "injection_detected_in_query")
        query = sanitize_query(query)

    # 1) 多轮改写（有历史才做）
    rewritten = rewrite_query(query, history or [], get_llm()) if (history and s.rewrite_enabled) else query

    # 2) 检索（self-RAG 时带反思重检）
    sr_meta = {"iterations": 1, "queries": [rewritten], "sufficient": None}
    if use_self_rag:
        chunks, sr_meta = retrieve_with_reflection(
            rewritten, top_k=top_k, max_iters=s.self_rag_max_iters, llm=get_llm()
        )
    else:
        chunks = retrieve(rewritten, top_k=top_k)
    if not chunks:
        return Answer(query=query, rewritten_query=rewritten, text=REFUSAL_MSG,
                      sources=[], is_refusal=True, self_rag_applied=use_self_rag,
                      iterations=sr_meta["iterations"])

    # 3) 生成（可选强制引用；citation 编号与检索列表 1:1）
    context = _build_context(chunks, guard=guard)
    system = SYSTEM_CITED if force_citation else SYSTEM_BASE
    try:
        text = get_llm().generate(system=system, user=f"【参考资料】\n{context}\n\n【问题】\n{rewritten}")
    except LLMUnavailable:
        # provider 全挂：优雅降级，不裸 500；仍返回已检索到的来源供参考
        return Answer(query=query, rewritten_query=rewritten,
                      text="服务暂时不可用（模型后端均不可用），请稍后重试。",
                      sources=[{"id": c.chunk_id, "source": c.source, "score": round(c.score, 4)} for c in chunks],
                      low_confidence=True)

    # 4) 引用校验：剔除越界引用
    num = len(chunks)
    _, invalid = split_valid_invalid(text, num)
    text = strip_invalid_citations(text, num)

    # 5) 忠实度 / 幻觉率
    ans = Answer(query=query, rewritten_query=rewritten, text=text,
                 sources=[{"id": c.chunk_id, "source": c.source, "score": round(c.score, 4)} for c in chunks],
                 invalid_citations=invalid,
                 self_rag_applied=use_self_rag,
                 iterations=sr_meta["iterations"],
                 evidence_sufficient=sr_meta.get("sufficient"))
    if check_faith:
        with STAGE_LATENCY.labels(stage="faithfulness").time():
            report = assess_faithfulness(context, text, get_llm(), threshold=s.faithfulness_threshold)
        ans.faithfulness = report["faithfulness"]
        ans.hallucination_rate = report.get("hallucination_rate")
        ans.grounded = report["grounded"]
        ans.is_refusal = report["is_refusal"]
        # 有内容但忠实度过低 -> 低置信（不假装确定）
        ans.low_confidence = (not report["is_refusal"]) and (not report["grounded"])

    # 5.5) 知识冲突检测（self-RAG 开启且启用时）
    if use_self_rag and s.conflict_check_enabled and len(chunks) >= 2:
        cres = detect_conflict(rewritten, context, get_llm())
        ans.conflict = cres["conflict"]
        ans.conflict_note = cres["explanation"]
        if ans.conflict:
            # 冲突时不偷选一个：标注、降置信，交调用方决策
            ans.grounded = False
            ans.low_confidence = True
            ans.text = "【注意】检索到的资料对该问题存在矛盾（" + ans.conflict_note + "），以下答案请谨慎采信：\n" + text
    # 质量信号计数
    if ans.conflict:
        RAG_QUALITY.labels(outcome="conflict").inc()
    if ans.is_refusal:
        RAG_QUALITY.labels(outcome="refusal").inc()
    elif ans.low_confidence:
        RAG_QUALITY.labels(outcome="low_confidence").inc()
    elif ans.grounded:
        RAG_QUALITY.labels(outcome="grounded").inc()

    log_event(
        _log, "info", "rag_query",
        refusal=ans.is_refusal, grounded=ans.grounded, low_confidence=ans.low_confidence,
        faithfulness=ans.faithfulness, n_sources=len(ans.sources),
        invalid_citations=len(ans.invalid_citations), rewritten=(ans.rewritten_query != query),
        self_rag=ans.self_rag_applied, iterations=ans.iterations, conflict=ans.conflict,
    )
    return ans
