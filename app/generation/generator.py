"""RAG 查询 pipeline（M3）：改写 -> 检索 -> 生成 -> 引用校验 -> 忠实度/置信标记。

为支持 SSE，把流程拆成两段：
- prepare_answer()：预算/改写/检索/上下文/提示词（不调用生成）
- finalize_answer()：引用校验、忠实度、冲突、指标与日志
非流式 answer_query() 与流式 stream_answer() 复用同一套准备与收尾，避免两套口径漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.generation.budget import should_degrade
from app.generation.citation import split_valid_invalid, strip_invalid_citations
from app.generation.faithfulness import REFUSAL_PHRASE, assess_faithfulness
from app.generation.llm import LLMUnavailable, get_llm
from app.generation.prompts import SYSTEM_BASE, SYSTEM_CITED
from app.generation.rewriter import rewrite_query
from app.generation.conflict import detect_conflict
from app.observability.logging import get_logger, log_event
from app.observability.metrics import COST_DEGRADED, RAG_QUALITY, STAGE_LATENCY
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
    degraded: bool = False     # M13：超成本预算已降级(rerank/忠实度/条数)


@dataclass
class PreparedAnswer:
    """准备阶段结果：检索与上下文已就绪，等待（流式或非流式）生成。"""

    query: str
    rewritten_query: str
    chunks: list[RetrievedChunk]
    retrieval_context: str
    context: str
    system: str
    degraded: bool
    self_rag_applied: bool
    iterations: int
    evidence_sufficient: bool | None
    check_faithfulness: bool
    refusal: Answer | None = None


def _build_context(chunks: list[RetrievedChunk], guard: bool = False) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        ctx = sanitize_query(c.context) if guard else c.context
        blocks.append(f"[{i}] (来源: {c.source})\n{ctx}")
    return "\n\n---\n\n".join(blocks)


def _memory_context(
    query: str, user_id: str | None,
    *, use_memory: bool | None = None, min_trust: str | None = None, k: int | None = None,
) -> str:
    """召回记忆作为背景证据：personal(用户私有) + knowledge(共享、按 trust 门控)。

    门控全靠 min_trust：draft 默认不参与作答（防污染），verified/curated 才进。use_memory/min_trust
    供实验显式覆盖（不依赖全局 config）。"""
    s = get_settings()
    enabled = s.memory_enabled if use_memory is None else use_memory
    if not enabled:
        return ""
    trust = min_trust or s.memory_min_trust_for_answer
    kk = k or s.memory_context_k
    from app.db.session import get_session
    from app.ingestion.embedder import get_embedder
    from app.memory.store import search_memories

    session = get_session()
    try:
        vec = get_embedder().encode_query(query)
        personal = search_memories(session, vec, scope="personal", owner_user_id=user_id,
                                   min_trust=trust, k=kk) if user_id else []
        knowledge = search_memories(session, vec, scope="knowledge",
                                    min_trust=trust, k=kk)
    finally:
        session.close()
    blocks = []
    if personal:
        blocks.append("【关于用户的已知信息】\n" + "\n".join(f"- {m.content}" for m in personal))
    if knowledge:
        playbook = [m for m in knowledge if getattr(m, "level", "fact") == "playbook"]
        facts = [m for m in knowledge if getattr(m, "level", "fact") != "playbook"]
        if playbook:
            blocks.append("【可复用的经验原则】\n" + "\n".join(f"- {m.content}" for m in playbook))
        if facts:
            blocks.append("【已沉淀的具体事实】\n" + "\n".join(
                f"- {m.content}" + (f"（来源:{m.source_ref}）" if m.source_ref else "") for m in facts))
    return "\n\n".join(blocks)


def _sources(chunks: list[RetrievedChunk]) -> list[dict]:
    return [{"id": c.chunk_id, "source": c.source, "score": round(c.score, 4)} for c in chunks]


def build_user_prompt(prep: PreparedAnswer) -> str:
    return f"【参考资料】\n{prep.context}\n\n【问题】\n{prep.rewritten_query}"


def prepare_answer(
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
    force_citation: bool | None = None,
    check_faithfulness: bool | None = None,
    self_rag: bool | None = None,
    user_id: str | None = None,
    use_memory: bool | None = None,
    memory_min_trust: str | None = None,
) -> PreparedAnswer:
    """预算降级 -> 多轮改写 -> 检索 -> 记忆上下文 -> 提示词，不调用生成。"""
    s = get_settings()
    force_citation = s.force_citation if force_citation is None else force_citation
    check_faith = s.faithfulness_enabled if check_faithfulness is None else check_faithfulness
    use_self_rag = s.self_rag_enabled if self_rag is None else self_rag
    guard = s.prompt_injection_guard
    if guard and detect_injection(query):
        log_event(_log, "warning", "injection_detected_in_query")
        query = sanitize_query(query)

    # 0) 成本预算：预估超预算则降级（缩 top_k、跳 rerank、关忠实度）
    degraded = False
    eff_top_k = top_k or s.retrieval_top_k
    do_rerank: bool | None = None
    if should_degrade(eff_top_k, s.request_token_budget):
        degraded = True
        eff_top_k = min(eff_top_k, s.degraded_top_k)
        if s.degrade_skip_rerank:
            do_rerank = False
        if s.degrade_skip_faithfulness:
            check_faith = False
        COST_DEGRADED.inc()
        log_event(_log, "info", "cost_degraded", top_k=eff_top_k, budget=s.request_token_budget)

    # 1) 多轮改写（有历史才做）
    rewritten = rewrite_query(query, history or [], get_llm()) if (history and s.rewrite_enabled) else query

    # 2) 检索（self-RAG 时带反思重检）
    sr_meta = {"iterations": 1, "queries": [rewritten], "sufficient": None}
    if use_self_rag:
        chunks, sr_meta = retrieve_with_reflection(
            rewritten, top_k=top_k, max_iters=s.self_rag_max_iters, llm=get_llm()
        )
    else:
        chunks = retrieve(rewritten, top_k=eff_top_k, rerank=do_rerank)

    if not chunks:
        refusal = Answer(query=query, rewritten_query=rewritten, text=REFUSAL_MSG,
                         sources=[], is_refusal=True, self_rag_applied=use_self_rag,
                         iterations=sr_meta["iterations"], degraded=degraded)
        return PreparedAnswer(
            query=query, rewritten_query=rewritten, chunks=[], retrieval_context="", context="",
            system="", degraded=degraded, self_rag_applied=use_self_rag,
            iterations=sr_meta["iterations"], evidence_sufficient=None,
            check_faithfulness=check_faith, refusal=refusal,
        )

    # 3) 生成前的上下文：记忆只用于个性化生成，不计入忠实度证据
    retrieval_context = _build_context(chunks, guard=guard)
    mem_block = _memory_context(rewritten, user_id,
                                use_memory=use_memory, min_trust=memory_min_trust)
    context = (mem_block + "\n\n---\n\n" + retrieval_context) if mem_block else retrieval_context
    system = SYSTEM_CITED if force_citation else SYSTEM_BASE
    return PreparedAnswer(
        query=query, rewritten_query=rewritten, chunks=chunks,
        retrieval_context=retrieval_context, context=context, system=system,
        degraded=degraded, self_rag_applied=use_self_rag,
        iterations=sr_meta["iterations"], evidence_sufficient=sr_meta.get("sufficient"),
        check_faithfulness=check_faith,
    )


def finalize_answer(
    prep: PreparedAnswer, raw_text: str, *, streamed_text: str | None = None,
    include_conflict_prefix: bool = True,
) -> Answer:
    """引用校验、忠实度、冲突检测、质量指标与结构化日志。"""
    s = get_settings()
    num = len(prep.chunks)
    _, invalid = split_valid_invalid(raw_text, num)
    clean = strip_invalid_citations(raw_text, num)
    text = clean if streamed_text is None else streamed_text

    ans = Answer(query=prep.query, rewritten_query=prep.rewritten_query, text=text,
                 sources=_sources(prep.chunks), invalid_citations=invalid,
                 self_rag_applied=prep.self_rag_applied, iterations=prep.iterations,
                 evidence_sufficient=prep.evidence_sufficient)

    # 4) 忠实度 / 幻觉率
    if prep.check_faithfulness:
        with STAGE_LATENCY.labels(stage="faithfulness").time():
            report = assess_faithfulness(prep.retrieval_context, text, get_llm(),
                                         threshold=s.faithfulness_threshold)
        ans.faithfulness = report["faithfulness"]
        ans.hallucination_rate = report.get("hallucination_rate")
        ans.grounded = report["grounded"]
        ans.is_refusal = report["is_refusal"]
        ans.low_confidence = (not report["is_refusal"]) and (not report["grounded"])

    # 5) 知识冲突检测（self-RAG 开启且启用时）
    if prep.self_rag_applied and s.conflict_check_enabled and len(prep.chunks) >= 2:
        cres = detect_conflict(prep.rewritten_query, prep.context, get_llm())
        ans.conflict = cres["conflict"]
        ans.conflict_note = cres["explanation"]
        if ans.conflict:
            # 冲突时不偷选一个：标注、降置信，交调用方决策
            ans.grounded = False
            ans.low_confidence = True
            if include_conflict_prefix:
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

    ans.degraded = prep.degraded
    log_event(
        _log, "info", "rag_query",
        refusal=ans.is_refusal, grounded=ans.grounded, low_confidence=ans.low_confidence,
        faithfulness=ans.faithfulness, n_sources=len(ans.sources),
        invalid_citations=len(ans.invalid_citations), rewritten=(ans.rewritten_query != prep.query),
        self_rag=ans.self_rag_applied, iterations=ans.iterations, conflict=ans.conflict,
    )
    return ans


def answer_query(
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
    force_citation: bool | None = None,
    check_faithfulness: bool | None = None,
    self_rag: bool | None = None,
    user_id: str | None = None,
    use_memory: bool | None = None,
    memory_min_trust: str | None = None,
) -> Answer:
    prep = prepare_answer(
        query, top_k=top_k, history=history, force_citation=force_citation,
        check_faithfulness=check_faithfulness, self_rag=self_rag, user_id=user_id,
        use_memory=use_memory, memory_min_trust=memory_min_trust,
    )
    if prep.refusal is not None:
        return prep.refusal
    try:
        text = get_llm().generate(system=prep.system, user=build_user_prompt(prep))
    except LLMUnavailable:
        # provider 全挂：优雅降级，不裸 500；仍返回已检索到的来源供参考
        return Answer(query=prep.query, rewritten_query=prep.rewritten_query,
                      text="服务暂时不可用（模型后端均不可用），请稍后重试。",
                      sources=_sources(prep.chunks), low_confidence=True)
    return finalize_answer(prep, text)
