"""RAG 查询 pipeline（M3）：改写 -> 检索 -> 生成 -> 引用校验 -> 忠实度/置信标记。"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.generation.citation import split_valid_invalid, strip_invalid_citations
from app.generation.faithfulness import REFUSAL_PHRASE, assess_faithfulness
from app.generation.llm import get_llm
from app.generation.prompts import SYSTEM_BASE, SYSTEM_CITED
from app.generation.rewriter import rewrite_query
from app.observability.logging import get_logger, log_event
from app.observability.metrics import RAG_QUALITY, STAGE_LATENCY
from app.retrieval.retriever import retrieve
from app.retrieval.store import RetrievedChunk

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


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = [f"[{i}] (来源: {c.source})\n{c.context}" for i, c in enumerate(chunks, start=1)]
    return "\n\n---\n\n".join(blocks)


def answer_query(
    query: str,
    top_k: int | None = None,
    history: list[dict] | None = None,
    force_citation: bool | None = None,
    check_faithfulness: bool | None = None,
) -> Answer:
    s = get_settings()
    force_citation = s.force_citation if force_citation is None else force_citation
    check_faith = s.faithfulness_enabled if check_faithfulness is None else check_faithfulness

    # 1) 多轮改写（有历史才做）
    rewritten = rewrite_query(query, history or [], get_llm()) if (history and s.rewrite_enabled) else query

    # 2) 检索
    chunks = retrieve(rewritten, top_k=top_k)
    if not chunks:
        return Answer(query=query, rewritten_query=rewritten, text=REFUSAL_MSG,
                      sources=[], is_refusal=True)

    # 3) 生成（可选强制引用；citation 编号与检索列表 1:1）
    context = _build_context(chunks)
    system = SYSTEM_CITED if force_citation else SYSTEM_BASE
    text = get_llm().generate(system=system, user=f"【参考资料】\n{context}\n\n【问题】\n{rewritten}")

    # 4) 引用校验：剔除越界引用
    num = len(chunks)
    _, invalid = split_valid_invalid(text, num)
    text = strip_invalid_citations(text, num)

    # 5) 忠实度 / 幻觉率
    ans = Answer(query=query, rewritten_query=rewritten, text=text,
                 sources=[{"id": c.chunk_id, "source": c.source, "score": round(c.score, 4)} for c in chunks],
                 invalid_citations=invalid)
    if check_faith:
        with STAGE_LATENCY.labels(stage="faithfulness").time():
            report = assess_faithfulness(context, text, get_llm(), threshold=s.faithfulness_threshold)
        ans.faithfulness = report["faithfulness"]
        ans.hallucination_rate = report.get("hallucination_rate")
        ans.grounded = report["grounded"]
        ans.is_refusal = report["is_refusal"]
        # 有内容但忠实度过低 -> 低置信（不假装确定）
        ans.low_confidence = (not report["is_refusal"]) and (not report["grounded"])
    # 质量信号计数
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
    )
    return ans
