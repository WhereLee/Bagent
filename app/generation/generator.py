"""RAG 查询 pipeline：检索 -> 组装上下文 -> MiMo 生成（带引用与拒答）。"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.db.session import get_session
from app.generation.llm import get_llm
from app.ingestion.embedder import get_embedder
from app.retrieval.store import RetrievedChunk, vector_search

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。只能依据【参考资料】回答。"
    "规则：\n"
    "1) 若参考资料不足以回答，明确回复『根据现有资料无法回答』，不要编造。\n"
    "2) 引用资料时在句末标注来源编号，如 [1][2]。\n"
    "3) 回答要简洁、准确，使用与用户相同的语言。"
)


@dataclass
class Answer:
    query: str
    text: str
    sources: list[dict]  # {id, source, score}


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (来源: {c.source})\n{c.content}")
    return "\n\n---\n\n".join(blocks)


def answer_query(query: str, top_k: int | None = None) -> Answer:
    s = get_settings()
    k = top_k or s.retrieval_top_k

    embedder = get_embedder()
    qvec = embedder.encode_query(query)

    session = get_session()
    try:
        chunks = vector_search(session, qvec, top_k=k)
    finally:
        session.close()

    # 无召回：直接拒答，不消耗 LLM
    if not chunks:
        return Answer(query=query, text="根据现有资料无法回答（知识库中未检索到相关内容）。", sources=[])

    context = _build_context(chunks)
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{query}"
    text = get_llm().complete(system=SYSTEM_PROMPT, user=user_prompt)

    sources = [
        {"id": c.chunk_id, "source": c.source, "score": round(c.score, 4)}
        for c in chunks
    ]
    return Answer(query=query, text=text, sources=sources)
