"""M9c 会话态：把 ResearchDoc 作为可变"活文档"按 session_id 存住，支持逐轮 refine。

进程内存储（带锁）；持久化到 DB 属后续。refine 重做某节并回填引用/论断。
"""
from __future__ import annotations

import threading
import uuid

from app.research import agent
from app.research.doc import ResearchDoc, cites_in
from app.research.evidence import gather_evidence
from app.generation.prompts import SECTION_SYSTEM, SECTION_USER
from app.generation.citation import strip_invalid_citations
from app.research.doc import Reference

_lock = threading.Lock()
_sessions: dict[str, ResearchDoc] = {}


def create(doc: ResearchDoc) -> str:
    sid = uuid.uuid4().hex[:12]
    with _lock:
        _sessions[sid] = doc
    return sid


def get(sid: str) -> ResearchDoc | None:
    with _lock:
        return _sessions.get(sid)


def clear() -> None:
    with _lock:
        _sessions.clear()


def refine_section(
    sid: str, index: int, *, llm, use_kb: bool = True, use_web: bool = False,
    max_iters: int = 2, kb_retrieve=None, web_retrieve=None,
) -> ResearchDoc | None:
    """重做第 index 节：重新取证并生成，回填全局引用编号。"""
    doc = get(sid)
    if doc is None or not (0 <= index < len(doc.sections)):
        return None
    title = doc.sections[index].title
    chunks, _ = gather_evidence(title, use_kb=use_kb, use_web=use_web,
                                max_iters=max_iters, llm=llm,
                                kb_retrieve=kb_retrieve, web_retrieve=web_retrieve)
    if not chunks:
        return doc
    refkey = {r.source: r.id for r in doc.references}
    local_to_global = {}
    for i, c in enumerate(chunks, 1):
        gid = refkey.get(c.content)
        if gid is None:
            ref = Reference(id=len(doc.references) + 1, source=c.source, url=c.metadata.get("url"),
                            source_type=c.metadata.get("source_type", "kb"),
                            trust=c.metadata.get("trust", "curated"), published=c.metadata.get("published"))
            doc.references.append(ref)
            refkey[c.content] = ref.id
            refkey.setdefault(c.source, ref.id)
            gid = ref.id
        local_to_global[i] = gid
    ctx = "\n\n".join(f"[{i}] {c.context}" for i, c in enumerate(chunks, 1))
    body = llm.generate(system=SECTION_SYSTEM, user=SECTION_USER.format(topic=doc.topic, section=title, context=ctx))
    body = strip_invalid_citations(body, len(chunks))
    body = agent._remap_cites(body, local_to_global)
    doc.sections[index].body = body
    doc.sections[index].cites = cites_in(body)
    return doc
