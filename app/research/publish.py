"""P2-c：把研究型 Agent 的产出沉淀为知识记忆（draft）——"问完/研究完即成资产"，
但走同一 trust 门控（未促升不参与作答），与 P0 的防污染机制一致。"""
from __future__ import annotations

from app.research.doc import ResearchDoc


_PLAYBOOK_HINTS = ("原则", "建议", "风险", "预案", "打法", "经验", "教训", "必做", "必须", "应当")


def _classify_level(text: str) -> str:
    """启发式：含原则/风险/建议类词的论断归为可迁移 playbook，否则具体 fact。"""
    return "playbook" if any(h in text for h in _PLAYBOOK_HINTS) else "fact"


def claims_to_facts(doc: ResearchDoc) -> list[dict]:
    """纯映射：论断 → {content, source_ref, level, tags}。"""
    ref_by_id = {r.id: r for r in doc.references}
    facts = []
    for claim in doc.claims:
        urls = [ref_by_id[i].url or ref_by_id[i].source for i in claim.reference_ids if i in ref_by_id]
        src = urls[0] if urls else "research"
        content = f"{doc.topic}：{claim.text}"
        facts.append({"content": content, "source_ref": src,
                      "level": _classify_level(claim.text),
                      "tags": {"activity_type": doc.meta.get("type") or "general"}})
    return facts


def publish(doc: ResearchDoc, *, session, embedder, tenant: str | None = None) -> int:
    """把论断写入 knowledge 记忆（trust=draft）。返回新增条数。"""
    from app.memory.store import add_memory

    facts = claims_to_facts(doc)
    if not facts:
        return 0
    vecs = embedder.encode_documents([f["content"] for f in facts])
    added = 0
    for f, v in zip(facts, vecs):
        _, is_new = add_memory(session, scope="knowledge", content=f["content"], embedding=v,
                              source_type="research", source_ref=f["source_ref"],
                              trust="draft", tenant_id=tenant, level=f["level"], tags=f["tags"])
        added += int(is_new)
    session.commit()
    return added
