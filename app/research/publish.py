"""P2-c：把研究型 Agent 的产出沉淀为知识记忆（draft）——"问完/研究完即成资产"，
但走同一 trust 门控（未促升不参与作答），与 P0 的防污染机制一致。"""
from __future__ import annotations

from app.research.doc import ResearchDoc


def claims_to_facts(doc: ResearchDoc) -> list[dict]:
    """纯映射：每条论断 → {content, source_ref}。content 带主题上下文，source_ref 汇总引用来源 URL。"""
    ref_by_id = {r.id: r for r in doc.references}
    facts = []
    for claim in doc.claims:
        urls = [ref_by_id[i].url or ref_by_id[i].source for i in claim.reference_ids if i in ref_by_id]
        src = urls[0] if urls else "research"
        content = f"{doc.topic}：{claim.text}"
        facts.append({"content": content, "source_ref": src})
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
                              trust="draft", tenant_id=tenant)
        added += int(is_new)
    session.commit()
    return added
