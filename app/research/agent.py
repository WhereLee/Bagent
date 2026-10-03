"""M9c 研究编排：列提纲 → 逐节多源取证并成文(行内引用) → 汇总 ResearchDoc。

全局参考文献池：各节证据映射到统一编号；冲突检测复用 M8。有界（max_sections/max_iters）。
llm 注入，retrieve/web 可注入，便于 hermetic 测试。
"""
from __future__ import annotations

import json
import re

from app.generation.citation import strip_invalid_citations
from app.generation.conflict import detect_conflict
from app.generation.prompts import PLAN_OUTLINE_SYSTEM, PLAN_OUTLINE_USER, SECTION_SYSTEM, SECTION_USER
from app.research.doc import Claim, Reference, ResearchDoc, Section, cites_in
from app.research.evidence import gather_evidence

_ARR = re.compile(r"\[.*\]", re.S)


def parse_outline(raw: str, max_sections: int) -> list[str]:
    m = _ARR.search(raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    titles = [str(x).strip() for x in data if str(x).strip()] if isinstance(data, list) else []
    return titles[:max_sections]


def plan_outline(topic: str, llm, max_sections: int) -> list[str]:
    raw = llm.complete(system=PLAN_OUTLINE_SYSTEM.format(max_sections=max_sections),
                       user=PLAN_OUTLINE_USER.format(topic=topic))
    out = parse_outline(raw, max_sections)
    return out or [topic]


def _remap_cites(body: str, local_to_global: dict[int, int]) -> str:
    return re.sub(r"\[(\d+)\]",
                  lambda m: f"[{local_to_global.get(int(m.group(1)), m.group(1))}]", body or "")


def research(
    topic: str, *, llm, use_kb: bool = True, use_web: bool = False,
    max_sections: int = 5, max_iters: int = 2, check_conflict: bool = True,
    kb_retrieve=None, web_retrieve=None, tenant: str | None = None,
) -> ResearchDoc:
    outline = plan_outline(topic, llm, max_sections)
    doc = ResearchDoc(topic=topic, outline=list(outline))
    refs: list[Reference] = []
    refkey: dict[str, int] = {}

    for title in outline:
        chunks, meta = gather_evidence(title, use_kb=use_kb, use_web=use_web,
                                       max_iters=max_iters, llm=llm,
                                       kb_retrieve=kb_retrieve, web_retrieve=web_retrieve,
                                       tenant=tenant)
        if not chunks:
            doc.sections.append(Section(title=title, body="（未采集到证据）", cites=[]))
            continue
        local_to_global = {}
        for i, c in enumerate(chunks, 1):
            gid = refkey.get(c.source)           # 以来源(source/url)为去重键，与 refine 一致
            if gid is None:
                refs.append(Reference(
                    id=len(refs) + 1, source=c.source, url=c.metadata.get("url"),
                    source_type=c.metadata.get("source_type", "kb"),
                    trust=c.metadata.get("trust", "curated"),
                    published=c.metadata.get("published"),
                ))
                gid = len(refs)
                refkey[c.source] = gid
            local_to_global[i] = gid

        ctx = "\n\n".join(f"[{i}] {c.context}" for i, c in enumerate(chunks, 1))
        body = llm.generate(system=SECTION_SYSTEM,
                            user=SECTION_USER.format(topic=topic, section=title, context=ctx))
        body = strip_invalid_citations(body, len(chunks))
        body = _remap_cites(body, local_to_global)
        sec = Section(title=title, body=body, cites=cites_in(body))
        doc.sections.append(sec)

        # 论断（按节聚合：引用到的来源 + 最低可信度）
        ref_trusts = [refs[i - 1].trust for i in sec.cites if 1 <= i <= len(refs)]
        trust = "verified" if ref_trusts and all(t in ("verified", "curated") for t in ref_trusts) else "draft"
        doc.claims.append(Claim(text=title, reference_ids=list(sec.cites), trust=trust))

        if check_conflict and len(chunks) >= 2:
            cres = detect_conflict(title, ctx, llm)
            if cres.get("conflict"):
                doc.conflicts.append({"section": title, "explanation": cres.get("explanation", "")})

    doc.references = refs
    doc.meta = {"use_kb": use_kb, "use_web": use_web, "sections": len(doc.sections)}
    return doc
