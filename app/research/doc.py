"""M9c 交付文档对象：带大纲、分节(行内引用)、论断表(来源+可信度+时效)、参考文献、冲突。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Reference:
    id: int
    source: str
    url: str | None = None
    source_type: str = "kb"        # kb/web/memory
    trust: str = "curated"
    published: str | None = None


@dataclass
class Section:
    title: str
    body: str
    cites: list[int] = field(default_factory=list)


@dataclass
class Claim:
    text: str
    reference_ids: list[int] = field(default_factory=list)
    trust: str = "draft"


@dataclass
class ResearchDoc:
    topic: str
    abstract: str = ""
    outline: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"# {self.topic}", ""]
        if self.abstract:
            lines += [self.abstract, ""]
        for i, s in enumerate(self.sections, 1):
            lines += [f"## {i}. {s.title}", s.body, ""]
        if self.conflicts:
            lines += ["## 冲突与待核实", ""]
            for c in self.conflicts:
                lines.append(f"- {c.get('explanation', c)}")
            lines.append("")
        if self.references:
            lines += ["## 参考文献", ""]
            for r in self.references:
                loc = r.url or r.source
                lines.append(f"[{r.id}] {r.source}（{r.source_type}/{r.trust}"
                             + (f", {r.published}" if r.published else "") + f"）{loc}")
        return "\n".join(lines)


def cites_in(text: str) -> list[int]:
    """按出现顺序去重提取 [n]。"""
    out, seen = [], set()
    for m in _CITE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
