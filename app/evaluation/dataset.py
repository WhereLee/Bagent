"""评估数据集：加载 golden 问答标注，并把"金标准片段"映射为库中相关的子块 id。

golden.jsonl 每行：
  {"question": "...", "gold_source": "相对路径", "gold_snippets": ["必须命中的原文片段", ...]}
相关性判定：某子块 content 归一化后包含任一 gold_snippet（且来源匹配），即视为相关。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from app.db.models import Chunk, Document

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", text)


@dataclass
class GoldenItem:
    question: str
    gold_source: str
    gold_snippets: list[str]
    relevant_ids: list[int] = field(default_factory=list)


def load_golden(path: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        items.append(
            GoldenItem(
                question=d["question"],
                gold_source=d["gold_source"],
                gold_snippets=d["gold_snippets"],
            )
        )
    return items


def label_relevance(session, items: list[GoldenItem]) -> list[GoldenItem]:
    """扫描全部子块，为每个 GoldenItem 填充 relevant_ids。"""
    rows = session.execute(
        select(Chunk.id, Chunk.content, Document.source)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.embedding.isnot(None), Document.is_deleted == False)  # noqa: E712
    ).all()
    for item in items:
        gold_norm = [_norm(s) for s in item.gold_snippets]
        matched: list[int] = []
        for cid, content, source in rows:
            if not source.endswith(item.gold_source):
                continue
            cn = _norm(content)
            if any(g in cn for g in gold_norm):
                matched.append(cid)
        item.relevant_ids = matched
    return items
