"""P0 Agent finalizer：证据归一化、引用清理与无证据低置信。"""
from __future__ import annotations

import json

from app.agent.finalize import collect_evidence, finalize_agent_answer


def _steps():
    return [
        {"action": "kb_search", "observation": json.dumps([
            {"source": "doc-a", "content": "A 的内容", "score": 0.9},
            {"source": "doc-a", "content": "重复来源", "score": 0.8},
        ], ensure_ascii=False)},
        {"action": "web_search", "observation": json.dumps([
            {"url": "https://x/1", "snippet": "网页证据"},
        ], ensure_ascii=False)},
    ]


def test_collect_evidence_dedupes_by_source():
    evidence = collect_evidence(_steps())
    assert [e.source for e in evidence] == ["doc-a", "https://x/1"]
    assert evidence[0].metadata["source_type"] == "kb"
    assert evidence[1].metadata["source_type"] == "web"


def test_finalize_agent_answer_has_sources_and_cleans_invalid_citation():
    answer = finalize_agent_answer("问题", _steps(), "结论 [1]，错误引用 [9]")
    assert answer.text.startswith("结论")
    assert "[9]" not in answer.text
    assert [s["source"] for s in answer.sources] == ["doc-a", "https://x/1"]
    assert answer.low_confidence is False


def test_finalize_agent_answer_without_evidence_is_low_confidence():
    answer = finalize_agent_answer("问题", [], "没有证据的答案")
    assert answer.text == "没有证据的答案"
    assert answer.sources == [] and answer.low_confidence is True
