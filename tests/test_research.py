"""M9c 研究编排的单元测试（fake llm + fake 检索，无 DB/网络）。"""
from app.research import agent, session
from app.research.doc import ResearchDoc, Section, cites_in
from app.research.evidence import _dedup_merge, gather_evidence
from app.retrieval.store import RetrievedChunk


def _chunk(i, content, *, stype="kb", trust="curated", url=None):
    md = {"source_type": stype, "trust": trust}
    if url:
        md["url"] = url
    return RetrievedChunk(chunk_id=i, document_id=i, source=f"src{i}", content=content,
                          score=1.0, metadata=md, context=content)


class _LLM:
    def complete(self, system, user):
        if "规划" in system:
            return '["背景", "关键事实", "结论"]'
        if "质检员" in system:            # 充分性：直接够
            return "SUFFICIENT"
        if "一致性审查员" in system:      # 冲突
            return '{"conflict": false, "explanation": ""}'
        if "记忆管理员" in system:
            return "[]"
        return "INSUFFICIENT"

    def generate(self, system, user):
        return "该节论断一[1]，论断二[2]。"


def test_parse_outline_caps():
    assert agent.parse_outline('["a","b","c","d"]', 3) == ["a", "b", "c"]
    assert agent.parse_outline("nope", 3) == []


def test_dedup_merge():
    a = [_chunk(1, "x")]
    b = [_chunk(2, "x"), _chunk(3, "y")]
    assert [c.content for c in _dedup_merge(a, b)] == ["x", "y"]


def test_gather_evidence_bound_and_merge():
    kb = lambda q, top_k=None, tenant=None: [_chunk(1, "证据A"), _chunk(2, "证据B")]
    web = lambda q: [_chunk(9, "网页证据", stype="web", trust="draft", url="http://e/1")]
    chunks, meta = gather_evidence("t", use_kb=True, use_web=True, max_iters=2, llm=_LLM(),
                                   kb_retrieve=kb, web_retrieve=web)
    assert meta["sufficient"] is True
    assert {c.metadata["source_type"] for c in chunks} == {"kb", "web"}


def test_research_assembly_and_references():
    kb = lambda q, top_k=None, tenant=None: [_chunk(1, f"{q}-a"), _chunk(2, f"{q}-b")]
    doc = agent.research("秦惠文王", llm=_LLM(), use_kb=True, use_web=False,
                         max_sections=3, kb_retrieve=kb)
    assert isinstance(doc, ResearchDoc)
    assert [s.title for s in doc.sections] == ["背景", "关键事实", "结论"]
    assert len(doc.references) == 6
    for s in doc.sections:
        assert all(1 <= i <= len(doc.references) for i in s.cites)
    assert len(doc.claims) == 3


def test_session_create_get_refine():
    doc = ResearchDoc(topic="t", sections=[Section("节1", "旧文[1]")], references=[])
    sid = session.create(doc)
    assert session.get(sid) is doc
    kb = lambda q, top_k=None, tenant=None: [_chunk(1, "新证据")]
    updated = session.refine_section(sid, 0, llm=_LLM(), use_kb=True, use_web=False,
                                     max_iters=1, kb_retrieve=kb)
    assert updated.sections[0].body != "旧文[1]"     # 已重做
    session.clear()


def test_cites_in_order():
    assert cites_in("a[2] b[1] c[2] d[3]") == [2, 1, 3]
