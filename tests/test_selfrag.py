"""Self-RAG 反思重检的单元测试（monkeypatch retrieve + 假 LLM，无网络/DB）。"""
from app.retrieval import selfrag
from app.retrieval.selfrag import parse_sufficiency, retrieve_with_reflection
from app.retrieval.store import RetrievedChunk


def _chunk(cid, text="c"):
    return RetrievedChunk(chunk_id=cid, document_id=1, source="s.md", content=text,
                          score=1.0, metadata={}, context=text)


class FakeLLM:
    def __init__(self, verdicts, newq="Q-NEW"):
        self.verdicts = verdicts
        self.i = 0
        self.newq = newq

    def complete(self, system, user):
        if "改写" in system:
            return self.newq
        v = self.verdicts[min(self.i, len(self.verdicts) - 1)]
        self.i += 1
        return v


def test_parse_sufficiency():
    assert parse_sufficiency("SUFFICIENT") is True
    assert parse_sufficiency("INSUFFICIENT") is False   # 不能被 "SUFFICIENT" 子串误判
    assert parse_sufficiency("hmm") is False


def test_reflection_stops_when_sufficient(monkeypatch):
    monkeypatch.setattr(selfrag, "retrieve", lambda q, **k: [_chunk(1)])
    llm = FakeLLM(["SUFFICIENT"])
    chunks, meta = retrieve_with_reflection("q", max_iters=3, llm=llm)
    assert meta["sufficient"] is True
    assert meta["iterations"] == 1  # 第一轮就够


def test_reflection_reformats_and_merges(monkeypatch):
    calls = {"n": 0}

    def fake_retrieve(q, **k):
        calls["n"] += 1
        return [_chunk(1 if q == "q" else 2)]

    monkeypatch.setattr(selfrag, "retrieve", fake_retrieve)
    llm = FakeLLM(["INSUFFICIENT", "SUFFICIENT"], newq="q")  # 第1轮不足->改写->第2轮够
    # 改写返回与已存在 query 相同会停，这里给不同词
    llm.newq = "q2"
    chunks, meta = retrieve_with_reflection("q", max_iters=3, llm=llm)
    assert meta["sufficient"] is True
    assert meta["iterations"] == 2
    assert {c.chunk_id for c in chunks} == {1, 2}  # 合并去重


def test_reflection_bounded_by_max_iters(monkeypatch):
    monkeypatch.setattr(selfrag, "retrieve", lambda q, **k: [_chunk(hash(q) % 1000)])
    llm = FakeLLM(["INSUFFICIENT", "INSUFFICIENT", "INSUFFICIENT"], newq="")
    # 改写返回空 -> 应立即停（不空转）
    chunks, meta = retrieve_with_reflection("q", max_iters=5, llm=llm)
    assert meta["iterations"] == 1  # 第一轮不足但改写为空 -> 停
