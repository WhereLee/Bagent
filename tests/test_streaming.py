"""SSE 流式作答与引用过滤的 hermetic 测试（不连 DB/模型）。"""
from __future__ import annotations

import pytest

from app.generation import generator, streaming
from app.generation.citation import CitationStreamFilter
from app.generation.generator import PreparedAnswer, finalize_answer
from app.generation.llm import LLMUnavailable
from app.retrieval.store import RetrievedChunk


def _chunk(cid: int = 1, text: str = "证据") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id=1, source="doc.md", content=text,
                          score=0.9, metadata={}, context=text)


class FakeLLM:
    def __init__(self, deltas: list[str] | None = None, *, fail: bool = False,
                 fail_after: int | None = None) -> None:
        self.deltas = deltas or []
        self.fail = fail
        self.fail_after = fail_after

    def stream(self, system, user):
        for i, d in enumerate(self.deltas):
            if self.fail_after is not None and i >= self.fail_after:
                raise LLMUnavailable("mid-stream boom")
            yield d
        if self.fail:
            raise LLMUnavailable("boom")

    def complete(self, system, user):
        return "SUPPORTED"


@pytest.fixture()
def one_chunk(monkeypatch):
    monkeypatch.setattr(generator, "retrieve", lambda *a, **k: [_chunk()])
    monkeypatch.setattr(generator, "_memory_context", lambda *a, **k: "")


def test_citation_stream_filter_buffers_split_citations():
    f = CitationStreamFilter(num_sources=1)
    out = f.feed("答案[") + f.feed("9] 和 [1") + f.feed("]") + f.flush()
    assert out == "答案 和 [1]"


def test_stream_answer_emits_meta_tokens_and_done(monkeypatch, one_chunk):
    fake = FakeLLM(["答案[", "9] 以及 [1", "]"])
    monkeypatch.setattr(streaming, "get_llm", lambda: fake)
    monkeypatch.setattr(generator, "get_llm", lambda: fake)

    events = list(streaming.stream_answer("问题", top_k=1))
    names = [name for name, _ in events]
    assert names == ["start", "meta", "token", "token", "token", "done"]
    meta = events[1][1]
    assert meta["rewritten_query"] == "问题" and meta["degraded"] is False
    assert meta["sources"] and meta["sources"][0]["source"] == "doc.md"
    tokens = "".join(d["text"] for name, d in events if name == "token")
    assert tokens == "答案 以及 [1]"
    done = events[-1][1]
    assert done["invalid_citations"] == [9]
    assert done["text"] == "答案 以及 [1]"
    assert done["sources"][0]["source"] == "doc.md"
    assert done["is_refusal"] is False


def test_stream_answer_refusal_without_chunks(monkeypatch):
    monkeypatch.setattr(generator, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(generator, "_memory_context", lambda *a, **k: "")
    events = list(streaming.stream_answer("问题", top_k=1))
    names = [name for name, _ in events]
    assert names[:2] == ["start", "meta"]
    done = events[-1][1]
    assert done["is_refusal"] is True
    assert "无法回答" in done["text"]


def test_stream_answer_llm_unavailable_before_tokens(monkeypatch, one_chunk):
    fake = FakeLLM([], fail=True)
    monkeypatch.setattr(streaming, "get_llm", lambda: fake)
    monkeypatch.setattr(generator, "get_llm", lambda: fake)
    events = list(streaming.stream_answer("问题", top_k=1))
    names = [name for name, _ in events]
    assert "error" in names and names[-1] == "done"
    done = events[-1][1]
    assert done["low_confidence"] is True
    assert "服务暂时不可用" in done["text"]


def test_stream_answer_llm_unavailable_mid_stream(monkeypatch, one_chunk):
    fake = FakeLLM(["部分"], fail=True, fail_after=1)
    monkeypatch.setattr(streaming, "get_llm", lambda: fake)
    monkeypatch.setattr(generator, "get_llm", lambda: fake)
    events = list(streaming.stream_answer("问题", top_k=1))
    tokens = "".join(d["text"] for name, d in events if name == "token")
    assert tokens == "部分"
    done = events[-1][1]
    assert done["text"] == "部分"
    assert done["low_confidence"] is True


def test_stream_answer_prepare_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(streaming, "prepare_answer", boom)
    events = list(streaming.stream_answer("问题"))
    names = [name for name, _ in events]
    assert names == ["start", "error", "done"]
    assert events[-1][1]["low_confidence"] is True


def test_streaming_conflict_does_not_rewrite_sent_text(monkeypatch):
    prep = PreparedAnswer(
        query="q", rewritten_query="q", chunks=[_chunk(1), _chunk(2)],
        retrieval_context="ctx", context="ctx", system="s",
        degraded=False, self_rag_applied=True, iterations=2,
        evidence_sufficient=True, check_faithfulness=True,
    )
    monkeypatch.setattr(generator, "get_llm", lambda: object())
    monkeypatch.setattr(generator, "assess_faithfulness", lambda *a, **k: {
        "faithfulness": 1.0, "hallucination_rate": 0.0,
        "grounded": True, "is_refusal": False,
    })
    monkeypatch.setattr(generator, "detect_conflict", lambda *a, **k: {
        "conflict": True, "explanation": "矛盾",
    })

    streamed = finalize_answer(prep, "答案[1]", streamed_text="答案[1]",
                               include_conflict_prefix=False)
    assert streamed.text == "答案[1]"
    assert streamed.conflict is True and streamed.conflict_note == "矛盾"

    non_stream = finalize_answer(prep, "答案[1]", include_conflict_prefix=True)
    assert non_stream.text.startswith("【注意】")


def test_llm_stream_falls_back_before_first_token(monkeypatch):
    from app.generation.llm import LLMClient, Provider

    client = object.__new__(LLMClient)
    client.chain = [Provider("a", "u", "k", "m"), Provider("b", "u", "k", "m")]
    client.max_retries = 1
    client.base_delay = 0.0
    client._retryable = (RuntimeError,)
    client._status_err = RuntimeError
    client._clients = {}

    def fake_stream_once(p, system, user, temperature):
        if p.name == "a":
            raise RuntimeError("boom")
        yield "ok"

    monkeypatch.setattr(client, "_stream_once", fake_stream_once)
    assert list(client.stream("s", "u")) == ["ok"]


def test_llm_stream_does_not_retry_after_partial_output(monkeypatch):
    from app.generation.llm import LLMClient, Provider

    client = object.__new__(LLMClient)
    client.chain = [Provider("a", "u", "k", "m")]
    client.max_retries = 3
    client.base_delay = 0.0
    client._retryable = (RuntimeError,)
    client._status_err = RuntimeError
    client._clients = {}

    def fake_stream_once(p, system, user, temperature):
        yield "partial"
        raise RuntimeError("mid-stream boom")

    monkeypatch.setattr(client, "_stream_once", fake_stream_once)
    with pytest.raises(LLMUnavailable):
        list(client.stream("s", "u"))


def test_stream_done_schedules_writeback(monkeypatch):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from app.api.main import app
    from app.api.routes import query as query_route

    def fake_stream(*args, **kwargs):
        yield "done", {"text": "答案", "sources": []}

    called = []

    def fake_writeback(user_id, query, answer_text):
        called.append((user_id, query, answer_text))

    monkeypatch.setattr(query_route, "stream_answer", fake_stream)
    monkeypatch.setattr(query_route, "_do_writeback", fake_writeback)
    monkeypatch.setattr(query_route, "get_settings", lambda: SimpleNamespace(memory_enabled=True))

    client = TestClient(app)
    with client.stream("POST", "/query/stream",
                       json={"query": "q", "user_id": "u1"}) as resp:
        assert resp.status_code == 200
        assert "event: done" in "".join(resp.iter_text())
    assert called == [("u1", "q", "答案")]
