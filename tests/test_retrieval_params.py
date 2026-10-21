"""检索运行参数的单元测试（不连库）：HNSW SET LOCAL 与 rerank 并发闸门。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.retrieval import rerank, store


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, stmt, *args, **kwargs):  # noqa: ANN001
        self.calls.append(str(stmt))
        return None


def test_hnsw_settings_are_optional_by_default(monkeypatch):
    monkeypatch.setattr(store, "get_settings", lambda: Settings())
    session = FakeSession()
    store._apply_hnsw_settings(session)
    assert session.calls == []


def test_hnsw_settings_emit_set_local(monkeypatch):
    s = Settings(hnsw_ef_search=80, hnsw_iterative_scan="relaxed_order")
    monkeypatch.setattr(store, "get_settings", lambda: s)
    session = FakeSession()
    store._apply_hnsw_settings(session)
    assert any("hnsw.ef_search = 80" in c for c in session.calls)
    assert any("hnsw.iterative_scan = 'relaxed_order'" in c for c in session.calls)


@pytest.mark.parametrize("bad", ["bad", "relaxed", "strict"])
def test_invalid_hnsw_iterative_scan_rejected(bad):
    with pytest.raises(ValidationError):
        Settings(hnsw_iterative_scan=bad)


def test_negative_hnsw_ef_search_rejected():
    with pytest.raises(ValidationError):
        Settings(hnsw_ef_search=-1)


def test_negative_rerank_concurrency_rejected():
    with pytest.raises(ValidationError):
        Settings(rerank_max_concurrency=-1)


def test_rerank_semaphore_default_is_unlimited(monkeypatch):
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings())
    rerank.reset_reranker()
    assert rerank._get_semaphore() is None


def test_rerank_semaphore_limits_concurrency(monkeypatch):
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_max_concurrency=1))
    rerank.reset_reranker()
    sem = rerank._get_semaphore()
    assert sem is not None
    assert sem.acquire(blocking=False)
    assert not sem.acquire(blocking=False)
    sem.release()
    rerank.reset_reranker()
