"""写回编排的单元测试（monkeypatch store，无 DB/网络）。"""
import numpy as np

from app.memory import writeback as wb


class _FakeLLM:
    def __init__(self, raw):
        self.raw = raw

    def complete(self, system, user):
        return self.raw


class _FakeEmbedder:
    def encode_documents(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


class _Session:
    def commit(self):
        pass

    def scalar(self, *a, **k):
        return None


def _run(monkeypatch, raw, similar, existing_support=1, existing_trust="draft"):
    added, bumped = [], []

    def fake_add(session, **kw):
        added.append(kw)
        class R: id = 99
        return R(), True

    def fake_bump(session, mid):
        bumped.append(mid)
        class Row:
            support = existing_support + 1
            trust = existing_trust
        return Row()

    monkeypatch.setattr(wb, "add_memory", fake_add)
    monkeypatch.setattr(wb, "bump_support", fake_bump)
    monkeypatch.setattr(wb, "update_content", lambda *a, **k: None)
    monkeypatch.setattr(wb, "find_similar", lambda *a, **k: similar)
    monkeypatch.setattr(wb, "_exists_hash", lambda session, content: False)
    summary = wb.run_writeback(
        session=_Session(), conversation="x", user_id="u1",
        llm=_FakeLLM(raw), embedder=_FakeEmbedder(),
        dedup_sim=0.92, promote_threshold=2,
    )
    return summary, added, bumped


def test_personal_pref_added_as_verified(monkeypatch):
    raw = '[{"content":"用户喜欢靠窗座位","scope":"personal","kind":"preference"}]'
    summary, added, _ = _run(monkeypatch, raw, similar=[])
    assert summary["add"] == 1
    assert added[0]["trust"] == "verified"      # 用户亲述偏好即权威
    assert added[0]["owner_user_id"] == "u1"


def test_knowledge_fact_added_as_draft(monkeypatch):
    raw = '[{"content":"秦惠文王任用张仪行连横","scope":"knowledge","kind":"fact"}]'
    summary, added, _ = _run(monkeypatch, raw, similar=[])
    assert summary["add"] == 1
    assert added[0]["trust"] == "draft"          # 知识结论默认草稿，防自毒


def test_similar_knowledge_bumps_and_promotes(monkeypatch):
    raw = '[{"content":"秦惠文王用张仪","scope":"knowledge","kind":"fact"}]'
    # 已有相似(id=5)，bump 后 support=2 达阈值 -> 促升
    summary, added, bumped = _run(monkeypatch, raw, similar=[(5, 0.95)],
                                  existing_support=1, existing_trust="draft")
    assert summary["update"] == 1 and summary["promote"] == 1
    assert added == [] and bumped == [5]


def test_personal_without_user_id_skipped(monkeypatch):
    raw = '[{"content":"喜欢红色","scope":"personal"}]'
    # run_writeback 用 user_id=None
    monkeypatch.setattr(wb, "find_similar", lambda *a, **k: [])
    monkeypatch.setattr(wb, "_exists_hash", lambda s, c: False)
    summary = wb.run_writeback(session=_Session(), conversation="x", user_id=None,
                               llm=_FakeLLM(raw), embedder=_FakeEmbedder(),
                               dedup_sim=0.92, promote_threshold=2)
    assert summary["skip"] == 1 and summary["add"] == 0
