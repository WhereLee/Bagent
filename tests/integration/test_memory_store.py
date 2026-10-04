"""记忆 store 的集成测试：真 pgvector + bge embedding，验证 trust 门控与双时态过滤。"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db.session import get_session, init_schema
from app.ingestion.embedder import get_embedder
from app.memory import store as ms

pytestmark = pytest.mark.integration


@pytest.fixture()
def owner():
    init_schema()
    return "it-" + uuid.uuid4().hex[:8]


def _v(text):
    return get_embedder().encode_documents([text])[0]


def test_add_search_trust_and_timewindow(owner):
    s = get_session()
    try:
        ms.add_memory(s, scope="personal", owner_user_id=owner, content="用户喜欢靠窗",
                      embedding=_v("用户喜欢靠窗座位"), trust="verified")
        ms.add_memory(s, scope="personal", owner_user_id=owner, content="用户讨厌辣",
                      embedding=_v("用户不喜欢吃辣"), trust="draft")
        ms.add_memory(s, scope="personal", owner_user_id=owner, content="过期记忆",
                      embedding=_v("这是一条已失效的记忆"), trust="verified",
                      invalid_at=datetime.now(timezone.utc) - timedelta(days=1))
        s.commit()

        # 只有 verified 参与（draft 被门控挡掉）
        hits = ms.search_memories(s, _v("座位偏好"), scope="personal", owner_user_id=owner,
                                 min_trust="verified", k=5)
        contents = [m.content for m in hits]
        assert "用户喜欢靠窗" in contents
        assert "用户讨厌辣" not in contents          # draft 过滤
        assert "过期记忆" not in contents            # invalid_at 过去 -> 过滤
    finally:
        s.rollback(); s.close()


def test_dedup_by_hash(owner):
    s = get_session()
    try:
        c = "幂等测试" + owner
        r1, new1 = ms.add_memory(s, scope="knowledge", content=c, embedding=_v(c), trust="draft")
        r2, new2 = ms.add_memory(s, scope="knowledge", content=c, embedding=_v(c), trust="draft")
        assert new1 is True and new2 is False        # 同内容 hash 幂等
        assert r1.id == r2.id
    finally:
        s.rollback(); s.close()


def test_promote_and_invalidate(owner):
    s = get_session()
    try:
        r, _ = ms.add_memory(s, scope="knowledge", content="促升测试" + owner,
                             embedding=_v("促升测试"), trust="draft")
        s.commit()
        mid = r.id
        assert ms.set_trust(s, mid, "verified")
        s.commit()
        assert s.get(ms.Memory, mid).trust == "verified"
        assert ms.invalidate(s, mid)
        s.commit()
        assert s.get(ms.Memory, mid).invalid_at is not None
    finally:
        s.rollback(); s.close()


def test_level_and_tag_filter(owner):
    s = get_session()
    try:
        ms.add_memory(s, scope="knowledge", content="重大活动必做风险预案" + owner,
                      embedding=_v("重大活动必做风险预案"), trust="verified",
                      level="playbook", tags={"type": "活动"})
        ms.add_memory(s, scope="knowledge", content="文昌老街坐标" + owner,
                      embedding=_v("文昌老街的位置"), trust="verified", level="fact", tags={"location": "文昌"})
        s.commit()
        pb = ms.search_memories(s, _v("活动该注意什么"), scope="knowledge", level="playbook",
                               min_trust="verified", k=5)
        assert all(m.level == "playbook" for m in pb) and pb
        tagged = ms.search_memories(s, _v("老街"), scope="knowledge", any_tags={"location": "文昌"},
                                    min_trust="verified", k=5)
        assert any("文昌老街" in m.content for m in tagged)
        miss = ms.search_memories(s, _v("任意"), scope="knowledge", any_tags={"location": "不存在"},
                                 min_trust="verified", k=5)
        assert miss == []
    finally:
        s.rollback(); s.close()


def test_record_feedback_drives_trust(owner):
    s = get_session()
    try:
        r, _ = ms.add_memory(s, scope="knowledge", content="回标促升" + owner,
                             embedding=_v("回标促升"), trust="draft")
        s.commit()
        assert ms.record_feedback(s, r.id, ok=True) == "promote"
        assert s.get(ms.Memory, r.id).trust == "verified"

        r2, _ = ms.add_memory(s, scope="knowledge", content="回标作废" + owner,
                              embedding=_v("回标作废"), trust="verified")
        s.commit()
        assert ms.record_feedback(s, r2.id, ok=False) == "noop"          # 1 次失败
        assert ms.record_feedback(s, r2.id, ok=False) == "invalidate"    # 2 次达阈值
        assert s.get(ms.Memory, r2.id).invalid_at is not None
    finally:
        s.rollback(); s.close()


def test_event_archive_playbook(owner):
    s = get_session()
    try:
        ev = ms.create_event(s, name="骑到文昌复盘", type="活动", created_by=owner)
        s.flush()
        ms.add_memory(s, scope="knowledge", content="重大活动必做风险预案" + owner, embedding=_v("风险预案"),
                      trust="verified", level="playbook", event_id=ev.id)
        ms.add_memory(s, scope="knowledge", content="文昌露营点A" + owner, embedding=_v("露营点"),
                      trust="verified", level="fact", event_id=ev.id)
        s.commit()
        pb = ms.event_playbook(s, ev.id)
        assert pb["event"]["name"] == "骑到文昌复盘"
        assert pb["playbook"] == ["重大活动必做风险预案" + owner]
        assert any(f["content"].startswith("文昌露营点A") for f in pb["facts"])
    finally:
        s.rollback(); s.close()
