"""M9c 会话态：把 ResearchDoc 作为可变"活文档"按 session_id 存住，支持逐轮 refine。

后端可换：默认进程内存；设 `redis_url` 则存 Redis（多 worker 共享活文档，JSON 往返）。
接口 create/get/refine_section/clear 不变。refine 重做某节并回填引用/论断后回写存储。
"""
from __future__ import annotations

import json
import threading
import uuid

from app.research import agent
from app.research.doc import Reference, ResearchDoc, cites_in
from app.research.evidence import gather_evidence
from app.generation.prompts import SECTION_SYSTEM, SECTION_USER
from app.generation.citation import strip_invalid_citations

_PREFIX = "bagent:sess:"


class _MemStore:
    def __init__(self) -> None:
        self._d: dict[str, tuple[ResearchDoc, str | None]] = {}
        self._lock = threading.Lock()

    def get(self, sid):
        with self._lock:
            return self._d.get(sid)

    def set(self, sid, doc, tenant):
        with self._lock:
            self._d[sid] = (doc, tenant)

    def clear(self):
        with self._lock:
            self._d.clear()


class _RedisStore:
    """跨进程共享：活文档以 JSON 存 Redis（TTL 兜底回收）。client 可注入以便单测。"""

    def __init__(self, client, ttl: int = 3600) -> None:
        self.r = client
        self.ttl = ttl

    def get(self, sid):
        raw = self.r.get(_PREFIX + sid)
        if raw is None:
            return None
        obj = json.loads(raw)
        return ResearchDoc.from_dict(obj["doc"]), obj.get("tenant")

    def set(self, sid, doc, tenant):
        self.r.setex(_PREFIX + sid, self.ttl,
                     json.dumps({"doc": doc.to_dict(), "tenant": tenant}, ensure_ascii=False))

    def clear(self):
        for k in self.r.scan_iter(match=_PREFIX + "*", count=200):
            self.r.delete(k)


_mem = _MemStore()
_store_override = None


def _store():
    if _store_override is not None:
        return _store_override
    from app.config import get_settings
    s = get_settings()
    if s.redis_url:
        import redis
        return _RedisStore(redis.Redis.from_url(s.redis_url, decode_responses=True))
    return _mem


def create(doc: ResearchDoc, tenant: str | None = None) -> str:
    sid = uuid.uuid4().hex[:12]
    _store().set(sid, doc, tenant)
    return sid


def get(sid: str, tenant: str | None = None) -> ResearchDoc | None:
    """tenant 非空时仅返回属于该租户的会话（跨租户 sid 不可读）。"""
    entry = _store().get(sid)
    if entry is None:
        return None
    doc, owner = entry
    if tenant is not None and owner != tenant:
        return None
    return doc


def clear() -> None:
    _store().clear()


def set_store(store) -> None:
    """测试用：注入存储。传 None 恢复默认。"""
    global _store_override
    _store_override = store


def refine_section(
    sid: str, index: int, *, llm, use_kb: bool = True, use_web: bool = False,
    max_iters: int = 2, kb_retrieve=None, web_retrieve=None, tenant: str | None = None,
) -> ResearchDoc | None:
    """重做第 index 节：重新取证并生成，回填全局引用编号，并回写存储。"""
    doc = get(sid, tenant)
    if doc is None or not (0 <= index < len(doc.sections)):
        return None
    title = doc.sections[index].title
    chunks, _ = gather_evidence(title, use_kb=use_kb, use_web=use_web,
                                max_iters=max_iters, llm=llm,
                                kb_retrieve=kb_retrieve, web_retrieve=web_retrieve,
                                tenant=tenant)
    if not chunks:
        return doc
    refkey = {r.source: r.id for r in doc.references}
    local_to_global = {}
    for i, c in enumerate(chunks, 1):
        gid = refkey.get(c.source)
        if gid is None:
            ref = Reference(id=len(doc.references) + 1, source=c.source, url=c.metadata.get("url"),
                            source_type=c.metadata.get("source_type", "kb"),
                            trust=c.metadata.get("trust", "curated"), published=c.metadata.get("published"))
            doc.references.append(ref)
            refkey[c.source] = ref.id
            gid = ref.id
        local_to_global[i] = gid
    ctx = "\n\n".join(f"[{i}] {c.context}" for i, c in enumerate(chunks, 1))
    body = llm.generate(system=SECTION_SYSTEM, user=SECTION_USER.format(topic=doc.topic, section=title, context=ctx))
    body = strip_invalid_citations(body, len(chunks))
    body = agent._remap_cites(body, local_to_global)
    doc.sections[index].body = body
    doc.sections[index].cites = cites_in(body)
    _store().set(sid, doc, tenant)   # 跨进程持久化（内存后端等价于原地更新，幂等）
    return doc
