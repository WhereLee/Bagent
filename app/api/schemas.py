"""HTTP 请求模型（集中定义，路由只做编排）。"""
from __future__ import annotations

from pydantic import BaseModel


class IngestReq(BaseModel):
    path: str


class DeleteReq(BaseModel):
    source: str


class QueryReq(BaseModel):
    query: str
    top_k: int | None = None
    self_rag: bool | None = None   # 覆盖默认：是否启用 self-RAG/冲突检测
    user_id: str | None = None     # 传入则启用个人记忆/写回
    agent: bool = False            # 走显式工具循环（有界 ReAct）而非单次管道


class ChatReq(BaseModel):
    query: str
    history: list[dict] = []  # [{"role":"user"|"assistant","content":str}]
    top_k: int | None = None
    self_rag: bool | None = None
    user_id: str | None = None


class StreamQueryReq(BaseModel):
    """SSE 单次 RAG 管线；agent 模式仍走 /query。"""

    query: str
    top_k: int | None = None
    self_rag: bool | None = None
    user_id: str | None = None


class StreamChatReq(BaseModel):
    query: str
    history: list[dict] = []
    top_k: int | None = None
    self_rag: bool | None = None
    user_id: str | None = None


class SearchReq(BaseModel):
    query: str
    top_k: int = 5


class MemoryPromoteReq(BaseModel):
    id: int
    trust: str  # draft|verified|curated


class MemoryInvalidateReq(BaseModel):
    id: int


class MemoryFeedbackReq(BaseModel):
    id: int
    ok: bool          # 这条经验按其所行，结果是好是坏


class ResearchReq(BaseModel):
    topic: str
    use_kb: bool = True
    use_web: bool | None = None   # 默认取 config.web_search_enabled
    max_sections: int = 4
    user_id: str | None = None


class RefineReq(BaseModel):
    index: int
    use_kb: bool = True
    use_web: bool | None = None


class AgentReq(BaseModel):
    goal: str
    max_steps: int = 6
    user_id: str | None = None


class EventReq(BaseModel):
    name: str
    type: str | None = None
    created_by: str | None = None
