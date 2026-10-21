"""单次 RAG 问答、多轮对话与 agent 模式。"""
from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatReq, QueryReq, StreamChatReq, StreamQueryReq
from app.config import get_settings
from app.db.session import get_session
from app.generation.generator import answer_query
from app.generation.llm import get_llm
from app.generation.streaming import stream_answer
from app.ingestion.embedder import get_embedder
from app.memory.writeback import run_writeback
from app.observability.logging import get_logger, log_event
from app.observability.metrics import STAGE_LATENCY
from app.tools import build_tools

router = APIRouter()
_log_api = get_logger("api")


@router.post("/query")
def query(req: QueryReq, background: BackgroundTasks) -> dict:
    if req.agent:
        return _query_agent(req.query, top_k=req.top_k, user_id=req.user_id)
    answer = answer_query(req.query, top_k=req.top_k, self_rag=req.self_rag,
                          user_id=req.user_id)
    _schedule_writeback(background, req.user_id, req.query, answer.text)
    return asdict(answer)


def _query_agent(query: str, *, top_k: int | None, user_id: str | None = None) -> dict:
    """agent 模式：有界工具循环产出带依据的答案；工具只走受控检索（主 LLM 不隐式联网）。"""
    from app.agent.react import run_agent
    from app.evaluation.trajectory import summarize_trajectories

    # compact=True：observation 只带短 snippet，避免超出 run_agent 的 1500 字截断
    all_tools = build_tools(user_id=user_id, compact=True)
    tools = {name: all_tools[name] for name in ("kb_search", "web_search")}
    traj = run_agent(
        f"请检索并回答（末尾给出依据来源）：{query}", llm=get_llm(), tools=tools,
        max_steps=4, finalize=True,
        check_faithfulness=get_settings().faithfulness_enabled,
    )
    answer = traj.get("answer") or {}
    return {
        "query": query,
        "text": answer.get("text") or traj.get("final") or "（未收敛）",
        "sources": answer.get("sources") or [],
        "mode": "agent",
        "status": traj["status"],
        "run_id": traj.get("run_id"),
        "grounded": bool(answer.get("grounded")),
        "low_confidence": bool(answer.get("low_confidence")),
        "invalid_citations": answer.get("invalid_citations") or [],
        "metrics": summarize_trajectories([traj]),
    }


@router.post("/chat")
def chat(req: ChatReq, background: BackgroundTasks) -> dict:
    """多轮问答：带上历史，服务端做查询改写后检索生成。"""
    answer = answer_query(req.query, top_k=req.top_k, history=req.history,
                          self_rag=req.self_rag, user_id=req.user_id)
    _schedule_writeback(background, req.user_id, req.query, answer.text)
    return asdict(answer)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_response(events, background: BackgroundTasks, user_id, query) -> StreamingResponse:
    """把 stream_answer 的事件转成 SSE；done 前调度与 /query 相同的写回。"""
    def gen():
        try:
            # HTTP 直方图对 SSE 只覆盖到首包；完整流时长单独用 sse_stream 计量
            with STAGE_LATENCY.labels(stage="sse_stream").time():
                for name, data in events:
                    if name == "done":
                        _schedule_writeback(background, user_id, query, data.get("text", ""))
                    yield _sse(name, data)
        except GeneratorExit:
            # 客户端提前断开：没有 done 就没有写回；记一条日志便于排查
            log_event(_log_api, "info", "sse_client_disconnected",
                      user_id=user_id, query=query[:60])
            raise

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/query/stream")
def query_stream(req: StreamQueryReq, background: BackgroundTasks):
    """SSE 流式问答（单次 RAG 管线；agent 模式仍走 /query）。"""
    events = stream_answer(req.query, top_k=req.top_k, self_rag=req.self_rag,
                           user_id=req.user_id)
    return _stream_response(events, background, req.user_id, req.query)


@router.post("/chat/stream")
def chat_stream(req: StreamChatReq, background: BackgroundTasks):
    """SSE 多轮流式问答：服务端先做查询改写，再流式生成。"""
    events = stream_answer(req.query, top_k=req.top_k, history=req.history,
                           self_rag=req.self_rag, user_id=req.user_id)
    return _stream_response(events, background, req.user_id, req.query)


def _schedule_writeback(background: BackgroundTasks, user_id, query, answer_text) -> None:
    if not (get_settings().memory_enabled and user_id):
        return
    background.add_task(_do_writeback, user_id, query, answer_text)


def _do_writeback(user_id: str, query: str, answer_text: str) -> None:
    """sleep-time：回答之后后台写回，失败不影响已返回的响应。"""
    s = get_settings()
    session = get_session()
    try:
        conv = f"用户：{query}\n助手：{answer_text}"
        run_writeback(session=session, conversation=conv, user_id=user_id,
                      llm=get_llm(), embedder=get_embedder(),
                      dedup_sim=s.memory_dedup_sim, promote_threshold=s.memory_promote_support)
    except Exception as e:  # noqa: BLE001
        session.rollback()
        _log_api.warning("writeback_failed", err=str(e)[:160])
    finally:
        session.close()
