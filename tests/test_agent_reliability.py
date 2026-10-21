"""P1：Agent 工具超时、重试、重复检测、结构化错误与 Trace 契约。"""
from __future__ import annotations

import time

from app.agent.react import run_agent
from app.observability.tracing import set_tracer_for_tests
from app.tools import KbSearchArgs, ToolSpec


class ScriptedLLM:
    def __init__(self, outs):
        self.outs = list(outs)
        self.i = 0

    def complete(self, system, user):
        out = self.outs[min(self.i, len(self.outs) - 1)]
        self.i += 1
        return out


def _spec(name: str, fn, **kwargs) -> ToolSpec:
    return ToolSpec(name=name, func=fn, description=name, args_model=KbSearchArgs, **kwargs)


def test_tool_timeout_returns_structured_error():
    def slow(query, top_k=5):
        time.sleep(0.2)
        return []

    spec = _spec("slow", slow, timeout_s=0.01, max_retries=0)
    llm = ScriptedLLM(['{"tool":"slow","args":{"query":"q"}}', '{"final":"ok"}'])
    result = run_agent("问题", llm=llm, tools={"slow": slow}, tool_specs={"slow": spec}, max_steps=3)
    assert result["status"] == "completed"
    assert any(s["action"] == "timeout:slow" for s in result["steps"])
    observation = next(s["observation"] for s in result["steps"] if s["action"] == "timeout:slow")
    assert '"error_type": "timeout"' in observation and '"retryable": true' in observation


def test_readonly_tool_retries_transient_error(monkeypatch):
    calls = {"n": 0}

    def flaky(query, top_k=5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("temporary")
        return [{"source": "s", "content": "ok"}]

    monkeypatch.setattr("app.agent.react.time.sleep", lambda _: None)
    spec = _spec("flaky", flaky, max_retries=1, retry_on=("unavailable",), idempotent=True)
    llm = ScriptedLLM(['{"tool":"flaky","args":{"query":"q"}}', '{"final":"ok"}'])
    result = run_agent("问题", llm=llm, tools={"flaky": flaky}, tool_specs={"flaky": spec}, max_steps=3)
    assert result["status"] == "completed" and calls["n"] == 2
    assert [s["action"] for s in result["steps"][:1]] == ["flaky"]


def test_duplicate_action_is_not_executed_twice():
    calls = {"n": 0}

    def counted(query, top_k=5):
        calls["n"] += 1
        return [{"source": "s", "content": "ok"}]

    llm = ScriptedLLM([
        '{"tool":"kb_search","args":{"query":"q"}}',
        '{"tool":"kb_search","args":{"query":"q"}}',
        '{"final":"ok"}',
    ])
    result = run_agent("问题", llm=llm, tools={"kb_search": counted}, max_steps=4)
    assert calls["n"] == 1
    assert [s["action"] for s in result["steps"]] == [
        "kb_search", "duplicate:kb_search", "final"
    ]
    duplicate = next(s for s in result["steps"] if s["action"].startswith("duplicate:"))
    assert '"error_type": "duplicate"' in duplicate["observation"]


def test_unknown_tool_error_is_structured_and_not_retried():
    calls = {"n": 0}

    def bad(query, top_k=5):
        calls["n"] += 1
        raise ValueError("bad query")

    llm = ScriptedLLM(['{"tool":"bad","args":{"query":"q"}}', '{"final":"ok"}'])
    result = run_agent("问题", llm=llm, tools={"bad": bad}, max_steps=3)
    assert result["status"] == "completed" and calls["n"] == 1
    assert any(s["action"] == "unknown:bad" for s in result["steps"])
    observation = next(s["observation"] for s in result["steps"] if s["action"] == "unknown:bad")
    assert '"error_type": "unknown"' in observation and '"retryable": false' in observation


class _FakeSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = _FakeSpan(name)
        self.spans.append(span)
        return span


def test_scope_blocks_tool_before_approval_or_execution():
    calls = {"n": 0}

    def research(topic, **kwargs):
        calls["n"] += 1
        return {}

    llm = ScriptedLLM(['{"tool":"research","args":{"topic":"x"}}', '{"final":"ok"}'])
    result = run_agent(
        "问题", llm=llm, tools={"research": research}, max_steps=3,
        confirm=lambda *_: True, scopes={"agent:read"},
    )
    assert calls["n"] == 0
    assert any(s["action"] == "forbidden:research" for s in result["steps"])
    assert result["status"] == "completed"


def test_agent_emits_agent_chat_and_tool_spans():
    tracer = _FakeTracer()
    set_tracer_for_tests(tracer)
    try:
        llm = ScriptedLLM([
            '{"tool":"kb_search","args":{"query":"q"}}',
            '{"final":"ok"}',
        ])
        run_agent("问题", llm=llm, tools={"kb_search": lambda **kwargs: []}, max_steps=3)
        names = [s.name for s in tracer.spans]
        assert "invoke_agent" in names
        assert "chat" in names
        assert "execute_tool kb_search" in names
        tool_span = next(s for s in tracer.spans if s.name == "execute_tool kb_search")
        assert tool_span.attributes["gen_ai.tool.name"] == "kb_search"
        chat_span = next(s for s in tracer.spans if s.name == "chat")
        assert chat_span.attributes["gen_ai.operation.name"] == "chat"
        assert "gen_ai.request.model" in chat_span.attributes
    finally:
        set_tracer_for_tests(None)
