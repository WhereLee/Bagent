"""P0 Agent usage sink：LLM 调用用量能汇总到当前 run/step。"""
from __future__ import annotations

from types import SimpleNamespace

from app.generation.llm import LLMClient
from app.observability.usage import collect_usage


def test_record_usage_reaches_current_sink():
    with collect_usage() as sink:
        LLMClient._record_usage(SimpleNamespace(prompt_tokens=11, completion_tokens=7))
        LLMClient._record_usage(SimpleNamespace(prompt_tokens=3, completion_tokens=2))
    assert sink.prompt_tokens == 14
    assert sink.completion_tokens == 9


def test_sink_is_scoped():
    with collect_usage() as sink:
        LLMClient._record_usage(SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    LLMClient._record_usage(SimpleNamespace(prompt_tokens=10, completion_tokens=10))
    assert sink.prompt_tokens == 1 and sink.completion_tokens == 1
