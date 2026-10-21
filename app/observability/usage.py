"""P0：当前线程/上下文的 LLM token 用量收集器。

LLMClient._record_usage 继续写全局 Prometheus，同时写入当前 usage sink；
agent 每步开一个 sink，把 token 汇总到 step/run，不改变 complete() 的签名。
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)


_usage_var: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("usage_sink", default=None)


def record_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    sink = _usage_var.get()
    if sink is not None:
        sink.add(prompt_tokens, completion_tokens)


@contextmanager
def collect_usage():
    sink = Usage()
    token = _usage_var.set(sink)
    try:
        yield sink
    finally:
        _usage_var.reset(token)
