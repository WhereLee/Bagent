"""LLM 客户端：多 provider（DeepSeek 主 / MiMo 备）+ 韧性（超时/指数退避重试/降级）。

设计：
- provider 由配置选择；主 provider 重试若干次仍失败则降级到备用 provider。
- 只对"可重试"错误（429/超时/连接/5xx）退避重试；4xx(鉴权/参数)直接换 provider 或失败。
- 全部 provider 失败 -> 抛 LLMUnavailable，由上层做优雅降级（不再裸 500）。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.observability.logging import get_logger, log_event
from app.observability.metrics import (
    LLM_FALLBACKS,
    LLM_RETRIES,
    LLM_REQUESTS,
    LLM_TOKENS,
    STAGE_LATENCY,
)

_log = get_logger("llm")


class LLMUnavailable(Exception):
    """所有 provider 都失败。"""


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    thinking: str | None = None  # MiMo 专有；DeepSeek 为 None


class LLMClient:
    def __init__(self) -> None:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
            OpenAI,
            RateLimitError,
        )

        self._OpenAI = OpenAI
        self._retryable = (APITimeoutError, APIConnectionError, RateLimitError)
        self._status_err = APIStatusError

        s = get_settings()
        self.timeout = s.llm_timeout
        self.max_retries = s.llm_max_retries
        self.base_delay = s.llm_retry_base_delay

        catalog = {
            "deepseek": Provider("deepseek", s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model),
            "mimo": Provider("mimo", s.mimo_base_url, s.mimo_api_key, s.mimo_model, s.mimo_thinking),
        }
        # 组装有序 provider 链：主 + 备（去重、剔除无 key 的）
        chain: list[Provider] = []
        for name in (s.llm_primary_provider, s.llm_fallback_provider):
            p = catalog.get(name)
            if p and p.api_key and p not in chain:
                chain.append(p)
        if not chain:
            raise RuntimeError("未配置任何可用的 LLM provider（检查 *_API_KEY 与 LLM_PRIMARY_PROVIDER）")
        self.chain = chain
        self._clients: dict[str, object] = {}

    def _client(self, p: Provider):
        c = self._clients.get(p.name)
        if c is None:
            c = self._OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=self.timeout)
            self._clients[p.name] = c
        return c

    @staticmethod
    def _is_retryable(err: Exception, retryable_types, status_err) -> bool:
        if isinstance(err, retryable_types):
            return True
        if isinstance(err, status_err) and getattr(err, "status_code", 0) >= 500:
            return True
        return False

    def _call_once(self, p: Provider, system: str, user: str, temperature: float) -> str:
        kwargs = {}
        if p.thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": p.thinking}}
        resp = self._client(p).chat.completions.create(
            model=p.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            LLM_TOKENS.labels(type="input").inc(usage.prompt_tokens or 0)
            LLM_TOKENS.labels(type="output").inc(usage.completion_tokens or 0)
        return resp.choices[0].message.content or ""

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        last_err: Exception | None = None
        for i, p in enumerate(self.chain):
            if i > 0:
                LLM_FALLBACKS.inc()
                log_event(_log, "warning", "llm_fallback", to=p.name)
            for attempt in range(self.max_retries):
                try:
                    out = self._call_once(p, system, user, temperature)
                    LLM_REQUESTS.labels(provider=p.name, outcome="ok").inc()
                    return out
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if not self._is_retryable(e, self._retryable, self._status_err):
                        LLM_REQUESTS.labels(provider=p.name, outcome="error").inc()
                        log_event(_log, "error", "llm_non_retryable", provider=p.name, err=str(e)[:180])
                        break  # 换下一个 provider
                    LLM_RETRIES.labels(provider=p.name).inc()
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.2)
                        time.sleep(delay)
            LLM_REQUESTS.labels(provider=p.name, outcome="error").inc()
        raise LLMUnavailable(f"所有 LLM provider 均不可用: {last_err}")

    def generate(self, system: str, user: str, temperature: float = 0.1) -> str:
        """带阶段时延埋点的生成调用。"""
        with STAGE_LATENCY.labels(stage="llm_generate").time():
            return self.complete(system, user, temperature)


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
