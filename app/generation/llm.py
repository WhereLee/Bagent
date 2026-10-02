"""生成用 LLM：Xiaomi MiMo（OpenAI 兼容端点）。"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.observability.metrics import LLM_TOKENS, STAGE_LATENCY


class LLMClient:
    def __init__(self) -> None:
        from openai import OpenAI

        s = get_settings()
        self.client = OpenAI(base_url=s.mimo_base_url, api_key=s.mimo_api_key)
        self.model = s.mimo_model
        self.thinking = s.mimo_thinking

    def complete(self, system: str, user: str, temperature: float = 0.1) -> str:
        kwargs = {}
        # MiMo 通过 extra_body.thinking 控制是否输出推理链
        if self.thinking in ("enabled", "disabled"):
            kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            LLM_TOKENS.labels(type="input").inc(usage.prompt_tokens or 0)
            LLM_TOKENS.labels(type="output").inc(usage.completion_tokens or 0)
        return resp.choices[0].message.content or ""

    def generate(self, system: str, user: str, temperature: float = 0.1) -> str:
        """带阶段时延埋点的生成调用。"""
        with STAGE_LATENCY.labels(stage="llm_generate").time():
            return self.complete(system, user, temperature)


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
