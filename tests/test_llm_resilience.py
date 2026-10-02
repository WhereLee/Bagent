"""LLM 多 provider 韧性的单元测试（不联网：绕开 __init__，注入假 _call_once）。"""
import pytest

from app.generation.llm import LLMClient, LLMUnavailable, Provider


def _mk(chain, max_retries=2, base_delay=0.0):
    c = LLMClient.__new__(LLMClient)  # 跳过需要 key/openai 的 __init__
    c.chain = chain
    c.max_retries = max_retries
    c.base_delay = base_delay
    c.timeout = 1
    c._retryable = (TimeoutError,)
    c._status_err = Exception
    c._clients = {}
    return c


def _p(name):
    return Provider(name, "http://x", "k", "m")


def test_is_retryable_classification():
    assert LLMClient._is_retryable(TimeoutError(), (TimeoutError,), Exception) is True
    # 非重试类型且无 5xx -> False
    assert LLMClient._is_retryable(ValueError(), (TimeoutError,), object) is False


def test_success_first_try():
    c = _mk([_p("a"), _p("b")])
    c._call_once = lambda p, s, u, t: "from_" + p.name
    assert c.complete("s", "u") == "from_a"


def test_retry_then_success_same_provider():
    state = {"n": 0}

    def flaky(p, s, u, t):
        state["n"] += 1
        if state["n"] < 3:
            raise TimeoutError("transient")
        return "recovered"

    c = _mk([_p("a")], max_retries=3)
    c._call_once = flaky
    assert c.complete("s", "u") == "recovered"


def test_fallback_after_primary_exhausts():
    def fn(p, s, u, t):
        if p.name == "a":
            raise TimeoutError("down")
        return "b_ok"

    c = _mk([_p("a"), _p("b")], max_retries=2)
    c._call_once = fn
    assert c.complete("s", "u") == "b_ok"


def test_all_providers_fail_raises_unavailable():
    def always(p, s, u, t):
        raise TimeoutError("down")

    c = _mk([_p("a"), _p("b")], max_retries=1)
    c._call_once = always
    with pytest.raises(LLMUnavailable):
        c.complete("s", "u")
