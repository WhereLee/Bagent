"""知识冲突检测与解析的单元测试。"""
from app.generation.conflict import detect_conflict, parse_conflict_verdict


class FakeLLM:
    def __init__(self, out):
        self.out = out

    def complete(self, system, user):
        return self.out


def test_parse_json_true():
    c, e = parse_conflict_verdict('{"conflict": true, "explanation": "端口值不一致"}')
    assert c is True and "端口" in e


def test_parse_json_false():
    c, _ = parse_conflict_verdict('{"conflict": false, "explanation": ""}')
    assert c is False


def test_parse_malformed_conservative():
    c, _ = parse_conflict_verdict("sorry I can't decide")
    assert c is False  # 解析失败保守判无冲突


def test_parse_empty():
    assert parse_conflict_verdict("") == (False, "")


def test_detect_conflict_uses_llm():
    r = detect_conflict("q", "[1] a\n[2] b", FakeLLM('{"conflict": true, "explanation": "x"}'))
    assert r["conflict"] is True


def test_detect_conflict_empty_context_no_call():
    # 空上下文直接不判冲突（不会调 LLM）
    r = detect_conflict("q", "", FakeLLM('{"conflict": true}'))
    assert r["conflict"] is False
