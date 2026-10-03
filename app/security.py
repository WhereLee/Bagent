"""安全：API-Key 鉴权、prompt 注入检测/中和、租户过滤（纯逻辑，可测）。

注意：prompt 注入没有银弹，这里是纵深防御的一层——对**用户输入**做检测/降级、对**检索文档**
做定界包裹（在 prompt 侧），并在系统提示中要求把参考资料当数据处理。
"""
from __future__ import annotations

import hmac
import re

# 常见注入模式（中英）：指令覆盖 / 角色伪造 / 提示词外泄
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
    re.compile(r"忽略(以上|上述|之前)(的)?(所有)?指令", re.I),
    re.compile(r"disregard\s+the\s+system\s+prompt", re.I),
    re.compile(r"new\s+instructions\s*:", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M),
    re.compile(r"<\|.*?\|>"),
    re.compile(r"</?s>", re.I),
]


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def sanitize_query(text: str) -> str:
    """把可疑注入片段中和掉（替换占位），而非直接拒绝——保证可用性。"""
    out = text
    for p in _INJECTION_PATTERNS:
        out = p.sub("[filtered]", out)
    return out


def constant_time_equal(a: str, b: str) -> bool:
    """API Key 比较，防时序侧信道。"""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# 租户过滤（⑥）真实现在 retrieval/store 与 memory/store 的 SQL 里（按 document 级
# metadata.tenant + memories.tenant_id），不在此提供易误导的通用 helper。
