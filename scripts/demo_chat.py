"""多轮改写演示（供手动验证 /chat 逻辑）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.generator import answer_query  # noqa: E402

history = [
    {"role": "user", "content": "XG-400 Pro这台设备"},
    {"role": "assistant", "content": "XG-400 Pro是面向中大型企业的高性能边缘网关。"},
]
a = answer_query("它的默认管理端口是多少？", history=history)
print("REWRITTEN:", a.rewritten_query)
print("ANS:", a.text)
print("FAITH:", a.faithfulness, "REFUSE:", a.is_refusal)
