"""一次性探测 MiMo 联网插件：真调 /chat/completions（tools=web_search），打印返回结构。

用 .env 的 mimo_api_key（不硬编码密钥）；只此一次调用（按次计费 ¥16/1000）。
用法：$env:MIMO_KEY_FROM_ENV=1; .venv\\...\\python.exe scripts/try_mimo_web.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.config import get_settings


def main() -> None:
    s = get_settings()
    if not s.mimo_api_key:
        raise SystemExit("mimo_api_key 未配置（.env）")
    url = s.mimo_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": s.mimo_model,
        "messages": [
            {"role": "system",
             "content": "你是 MiMo。回答需基于联网结果。请同时列出参考来源 URL。"},
            {"role": "user", "content": "2026 年诺贝尔文学奖得主是谁？请给出来源链接。"},
        ],
        "tools": [{"type": "web_search", "force_search": True, "max_keyword": 3}],
        "temperature": 0.3,
        "stream": False,
    }
    r = httpx.post(url, headers={"api-key": s.mimo_api_key, "Content-Type": "application/json"},
                   json=body, timeout=60)
    print("HTTP", r.status_code)
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        print("非 JSON 响应：", r.text[:1000]); return

    print("顶层键：", list(data.keys()))
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message", {})
    print("message 键：", list(msg.keys()))
    print("finish_reason:", ch.get("finish_reason"))
    content = msg.get("content")
    print("\n--- content (前 1200 字) ---\n", (content or "")[:1200])
    for k in ("tool_calls", "annotations", "url_citation", "references", "search_results", "citations"):
        if k in msg:
            print(f"\n--- msg.{k} ---\n", json.dumps(msg[k], ensure_ascii=False)[:1200])
    for k in ("annotations", "search_results", "citations", "references"):
        if k in data:
            print(f"\n--- 顶层 {k} ---\n", json.dumps(data[k], ensure_ascii=False)[:1200])
    print("\nusage:", json.dumps(data.get("usage", {}), ensure_ascii=False))
    # 把整包结构键也 dump 一点，便于判断从哪取结果
    print("\nraw keys probe:", json.dumps({k: type(v).__name__ for k, v in data.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
