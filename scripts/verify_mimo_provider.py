"""P2-a：用真实 MiMoWebProvider 走一次联网检索，把原始返回与解析结果落盘，钉死字段结构。

消耗一次按量调用。用法：.venv\\python.exe scripts/verify_mimo_provider.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.search.mimo import MiMoWebProvider  # noqa: E402


def main() -> None:
    s = get_settings()
    url = s.mimo_base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": s.mimo_model,
        "messages": [
            {"role": "system", "content": "你是检索器。只用联网工具找到的网页作答，逐条给出标题、URL、摘要。"},
            {"role": "user", "content": "2026 年 10 月最新发布的国产大模型有哪些？给来源链接。"},
        ],
        "tools": [{"type": "web_search", "force_search": True, "max_keyword": 3}],
        "temperature": 0.2, "stream": False,
    }
    r = httpx.post(url, headers={"api-key": s.mimo_api_key, "Content-Type": "application/json"}, json=body, timeout=90)
    data = r.json()
    out = Path("logs/mimo_verify.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    msg = (data.get("choices") or [{}])[0].get("message", {})
    anns = msg.get("annotations") or []
    print("HTTP", r.status_code, "| 顶层键:", list(data.keys()))
    print("message 键:", list(msg.keys()))
    print("annotations 数量:", len(anns))
    if anns:
        print("首个 annotation 结构:", json.dumps(anns[0], ensure_ascii=False)[:600])
    parsed = MiMoWebProvider._parse(data, 8)
    print("\n解析出的 SearchResult:", json.dumps(
        [{"title": r.title[:40], "url": r.url, "snippet": r.snippet[:40]} for r in parsed],
        ensure_ascii=False, indent=2))
    print(f"\n原始返回已存 → {out}")


if __name__ == "__main__":
    main()
