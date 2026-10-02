"""在 RGB(中文) 上评测生成侧证据链：噪声鲁棒 + 负例拒答。

RGB 自带文档(positive=金标准/negative=噪声)，不经我们的库检索，直接构造上下文测生成：
- 噪声鲁棒：给 positive+negative，测答案命中(gold 出现)与 faithfulness。
- 负例拒答：只给 negative(无 gold)，测系统是否正确拒答。
集成脚本（调 MiMo），不进 CI。先浅克隆 RGB 到 data/_ext/RGB。
用法： .venv\\Scripts\\python.exe scripts\\eval_rgb.py [-n 15]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR  # noqa: E402
from app.generation.faithfulness import REFUSAL_PHRASE, assess_faithfulness  # noqa: E402
from app.generation.llm import get_llm  # noqa: E402
from app.generation.prompts import SYSTEM_BASE, SYSTEM_CITED  # noqa: E402

_WS = re.compile(r"\s+")
RGB_PATH = ROOT_DIR / "data" / "_ext" / "RGB" / "data" / "zh.json"


def _norm(s: str) -> str:
    return _WS.sub("", s or "")


def _context(docs: list[str]) -> str:
    return "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, start=1))


def _answer(system: str, query: str, ctx: str) -> str:
    return get_llm().complete(system=system, user=f"【参考资料】\n{ctx}\n\n【问题】\n{query}")


def load_items(n: int) -> list[dict]:
    if not RGB_PATH.exists():
        sys.exit("RGB 数据缺失，请先: git clone --depth 1 https://github.com/chen700564/RGB.git data/_ext/RGB")
    items = []
    with RGB_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("positive") and r.get("negative"):
                items.append(r)
            if len(items) >= n:
                break
    return items


def main(n: int) -> None:
    items = load_items(n)
    llm = get_llm()

    hit = faith_sum = faith_cnt = 0          # 噪声鲁棒（cited）
    rej_cited = rej_base = 0                 # 负例拒答：强制引用 vs 基础提示
    total = len(items)

    for r in items:
        golds = [_norm(a) for a in r["answer"] if isinstance(a, str)]
        pos, neg = r["positive"], r["negative"]

        # 噪声鲁棒：positive + 全部 negative 作上下文
        ctx = _context(pos + neg[:5])
        ans = _answer(SYSTEM_CITED, r["query"], ctx)
        if any(g and g in _norm(ans) for g in golds):
            hit += 1
        rep = assess_faithfulness(ctx, ans, llm)
        if rep["faithfulness"] is not None:
            faith_sum += rep["faithfulness"]; faith_cnt += 1

        # 负例拒答：只给 negative（无 gold）
        ctx_neg = _context(neg[:5])
        ans_cited = _answer(SYSTEM_CITED, r["query"], ctx_neg)
        ans_base = _answer(SYSTEM_BASE, r["query"], ctx_neg)
        if ans_cited.strip().startswith(REFUSAL_PHRASE):
            rej_cited += 1
        if ans_base.strip().startswith(REFUSAL_PHRASE):
            rej_base += 1

    print(f"\n# RGB(zh) 生成侧评测, n={total}\n")
    print(f"噪声鲁棒-答案命中率(cited)     : {hit/total:.3f}")
    print(f"噪声鲁棒-平均faithfulness(cited): {faith_sum/faith_cnt:.3f}" if faith_cnt else "faithfulness: n/a")
    print(f"负例拒答-强制引用 SYSTEM_CITED   : {rej_cited/total:.3f}")
    print(f"负例拒答-基础提示 SYSTEM_BASE    : {rej_base/total:.3f}")
    print("\n说明：拒答率越高越好(负例)；对比两种提示可量化'收紧引用/提示'对拒答的影响。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=15)
    main(ap.parse_args().n)
