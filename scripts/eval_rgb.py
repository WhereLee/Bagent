"""在 RGB(中文) 上评测生成侧证据链：噪声鲁棒/负例拒答 + 反事实 + 信息整合。

RGB 自带文档(positive=金标准/positive_wrong=错误信息/negative=噪声)，评测不经我们的检索库，
直接构造上下文测生成侧。子集：
- noise_reject(zh.json)：噪声鲁棒(命中+faithfulness)、负例拒答(只给噪声→应拒)。
- counterfactual(zh_fact.json)：只给"错误文档"→模型是否被带偏(答 fakeanswer)。
- integration(zh_int.json)：需综合两条事实→两条都答对率。
集成脚本(调 LLM)，不进 CI。先浅克隆 RGB 到 data/_ext/RGB。
用法： python scripts/eval_rgb.py -n 6 --subsets noise_reject counterfactual integration
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR, get_settings  # noqa: E402
from app.generation.faithfulness import REFUSAL_PHRASE, assess_faithfulness  # noqa: E402
from app.generation.llm import get_llm  # noqa: E402
from app.generation.prompts import SYSTEM_CITED  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402

_RGB = ROOT_DIR / "data" / "_ext" / "RGB" / "data"
_WS = re.compile(r"\s+")


def _norm(s):
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    return _WS.sub("", str(s))


def _flat_docs(x):
    """positive/negative 可能是嵌套 list，展平为字符串文档列表。"""
    out = []
    for e in (x or []):
        out.extend(_flat_docs(e) if isinstance(e, list) else [str(e)])
    return out


def _read(fname):
    p = _RGB / fname
    if not p.exists():
        sys.exit(f"缺 {p}；先 git clone --depth 1 https://github.com/chen700564/RGB.git data/_ext/RGB")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ctx(docs):
    return "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1))


def _ans(system, query, ctx):
    return get_llm().complete(system=system, user=f"【参考资料】\n{ctx}\n\n【问题】\n{query}")


def eval_noise_reject(n):
    rows = _read("zh.json")[:n]
    hit = faith_sum = 0.0
    rej = 0
    cnt = 0
    for r in rows:
        pos, neg = _flat_docs(r.get("positive")), _flat_docs(r.get("negative"))
        golds = [_norm(a) for a in (r["answer"] if isinstance(r["answer"], list) else [r["answer"]])]
        ctx = _ctx(pos + neg[:5])
        a = _ans(SYSTEM_CITED, r["query"], ctx)
        if any(g and g in _norm(a) for g in golds):
            hit += 1
        rep = assess_faithfulness(ctx, a, get_llm())
        if rep["faithfulness"] is not None:
            faith_sum += rep["faithfulness"]; cnt += 1
        # 只给噪声 -> 应拒
        a2 = _ans(SYSTEM_CITED, r["query"], _ctx(neg[:5]))
        if a2.strip().startswith(REFUSAL_PHRASE):
            rej += 1
    m = len(rows)
    print(f"[noise_reject n={m}] 命中={hit/m:.3f} faithfulness={(faith_sum/cnt if cnt else 0):.3f} 负例拒答={rej/m:.3f}")


def eval_counterfactual(n):
    rows = _read("zh_fact.json")[:n]
    misled = correct = 0
    for r in rows:
        wrong = _flat_docs(r.get("positive_wrong"))
        true_ans, fake_ans = _norm(r["answer"]), _norm(r.get("fakeanswer", ""))
        a = _ans(SYSTEM_CITED, r["query"], _ctx(wrong[:3]))
        na = _norm(a)
        if fake_ans and fake_ans in na:
            misled += 1
        elif true_ans and true_ans in na:
            correct += 1
    m = len(rows)
    print(f"[counterfactual n={m}] 被错误文档带偏率={misled/m:.3f} 顶住错误答对率={correct/m:.3f}"
          "  (被带偏=忠实于错误上下文，是纯 RAG 的固有软肋)")


def eval_integration(n):
    rows = _read("zh_int.json")[:n]
    both = 0
    for r in rows:
        pos = _flat_docs(r.get("positive"))
        a1, a2 = _norm(r.get("asnwer1", r.get("answer1", ""))), _norm(r.get("answer2", ""))
        na = _norm(_ans(SYSTEM_CITED, r["query"], _ctx(pos)))
        if (a1 and a1 in na) and (a2 and a2 in na):
            both += 1
    m = len(rows)
    print(f"[integration n={m}] 两条子事实都答对率={both/m:.3f}")


SUBSETS = {
    "noise_reject": eval_noise_reject,
    "counterfactual": eval_counterfactual,
    "integration": eval_integration,
}

if __name__ == "__main__":
    configure_logging(get_settings().log_level)
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--subsets", nargs="*", default=["noise_reject", "counterfactual", "integration"])
    a = ap.parse_args()
    print(f"\n# RGB(zh) 生成侧评测 n={a.n}\n")
    for name in a.subsets:
        SUBSETS[name](a.n)
