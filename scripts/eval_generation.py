"""生成质量消融：对比不同生成配置（是否强制引用）的忠实度/幻觉率/拒答表现。

用法： .venv\\Scripts\\python.exe scripts\\eval_generation.py [-n 8]
调用 MiMo（生成+逐句 judge），属集成评测，不进 CI。
- 可答题(golden.jsonl)：expected_refusal=False，看 faithfulness、幻觉率、是否误拒。
- 不可答题(golden_unanswerable.jsonl)：expected_refusal=True，看拒答准确率。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR  # noqa: E402
from app.evaluation.generation_metrics import (  # noqa: E402
    GenSample, avg_faithfulness, hallucination_rate, over_refusal_rate, refusal_accuracy,
)
from app.generation.faithfulness import REFUSAL_PHRASE, assess_faithfulness  # noqa: E402
from app.generation.generator import answer_query  # noqa: E402
from app.generation.llm import get_llm  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.retrieval.retriever import retrieve  # noqa: E402

CONFIGS = [
    ("base(不强制引用)", dict(force_citation=False)),
    ("cited(强制引用)", dict(force_citation=True)),
    ("no-RAG(无证据对照)", dict(force_citation=True, no_rag=True)),  # 负向对照
]

NO_RAG_SYSTEM = (
    "你是一个知识渊博的助手，请凭自身知识尽可能详细地回答用户问题。"
)


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run_pipeline(question: str, **kwargs):
    """no_rag 对照：凭模型自身知识回答，再拿检索上下文去判忠实度。"""
    if kwargs.get("no_rag"):
        chunks = retrieve(question)
        context = "\n\n".join(c.context for c in chunks)
        text = get_llm().complete(system=NO_RAG_SYSTEM, user=question)
        rep = assess_faithfulness(context, text, get_llm())
        refused = rep["is_refusal"]

        class _A:
            pass
        a = _A()
        a.text, a.faithfulness, a.is_refusal, a.low_confidence = text, rep["faithfulness"], refused, not rep["grounded"]
        return a
    return answer_query(question, **kwargs)


def _is_refusal(ans) -> bool:
    return ans.is_refusal or ans.text.strip().startswith(REFUSAL_PHRASE)


def run(n_answerable: int) -> None:
    configure_logging()
    gdir = ROOT_DIR / "data" / "golden"
    answerable = _load(gdir / "golden.jsonl")[:n_answerable]
    unanswerable = _load(gdir / "golden_unanswerable.jsonl")

    print(f"\n# Generation ablation: {len(answerable)} answerable + {len(unanswerable)} unanswerable\n")
    cols = ["config", "avg_faith", "halluc_rate", "refusal_acc", "over_refusal", "low_conf(答了但存疑)"]
    print(f"{cols[0]:<22}" + "".join(f"{c:>16}" for c in cols[1:]))
    print("-" * (22 + 16 * (len(cols) - 1)))

    for name, kwargs in CONFIGS:
        samples: list[GenSample] = []
        low_conf = 0
        for item in answerable:
            ans = _run_pipeline(item["question"], **kwargs)
            samples.append(GenSample(item["question"], False, _is_refusal(ans), ans.faithfulness))
            low_conf += 1 if ans.low_confidence else 0
        for item in unanswerable:
            ans = _run_pipeline(item["question"], **kwargs)
            samples.append(GenSample(item["question"], True, _is_refusal(ans), ans.faithfulness))
            low_conf += 1 if ans.low_confidence else 0

        print(
            f"{name:<22}"
            f"{avg_faithfulness(samples):>16.3f}"
            f"{hallucination_rate(samples):>16.3f}"
            f"{refusal_accuracy(samples):>16.3f}"
            f"{over_refusal_rate(samples):>16.3f}"
            f"{low_conf:>16}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8, help="可答题取样数")
    run(ap.parse_args().n)
