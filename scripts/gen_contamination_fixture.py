"""生成写回污染评测用的对抗夹具（合成、确定、可复现）。

每个实体：库里一个正确端口值、写回记忆里一个错误端口值（来源标"网页"）。对错值都用固定位数、
整词匹配避免子串误判。程序化生成到 ~100 条探针（非手写，够统计）。
用法： .venv\\Scripts\\python.exe scripts/gen_contamination_fixture.py [--n 100]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "golden" / "contamination_fixture.json"


def build(n: int) -> dict:
    lines = ["手持测试仪默认管理端口对照表。"]
    probes = []
    for i in range(1, n + 1):
        ent = f"ZK-{i:04d}"
        correct = 40000 + i        # 5 位，唯一
        wrong = 70000 + i          # 5 位，与所有正确值不重叠
        lines.append(f"{ent} 的默认管理端口是 {correct}。")
        probes.append({
            "id": ent,
            "query": f"{ent} 的默认管理端口是多少？",
            "correct_value": str(correct),
            "wrong_value": str(wrong),
            "wrong_memory": f"{ent} 的默认管理端口是 {wrong}（来源：某网页抓取）。",
        })

    # KB 沉默场景：库里**没有**这些实体，只有写回的错误记忆——这才是写回真正危险的地方。
    silent = []
    for i in range(1, n + 1):
        ent = f"QN-{i:04d}"
        wrong = 80000 + i          # 库内不存在，写回记忆是**唯一来源**
        silent.append({
            "id": ent,
            "query": f"{ent} 的默认管理端口是多少？",
            "correct_value": "00000",          # 永不可知 → 只会落在 wrong/abstain
            "wrong_value": str(wrong),
            "wrong_memory": f"{ent} 的默认管理端口是 {wrong}（来源：某网页抓取）。",
        })

    return {
        "clean_docs": [{"source": "testbench_ports.md", "text": "\n".join(lines)}],
        "probes": probes,
        "kb_silent_probes": silent,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    data = build(a.n)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(data['probes'])} 条探针 → {OUT}")


if __name__ == "__main__":
    main()
