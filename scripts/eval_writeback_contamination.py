"""写回记忆污染 A/B 实验（端到端真跑）。

三臂（隔离库 bagent_exp，rerank 关，因变量是记忆召回不是检索）：
- A0 记忆关：只靠 KB（地板，应基本 correct）。
- A1 naive：draft 记忆**参与作答**（min_trust=draft）→ 看污染。
- A2 门控：同一批 draft 记忆，但 min_trust=verified **挡在作答外**（现状）。
- A2b 合谋泄漏：同一错误事实从 2 来源佐证→促升 verified→即使门控也召回（诚实暴露多源≠真相）。
判定用整词四类，污染率带 bootstrap CI。用法：python scripts/eval_writeback_contamination.py [--n 30] [--repeats 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PG_DATABASE"] = "bagent_exp"          # 隔离实验库，别污染 demo/bench
os.environ.setdefault("RERANK_ENABLED", "false")  # 记忆才是自变量
os.environ.setdefault("RETRIEVAL_CACHE_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import ROOT_DIR  # noqa: E402
from app.db.session import get_session, init_schema  # noqa: E402
from app.evaluation.contamination import (  # noqa: E402
    classify_answer, outcome_across_repeats, outcome_report,
)
from app.generation.generator import answer_query  # noqa: E402
from app.ingestion.chunking import chunk_parent_child  # noqa: E402
from app.ingestion.embedder import get_embedder  # noqa: E402
from app.memory import store as ms  # noqa: E402
from app.retrieval.store import index_document  # noqa: E402


def ensure_db() -> None:
    import psycopg
    s = __import__("app.config", fromlist=["get_settings"]).get_settings()
    try:
        with psycopg.connect(s.sqlalchemy_url.replace("+psycopg", "").replace(
                "/bagent_exp", "/postgres"), autocommit=True) as c:
            if c.execute("SELECT 1 FROM pg_database WHERE datname='bagent_exp'").fetchone() is None:
                c.execute("CREATE DATABASE bagent_exp")
    except Exception as e:  # noqa: BLE001
        print(f"[db] 建库注意: {e}")


def reset_memories(session) -> None:
    from sqlalchemy import delete
    from app.db.models import Memory
    session.execute(delete(Memory))
    session.commit()


def seed_kb(embedder) -> None:
    fx = json.loads((ROOT_DIR / "data" / "golden" / "contamination_fixture.json").read_text(encoding="utf-8"))
    session = get_session()
    try:
        for d in fx["clean_docs"]:
            parents = chunk_parent_child(d["text"], parent_tokens=800, child_tokens=200, child_overlap=40)
            ct = [c.text for p in parents for c in p.children]
            index_document(session, d["source"], d["text"], "md", parents, embedder.encode_documents(ct))
        session.commit()
    finally:
        session.close()


def inject(embedder, probes, *, corroborate: bool) -> None:
    """把每条错误结论作为 knowledge 记忆写入（trust=draft）；corroborate=True 再注一次 → 促升。"""
    session = get_session()
    try:
        for pr in probes:
            vec = embedder.encode_documents([pr["wrong_memory"]])[0]
            ms.add_memory(session, scope="knowledge", content=pr["wrong_memory"],
                          embedding=vec, source_type="web", source_ref="网页", trust="draft")
            if corroborate:
                v2 = embedder.encode_documents([pr["wrong_memory"] + "（另一来源）"])[0]
                sim = ms.find_similar(session, v2, scope="knowledge", owner_user_id=None, k=3)
                op, tid = __import__("app.memory.lifecycle", fromlist=["decide_op"]).decide_op(
                    sim, 0.92, exact_hash_match=False)
                if tid:
                    row = ms.bump_support(session, tid)
                    if row is not None:
                        row.trust = "verified"   # 模拟多源促升
        session.commit()
    finally:
        session.close()


def run_arm(embedder, probes, arm: str, repeats: int) -> list[str]:
    outcomes = []
    for pr in probes:
        per = []
        for _ in range(repeats):
            if arm == "A0":
                ans = answer_query(pr["query"], use_memory=False)
            else:
                trust = "draft" if arm == "A1" else "verified"   # A2/A2b 都 verified
                ans = answer_query(pr["query"], use_memory=True, memory_min_trust=trust)
            per.append(classify_answer(ans.text, pr["correct_value"], pr["wrong_value"]))
        outcomes.append(outcome_across_repeats(per))
    return outcomes


def run_scenario(embedder, probes, repeats):
    """一个场景下跑四臂。A0 无记忆；注入 draft 后 A1(draft参与)/A2(verified门控)；再合谋促升 A2b。"""
    res = {}
    s = get_session(); reset_memories(s); s.close()
    res["A0_memory_off"] = run_arm(embedder, probes, "A0", repeats)
    s = get_session(); reset_memories(s); s.close()
    inject(embedder, probes, corroborate=False)
    res["A1_naive_draft"] = run_arm(embedder, probes, "A1", repeats)
    res["A2_gated"] = run_arm(embedder, probes, "A2", repeats)
    s = get_session(); reset_memories(s); s.close()
    inject(embedder, probes, corroborate=True)
    res["A2b_collusion_leak"] = run_arm(embedder, probes, "A2b", repeats)
    return res


def print_table(title, results):
    print(f"\n## {title}")
    print(f"{'arm':<20}{'contam':>9}{'CI95':>15}{'conflict':>10}{'correct':>9}{'abstain':>9}")
    print("-" * 72)
    reports = {}
    for name, outs in results.items():
        rep = outcome_report(outs); reports[name] = rep
        lo, hi = rep["contamination_ci"]
        print(f"{name:<20}{rep['contamination_rate']*100:>8.1f}%"
              f"{f'±{(hi-lo)/2*100:.1f}%':>15}{rep['rates']['conflict']*100:>9.1f}%"
              f"{rep['rates']['correct']*100:>8.1f}%{rep['rates']['abstain']*100:>8.1f}%")
    return reports


def main(n: int, repeats: int) -> None:
    ensure_db()
    init_schema()
    emb = get_embedder()
    seed_kb(emb)
    fx = json.loads((ROOT_DIR / "data" / "golden" / "contamination_fixture.json").read_text(encoding="utf-8"))
    scenarios = {
        "conflict(KB有正确值+错误记忆并存)": fx["probes"][:n],
        "kb_silent(库无此事实、错误记忆是唯一来源)": fx["kb_silent_probes"][:n],
    }
    print("\n# 写回污染 A/B（污染=采纳错值；conflict=同时给对/错值=主动标分歧；abstain=未采纳）")
    all_reports = {}
    for title, probes in scenarios.items():
        all_reports[title] = print_table(title, run_scenario(emb, probes, repeats))
    out = ROOT_DIR / "logs" / "contamination_ab.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n判读：预期 conflict 场景 naive 污染低（KB真相在场、只会标分歧）；"
          "kb_silent 才见真高污染(A1) vs 安全拒答/abstain(A2)，A2b 促升后泄漏=多源合谋绕过门控（已知边界）。明细 → logs/contamination_ab.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    main(a.n, a.repeats)
