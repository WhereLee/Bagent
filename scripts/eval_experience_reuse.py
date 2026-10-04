"""P3-A 评测：经验分层能否提升"跨场景可复用"——扁平语义召回 vs 分层(playbook 保底槽)召回。

只用本地 bge 嵌入 + numpy，无 LLM/无 DB，确定性可复现。核心命题：抽象打法(playbook)与
新任务几乎不共享表面词，纯扁平 top-k 常被具体 fact 挤掉；给 playbook 保留一个召回槽，
能否提高"命中可迁移经验"的比率。用法：.venv\\python.exe scripts/eval_experience_reuse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.stats import bootstrap_ci, mean  # noqa: E402
from app.ingestion.embedder import get_embedder  # noqa: E402

# 每条 (text, level)；playbook 是跨场景可迁移打法，fact 是具体经验
CORPUS = [
    ("年会当天设备调试留 90 分钟缓冲", "fact"),
    ("骑行出发前清点备胎与补剂", "fact"),
    ("露营选址避开河滩低洼处", "fact"),
    ("婚礼彩排须提前一天到场走台", "fact"),
    ("线上直播须做双链路推流备份", "fact"),
    ("重大活动必做风险预案并指定责任人", "playbook"),
    ("任何现场执行都要预留缓冲时间应对意外", "playbook"),
    ("关键资源与人员须有备份以防单点故障", "playbook"),
    ("活动前做一次全流程走查/彩排", "playbook"),
]
# 新任务查询 → 期望命中的可迁移 playbook（表面词刻意不同）
PROBES = [
    ("组织一场新产品发布会", "重大活动必做风险预案并指定责任人"),
    ("筹办学院毕业晚会现场", "任何现场执行都要预留缓冲时间应对意外"),
    ("策划一次部门季度团建出游", "重大活动必做风险预案并指定责任人"),
    ("筹备一场线下读书分享会", "活动前做一次全流程走查/彩排"),
    ("组织志愿者社区服务活动", "关键资源与人员须有备份以防单点故障"),
]


def _topk(scores, items, k):
    order = np.argsort(scores)[::-1][:k]
    return [items[i] for i in order]


def main(k: int) -> None:
    emb = get_embedder()
    texts = [c[0] for c in CORPUS]
    levels = [c[1] for c in CORPUS]
    docs = emb.encode_documents(texts)
    flat_hits, layered_hits = [], []
    for q, target in PROBES:
        qv = emb.encode_query(q)
        sims = docs @ qv
        flat = _topk(sims, list(zip(texts, levels)), k)
        hit_flat = float(any(t == target and lv == "playbook" for t, lv in flat))
        # 分层：playbook 槽保底 top-1 playbook + 其余按相似度
        pb_idx = [i for i, lv in enumerate(levels) if lv == "playbook"]
        best_pb = pb_idx[int(np.argmax([sims[i] for i in pb_idx]))]
        fact_slot = _topk(sims, list(range(len(texts))), k - 1)
        layered_texts = {texts[i] for i in fact_slot} | {texts[best_pb]}
        hit_layered = float(target in layered_texts)
        flat_hits.append(hit_flat)
        layered_hits.append(hit_layered)

    lo_f, hi_f = bootstrap_ci(flat_hits)
    lo_l, hi_l = bootstrap_ci(layered_hits)
    print(f"\n# 跨场景可复用命中率 (K={k}, n={len(PROBES)})：命中该任务的可迁移 playbook\n")
    print(f"扁平语义召回      : {mean(flat_hits):.2f}  (CI {lo_f:.2f}–{hi_f:.2f})")
    print(f"分层(playbook保底): {mean(layered_hits):.2f}  (CI {lo_l:.2f}–{hi_l:.2f})")
    print("\n解读：playbook 与新任务表面不相似，扁平易被具体 fact 挤出 top-k；"
          "给可迁移经验保留召回槽提升复用命中。样本极小(合成)，看趋势非统计显著。")


if __name__ == "__main__":
    main(k=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
