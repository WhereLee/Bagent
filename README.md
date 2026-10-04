# Bagent RAG

一个**生产级、可独立部署**的检索增强生成（RAG）系统。目标不是"跑通一个 demo"，
而是把 RAG 全链路的每一步（切块、向量化、混合检索、重排、生成、评估）都做成**可量化、
可解释、可迭代**的工程，能够经受"你凭什么说效果更好"的追问。

- LLM：Xiaomi MiMo（`mimo-v2.6-flash`，OpenAI 兼容端点，按量计费经济型）
- Embedding：`bge-small-zh-v1.5`（本地 CPU 推理，权重落在 `models/`，不写 C 盘）
- 向量库：PostgreSQL 18 + pgvector 0.8.2（HNSW / cosine）
- 服务：FastAPI

## 这是什么（产品定位）

底层是一个可量化的 RAG/Agent 引擎；上层产品是**“组织经验复用 + 联网补全 + 对话打磨 + 复盘回写”的闭环系统**：
沉淀过去同类事项（活动/年会/策划/会议复盘/事故 postmortem）的经验 → 新任务来时**检索历史经验 ⊕ 联网补当下信息** → 与人来回讨论 → 把本次结论**归档成新经验**（受 trust 门控，防污染）。
可复用的是**这套循环**（领域无关，换语料/标签即用）。例：骑行社长筹备“骑到文昌”，系统汇总历任同类活动打法+文昌实时信息，讨论后归档供下任复用。

> 实现细节与分层评测见 `docs/M12-组织经验复用系统-锚定与计划.md`。

---

## 架构：两条 pipeline

```
┌── 索引 pipeline（离线） ─────────────────────────────────┐
│  文档 → 解析 → 清洗 → 切块 → 向量化 → 写入 pgvector        │
└──────────────────────────────────────────────────────────┘
                          │  documents / chunks(embedding)
                          ▼
┌── 查询 pipeline（在线） ─────────────────────────────────┐
│  问题 →(改写)→ 向量化 → 召回 →(混合/重排)→ 拼上下文 →      │
│  MiMo 生成 →(引用/拒答)→ 答案                             │
└──────────────────────────────────────────────────────────┘
```

## 目录结构

```
app/
  config.py            # 统一配置 + HF 缓存重定向
  db/       schema.sql models.py session.py
  ingestion/ parser.py chunking.py embedder.py indexer.py
  retrieval/ store.py                 # 向量检索（M2 加混合检索/rerank）
  generation/ llm.py generator.py     # MiMo 客户端 + 查询编排
  api/      main.py                   # FastAPI
scripts/    download_models.py init_db.py rag_cli.py
data/       corpus/ golden/           # 语料 / 评估集
tests/  docs/  models/
```

## 已知限制与运维约束（诚实列出，非黑盒）

- **进程内状态四处**（BM25 内存索引、检索缓存、令牌桶、research 会话）：**必须单进程部署**（start_app.sh 未加 `--workers`）。加 `--workers N` 会造成这四个状态不一致（缓存/限流/租户视图分叉）。要横向扩展需先把这些外置（Redis/共享存储）。
- **BM25 规模上限**：打分是 Python 层逐查询词遍历全量 tf，且任一写入即全库重建(含全量 jieba 分词)。万级子块就会开始吃秒级；过阈应切 pg_search/Elasticsearch 作词法腿（接口已抽象）。
- **HNSW + WHERE 过滤召回退化**：vector_search 把 is_deleted/tenant 写在同一 WHERE；pgvector 默认 `ef_search=40`、`hnsw.iterative_scan=off`，过滤选择率低时会**真少返**（非仅排序）。上量后需在评测集实测，必要时 `SET hnsw.iterative_scan=relaxed_order`/调大 ef_search 或按租户部分索引/分区。集成测试库小看不出。
- **无迁移工具**：schema 靠幂等 SQL + `ALTER ... IF NOT EXISTS`；**换 embedding 模型需改 `vector(512)` 并全量重建**（维度与模型绑定）。未引 Alembic 是有意的：项目规模下幂等 DDL 足够，上生产多环境时再引。
- **/ingest 白名单**：仅允许读 `INGEST_ALLOWED_ROOT`(默认 data/)下的文件，防任意路径读取；CLI/脚本入库不受此限（信任本地）。

## 快速开始

```powershell
# 1. 依赖（torch 走 CPU 版）
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 下载 embedding 模型到 models/
.\.venv\Scripts\python.exe scripts\download_models.py

# 3. 初始化库表（需先在 .env 配好 PG*）
.\.venv\Scripts\python.exe scripts\init_db.py

# 4a. 命令行直接跑
.\.venv\Scripts\python.exe scripts\rag_cli.py ingest data\corpus
.\.venv\Scripts\python.exe scripts\rag_cli.py query "XG-200 的默认管理端口是多少？"

# 4b. 或起 HTTP 服务
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload
#   POST /ingest  {"path": "data\\corpus"}
#   POST /query   {"query": "...", "top_k": 5}
```

## 设计对齐的"面试深挖点"

本项目专门针对 RAG 面试的高频追问来落地功能（详见 `docs/design.md`）：

| 追问 | 本项目做法 |
|---|---|
| 纯向量检索够不够 | 混合检索 BM25 + 稠密，RRF 融合 |
| 为什么要 Rerank | Cross-Encoder 精排，粗召回 topN → 精排 topK |
| Chunk 怎么切、多大 | 多策略 + 父子块 + overlap，**用消融实验定量比较** |
| 怎么证明效果更好 | 评估体系：recall@k / MRR / NDCG + faithfulness，golden set + 回归 |
| 怎么减幻觉 | 引用溯源（回链 chunk id）+ 无依据拒答 |
| 多轮怎么办 | Query 改写 / 指代消解 |

## 里程碑

- **M1 ✅**：端到端闭环 —— 解析→切块→向量→pgvector→MiMo 生成，含引用与拒答、FastAPI、CLI。
- **M2 ✅**：检索质量 —— 混合检索(BM25+jieba / 稠密 / RRF) + Cross-Encoder rerank + 父子块(small-to-big)；golden set + recall@k/MRR/nDCG 消融。
- **M3 ✅**：生成质量+证据链 —— faithfulness/幻觉率(LLM-as-judge)、引用校验与拒答收紧、多轮 query 改写、生成侧消融(含负向对照)。
- **M4 ✅**：上线运维 —— 结构化 JSON 日志 + request_id、Prometheus 指标(/metrics)、令牌桶限流(429)、Locust 压测、Docker/compose 部署、CI 增强(secret-scan + docker-build)。
- **M5 ✅**：评估可信度与工程补齐 —— ✅接入标准基准 C-MTEB/DuRetrieval(真 qrels、文档级)、✅修 #3 增量正确性 bug、✅#2 candidate 扫参、✅#1 chunk 尺寸扫参、✅生成侧接 RGB（拒答/噪声鲁棒）。
- **M6 ✅**：生产化 —— 多 LLM provider(DeepSeek 主/MiMo 备)+超时/重试退避/降级、检索结果缓存+自适应早退、API-Key 鉴权、prompt 注入防护。
- **M7 ✅**：测试与评估纵深 —— CI 新增 **integration job**（真 pgvector 服务容器跑端到端检索/增量回归）、注入对抗测试集、bootstrap 置信区间、RGB 反事实/信息整合子集。
- **M8 ✅**：Self-RAG 与知识冲突 —— 证据充分性自检 + 不足时改写重检(带上限) + 多源矛盾检测与降级；为接入联网多源预留中枢。
- **M9a ✅**：联网检索层 —— `SearchProvider` 抽象 + **SearXNG provider** + trafilatura 抽正文/时效 + **SSRF/注入清洗** + mock provider + 自架部署物料；默认关，结果以 draft 进多源佐证（实测引擎用 sogou/360，baidu 对机房 IP 弹验证码）。
- **M9b ✅**：写回记忆 —— **个人/知识双库分离** + **双时态**(valid_at/invalid_at) + **trust 生命周期**(draft→verified→curated，多源佐证促升) + **sleep-time 异步写回**(不阻塞回答) + 检索按 trust 门控。默认关。
- **M9c ✅**：研究型 Agent —— **多源取证**(KB⊕联网进 self-RAG) + **文档编排**(列提纲→逐节取证成文→行内引用/参考文献/论断表/冲突) + **会话态活文档** `/research` `/research/{id}` `/research/{id}/refine`。
- **M10(⑥) ✅**：多租户**数据面强制** —— HMAC 签名租户令牌(fail-closed) + 入库盖章 + **检索三路一致过滤**(dense/BM25/缓存) + 记忆按租户隔离 + 跨租户读写越权拦截。
- **M11(P0) ✅**：写回记忆**抗污染端到端 A/B**（三臂+合谋泄漏+四类判定+双场景）；修复 knowledge 记忆只写不读缺口。证据(n=60)：naive 68.3%→门控 0%→合谋泄漏 73.3%（已知边界）。
- **M11(P1) ✅**：MiMo 受控联网检索源（主源空/错时**配置降级**，非主 LLM 自决）+ **共享工具注册表 `app/tools.py`** + **MCP server** + **显式有界 agent 循环与轨迹评测**（`/agent`，收敛率/非法动作率/平均步数）。
- **M11(P2) ✅**：MiMo **端到端真验**（确认返回为平铺 `url_citation`、摘要字段实为 `summary`，修解析+回归测试）+ **wiki 沉淀** `/research/{sid}/publish`（论断→knowledge 记忆 draft，受同一 trust 门控）+ **agent 循环接入 `/query`**（`agent=true` 走工具循环产带依据答案）。
- **M12 产品锚定 ✅**：把引擎向上定成**“组织经验复用 + 联网补全 + 对话打磨 + 复盘回写”闭环**（`docs/M12-*.md`）。**P3-A 经验分层**(fact/playbook+tags；含可复用评测——诚实证伪：扁平时优，不塞更差启发式) + **P3-B 复盘整体归档**(Event 实体 + `event_playbook`)。

- **M12 P3-C ✅**：**结果回标**（`/memory/feedback`：use_success/use_fail 驱动 outcome_action 促升/作废），用“用过成没成”补强“多源一致≠真相”。
- **M13 生产化+上量 ✅**：**成本预算与自动降级**(`request_token_budget`超预算缩top_k/跳rerank/关忠实度, `Answer.degraded`+`COST_DEGRADED`) + **共享状态外部化 Redis**(检索缓存/限流固定窗口/研究会话均可切，接口不变，默认内存) + **SLO/告警**(`deploy/prometheus/slo.rules.yml`+`docs/SLO.md`) + **评测上量 n=600**。

## 检索质量验证（消融，本地复现：`scripts/eval_retrieval.py`）

基于 `data/golden/golden.jsonl`（14 条覆盖 3 篇可混淆手册的标注）：

| 配置 | recall@3 | mrr | ndcg@3 | recall@5 | ndcg@5 |
|---|---|---|---|---|---|
| dense | 0.679 | 0.631 | 0.613 | 1.000 | 0.752 |
| hybrid | 0.857 | 0.631 | 0.692 | 1.000 | 0.748 |
| hybrid+rerank | 0.786 | 0.714 | 0.732 | 0.929 | 0.793 |

结论：**hybrid 显著提升 recall@3（+0.18）**，**rerank 显著提升排序质量（MRR/nDCG）**；
rerank 会轻微拉低 recall@k（重排把边缘相关块排出截断）——该取舍在 M3 调候选池/阈值。

**Reranker 量化（三方 A/B，`scripts/eval_rerank_ab.py` 本机实测）**：导出 ONNX int8（`scripts/export_reranker_onnx.py`）后，磁盘 **1.1G→279MB**；DuRetrieval(cand=20,K=10,n=40)：

| backend | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| st-fp32 | 0.905 | 1.000 | 0.943 |
| st-int8(torch) | 0.905 | 0.988 | 0.934 |
| **onnx-int8** | 0.902 | **1.000** | 0.938 |

→ **生产用 `RERANKER_BACKEND=onnx` + int8**：mrr 与 fp32 持平、nDCG/recall 掉 <0.005（均在 95%CI 内），比 torch 动态 int8 更稳。Reranker 可插拔 st/onnx。

## 生成质量验证（消融，本地复现：`scripts/eval_generation.py`）

4 道可答题 + 5 道不可答题；faithfulness 用 LLM-as-judge 逐句蕴含判定；
**no-RAG 为负向对照**（去掉检索证据、只凭模型知识回答）：

| 配置 | 平均忠实度 | 幻觉率 | 拒答准确率 | 低置信(答了但存疑) |
|---|---|---|---|---|
| base(不强制引用) | 1.000 | 0.000 | 1.000 | 0 |
| cited(强制引用) | 1.000 | 0.000 | 1.000 | 0 |
| **no-RAG(无证据)** | **0.000** | **1.000** | **0.444** | 9 |

结论：负向对照证明 faithfulness 确实能**抓到幻觉**（去证据后塔到 0）；带证据链则满忠实、
且不可答题正确拒答。base/cited 本语料上未拉开差距（语料干净、模型守规矩），不粉饰。

## 可观测 / 限流 / 压测 / 部署（M4）

- **指标**：`GET /metrics`（Prometheus）—— HTTP 请求数/时延直方图、各阶段耗时(retrieval/rerank/llm/faithfulness)、
  LLM token 用量、拒答与低置信计数、限流拒绝计数。
- **日志**：结构化 JSON，每请求一个 `request_id` 贯穿（响应头 `X-Request-ID` 同步返回）。
- **限流**：令牌桶，按客户端 IP；超限返回 **429 + Retry-After**（burst=10、5/s）。实测并发 30 → **10 放行 / 20 拒绝**。
- **多轮**：`POST /chat` 传 `history`，服务端做查询改写（指代消解）后检索生成。
- **压测**：`pip install -r requirements-dev.txt` 后
  `locust -f locustfile.py --host http://127.0.0.1:8000 --headless -u 50 -r 10 -t 60s`。
- **部署**：`docker compose up -d --build`（起 pgvector 库 + app；模型权重挂载 `./models`，不入镜像）。

### HTTP 端点
`GET /health`、`GET /metrics`、`POST /ingest`、`POST /delete`、`POST /search`、`POST /query`、`POST /chat`（文档 `/docs`）。
若设了 `API_KEY`，除 `/health`/`/metrics`/`/docs` 外均需请求头 `X-API-Key`（恒定时间比较）。

## 测试与评估纵深（M7）

- **集成测试进 CI**：`integration-tests` job 用 `pgvector/pgvector:pg16` 服务容器，跑 `tests/integration/test_retrieval_e2e.py`：真库入库→dense/hybrid 召回正确源→**删除后两路都不再召回**。unit job 改为 `-m "not integration"`（保持秒级）。CI 现共 4 job。
- **注入对抗测试**：`tests/test_injection_adversarial.py` 用一批真实 payload 验证“防护真拦得住”（含发现并补全了 `new instructions:` 漏网模式）。
- **bootstrap 置信区间**：`app/evaluation/stats.py` + `eval_benchmark` 输出“均值±95%CI”，不再报单点。
- **评测上量（M13）**：查询样本从 80→**600**（同 4000 段语料，doc级 K=10，`--golden-only` 复用本地缓存不重嵌）：dense recall@10 **0.903±0.018**、hybrid **0.888±0.018**——CI 由旧小样本的 ±0.099 收窄到 ±0.018。**诚实结论：大样本下 dense 与 hybrid 统计上打平**（差 0.015＜半CI），M5“hybrid 更优”是小样本假象；真正拉分的是 rerank（M5 hybrid+rerank c20 mrr 0.988，未在 600 上重跑，如实标注）。（更大语料维度如 T2Ranking 仍为延后项。）
- **RGB 子集**：`eval_rgb.py` 扩展 counterfactual/integration。**诚实发现**：反事实集上“只信上下文”的严格 grounding 会被**错误文档 100% 带偏**（复述谬误）——纯 RAG 的固有软肋，也是为何需要自检式/多源佐证检索（③）。

## 受控联网与 MCP（P1）

- **联网两条受控路径**：`SEARCH_PROVIDER=searxng`（自架、零 token、主用）；`MiMoWebProvider`（MiMo 联网插件，**按次计费、由 `SEARCH_FALLBACK` 在主源空/错时降级触发**）。**硬约束：主生成 LLM（DeepSeek/MiMo）绝不挂 `web_search`**——按量模型联网会几乎必然 cache-miss 导致成本暴涨；联网只走上述两条检索调用。`WEB_SEARCH` 指标按 provider 计量。
- **共享工具注册表** `app/tools.py`：`kb_search/kb_answer/web_search/research/memory_search` 包成 JSON 安好的工具，MCP 与未来 agent 循环**复用同一套实现**。
- **显式 agent 循环** `app/agent/react.py`（`/agent`）：有界 ReAct，**主 LLM 每步只出一个动作 JSON、自身绝不挂 web_search**；max_steps 预算、非法/未知动作计数、HITL 确认门。**轨迹评测** `app/evaluation/trajectory.py`：收敛率/非法动作率/平均步数。
- **MCP server** `app/mcp_server.py`（`python -m app.mcp_server`）：把上述工具暴露为 MCP，Claude/Cursor 可接入。**依赖与 fastapi 的 starlette 版本互斥 → 用 `requirements-mcp.txt` 单独 venv 跑**。
- **wiki 沉淀** `/research/{sid}/publish`：研究文档论断→ knowledge 记忆（**trust=draft**，受 P0 同一门控，不直接污染作答），把"研究完"变成团队可复用资产。
- **agent 接入 `/query`**：`{"query":..., "agent":true}` 走有界工具循环产出带依据答案（附轨迹指标）；`verify_mimo_provider.py` 记录 MiMo 真实返回结构。

## 写回记忆（M9b）

`app/memory/`，`MEMORY_ENABLED=true` 且 `/query /chat` 传 `user_id` 时启用（默认关）：
- **双库分离**：`memories` 表按 `scope` 分 personal(用户私有, 注入作答) 与 knowledge(共享) —— 避免“把偏好当共享知识”/“把未证网络结论当常识”。
- **双时态**：`valid_at/invalid_at`（联网知识会过期），检索自动排除已失效行。
- **trust 生命周期**：个人偏好亲述即 verified；知识结论先 **draft**（默认不作答，防自毒），多次佐证 support≥阈值促升 verified；`/memory/promote`、`/memory/invalidate` 手动升降。
- **异步写回**：`BackgroundTasks` 跑 `run_writeback`（抽取→去重决策 ADD/UPDATE/NOOP→写入），不阻塞回答（sleep-time）。
- 135 单测(含生命周期/解析/写回编排) + 6 集成(真 pgvector 验证 trust 门控与双时态)。

## 写回记忆抗污染——端到端 A/B 证据（P0）

`scripts/eval_writeback_contamination.py`（隔离库 bagent_exp，rerank 关，真跑）。**两类场景×四臂**，四类判定（correct/wrong污染/conflict暴露/abstain），污染率带 bootstrap 95%CI。
**kb_silent（库里无该事实、写回错误记忆是唯一来源）——写回真正的风险：**

| 臂 | 污染率 | 解读 |
|---|---|---|
| A0 记忆关 | 0% | 地板 |
| A1 naive（draft 参与作答）| **68.3%±11.7** | 把错误记忆当唯一来源→答错 |
| **A2 门控（draft 不作答）**| **0%** | 100% 安全 abstain |
| A2b 合谋（2 源促升为 verified）| **73.3%** | 多源佐证把错误抬进 verified→**绕过门控（已知边界）** |

conflict 场景（KB 有真相）各臂污染均 0%（naive 只标分歧）。**结论（n=60）：trust 门控把写回污染从 68.3% 压到 0%（转为安全拒答）；但“多源一致”≠真相，合谋促升可绕过门控（甚升至 73.3%）——已如实标为边界（需来源真实性/人工促升）。** 同时本实验暴露并修复了一个真缺口：knowledge 记忆此前根本没接回作答路径，现已按 trust 接通。
（`scripts/eval_writeback_contamination.py`，隔离库 bagent_exp、n=60、四类判定、污染率带 bootstrap 95%CI。）

## 多租户数据面强制（⑥/M10）

按既定拆分：**管理面(Java) 不属本仓**；本仓做 **数据面强制**（不可外包给网关）：
- `app/tenant.py`：Java 签发 `tenant|<hmac>`，Bagent 用 `TENANT_SECRET` 验签；**绝不信任裸传 header**；未启用时不过滤（兼容单租户），启用时缺/错令牌直接 **403 fail-closed**。
- **三路一致**：`index_document` 将 tenant 写入 document/chunk 的 metadata；`vector_search`、`BM25`、检索缓存 key 都按 tenant 过滤（一处漏了就越权）。
- **记忆隔离**：`memories.tenant_id` 列；写入/去重/检索/列举均限本租户；跨租户删除被 `delete_document` 拦下。
- 146 单测 + 9 集成（含跨租户 dense/hybrid 不召回对方文档、跨租户删除拒绝、fail-closed 403）。

## 研究型 Agent（M9c）

`app/research/`：`POST /research {topic, use_kb, use_web}` → 产出带引用/时效/冲突标注的 `ResearchDoc`（markdown + 结构化），返回 `session_id`：
- **多源取证**（`evidence.py`）：每节 KB⊕联网经 self-RAG 充分性反思（带上限），去重合并。
- **编排**（`agent.py`）：`plan_outline`(LLM列提纲) → 逐节取证成文 → 全局参考文献编号重映射 → 论断表 + 冲突(复用 M8)。有界 max_sections/max_iters。
- **会话态**（`session.py`）：进程内存活文档，`/research/{id}/refine` 逐轮重做某节并回填引用。
- 141 单测（含提纲解析/去重/装配/会话 refine，全 fake 不联网）。`use_web` 默认取 `WEB_SEARCH_ENABLED`。

## 联网检索层（M9a）

`app/search/`：**可插拔 `SearchProvider`**（`searxng` / `mock`，将来 `baidu_api`）+ **安全清洗**（SSRF 白名单：拒内网/回环/云元数据/非http；注入中和；长度限制）+ **trafilatura 抽正文/时间**。`web_search(q)` 把结果归一为 `RetrievedChunk(source_type=web, trust=draft)`，供 self-RAG 多源佐证。
默认关（`WEB_SEARCH_ENABLED=false`）；自架后端见 `deploy/searxng/`（docker-compose + settings + 运行手册）。119 单测覆盖 provider 解析/编排/SSRF，全 hermetic（mock/假 client，不联网）。

## Self-RAG 与知识冲突（M8）

config 默认关（不静默改行为/加成本），`/query` 传 `self_rag:true` 或设 `SELF_RAG_ENABLED=true` 启用：
- **反思重检**（`app/retrieval/selfrag.py`）：检索→LLM 判证据充分性→不足则改写 query 重检，**最多 N 轮**（防死循环）。
- **知识冲突**（`app/generation/conflict.py`）：多条上下文就同一对象同一属性给出不同值→**不偷选一个**，标 `conflict`+降置信+前缀提示。
- 端到端验证（`scripts/demo_selfrag.py`）：两份对 XG-777 端口矛盾(8443 vs 9000)→系统检出冲突并正确区分“[3][4][5] 是 XG-200 不同对象不算矛盾”。
- **诚实边界**：纯 KB 下 self-RAG 能治“证据不足硬答”与“KB 内部矛盾”，但“整份权威上下文都错”的 counterfactual 需接入第二个可信源（联网）才能彻底解——这套中枢就是为它预留。

## 多 provider 与生产韧性（M6）

- **多 provider + 降级**：`LLM_PRIMARY_PROVIDER=deepseek`、`LLM_FALLBACK_PROVIDER=mimo`；主 provider 遇 429/超时/连接/5xx **指数退避重试**，仍失败则**降级到备用**；全挂则抛 `LLMUnavailable` → `/query` **优雅降级**（不裸 500，仍返回已检索来源）。
- **检索缓存**：`retrieve()` 按 `(query,mode,rerank,candidate_n,top_k,parent)` 的 LRU+TTL 缓存，命中跳过 embed+rerank；写入/删除时主动失效（多进程需换 Redis，接口不变）。
- **自适应早退**：`RERANK_SKIP_THRESHOLD` 配置后，dense 首分超阈时跳过 rerank 降延迟（默认关）。
- **prompt 注入防护**：对用户输入与检索文档都做常见注入模式的检测/中和（纵深防御一层，非银弹）。
- **压测（DeepSeek 主）**：`/query` 8 并发 23 次、**0 失败、avg 1.3s、p99 2.3s**——瓶颈已回到本地 rerank（provider 不再是短板）。
- 指标：`bagent_llm_requests_total/retries/fallbacks`、`bagent_retrieval_cache_total{hit|miss}`、`bagent_rate_limit_rejected_total`。

## 标准基准评估（M5）

不靠自造小语料自证——接入 **C-MTEB/DuRetrieval**（中文通用网页段落检索，带真 qrels），
在**独立 bench 库**上评估，**文档级相关性**（与切块尺寸解耦），固定 seed + 本地缓存可复现。

`scripts/prepare_dataset.py` 采样入库（`--passages`/`--queries` 可调，默认 1200/80）→ `scripts/eval_benchmark.py` 出数（doc-level K=10）。

**小池（1200 段/40 查询）**——recall 偏高（池小→干扰少）：

| 配置 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| dense | 0.963 | 0.946 | 0.941 |
| hybrid | 0.976 | 0.942 | 0.933 |
| hybrid+rerank c20 | 0.988 | **0.988** | **0.986** |

**放大池（4000 段/11569 子块/80 查询）**——recall 回落到真实水平，且 rerank 增益更健康：

| 配置 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| dense | 0.893 | 0.927 | 0.905 |
| hybrid | 0.902 | 0.936 | 0.899 |
| hybrid+rerank c20 | **0.915** | **0.968** | **0.929** |

诚实发现：小池→大池，dense recall@10 从 0.963 降到 0.893，印证“池小会虚高 recall”；大池上 rerank 仍是主要驱动（recall +0.022、mrr +0.041、ndcg +0.024），hybrid 单独用在小数据上 MRR/nDCG 反而微降——“混合需与 rerank 配合”而非无条件变好。

`scripts/chunk_sweep.py` 输出 #1 的 chunk 尺寸消融（真 gold、文档级）。

**#1 chunk 尺寸扫参**（doc-level K=10, 40 查询，同口径对比）：

| child_tokens | 配置 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|
| 150 | dense | 0.963 | 0.946 | 0.941 |
| 150 | hybrid | 0.976 | 0.942 | 0.933 |
| **300** | dense | **0.997** | 0.963 | **0.967** |
| **300** | hybrid | 0.987 | **0.975** | 0.962 |

结论：**更大子块(150→300)在本语料上提升 recall/nDCG**（切太小会把 gold 段落切断→召回受损）；
但它是多目标（与 rerank 交互，之前 child=150+rerank c20 排序最优），非“越大越好”。现取 2 个尺寸点，更多点可重跑 `chunk_sweep.py`。

### 顺带修掉的 #3 正确性 bug
旧版 BM25 词法索引**不过滤 `is_deleted`**、软删不触发重建——会召回已删文档。已修：载入/版本检测均
join documents 过滤 `is_deleted`；新增按 `source` 的 upsert（内容变更则替换旧块）与 `delete_document`。
`scripts/verify_incremental.py` 实测：删除后 dense 与 lexical 两路均不再召回。

### 生成侧基准（RGB 中文，`scripts/eval_rgb.py`，n=12）
用 RGB 自带文档（positive=金标准 / negative=噪声）直接测生成侧证据链：

| 能力 | 指标 | 值 |
|---|---|---|
| 噪声鲁棒 | 答案命中率(cited) | 0.917 |
| 噪声鲁棒 | 平均 faithfulness(cited) | 0.921 |
| 负例拒答 | 强制引用 SYSTEM_CITED | **0.833** |
| 负例拒答 | 基础提示 SYSTEM_BASE | 0.750 |

结论：M3 的“收紧引用+拒答”提示在**公认基准 RGB** 上把负例拒答率从 0.750 提到 **0.833（+8.3pp）**，
噪声下仍有 0.92 faithfulness。诚实标注：拒答 83% 非 100%（仍有误答，阈值可再调），n=12 偏小。

## 性能压测（Locust，单 worker / CPU / 无 GPU）

`locust -f locustfile.py --headless -u 10 -r 5 -t 40s --host http://127.0.0.1:8000`（`/search` 走完整 retrieve 链，`/query` 默认关）：

| /search 配置 | Avg | p99 | 吞吐 | 失败 |
|---|---|---|---|---|
| hybrid+rerank candidate_n=20 | 26.1s | ~30s | 0.29 req/s | 0 |
| hybrid+rerank candidate_n=5 | 4.2s | 5.8s | **1.58 req/s** | 0 |
| /health | 16ms | — | — | 0 |

**瓶颈定位（靠 /metrics 阶段计时）**：retrieval 阶段 25.6s 里 **rerank 占 25.5s**，embedding+pgvector 仅 ~0.1s
——CPU 上的 Cross-Encoder rerank 是吞吐天花板，比其余环节慢 ~180×。`candidate_n` 同时是**质量旋钮与延迟旋钮**（降到5 → 吞吐×5、p99 30s→5.8s）。
**生产优化方向**：rerank 移到独立批量推理服务(TEI/Triton/GPU)、多 worker 水平扩（需处理 BM25 内存副本与 Prometheus 多进程）、或自适应 candidate_n。（本环境无 GPU 是根本约束。）
