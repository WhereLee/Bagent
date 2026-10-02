# Bagent RAG

一个**生产级、可独立部署**的检索增强生成（RAG）系统。目标不是"跑通一个 demo"，
而是把 RAG 全链路的每一步（切块、向量化、混合检索、重排、生成、评估）都做成**可量化、
可解释、可迭代**的工程，能够经受"你凭什么说效果更好"的追问。

- LLM：Xiaomi MiMo（`mimo-v2.6-flash`，OpenAI 兼容端点，按量计费经济型）
- Embedding：`bge-small-zh-v1.5`（本地 CPU 推理，权重落在 `models/`，不写 C 盘）
- 向量库：PostgreSQL 18 + pgvector 0.8.2（HNSW / cosine）
- 服务：FastAPI

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

## 检索质量验证（消融，本地复现：`scripts/eval_retrieval.py`）

基于 `data/golden/golden.jsonl`（14 条覆盖 3 篇可混淆手册的标注）：

| 配置 | recall@3 | mrr | ndcg@3 | recall@5 | ndcg@5 |
|---|---|---|---|---|---|
| dense | 0.679 | 0.631 | 0.613 | 1.000 | 0.752 |
| hybrid | 0.857 | 0.631 | 0.692 | 1.000 | 0.748 |
| hybrid+rerank | 0.786 | 0.714 | 0.732 | 0.929 | 0.793 |

结论：**hybrid 显著提升 recall@3（+0.18）**，**rerank 显著提升排序质量（MRR/nDCG）**；
rerank 会轻微拉低 recall@k（重排把边缘相关块排出截断）——该取舍在 M3 调候选池/阈值。

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
`GET /health`、`GET /metrics`、`POST /ingest`、`POST /search`、`POST /query`、`POST /chat`（文档 `/docs`）。
