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
- **M5 ✅**：评估可信度与工程补齐 —— ✅接入标准基准 C-MTEB/DuRetrieval(真 qrels、文档级)、✅修 #3 增量正确性 bug、✅#2 candidate 扫参、✅#1 chunk 尺寸扫参、✅生成侧接 RGB（拒答/噪声鲁棒）。
- **M6 ✅**：生产化 —— 多 LLM provider(DeepSeek 主/MiMo 备)+超时/重试退避/降级、检索结果缓存+自适应早退、API-Key 鉴权、prompt 注入防护。
- **M7 ✅**：测试与评估纵深 —— CI 新增 **integration job**（真 pgvector 服务容器跑端到端检索/增量回归）、注入对抗测试集、bootstrap 置信区间、RGB 反事实/信息整合子集。

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
`GET /health`、`GET /metrics`、`POST /ingest`、`POST /delete`、`POST /search`、`POST /query`、`POST /chat`（文档 `/docs`）。
若设了 `API_KEY`，除 `/health`/`/metrics`/`/docs` 外均需请求头 `X-API-Key`（恒定时间比较）。

## 测试与评估纵深（M7）

- **集成测试进 CI**：`integration-tests` job 用 `pgvector/pgvector:pg16` 服务容器，跑 `tests/integration/test_retrieval_e2e.py`：真库入库→dense/hybrid 召回正确源→**删除后两路都不再召回**。unit job 改为 `-m "not integration"`（保持秒级）。CI 现共 4 job。
- **注入对抗测试**：`tests/test_injection_adversarial.py` 用一批真实 payload 验证“防护真拦得住”（含发现并补全了 `new instructions:` 漏网模式）。
- **bootstrap 置信区间**：`app/evaluation/stats.py` + `eval_benchmark` 输出“均值±95%CI”，不再报单点。实测 dense recall@10 0.859±0.099 vs hybrid 0.889±0.065。
- **RGB 子集**：`eval_rgb.py` 扩展 counterfactual/integration。**诚实发现**：反事实集上“只信上下文”的严格 grounding 会被**错误文档 100% 带偏**（复述谬误）——纯 RAG 的固有软肋，也是为何需要自检式/多源佐证检索（③）。

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
