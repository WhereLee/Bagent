# 设计说明与关键决策

本文记录 Bagent RAG 的架构决策、选型理由与被否决的备选，便于对外解释"为什么这么做"。

## 1. 定位

独立可部署的 RAG 服务，非某个大 Agent 的子模块。因此优先保证：接口自洽、能力自证
（自带评估/测试/部署），不依赖外部项目背书。

## 2. 技术栈选型（结论 + 理由 + 被否）

| 层 | 选择 | 理由 | 被否 & 原因 |
|---|---|---|---|
| 语言 | Python | embedding/rerank/评估生态一等公民 | Go/Java：与语言统一收益 < 生态收益 |
| LLM | MiMo v2.6-flash | OpenAI 兼容、经济、支持长上下文 | 本地部署大模型：无 GPU，不现实 |
| Embedding | bge-small-zh-v1.5 | 中文强、CPU 可跑、维度 512 | bge-m3：更强但更重，留作 M2 升级位 |
| 向量库 | pgvector | 向量+原文+元数据+事务一库搞定，增量最省心 | Milvus：当前数据量下属过度设计 |
| 近邻索引 | HNSW / cosine | 召回质量优于 IVFFlat，bge 向量已归一化 | IVFFlat：需先训练、召回不稳 |
| 服务 | FastAPI | 异步、OpenAPI 白送、易上 SSE | — |

## 3. 数据模型

- `documents`：一个源文件一行，`doc_hash` 唯一约束 → 支撑**去重与增量**（同内容重复入库直接跳过）。
- `chunks`：文本块 + `vector(512)` + `metadata`（溯源：页码/标题）+ `parent_id`（M2 父子块预留）。
- 维度与 embedding 模型绑定：**换模型必须重建索引**（记录为一条运维约束）。

## 4. 切块策略

M1：递归切分（段落→句子→定长回退）+ 重叠窗口。切块大小不拍脑袋定死，
M2/M3 会用 **golden set 上的 recall@k 消融实验**比较不同 chunk_size/overlap 的效果，用数据选参。

## 5. 检索与生成（演进路线）

- M1：纯稠密向量 TopK + 引用编号 + 无召回直接拒答（省一次 LLM 调用）。
- M2：加 **BM25 稀疏召回**与**稠密**做 **RRF 融合**，再用 **Cross-Encoder rerank** 精排（topN→topK）。
- M3：加**多轮 query 改写**、**faithfulness 校验**（答案与证据比对）。

## 6. 评估（差异化核心）

- 检索侧：recall@k、MRR、NDCG，基于人工标注 `data/golden/` 的 query→相关 chunk。
- 生成侧：faithfulness / answer relevance（Ragas）。
- 每次改动跑回归，形成"数据→检索→生成→评估→反馈"闭环。这是与玩具 RAG 的关键区别。

## 7. 已知取舍

- Embedding 用 small 模型：牺牲部分召回上限，换取 CPU 可用与迭代速度；后续可平滑换 m3 并重建。
- 表格/图片：M1 只做基础文本抽取，结构化解析（表格）在 M2 增强——**流程完整，深度渐进**。

## 8. M2：混合检索与重排

**词法腿（关键决策）**：面试默认的 BM25，在 Windows + PostgreSQL 18 上无法便捷安装 pg_search(Rust) 扩展。
选型对比：pg_search(真 BM25、DB 原生、可移植性差) / pg_trgm(非 BM25、中文噪声) /
**Python BM25 + jieba（采用）**。取舍在于：算法是真 BM25(k1/b/IDF)，但索引载体是进程内存；
一致性用 generation 标记 (count, max_id) 触发重建。**规模化迁移路径：语料变大后切 pg_search/ES，
对外接口 `search(query, top_k)` 不变。** 这是环境约束导致的技术选择，非"项目小用不上"式简化。

**父子块(small-to-big)**：父块（~600 token）只存文本、embedding=NULL；子块（~150 token、带 overlap）
存 embedding 供检索。检索命中子块，`expand_to_parents` 把喂给 LLM 的 context 扩为父块，
在"匹配精度"与"上下文完整"间兼得。vector_search 以 `embedding IS NOT NULL` 天然只召回子块。

**RRF**：用排名而非分数融合（规避稠密相似度与 BM25 量纲不可比），k=60。

**Rerank**：`BAAI/bge-reranker-base`(Cross-Encoder)，粗召回 topN → 逐对联合打分 → topK。

**消融发现（golden 14 条/3 篇可混淆手册）**：hybrid 显著提升 recall@3(+0.18)；rerank 提升 MRR/nDCG，
但会把边缘相关块排出小 k 截断，轻微拉低 recall@k——该取舍在 M3 通过候选池大小/阈值调优，不回避。

## 9. M3：生成质量与证据链

**Faithfulness（幻觉率）**：采用 LLM-as-judge 的逐句蕴含判定（答案拆句→每句判断是否被检索上下文支持→
支持句/总句）。**自实现而非引入 Ragas**：Ragas 依赖链重，与本地轻部署约束冲突；但方法与 Ragas faithfulness
同源。代价：judge 要调 LLM，故生成评测属集成测试（不进 CI）；而拆句/判定解析/引用校验等纯逻辑进 CI。

**两级拒答**：① 无召回→硬拒（不消耗 LLM）；② 有召回但 faithfulness < 阈值(0.6) → 标记 `low_confidence`，
不假装确定。评测中用 `over_refusal_rate` 监控"拒答收紧不能误拒可答题"。

**引用校验**：`citation.py` 纯逻辑校验 `[n]` 是否落在实际检索数内，剔除越界引用，避免凭空引用。

**多轮改写**：`rewriter.py` 用 LLM 结合历史把追问改写为自包含检索查询（指代消解），无历史则跳过。

**评估中的关键手法——负向对照**：小语料+强模型下 base/cited 都满忠实，看不出差异。为证明"评估器能抓到问题"，
加入 no-RAG 对照（去检索证据只凭模型知识回答）：faithfulness 塔到 0、幻觉率1.0、拒答准确降到0.444。
这直接验证了指标有效性与证据链的价值，而不是自证一个永远满分的空评估。
