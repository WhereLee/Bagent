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
