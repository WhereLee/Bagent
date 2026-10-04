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

## 10. M4：可观测、限流、部署

**可观测**：结构化 JSON 日志 + contextvar 存 `request_id` 贯穿全请求（响应头回写）；Prometheus 指标
（HTTP 数/时延、阶段耗时、LLM token、质量信号、限流拒绝）。用 `prometheus-client`（行业标准、依赖轻）。
边界：多进程 worker 需 `PROMETHEUS_MULTIPROC_DIR`+multiprocess 模式；当前单 worker 用默认 REGISTRY。

**限流**：令牌桶（rate+burst），按 IP。**内存存桶 + 多实例迁 Redis(Lua 原子扣减) 的路径写死**：
`RateLimiter.allow(key)` 接口不变、只换存储。与 BM25 载体同理：算法不变、载体可换。

**压测**：Locust（`locustfile.py`），`/search` 主力(纯检索、不花 LLM)，`/query` 低权重。属集成，不进 CI。

**Docker**：镜像只装代码+依赖；**模型权重不打进镜像**（1GB+且离线环境），compose 挂载 `./models` +
`HF_HUB_OFFLINE=1`。db 用 `pgvector/pgvector:pg16`，宿主机映射 5433 避开本机 5432。

**CI 三 job**：`unit-tests`（hermetic）、`secret-scan`（守 .env/密钥不入库）、`docker-build`（验证镜像可构建，仅 push 触发）。

## 11. M5：接入标准基准 + 补齐 #1–#4

**为何用标准基准而非自造/爬取**：自造 4 篇语料的数字只“看趋势”；爬豆瓣 UGC 有 ToS/版权/不可复现风险。
改用 **C-MTEB/DuRetrieval**（中文通用、Apache-2.0、自带 qrels），评估可信且可复现。

**文档级相关性（解耦切块）**：bench 中每个 passage 当一个文档(source=pid)；相关性 = 召回块的 source 是否 ∈ 该 query 的相关
 passage 集。这样 recall 不再受“gold 被切块边界切断”干扰，才能公平地做 #1 扫参。

**独立 bench 库**：基准评测写 `bagent_bench`，不污染 demo 库；`prepare_dataset.py` 首次拉取后本地缓存
原始数据（避免重复流扫、更稳）。

**#2 两段式与 candidate 旋钮**：recall 应在“检索层候选集”与“答案层 top_k”分开看；rerank 只重排不新增候选，
因此增大 candidate_n 提召回、但可能稀释 top_k 排序——c20 排序优、c50 召回优，是真实的 recall↔精度前沿。

**#3 正确性修复**（非新功能，是 bug）：旧 BM25 载入不过滤 `is_deleted`、软删不改 count/max_id 故不重建——已删文档仍被召回。
修法：载入与 generation 都 join documents 过滤软删、版本含 `max(updated_at)`；`index_document` 改为按 `source` upsert（内容变更替换旧块），
新增 `delete_document`。`verify_incremental.py` 实证两路都不再召回已删文档。

**性能取舍**：rerank 在 CPU 上对大候选池很贵（c100 全量扫参本机会超时）；已将精排限为“前 candidate_n 个”，
且扫参默认只跑 dense/hybrid（#1 关心检索层尺寸效应）。属真实硬件约束，非简化功能。

**生成侧基准 RGB**：RGB 自带 positive/negative 文档，评测不经我们的库检索，直接构造上下文测生成侧：
噪声鲁棒（cited 答案命中 0.917 / faithfulness 0.921）与负例拒答（强制引用 0.833 vs 基础提示 0.750）。
意义：把 M3 “收紧引用+拒答提示”的收益放到公认基准上量化（+8.3pp 拒答），而不是自造集自证。

## 12. 压测实测与 /search 口径修正

- 压测发现 **/search 端点当时仍走 M1 的纯 dense vector_search**，与 /query 的 hybrid+rerank 口径不一致 → 已改为一律走 retrieve()，保证压测/预览测的是真实检索链。
- Locust 实测(单worker/CPU/无GPU)：/search 带 rerank、10 并发下 avg ~26s、吞吐 0.3 req/s；/metrics 阶段计时归因 = **rerank 占 25.5/25.6s**，embedding+pgvector 仅 ~0.1s → **CPU 上 Cross-Encoder rerank 是吞吐天花板**。
- candidate_n 20→5：吞吐 0.3→1.58 req/s、p99 30s→5.8s —— candidate_n 是质量(#2)与延迟的双重旋钮。
- 缓解方向(未做，记录)：rerank 独立批量推理服务/GPU、多 worker(需解 BM25 内存副本 + Prometheus 多进程)。

## 13. M6：多 provider 韧性、缓存与安全

**多 provider 路由与降级**：DeepSeek 主、MiMo 备（配置切换）。只对可重试错误(429/超时/连接/5xx)做指数退避重试；4xx 直接换 provider；全挂抛 LLMUnavailable，由 /query 优雅降级为"服务暂不可用"而非裸 500。指标 llm_requests/retries/fallbacks。
理由：MiMo RPM=100 是压测暴露的现实约束；多 provider + 退避是生产必备的下游韧性，DeepSeek 并发更高。

**检索缓存**：retrieve() 结果 LRU+TTL，key 含全部影响结果的参数；写入/删除显式 clear。瓶颈定位显示 CPU rerank 是天花板 → 缓存直接省掉热点查询的 embed+rerank。多进程下各自缓存 + TTL 兜底，分布式一致需 Redis（接口不变）。

**自适应早退**：dense 首分超阈跳过 rerank（默认关，避免伤召回）。

**安全**：API-Key 中间件（恒定时间比较，/health /metrics 豁免）；prompt 注入对输入与检索文档做检测/中和（纵深防御一层，明确不是银弹）；租户过滤以参数化 SQL/后过滤实现（metadata.tenant，入库写入）。

## 14. M8：Self-RAG 与知识冲突

**动机**：M7 的 RGB 反事实测出"只信上下文"的严格 grounding 被错误文档 100% 带偏；且现链路是"检索一次→生成"，证据不足也硬答。

**形态**：在检索与生成间加一层"反思-自检-重检"中枢 + 冲突检测：
- 充分性自检(LLM)→不足改写重检，**max_iters 上限 + 改写无进展即停**（防 agent 死循环，是高频考点）。
- 冲突检测：LLM 判"同一对象同一属性是否给出不同值"，**区分不同对象不算矛盾**（demo 实证）；冲突时不偷选，标 conflict+降置信+前缀提示，交调用方决策。
- 纯解析(parse_sufficiency/parse_conflict_verdict)与 LLM 调用分离，进单测；LLM 判定属集成。config 默认关，不静默改行为/加成本。

**架构定位**：这套是"多源 agentic 检索"的中枢；联网搜索作为**第二个证据源**接进来即可复用同一充分性/冲突/佐证机制——这也是为什么 self-RAG 先于联网做是对的（先建中枢，后接源）。纯 KB 下它治不了"整份上下文都错"的 counterfactual，需多源交叉。

## 15. Reranker int8 量化（决策+实测）
为把 rerank 留在 4G 服务器本地跑、又不被"1.1G 模型传输/CPU 吞吐"卡：Reranker 做成可插拔 fp32/int8（PyTorch 动态量化 quantize_dynamic，只量 Linear，稳、无 ONNX 依赖链）。
本机实测：体积 1112→770MB(−31%)、打分 ×2.4 快。质量用 eval_rerank_ab.py 在 DuRetrieval A/B(cand=20,K=10,n=40)：recall@10 +0.000、MRR −0.012、nDCG −0.009，均在 95%CI 内 → int8 采纳。
教训：toy 样本的排名一致性 ρ=0.875 是 n=3 假警报；量化掉不掉要在真实基准上测 Δ±CI 判，别信经验数。默认 RERANKER_INT8=false，生产按需开；换领域复测。

## 16. Reranker ONNX int8（磁盘级量化）
澄清：前面 torch 动态 int8 是"加载时在内存量化"，磁盘仍是 1.1G fp32。要做"磁盘也小"需导出量化后的模型：
scripts/export_reranker_onnx.py 用 torch.onnx.export(fp32 .onnx) → onnxruntime.quantization 动态 int8 → model_int8.onnx（279MB）。
Reranker 增 OnnxReranker 后端（同 .rerank 接口，喂 input_ids/attention_mask），config reranker_backend=st|onnx 选择。
三方 A/B(eval_rerank_ab.py, DuRetrieval cand=20 K=10 n=40)：onnx-int8 recall@10 0.902/mrr 1.000/ndcg 0.938，
对比 st-fp32(0.905/1.000/0.943) Δ 均在 95%CI 内、且优于 torch st-int8(mrr0.988)。生产建议 RERANKER_BACKEND=onnx。
依赖：onnxruntime(运行时,可选) 入 requirements；onnx(导出) 入 requirements-dev。onnx 产物不入 git(models/ 忽略)。

## 17. 服务器量化落地约束（实测）
- fp32 ONNX 导出在 4G 服务器可行；但 onnxruntime int8 量化峰值内存 >3.1G，**在 4G 机上 OOM(rc=137) 不可行**。
- 正解：在内存充裕的机器(本机)导出+量化出 278MB model_int8.onnx → 传上服务器(md5 校验) → 就地删 1.1G fp32 源与中间 fp32.onnx。
- 服务器运行时(⑤)务必装 **CPU 版 torch**(--index-url .../whl/cpu)，别装成 CUDA 版(占 5.7G 无用)；用完清 pip 缓存。
- 教训：构建产物若在目标机做不动，就"在能做的机器产小成品传过去"，而非传大原料；且 scp 大文件不可中断取消(会得到大小对但内容坏的文件)，务必 md5 校验。

## 18. M9b 写回记忆
双库分离(memories 表 scope=personal/knowledge)防“偏好当共享知识/未证结论当常识”；双时态 valid_at/invalid_at 表联网知识会过期；trust 生命周期 draft→verified→curated：个人偏好亲述即 verified(用户是自身偏好权威且私有低风险)，知识结论先 draft 且默认不参与作答(trust 门控)，多次佐证 support≥阈值促升。写回走 FastAPI BackgroundTasks(sleep-time 异步，不阻塞回答)：抽取事实(LLM, 纯解析可测)→按相似度 decide_op ADD/UPDATE/NOOP→写入。/memory 列举/promote/invalidate。默认 MEMORY_ENABLED=false。
关键防自毒：未核实来源(web/research)默认 draft 且被 search_memories 的 min_trust 挡在作答之外，只有被多源佐证或人工 promote 才参与——这条生命周期就是为联网时代准备的。

## 19. M9c 研究型 Agent（编排 + 文档 + 会话态）
读循环之上长出编排层：POST /research → plan_outline(LLM) → 逐节 gather_evidence(KB⊕web, self-RAG 充分性反思, 去重) → 成文(行内[n]) → 全局参考文献编号重映射(local→global, 内容去重共享) → 论断表(按节引用+最低trust) → 冲突(复用 M8 detect_conflict)。有界 max_sections/max_iters，空证据节显式标注不编造。
会话态 session.py：进程内活文档 dict + 锁；/research/{id}/refine 重做某节回填引用。持久化 DB 属后续。
交付物 schema(doc.py)：ResearchDoc{topic/outline/sections/claims/references/conflicts} + to_markdown。联网结果作为 draft 源进入，由 M9b 的 trust 生命周期管住是否可作答复。

## 20. ⑥ 多租户数据面强制
按"管理面(Java)/数据面(Python)"拆分，本仓只做数据面强制（不可外包网关：dense/BM25/缓存需原子一致过滤，网关看不到块级归属）。
- 信任根 app/tenant.py：HMAC 签名租户令牌 	enant|sig，verify 失败/缺失 fail-closed(403)；不启用则不过滤（兼容单租户，默认）。端点经 X-Tenant header → resolve_tenant。
- 入库盖章：index_document 把 tenant 写进 document/chunk 的 metadata JSONB（不改表结构）。
- 三路一致：vector_search(d.metadata->>'tenant')、BM25(内存索引存 tenant 列表+search 过滤)、缓存 key 含 tenant。delete_document 带 tenant 只能删本租户。
- 记忆：memories.tenant_id 列(幂等 ALTER)；add/search/find_similar/list 全按 tenant 分域去重与过滤。
测试：146 单测(含 TestClient fail-closed 403) + 9 集成(跨租户 dense/hybrid 不召回对方、跨租户删除拒绝)。默认 TENANT_ENFORCEMENT_ENABLED=false 不改现有行为。

## 21. P0 写回记忆抗污染 A/B（含自曝边界）
先重审原计划、补回被简化的降级：三臂(A0记忆关/A1 naive/A2门控)+合谋泄漏臂(A2b)、四类判定(把"同时给对错值"识别为主动暴露冲突而非污染)、双场景(conflict / kb_silent)、n 提到~100程序化生成+每探针重复压 LLM 抖动、隔离库 bagent_exp。
关键发现1(设计缺陷自我纠正)：只测 conflict(KB 有真相)会得到"污染恒 0%"的**无区分度**结果——因为真相当场、错误值翻不了车；真正的风险在 **kb_silent**(库无该事实、写回记忆是唯一来源)，naive 68.3%±11.7 污染 → 门控 0%(安全拒答) → 合谋促升 73.3% 泄漏（n=60）。
关键发现2(真缺口)：knowledge 记忆此前只写不读，/query 仅召回 personal → "防污染门控"守的是空路径；已修：_memory_context 同时按 trust 召回 personal+knowledge。
边界(不粉饰)：多源佐证≠真相，两个都错且一致会被促升绕过门控(73.3%)→ 需来源真实性/多样性、促升人工确认。

## 22. P1 受控联网(MiMo) + 共享工具/MCP
成本护栏(用户明确)：主生成 LLM(DeepSeek/MiMo) 绝不挂 web_search——按量模型联网几乎必然 cache-miss→成本暴涨。联网只两条受控检索路径：SearxNG(自架/零token/主用) + MiMoWebProvider(MiMo联网插件, 按次计费, 由 SEARCH_FALLBACK 在主源空/错时降级; 独立检索调用非主模型自决)。MiMo 返回取 url_citation annotations + 正文链接兜底归一 SearchResult→既有 SSRF/清洗。WEB_SEARCH 指标按 provider 计量。
共享工具注册表 app/tools.py(kb_search/kb_answer/web_search/research/memory_search)：MCP server 与未来 agent 循环复用同一实现。app/mcp_server.py 用 FastMCP 绑定 TOOLS。
依赖教训(实测)：mcp 的 sse-starlette 要 starlette>=0.49，fastapi 锁 starlette<0.47 → 二者同 venv 冲突 → MCP server 必须独立 venv(requirements-mcp.txt)，web 应用 venv 不装 mcp。test_tools 用 importorskip 保证 app 环境不装 mcp 也绿。

## 23. P1-C 显式 agent 循环 + 轨迹评测
app/agent/react.py：有界 ReAct，动作空间=app.tools.TOOLS(与 MCP 同源)。主 LLM 每步只输出一个动作 JSON(final 或 tool+args)，parse_action 稳健解析失败计 invalid；**主 LLM 绝不挂 web_search**(成本护栏)，联网只经 web_search 工具走受控检索。max_steps 预算防死循环；未知工具/异常计入 invalid 并反馈重试；HITL: requires_confirm(默认 research)无 confirm 放行→status awaiting_confirmation(先做门标志，不接 UI)。
app/evaluation/trajectory.py：收敛率(status=final)/非法动作率/平均步数，复用 bootstrap_ci——把"agent 走得对不对"变数据而非只看最终答案(面经点名的加分项)。/agent 端点复用之。测试 166 单测(含 8 agent)。

## 24. P2 MiMo端到端验证 / wiki沉淀 / agent入query
- MiMo provider 真调验证(verify_mimo_provider.py)：返回 message.annotations 为**平铺 url_citation**（非嵌套），字段 url/title/**summary**/site_name；解析器原取 snippet/content→摘要全空，已修(优先 summary)并加回归 test_mimo_parse_annotations。
- wiki 沉淀 app/research/publish.py：claims_to_facts(纯映射) + publish→ add_memory(scope=knowledge,trust=draft)。/research/{sid}/publish 端点。走 P0 同一 trust 门控：draft 不直接进作答，需佐证/人工促升——"沉淀"与"防污染"不矛盾。
- agent 入 /query：query 加 agent=true → _query_agent 走 run_agent(工具=受控 kb_search/web_search)，返回 final+sources+轨迹指标。主 LLM 仍不隐式联网。
测试 168 单测。n=60 污染确认跑已完成：kb_silent naive 68.3%±11.7 → 门控 0% → 合谋 73.3%，已刷新 §21/README。

## 25. P3-A 经验分层（含自我证伪）
落地 memories.level(fact|playbook)+tags(JSONB) 基础设施、search_memories 按 level/标签(@>)过滤、find_similar 同域、抽取与 publish 按内容启发式打 level、_memory_context 分块呈现。
诚实评测(eval_experience_reuse.py，本地 bge、K=3、n=5)：跨场景可迁移 playbook 命中率 **扁平语义 1.00 vs 分层保底槽 0.60** —— 我的"给 playbook 保留召回槽"启发式**反而更差**（argmax 选错 playbook）。结论：不将该检索启发式并入产品；level/tags 只作**过滤/溯源/呈现**（真实有用、已过测试）。分层是否真提升可复用，需更对抗/真实的语料再判，此处不硬凑正向数。这是"评测证伪自己"的示范。

## 26. P3-B 复盘整体归档（Event）
events 表(name/type/tags/summary/created_by) + memories.event_id 关联。store.create_event / get_event / list_memories(event_id) / event_playbook(event_id)→{event, playbook[], facts[]}。publish 接受 event_id（可按 event_name 自动建）。API：POST /events、GET /events/{id}/playbook、/research/{sid}/publish 带事件归属。把"一次对话/活动"归成整份可检索可引用的经验档案，补上"复盘整体归档"这块。测试 12 集成含事件档案。

## 28. P3-C 结果回标（经验靠成败治理）
memories.use_success/use_fail 计数 + lifecycle.outcome_action(纯)：成功→促升一级(draft→verified→curated)；失败累计达阈值(默认2)→作废(invalid_at)。store.record_feedback(id,ok,tenant 校验) + API POST /memory/feedback。补上 P0 暴露的"多源一致≠真相"缺口——来源数之外再加"用过成没成"这个更硬的信号。测试：outcome_action 单测 + record_feedback 集成(促升/达阈值作废)。
注：集成测试数据须按 owner 后缀唯一化，否则 add_memory 内容哈希去重会撞上一轮 commit 残留(rollback 撤不掉已提交行)。
