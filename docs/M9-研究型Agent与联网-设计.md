# M9 设计：研究型 Agent + 联网（SearXNG/百度）+ 写回记忆

> 本文是 2026-10-03 讨论的决策记录，作为实现依据。多租户/权限(②)明确**放最后**。

## 1. 产品定位（不是一次性问答，是"研究型 Agent"）
用户给一个主题/问题，系统产出一份**带引用、带时效、标注冲突的文档**；过程可讨论迭代；
并且把**这一轮的讨论/联网成果作为分可信度的知识写回**，越用越懂。
参照的成熟开源（读循环/文档合成）：LangChain `open_deep_research`、`GPT-Researcher`、Stanford `STORM`、`dzhng/deep-research`。
它把 M8 的 self-RAG 当作"每节研究够不够再查一轮"的内层原语——**是在读循环上长出编排层+文档对象+写回，不是重做**。

## 2. 已定的设计决策
- **交付物形态**：STORM 式"提纲→逐节研究成文" + GPT-Researcher 式"行内引用+来源表"。
  文档 schema：`标题/摘要/大纲树/各节正文(行内[n])/论断表(claim·confidence·source_url·crawled_at)/冲突与待核实/参考文献`。
- **记忆 = 两库分离**：`个人记忆`(用户级私有) 与 `知识库`(共享、分级)。混用=泄露/自毒。
- **记忆实现（不上图数据库）**：pgvector 上做 **Mem0 式抽取(ADD/UPDATE/DELETE)** + **Graphiti-lite 双时态字段**(`valid_at/invalid_at/created_at`) + **Letta 式 sleep-time 异步写回** + **Text2Mem 式 promote/demote/expire 可信度生命周期**。
- **可信度/反污染**：联网/讨论得到的默认落 **draft**，**未经多源佐证或人工核实不参与下次作答**。
- **并发语义**：检索+生成走关键路径；写回是**后台 fire-and-forget**；本轮结果直接进本轮上下文(读己之写)，KB 异步生效(最终一致)；写回按内容 hash 幂等 + 触发增量索引(M5)。
- **联网供给**：抽象 `SearchProvider` 可插拔；**先自架 SearXNG(只开百度系)**，将来无痛切托管百度 API。

## 3. 联网 / SearXNG 决策与知识（讨论沉淀）
### 3.1 两条路
- A 自架 **SearXNG**（元搜索，代理百度/360/搜狗，返回 JSON）：零 key、零按次费、可本地测、最自证；成本=运维+扛上游限流。
- B **百度托管 API**（千帆 AI 搜索 / 博查等）：省事、有 SLA；按次付费、要 key/可能备案。
选 **A 起步**（契合"百度就行 + 第一次做联网"），`SearchProvider` 抽象保证可切 B。

### 3.2 面向 AI 的搜索 API 两大类（选型时用）
- **SERP API**(SerpAPI/Serper)：原始 SERP，便宜快，要自己清洗；⚠️Google 2025-12 起诉 SerpAPI、微软 2025 关停 Bing API，法律/连续性风险。
- **AI/Web-Search API**(Tavily/Exa/Brave)：LLM-ready(已清洗/摘要)，贵一档；Brave 2026-02 取消免费档、Bing 索引关停后 Brave 是唯一大独立索引。

### 3.3 自架 SearXNG 的坑（实现/运维要点）
1. 官方镜像**默认不含 baidu 引擎文件**→ `[Errno 2] baidu.py not found`；需补/挂载 engines。
2. **JSON API 默认关**→ `settings.yml` 里 `search.formats: [html, json]` 必须开，否则 "Invalid JSON response"。
3. 私有实例 `server.limiter: false`；需 redis/valkey。
4. **上游百度反爬/限流/封 IP**：必须控频(≥1s)，agent 多轮会放大调用→更易被封。零 key ≠ 零风险。
5. engines 关 google/bing/duckduckgo(CN 难达)，留 baidu/360/sogou（wikipedia 可选）。
6. **只绑内网/127.0.0.1**（它是能出网的 web 服务）。

### 3.4 数据流（SearXNG 是"给链接+摘要的源"，消化自己做）
```
self-RAG 判定需联网 → SearxNGProvider.search(q, engines=baidu, format=json)
  → top-k {title,url,snippet} → trafilatura 抓正文+抽 publish_time
  → 清洗: 去HTML/限长/注入过滤/SSRF白名单(禁内网&非http)
  → 归一 RetrievedChunk(source_type=web, url, crawled_at, trust=draft)
  → 多源佐证/冲突/引用(复用M8) → 异步写回 draft
```
原则：**只存"消化后的断言+URL"，不存网页生料**（版权+噪音+重复embed）。

## 4. 分阶段落地
- **M9a（本阶段）**：`SearchProvider` 抽象 + `SearxNGProvider` + trafilatura 全文/时效 + SSRF/注入清洗 + mock provider + hermitic 测试 + searxng docker-compose/runbook。跑通链路，实网验证留到能起 SearXNG 的机器上。
- **M9b**：记忆双库数据模型 + 异步写回管道 + trust 生命周期 + 检索按 trust 过滤。
- **M9c**：会话态 + 活文档对象 + plan→分节→合成的编排（用 M9a 的 web 源当证据之一）。
- **多租户(②)**：最后做（数据面强制在 M9b 的 user-scope 上顺势落地）。
