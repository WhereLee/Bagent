# 调研：AI 应用 / Agent 开发岗面试与作品集（2026）

> 用途：简历项目准备与面试方向校准的参考底稿。结论基于公开真实面经/资料，来源见文末；数字为二手统计，看趋势。

## 0. 一句话结论
概念题背后一定是"你们具体怎么做的、效果怎么量化"——**项目深挖占一半时间，权重高于任何八股**；差异化不靠"我做了 RAG+Agent"（及格线），靠**评测数字 + 决策链 + 失败处理 + 成本取舍**。

## 1. 面试官问什么（按公司风格，来自约 2828 条真实记录）
| 公司 | 侧重 | 代表真题 | 拉开差距点 |
|---|---|---|---|
| 字节 | RAG 全链路逐段深挖 | 索引→分块→召回→生成；**项目如何评测、有哪些指标** | 评测维度（召回质量、答案忠实度）|
| 腾讯 | 工程场景给方案 | 长文超上下文怎么办；检索无结果怎么办；Agent vs ChatBot vs RAG 区别 | 兜底/降级设计 |
| 快手 | 设计权衡 | **workflow vs 自由规划(agent)** 怎么取舍 | 边界意识 |
| 京东 | 多 Agent 架构 + 讲自己项目 | 描述你项目的 Multi-Agent 设计 | 能画架构图、说清取舍 |
| 百度 | 细节抠 | 分块造成的语义割裂怎么解 | 真调过才答得出 |
| Shopee | 概念辨析 | Agent vs workflow；Agent vs ChatBot | 三句话说清定义 |
- **通考**：**幻觉**（要分工程手段：RAG/校验/兜底 vs 算法手段：对齐/微调/置信过滤）、Function Calling/工具调用、评测、部署与成本。
- **2026 新增高频**（一年前几乎没有）：**上下文工程(context engineering)**、**Skill/Harness**、**HITL 人工确认与权限管控**。

## 2. 候选人在做什么项目（同质化警报）
来自同一批课/coze/90 天教程，高度雷同：企业知识库 RAG 问答、客服/多模态客服 Agent、简历优化 Agent、网页分析 Agent、会议纪要、deep-research 复刻。
→ **"我做了个 RAG + Agent" = 及格线，不是亮点。**

## 3. 面试官的评价标准（正/负信号）
**负信号（会被判"调包/背书"）**：不可追问的宏大陈述（"做了个 Agent""支持多租户"无数据）；只堆名词不讲"为什么用这个不用那个"；只有 API 调用没有优化。
**正信号（加分）**：
- **评测即故事**：量化数据（如 recall@3 0.68→0.86）、消融、失败案例（反事实暴露 100% 带偏）。
- **决策链**：问题→架构→评测数字→失败处理→成本取舍。
- **一个"从 X% 调到 Y%"的优化叙事**（比项目规模更说服力）。
- **边界意识**：何时用 workflow、何时才用 agent、为什么不无脑多 agent。
- **最小可行证据**：哪怕 20–50 题评测集 + 打分脚本，也胜过无数据的大系统。
- **MCP 封装**：把核心能力（检索/记忆/研究）封成 MCP server，体现工具化/可组合。
- **轨迹评测**：不止看答案，看 agent 的推理路径。
- **能讲清"线上事故怎么定位"**（可观测/成本是 AI 后端加分项）。

## 4. 招聘市场分层（看趋势）
AI 招聘呈"顶端紧缺、中端饱和、底端收缩"。应用/智能体层（RAG/Agent/AI 应用开发）要求熟练 LangChain/LangGraph/向量库，**更吃工程能力而非算法**；纯调 API/coze 拖 demo 不足以拿 offer。

## 5. 对本项目（Bagent）的对位判断
**已超过同类中位数**的地方：真评测（DuRetrieval/RGB + recall/MRR/nDCG + bootstrap CI + 消融）、自曝缺陷（counterfactual 100% 带偏）、混合检索/RRF/rerank、可观测+CI+量化 A/B、韧性降级、安全（SSRF/注入/fail-closed）。
**需补/要显性化**：agent 编排的显式循环 + 轨迹评测、写回记忆端到端有效性证据、MCP 封装、workflow-vs-agent 边界的清楚叙事、成本/延迟量化与调优故事。

## 6. 名词三层（避免把不同层的东西并列）
① 能力构件：RAG、function calling/tool use、structured output、agent memory、grounding/citation、guardrails、context engineering、embedding。
② 编排范式：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer、reflection(self-RAG/ReAct/Reflexion)、plan-and-execute、agent(自主循环)、multi-agent、HITL。
③ 协议/框架/基建：MCP、A2A、LangChain/LangGraph、LlamaIndex、向量库、eval/observability(LangSmith/Langfuse)。
合成词记法：**agentic RAG**、**context engineering**、**deep-research agent**、**workflow vs agent**(Anthropic 区分)。

## 来源（均为公开资料，数字看趋势）
牛客《2025-2026 AI 应用开发与 Agent 大厂面试高频问题》(约 2828 条真实记录)；面灵 AI《大模型与 AI Agent 面试题汇编 265 题》；卡码笔记/多份 CSDN 在职面试官经验帖；dev.to《AI Engineer Interview Playbook》；Ed Donner LLM Engineering 课程项目集；Xiaomi MiMo 联网服务插件定价页。
