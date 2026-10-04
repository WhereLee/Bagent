# SLO / 错误预算 / 成本降级（M13）

## 1. SLO 定义
| SLI | 目标(SLO) | 度量 | 依据指标 |
|---|---|---|---|
| 可用性 | 错误率 < 1% / 30天 | 5m 5xx 占比 | `bagent_http_requests_total{status=~"5.."}` |
| 延迟 | p95 < 3s | 5m p95 | `bagent_http_request_seconds`（直方图）|
| 检索缓存命中 | > 30%（热点查询） | 15m hit 率 | `bagent_retrieval_cache_total{result}` |
| 主 provider 韧性 | 降级率 < 5% | 5m fallback 率 | `bagent_llm_fallbacks_total` |

**误差预算(error budget)**：30 天窗口下 1% 即预算上限；告警用**消耗速率(burn rate)** 触发（见 `deploy/prometheus/slo.rules.yml`），错误率>1% 持续 5m 触发 page。

## 2. 阈值为何这么定（不空谈）
- p95 用 3s 而非行业"200ms"：**诚实反映本栈瓶颈是 CPU rerank**（M6 实测 /search 单 worker ~0.29 req/s、p99 数十秒量级在 candidate_n=20 时）。阈值随 candidate_n/后端(onnx int8)/是否外置推理服务而变，**上线后按实测校准**，不写一个跑不到的漂亮数。
- 延迟预算与成本预算联动：p95 逼近 SLO 时，可临时调低 `retrieval_candidate_n` 或启用成本降级。

## 3. 成本预算与自动降级（`request_token_budget`）
预估一次作答 token 开销（系统提示 + `top_k`×每块 + 生成），**超预算即降级**（`app/generation/budget.py`，纯逻辑可单测）：
- 缩小 `top_k → degraded_top_k`；
- 跳过 rerank（省 CPU，`degrade_skip_rerank`）；
- 关一次忠实度 LLM 调用（`degrade_skip_faithfulness`）；
- 计数 `bagent_cost_degraded_total`，`Answer.degraded=true` 让调用方可见。
默认 `request_token_budget=0`（不降级），需要控成本时才设。这是"按量 LLM 下主动控成本"的机制，不是事后看板。

## 4. 共享状态外部化（多 worker / 多实例，`redis_url`）
接口不变、后端可切（置空=进程内存）：
- **检索缓存** → Redis：跨进程共享，避免每 worker 各存一份 + 缓存戳化；值 JSON 化（`list[dict]`）。
- **限流** → Redis 固定窗口：全局计数才在多实例下成立（**注明：与单实例内存令牌桶语义近似但不完全等价**，窗口边界会放过 1~2× 突发）。
- **研究会话活文档** → Redis：`ResearchDoc` to_dict/from_dict JSON 往返，跨 worker 可继续 refine。
> 诚实边界：Redis 化解决的是**状态一致性与 web 层水平扩展**，**不提升 CPU rerank 吞吐**（那要靠降 candidate_n / onnx int8 / 独立推理服务 / GPU）。

## 5. 加载方式
Prometheus `rule_files: [deploy/prometheus/slo.rules.yml]`；Grafana 以 `job:bagent_*` 记录规则建面板 + AlertManager 接 `severity=page|ticket`。
