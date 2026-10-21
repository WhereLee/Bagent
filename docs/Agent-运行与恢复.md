# Agent 运行与恢复

## 1. 状态与持久化

一次 Agent run 写入 `agent_runs`，每一步写入 `agent_steps`。状态包括：

`running / awaiting_approval / completed / rejected / failed / cancelled / budget_exceeded`

运行期间使用 `lease_owner + lease_until` 防止多个实例同时执行同一个 run。

## 2. 正常执行

```text
start_run
  -> acquire_lease
  -> _drive
       -> renew_lease（每次循环）
       -> LLM -> tool -> step 落库
       -> final / approval / budget / error
  -> release_lease
```

审批恢复时，待执行动作先写入 step，再清除 `pending_action`；崩溃恢复 worker 也能处理 `running + pending_action` 的 run。

## 3. 崩溃恢复 worker

```bash
python scripts/agent_worker.py --once
python scripts/agent_worker.py --poll 5
```

worker 扫描 `running` 且 lease 过期的 run，重新抢租约后用已持久化 step 继续。

指标：

- `bagent_agent_recovery_total{status=success|failed|skipped}`
- `bagent_agent_worker_last_scan_timestamp_seconds`

Docker Compose 的 `agent-worker` 服务会写 `/tmp/bagent-worker-heartbeat`，healthcheck 只检查该文件是否存在；生产可替换为更严格的新鲜度探针。

## 4. 副作用操作账本

`tool_operations` 为每个 side-effect 工具操作记录：

- 稳定 idempotency key：`run_id:step_index:tool:sha256(args)[:16]`
- state：`started / succeeded / failed / unknown`
- result、error、external_ref、attempts

行为：

- 已 `succeeded`：直接回放结果，不重复执行。
- `started/unknown`：返回 `unknown_outcome`，不自动重试。
- 账本不可用：不执行副作用工具。

查询：

```http
GET /agent/runs/{run_id}/operations
```

## 5. 工具完整结果

`agent_steps.observation` 只保留 1500 字预览；完整结果在 `observation_full`。默认列表/SSE 不返回完整结果。

```http
GET /agent/runs/{run_id}/steps/{step_index}
```

## 6. 上下文压缩

`compact_history` 保留 goal + 最近步骤，较早步骤折叠为动作摘要；完整轨迹仍可回查。

配置：

```dotenv
AGENT_CONTEXT_MAX_CHARS=12000
AGENT_KEEP_RECENT_STEPS=4
AGENT_COMPACT_OBSERVATION_CHARS=240
```
