# 自托管 SearXNG 运行手册（Bagent 联网检索后端）

零 API key 的元搜索：Bagent 通过 `http://127.0.0.1:8080/search?...&format=json` 拿百度系结果。
**成本不是零**：要自己养这个服务，并扛上游百度的反爬/限流。

## 启动
```bash
cd deploy/searxng
docker compose up -d --build      # 首次会生成 ./searxng 配置
# 用仓库里的 settings.yml 覆盖生成的默认（先改 secret_key），再重启：
docker compose restart searxng
```

## 必踩的三个坑
1. **JSON 默认关**：`settings.yml` 里 `search.formats` 必须含 `json`，否则 API 报 `Invalid JSON response`。（本目录已配好）
2. **官方镜像默认没有 baidu 引擎文件**：可能报 `[Errno 2] .../engines/baidu.py not found`。
   处理：启用 compose 里注释的 `./engines` 挂载，把 baidu/360search/sogou 引擎 .py 放入 `deploy/searxng/engines/`
   （从 SearXNG 上游 `searx/engines/` 取对应文件），重启。
3. **上游限流/封 IP**：百度会对云服务器 IP 的频繁请求弹验证码。已要求 **`SEARCH_MIN_INTERVAL≥1s` 节流**；
   别把并发/调用堆高。被封就换出口 IP 或改走托管百度 API。

## 验证
```bash
curl "http://127.0.0.1:8080/search?q=%E7%A7%A6%E6%83%A0%E6%96%87%E7%8E%8B&format=json&engines=baidu"
# 期望：{"results":[{"title":..,"url":..,"content":..}, ...]}
```

## 接进 Bagent
`.env` 设：
```
WEB_SEARCH_ENABLED=true
SEARCH_PROVIDER=searxng
SEARXNG_BASE_URL=http://127.0.0.1:8080
SEARXNG_ENGINES=baidu
```
联网结果以 `trust=draft` 进入 self-RAG 多源佐证（默认不作为可核实答案）。

## 安全
- compose 只绑 `127.0.0.1:8080`，**别公网暴露**（它能出网，是 SSRF 跳板风险）。
- Bagent 侧已对 web 结果做 SSRF 白名单 + 注入中和 + 长度限制（`app/search/clean.py`）。
- 只把"消化后的摘要/断言 + URL"入库，不存网页原文（版权/噪音）。

## 什么时候换托管百度 API
需要稳定 SLA、不想跟反爬搏斗、或量大了 → 实现一个 `baidu_api` 的 `SearchProvider`（同接口），
在 `app/search/factory.py` 注册即可切换，业务层不动。
