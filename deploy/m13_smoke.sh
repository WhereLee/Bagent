#!/usr/bin/env bash
# M13 冒烟：线上 /query 返回含 degraded 字段（成本降级标记），且未破坏正常问答。
set -e
BASE=http://127.0.0.1:8000
curl -s -X POST "$BASE/query" -H 'Content-Type: application/json' \
  -d '{"query":"XG-200 的默认管理端口"}' -o /tmp/q13.json
echo "keys: $(grep -oE '"(text|sources|degraded|is_refusal|faithfulness)"' /tmp/q13.json | tr '\n' ' ')"
echo "degraded present: $(grep -c degraded /tmp/q13.json)"
