#!/usr/bin/env bash
# ⑤ 冒烟：ingest → query（本地知识库问答）→ research（联网多源研究）
set -uo pipefail
B=http://127.0.0.1:8000

echo "[ingest] gateway_manual.md（首次会加载 embedding 模型，稍慢）"
curl -s -X POST $B/ingest -H 'Content-Type: application/json' -d '{"path":"/opt/bagent/Bagent/data/corpus/gateway_manual.md"}' | head -c 400; echo

echo; echo "[query] 本地知识库：XG-200 默认管理端口"
curl -s -X POST $B/query -H 'Content-Type: application/json' -d '{"query":"XG-200 的默认管理端口是多少？"}' | head -c 500; echo

echo; echo "[research] 联网多源研究：秦惠文王"
curl -s -X POST $B/research -H 'Content-Type: application/json' -d '{"topic":"秦惠文王","use_kb":false,"use_web":true,"max_sections":3}' | head -c 900; echo
