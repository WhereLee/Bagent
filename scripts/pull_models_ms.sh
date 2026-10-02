#!/usr/bin/env bash
# 用 ModelScope（国内可达）拉 BGE 两模型到服务器本地目录。embed 前台、reranker 后台。
set -uo pipefail
VENV=/opt/bagent/.dlvenv
PY="$VENV/bin/python"
[ -x "$PY" ] || PY=python3

"$PY" -c "import modelscope" 2>/dev/null || { echo "[setup] pip install modelscope..."; "$PY" -m pip install -q modelscope >/dev/null 2>&1; }

BASE=/opt/bagent/Bagent/models
mkdir -p "$BASE"

echo "[embed] modelscope 下载 bge-small-zh-v1.5 ..."
"$PY" -c "from modelscope import snapshot_download as s; print(s('BAAI/bge-small-zh-v1.5', local_dir='$BASE/bge-small-zh-v1.5'))" 2>&1 | tail -3 || echo "embed 失败"

echo "[reranker] 后台 modelscope 下载 bge-reranker-base ..."
nohup "$PY" -c "from modelscope import snapshot_download as s; print(s('BAAI/bge-reranker-base', local_dir='$BASE/bge-reranker-base'))" >/opt/bagent/pull_ms_reranker.log 2>&1 &
echo "  PID $! 日志 /opt/bagent/pull_ms_reranker.log"

echo "=== 现状 ==="
du -sh "$BASE"/* 2>/dev/null || true
