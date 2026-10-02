#!/usr/bin/env bash
# 在服务器拉两个模型到 HF 缓存。embedding 前台、reranker(1.1G) 后台 nohup。
set -uo pipefail
cd /opt/bagent/Bagent || exit 1

# 选一个能 import huggingface_hub 的 python
pick_py() {
  for c in /opt/bagent/.dlvenv/bin/python python3; do
    if $c -c "import huggingface_hub" >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}

PY="$(pick_py)" || PY=""
if [ -z "$PY" ]; then
  echo "[setup] 装 huggingface_hub ..."
  if python3 -m venv /opt/bagent/.dlvenv >/dev/null 2>&1; then
    /opt/bagent/.dlvenv/bin/pip install -q -U pip >/dev/null 2>&1
    /opt/bagent/.dlvenv/bin/pip install -q "huggingface_hub" >/dev/null 2>&1
  fi
  PY="$(pick_py)"
  if [ -z "$PY" ]; then   # venv 不行 → 系统级
    python3 -m pip install --user --break-system-packages -q "huggingface_hub" >/dev/null 2>&1 \
      || python3 -m pip install --break-system-packages -q "huggingface_hub" >/dev/null 2>&1
    PY="$(pick_py)"
  fi
fi
if [ -z "$PY" ]; then echo "[FATAL] 无法 import huggingface_hub"; exit 2; fi
echo "[setup] 用 $PY"

echo "[embed] 前台拉 bge-small-zh-v1.5 (~100MB)..."
$PY scripts/pull_models.py embed || echo "  embed 失败"

echo "[reranker] 后台拉 bge-reranker-base (~1.1G)... 日志 /opt/bagent/pull_reranker.log"
nohup $PY scripts/pull_models.py reranker >/opt/bagent/pull_reranker.log 2>&1 &
echo "  PID $!"

echo "=== 现状 ==="
du -sh /opt/bagent/Bagent/models/hub/models--BAAI--* 2>/dev/null || true
