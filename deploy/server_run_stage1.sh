#!/usr/bin/env bash
# ⑤阶段1：pgvector 容器 + 应用 venv(依赖+torch) + 重取 reranker tokenizer。全部幂等。
set -uo pipefail
AL=https://mirrors.aliyun.com/pypi/simple/

echo "[db] pgvector 容器"
if ! sudo docker ps --format '{{.Names}}' | grep -qx bagent-db; then
  sudo docker rm -f bagent-db 2>/dev/null || true
  sudo docker run -d --name bagent-db --restart unless-stopped \
    -e POSTGRES_USER=bagent -e POSTGRES_PASSWORD=bagentpw -e POSTGRES_DB=bagent_rag \
    -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16 && echo "  db up (127.0.0.1:5433)"
else echo "  db 已在"; fi

echo "[tok] 重取 reranker tokenizer (~20M)"
DL=/opt/bagent/.dlvenv/bin/python
$DL - <<'PY'
from modelscope import snapshot_download
snapshot_download('BAAI/bge-reranker-base',
                  local_dir='/opt/bagent/Bagent/models/bge-reranker-base',
                  allow_patterns=['config.json','tokenizer*','vocab*','special*','*.txt','sentence_bert_config.json'])
print("tokenizer done")
PY

echo "[app] 应用 venv + 依赖(含 torch)"
RT=/opt/bagent/.appenv
[ -x "$RT/bin/python" ] || python3 -m venv "$RT"
"$RT/bin/pip" install -q -U pip -i $AL
"$RT/bin/pip" install -q -i $AL torch
"$RT/bin/pip" install -q -i $AL -r /opt/bagent/Bagent/requirements.txt
"$RT/bin/python" -c "import torch, sentence_transformers, fastapi, pgvector; print('APPDEPS_OK torch', torch.__version__)"
echo "STAGE1_DONE"
