#!/usr/bin/env bash
# 单独进程对已导出的 fp32.onnx 做 int8 量化（不载 torch，峰值内存低）；成功则删 fp32 产物+源模型目录。
set -uo pipefail
PY=/opt/bagent/.rtenv/bin/python
O=/opt/bagent/Bagent/models/reranker-onnx
M=/opt/bagent/Bagent/models/bge-reranker-base

echo "=== 量化前内存 ==="; free -m | head -2
# 临时停 searxng/redis 腾内存
sudo docker stop searxng redis >/dev/null 2>&1 && echo "(临时停 searxng/redis)"

$PY - <<'PYEOF'
import os
from onnxruntime.quantization import quantize_dynamic, QuantType
o = "/opt/bagent/Bagent/models/reranker-onnx"
src, dst = os.path.join(o, "model_fp32.onnx"), os.path.join(o, "model_int8.onnx")
quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
print("INT8_MB", round(os.path.getsize(dst) / 1e6, 1))
PYEOF
RC=$?

sudo docker start searxng redis >/dev/null 2>&1 && echo "(已重启 searxng/redis)"

if [ -f "$O/model_int8.onnx" ]; then
  rm -f "$O/model_fp32.onnx"; rm -rf "$M"
  echo "已删 fp32.onnx + 源模型目录"
else
  echo "int8 未生成 (rc=$RC)，保留源，不删"; exit 3
fi
echo "=== models 现状 ==="; du -sh /opt/bagent/Bagent/models/* 2>/dev/null
df -h / | tail -1
