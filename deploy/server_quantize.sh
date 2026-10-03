#!/usr/bin/env bash
# 服务器端：建运行时 venv(torch-cpu+transformers+onnx+onnxruntime) → 就地导出 int8 onnx → 删 fp32 源。
set -uo pipefail
RT=/opt/bagent/.rtenv
PY="$RT/bin/python"
PIP="$PY -m pip"
AL=https://mirrors.aliyun.com/pypi/simple/

[ -x "$PY" ] || python3 -m venv "$RT" || { echo "venv 失败(需 python3-venv)"; exit 1; }
$PIP install -q -U pip -i $AL

if ! $PY -c "import torch" 2>/dev/null; then
  echo "安装 torch (cpu) ..."
  $PIP install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || $PIP install -q -f https://mirrors.aliyun.com/pytorch-wheels/cpu/ torch
fi
$PIP install -q transformers onnx onnxruntime -i $AL
$PY -c "import torch,transformers,onnx,onnxruntime; print('DEPS_OK torch',torch.__version__)" || { echo "依赖不全"; exit 2; }

M=/opt/bagent/Bagent/models/bge-reranker-base
O=/opt/bagent/Bagent/models/reranker-onnx
$PY /opt/bagent/Bagent/scripts/export_onnx_standalone.py "$M" "$O"

if [ -f "$O/model_int8.onnx" ]; then
  echo "int8 已生成，删除 fp32 产物与源模型目录"
  rm -f "$O/model_fp32.onnx"
  rm -rf "$M"
else
  echo "int8 未生成，保留源，不删"; exit 3
fi

echo "=== models 现状 ==="; du -sh /opt/bagent/Bagent/models/* 2>/dev/null
df -h / | tail -1
