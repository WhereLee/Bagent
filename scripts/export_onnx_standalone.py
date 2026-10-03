"""独立 ONNX 导出：把本地模型目录导成 fp32 onnx 并 int8 量化。无 app 依赖（只需 torch/transformers/onnx/onnxruntime）。
用法: python export_onnx_standalone.py <model_dir> <out_dir>
"""
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
model.eval()

enc = tok(["q"], ["d"], padding=True, truncation=True, return_tensors="pt")
fp32 = os.path.join(out, "model_fp32.onnx")
print("export fp32 onnx ...", flush=True)
torch.onnx.export(
    model, (enc["input_ids"], enc["attention_mask"]), fp32,
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
    opset_version=17, dynamo=False,
)
int8 = os.path.join(out, "model_int8.onnx")
print("quantize int8 ...", flush=True)
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
print(f"fp32 {os.path.getsize(fp32)/1e6:.1f} MB, int8 {os.path.getsize(int8)/1e6:.1f} MB -> {int8}", flush=True)
