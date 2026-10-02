"""把 bge-reranker-base 导出为 ONNX，并做 int8 动态量化 → 得到磁盘上的小模型。

产物：models/reranker-onnx/model_fp32.onnx 与 model_int8.onnx（后者约 ~300MB 级）。
用法： .venv\\Scripts\\python.exe scripts/export_reranker_onnx.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from app.config import ROOT_DIR, get_settings  # noqa: E402

OUT_DIR = ROOT_DIR / "models" / "reranker-onnx"


def main() -> None:
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer

    name = get_settings().reranker_model_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(name)
    ce = CrossEncoder(name, device="cpu")
    model = ce.model
    model.eval()

    enc = tok(["q"], ["d"], padding=True, truncation=True, return_tensors="pt")
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    fp32_path = OUT_DIR / "model_fp32.onnx"

    print("导出 fp32 ONNX ...", flush=True)
    torch.onnx.export(
        model, (input_ids, attention_mask), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    int8_path = OUT_DIR / "model_int8.onnx"
    print("int8 动态量化 ...", flush=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    for p in (fp32_path, int8_path):
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB")
    print(f"\n完成 → {OUT_DIR}")
    print("运行时用 RERANKER_BACKEND=onnx + RERANKER_ONNX_PATH=models/reranker-onnx/model_int8.onnx")


if __name__ == "__main__":
    main()
