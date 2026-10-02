"""Reranker 后端选择的单元测试（monkeypatch，不加载真模型）。"""
import pytest

from app.config import get_settings
from app.retrieval import rerank


def test_backend_default_is_st(monkeypatch):
    monkeypatch.setattr(rerank, "Reranker", lambda *a, **k: "ST")
    s = get_settings().model_copy(update={"reranker_backend": "st"})
    monkeypatch.setattr(rerank, "get_settings", lambda: s)
    assert rerank._build_reranker() == "ST"


def test_backend_onnx_selected(monkeypatch):
    called = {}

    def fake_onnx(path):
        called["path"] = str(path)
        return "ONNX"

    monkeypatch.setattr(rerank, "OnnxReranker", fake_onnx)
    s = get_settings().model_copy(update={"reranker_backend": "onnx",
                                          "reranker_onnx_path": "models/reranker-onnx/model_int8.onnx"})
    monkeypatch.setattr(rerank, "get_settings", lambda: s)
    assert rerank._build_reranker() == "ONNX"
    assert called["path"].endswith("model_int8.onnx")
