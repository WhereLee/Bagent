"""预下载 embedding 模型到项目 models/ 目录（不落 C 盘）。

用法： .venv\\Scripts\\python.exe scripts\\download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def main() -> None:
    s = get_settings()  # 触发 HF_HOME 指向 models/
    print(f"HF_HOME={__import__('os').environ.get('HF_HOME')}")
    print(f"下载模型: {s.embedding_model_name} -> {s.models_dir}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(s.embedding_model_name, cache_folder=str(s.models_dir), device="cpu")
    dim = model.get_sentence_embedding_dimension()
    print(f"完成。向量维度={dim}")
    # 冒烟测试
    v = model.encode(["你好，世界"], normalize_embeddings=True)
    assert v.shape[1] == dim
    print("冒烟测试通过。")


if __name__ == "__main__":
    main()
