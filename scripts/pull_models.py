#!/usr/bin/env python3
"""用 huggingface_hub 拉模型到 HF 缓存（供 Bagent 离线加载）。
用法: pull_models.py [embed|reranker|all]
"""
import os
import sys

os.environ.setdefault("HF_HOME", "/opt/bagent/Bagent/models")
os.environ.setdefault("HF_HUB_CACHE", "/opt/bagent/Bagent/models/hub")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402

MODELS = {
    "embed": "BAAI/bge-small-zh-v1.5",
    "reranker": "BAAI/bge-reranker-base",
}


def pull(which: str) -> None:
    names = list(MODELS) if which == "all" else [which]
    for n in names:
        repo = MODELS[n]
        print(f"[{n}] snapshot_download {repo} ...", flush=True)
        p = snapshot_download(repo)
        print(f"[{n}] -> {p}", flush=True)


if __name__ == "__main__":
    pull(sys.argv[1] if len(sys.argv) > 1 else "all")
