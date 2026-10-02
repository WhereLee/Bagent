"""探测 C-MTEB/DuRetrieval 是否可通过镜像流式访问、以及字段结构。"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset  # noqa: E402


def try_name(name: str) -> None:
    print(f"\n=== {name} ===")
    try:
        ds = load_dataset(name, streaming=True)
    except Exception as e:  # noqa: BLE001
        print("load 失败:", repr(e)[:200])
        return
    # 可能是 dict of splits/configs
    keys = list(ds.keys()) if hasattr(ds, "keys") else ["<dataset>"]
    print("keys:", keys)
    for k in keys[:4]:
        try:
            it = ds[k]
            row = next(iter(it))
            print(f"  [{k}] cols={list(row.keys())} sample={ {c: (str(row[c])[:40]) for c in list(row)[:3]} }")
        except Exception as e:  # noqa: BLE001
            print(f"  [{k}] 读取失败:", repr(e)[:120])


if __name__ == "__main__":
    for n in ["C-MTEB/DuRetrieval", "C-MTEB/DuRetrieval-qrels"]:
        try_name(n)
