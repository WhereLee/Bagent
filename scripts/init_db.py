"""初始化数据库 schema（幂等）。

用法： .venv\\Scripts\\python.exe scripts\\init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import init_schema  # noqa: E402


def main() -> None:
    init_schema()
    print("schema 初始化完成（documents / chunks / HNSW 索引）。")


if __name__ == "__main__":
    main()
