"""P2：运行 Agent 崩溃恢复 worker。"""
from __future__ import annotations

import argparse

from app.agent.worker import run_forever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只扫描一次后退出")
    args = parser.parse_args()
    run_forever(once=args.once)


if __name__ == "__main__":
    main()
