"""P2：运行 Agent 崩溃恢复 worker。"""
from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from app.agent.worker import run_forever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只扫描一次后退出")
    parser.add_argument("--poll", type=float, default=None, help="轮询间隔秒数")
    args = parser.parse_args()
    run_forever(once=args.once, poll_s=args.poll)


if __name__ == "__main__":
    main()
