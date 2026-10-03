#!/usr/bin/env bash
# 用 setsid 把 uvicorn 脱离会话启动，确保 ssh 退出后仍存活。
cd /opt/bagent/Bagent || exit 1
pkill -f "uvicorn app.api.main" 2>/dev/null
sleep 2
setsid /opt/bagent/.appenv/bin/uvicorn app.api.main:app --host 127.0.0.1 --port 8000 </dev/null >/tmp/bagent_app.log 2>&1 &
echo "launched pid $!"
