#!/usr/bin/env bash
# 用 setsid 把 uvicorn 脱离会话启动，确保 ssh 退出后仍存活。
# APP_ROOT_PATH 为空时不加 --root-path（本地直连）；子路径反代时设为 /bagent。
cd /opt/bagent/Bagent || exit 1
ROOT_ARG=""
[ -n "${APP_ROOT_PATH:-}" ] && ROOT_ARG="--root-path ${APP_ROOT_PATH}"
pkill -f "uvicorn app.api.main" 2>/dev/null
sleep 2
# shellcheck disable=SC2086  # ROOT_ARG 需要按空格分词
setsid /opt/bagent/.appenv/bin/uvicorn app.api.main:app --host 127.0.0.1 --port 8000 ${ROOT_ARG} </dev/null >/tmp/bagent_app.log 2>&1 &
echo "launched pid $! ${APP_ROOT_PATH:+(root_path=$APP_ROOT_PATH)}"
