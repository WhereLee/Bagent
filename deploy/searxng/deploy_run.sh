#!/usr/bin/env bash
# 在服务器起 SearXNG（+valkey），仅绑 127.0.0.1:8080，并做一次 json 冒烟。
set -uo pipefail
cd /opt/bagent/Bagent/deploy/searxng || exit 1

mkdir -p searxng
cp -f settings.yml searxng/settings.yml
S=$(openssl rand -hex 32)
sed -i "s/CHANGE_ME-openssl-rand-hex-32/$S/" searxng/settings.yml

if ss -ltn | grep -q ':8080 '; then echo "PORT8080_BUSY"; exit 3; fi

sudo docker rm -f searxng redis 2>/dev/null || true
sudo docker run -d --name redis --restart unless-stopped valkey/valkey:8-alpine \
  valkey-server --save 30 1 >/dev/null && echo "redis up"
sudo docker run -d --name searxng --restart unless-stopped -p 127.0.0.1:8080:8080 \
  -e SEARXNG_BASE_URL=http://localhost:8080/ \
  -v "$PWD/searxng:/etc/searxng:rw" --link redis searxng/searxng:latest >/dev/null && echo "searxng up"

echo "等待启动..."; sleep 20
echo "=== searxng 日志尾 ==="; sudo docker logs --tail 20 searxng 2>&1 | tail -20
echo "=== 冒烟: engines=baidu ==="
curl -s "http://127.0.0.1:8080/search?q=%E6%B5%8B%E8%AF%95&format=json&engines=baidu" | head -c 600
echo
