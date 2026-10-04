#!/usr/bin/env bash
# M12 复盘归档闭环冒烟（不依赖 LLM）：建事件 → 取整份 playbook
set -e
BASE=http://127.0.0.1:8000
RESP=$(curl -s -X POST "$BASE/events" -H 'Content-Type: application/json' \
  -d '{"name":"文昌骑行复盘","type":"活动","created_by":"president"}')
echo "create_event: $RESP"
EID=$(echo "$RESP" | grep -oE '"id":[0-9]+' | grep -oE '[0-9]+')
echo "event_id=$EID"
echo "playbook:"
curl -s "$BASE/events/$EID/playbook"
echo
