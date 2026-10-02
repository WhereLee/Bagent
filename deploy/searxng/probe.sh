#!/usr/bin/env bash
Q="%E7%A7%A6%E6%83%A0%E6%96%87%E7%8E%8B"   # 秦惠文王
for e in sogou 360search baidu quark; do
  echo "=== engines=$e ==="
  curl -s "http://127.0.0.1:8080/search?q=$Q&format=json&engines=$e" | head -c 320
  echo
done
echo "=== default(所有启用引擎) ==="
curl -s "http://127.0.0.1:8080/search?q=$Q&format=json" | head -c 500
echo
