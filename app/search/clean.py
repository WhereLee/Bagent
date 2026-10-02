"""Web 内容安全清洗：SSRF 防护 + 注入中和 + 长度限制。

联网抓回的内容一律当**敌意输入**：
- 只允许 http/https；拒绝内网/回环/链路本地/元数据地址（含解析后为内网的域名）→ 防 SSRF；
- 文本过 M6 的注入检测/中和 → 防"网页里藏忽略指令"；
- 截断超长，降低 token 与注入面。
"""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

from app.security import sanitize_query

# 明确的敏感主机名
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata", "instance", ""}
_WS = re.compile(r"\s+")


def is_safe_url(url: str, *, resolve_dns: bool = True) -> bool:
    """url 是否可安全抓取（仅 http/https，且不指向内网/回环/链路本地/元数据）。"""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname or ""
    if host.lower() in _BLOCKED_HOSTS:
        return False

    hosts_to_check = [host]
    if resolve_dns:
        try:
            infos = socket.getaddrinfo(host, None)
            hosts_to_check += [i[4][0] for i in infos]
        except (socket.gaierror, UnicodeError, OSError):
            return False  # 解析不了，宁可不抓

    for h in hosts_to_check:
        try:
            ip = ipaddress.ip_address(h)
        except ValueError:
            continue  # 非字面 IP（域名已解析进列表，字面量已覆盖）
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
        if str(ip).startswith("169.254."):  # 云元数据
            return False
    return True


def clean_web_text(text: str, max_chars: int = 4000) -> str:
    """折叠所有空白 + 截断 + 注入中和。返回可安全进 prompt 的文本。"""
    t = _WS.sub(" ", (text or "").strip())
    if max_chars and len(t) > max_chars:
        t = t[:max_chars]
    return sanitize_query(t)
