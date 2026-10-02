"""从网页 URL 抽取正文与发布时间（trafilatura，缺库/失败则降级为不抽取）。

SearXNG/托管 API 一般只给 snippet；这里对少数高价值 URL 补"全文 + 时间"，
再由 clean 做安全清洗。抓取失败/超时/非安全 → 返回 None，调用方回退到 snippet。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.observability.logging import get_logger, log_event
from app.search.clean import is_safe_url

_log = get_logger("search.extract")


@dataclass
class Extracted:
    text: str
    published: str | None


def extract(url: str, *, timeout: float = 8.0, user_agent: str = "BagentRAG/0.1") -> Extracted | None:
    if not url or not is_safe_url(url):
        return None
    try:
        import trafilatura
    except Exception:  # noqa: BLE001  未安装 trafilatura → 降级
        log_event(_log, "debug", "trafilatura_unavailable")
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, favor_recall=True) or ""
        meta = trafilatura.extract_metadata(downloaded)
        published = getattr(meta, "date", None) if meta else None
        return Extracted(text=text.strip(), published=published)
    except Exception as e:  # noqa: BLE001
        log_event(_log, "debug", "extract_failed", url=url[:120], err=str(e)[:120])
        return None
