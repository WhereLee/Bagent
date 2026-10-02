"""结构化 JSON 日志 + 请求级 request_id 贯穿。

request_id 用 contextvar 存储：中间件在请求入口设置，日志 Formatter 自动带上，
使同一请求的所有日志行可用 request_id 关联（可观测的"追踪"基础）。
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        # 允许通过 extra={"fields": {...}} 附加结构化字段
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    # 幂等：清掉默认 handler，避免重复
    root.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_fields(logger: logging.Logger, level: str, msg: str, **fields) -> None:
    """带结构化字段地记录一条日志。"""
    logger.log(getattr(logging, level.upper()), msg, extra={"fields": fields})
