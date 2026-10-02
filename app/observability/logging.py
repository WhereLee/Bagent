"""结构化 JSON 日志：统一根 logger、接管 uvicorn、压制第三方噪声、request_id 贯穿。

设计目标（解决"日志看着总有问题"）：
- 单一出口：根 logger 只挂一个 JSON handler，所有日志（含我们、uvicorn、被转发的警告）统一 JSON。
- 噪声治理：transformers/sentence_transformers/httpx 等第三方 logger 降到 WARNING；
  uvicorn.access 交给我们中间件统一记录（避免重复且格式不一）；Python warnings 走 logging
  并过滤 jieba 触发的 pkg_resources DeprecationWarning。
- 可贯穿：contextvar 存 request_id，Formatter 自动带上，同一请求日志可关联。
- 可复用：FastAPI 与脚本都调用同一个 configure_logging。
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import warnings
from datetime import datetime, timezone

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# 第三方：只关心 WARNING 及以上，正常 INFO 噪声丢弃
_NOISY_LOGGERS = (
    "transformers", "sentence_transformers", "huggingface", "huggingface_hub",
    "httpx", "httpcore", "urllib3", "filelock", "PIL", "numpy", "numba", "fsspec",
    "datasets", "pyarrow", "asyncio", "jieba",
)
# 由我们中间件统一记录请求，关掉 uvicorn 的 access 逐条日志以免重复/非 JSON
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # 允许 module 直接放结构化字段之外的标准项（如 funcName）
        if hasattr(record, "funcName") and record.levelno >= logging.WARNING:
            payload["func"] = record.funcName
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", stream=None) -> None:
    """安装统一 JSON 日志。幂等：重复调用只重装 handler，不叠加。"""
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)

    # 接管 uvicorn：清空其自带 handler，交给 root（统一 JSON）；access 交给中间件
    for name in _UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # 压制第三方噪声
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # warnings 走 logging；过滤已知的无害弃用（jieba -> pkg_resources）
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*pkg_resources.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*setuptools.*")
    warnings.showwarning = _showwarning_to_log


def _showwarning_to_log(message, category, filename, lineno, file=None, line=None) -> None:
    """把 Python warning 路由到 logging（取代 3.13 已移除的 warnings.captureWarnings）。"""
    logging.getLogger("py.warnings").warning(
        "%s:%s: %s: %s", filename, lineno, category.__name__, message
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: str, msg: str, **fields) -> None:
    """记录一条带结构化字段的日志。"""
    logger.log(getattr(logging, level.upper()), msg, extra={"fields": fields})
