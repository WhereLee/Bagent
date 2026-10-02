"""可观测纯逻辑单元测试：JSON 日志格式 + Prometheus 渲染。"""
import json
import logging

from app.observability.logging import JsonFormatter, request_id_var
from app.observability.metrics import HTTP_REQUESTS, render_metrics


def test_json_formatter_emits_request_id_and_fields():
    request_id_var.set("abc123")
    rec = logging.LogRecord("unit", logging.INFO, "path", 1, "hello", None, None)
    rec.fields = {"duration_ms": 12.3, "path": "/query"}
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "hello"
    assert payload["request_id"] == "abc123"
    assert payload["level"] == "INFO"
    assert payload["duration_ms"] == 12.3
    assert payload["path"] == "/query"


def test_json_formatter_is_valid_json_without_fields():
    rec = logging.LogRecord("unit", logging.WARNING, "path", 1, "plain", None, None)
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "plain"
    assert payload["level"] == "WARNING"


def test_render_metrics_exposes_custom_metric():
    HTTP_REQUESTS.labels(method="GET", path="/probe", status="200").inc()
    body, ctype = render_metrics()
    assert isinstance(body, bytes)
    assert b"bagent_http_requests_total" in body
    assert "text/plain" in ctype or "openmetrics" in ctype


def test_log_event_attaches_fields():
    from app.observability.logging import get_logger, log_event

    logger = get_logger("test_evt")
    logger.setLevel(logging.INFO)
    captured: list[logging.LogRecord] = []

    class Cap(logging.Handler):
        def emit(self, record):
            captured.append(record)

    h = Cap()
    logger.addHandler(h)
    logger.propagate = False
    try:
        log_event(logger, "info", "hello", x=1, path="/query")
        assert captured and captured[0].fields == {"x": 1, "path": "/query"}
    finally:
        logger.removeHandler(h)
        logger.propagate = True


def test_configure_logging_single_handler_and_noise_suppression():
    from app.observability.logging import JsonFormatter, configure_logging

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("INFO")
        # 存因 JSON handler（pytest 可能另挂自己的捕获 handler，故不断言“仅 1 个”）
        assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)
        # 第三方噪声降级
        assert logging.getLogger("transformers").level == logging.WARNING
        assert logging.getLogger("jieba").level == logging.WARNING
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
        # uvicorn 交给 root 统一 JSON
        assert logging.getLogger("uvicorn").propagate is True
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
