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
