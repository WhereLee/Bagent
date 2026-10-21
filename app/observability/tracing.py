"""P1：轻量 TraceSpan 适配层。

默认 no-op；只有 OTEL_ENABLED=true 且安装了 requirements-otel.txt 才启用 OTel。
默认不采集 prompt/工具参数/结果内容，只输出有界标识、耗时、token、错误类型。
GenAI 语义约定仍在演进，字段集中在这里，不散落到业务代码。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from app.config import get_settings
from app.observability.logging import get_logger

_log = get_logger("tracing")
_tracer = None
_tracer_initialized = False


class TraceSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def end(self, error: str | None = None) -> None:
        raise NotImplementedError


class _NoopSpan(TraceSpan):
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def end(self, error: str | None = None) -> None:
        return None


class _OtelSpan(TraceSpan):
    def __init__(self, span) -> None:
        self._span = span
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        if not self._ended:
            self._span.set_attribute(key, value)

    def end(self, error: str | None = None) -> None:
        if self._ended:
            return
        if error:
            self._span.set_attribute("error.type", error)
        self._ended = True


def _get_tracer():
    global _tracer, _tracer_initialized
    if _tracer_initialized:
        return _tracer
    _tracer_initialized = True
    s = get_settings()
    if not s.otel_enabled:
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(resource=Resource.create({"service.name": s.otel_service_name}))
        if s.otel_exporter_otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=s.otel_exporter_otlp_endpoint))
            )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("bagent.agent")
        return _tracer
    except Exception as e:  # noqa: BLE001
        _log.warning("otel_init_failed", err=str(e)[:160])
        return None


def set_tracer_for_tests(tracer) -> None:
    """测试注入；传 None 恢复按配置初始化。"""
    global _tracer, _tracer_initialized
    _tracer = tracer
    _tracer_initialized = True


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[TraceSpan]:
    tracer = _get_tracer()
    if tracer is None:
        span = _NoopSpan()
        try:
            yield span
        finally:
            span.end()
        return
    with tracer.start_as_current_span(name) as otel_span:
        span = _OtelSpan(otel_span)
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:  # noqa: BLE001
            span.end(error=type(e).__name__)
            raise
        finally:
            span.end()
