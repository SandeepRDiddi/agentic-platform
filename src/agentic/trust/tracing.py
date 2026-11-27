"""Tracing (trust layer: "Langfuse, OTel").

One mechanism: OpenTelemetry. Spans are exported over OTLP/HTTP to
  * an OTel collector / Jaeger / Tempo / Azure Monitor (OTEL_EXPORTER_OTLP_ENDPOINT), and/or
  * Langfuse's OTLP endpoint (LANGFUSE_HOST + keys) — LLM spans use GenAI semantic conventions
    (gen_ai.*), which Langfuse renders as generations with model, tokens and cost.
Trace context is propagated across HTTP (traceparent) and Kafka (message headers), so one task =
one trace from channel to worker.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode

from agentic.config import Settings

log = logging.getLogger(__name__)
_configured = False
_httpx_instrumented = False


def setup_tracing(settings: Settings, service: str, extra_exporters: list[SpanExporter] | None = None) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service,
                "service.version": settings.service_version,
                "deployment.environment": settings.env,
            }
        )
    )
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        if settings.otel_exporter_otlp_endpoint:
            ep = settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=ep)))
        auth = settings.langfuse_auth_header()
        if settings.langfuse_host and auth:
            ep = settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces"
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=ep, headers={"Authorization": auth}))
            )
    except Exception as e:  # tracing must never take the service down
        log.error("tracing exporter setup failed: %s", e)
    for ex in extra_exporters or []:
        provider.add_span_processor(BatchSpanProcessor(ex))
    trace.set_tracer_provider(provider)


def instrument_fastapi(app: Any) -> None:
    global _httpx_instrumented
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
        if not _httpx_instrumented:
            HTTPXClientInstrumentor().instrument()
            _httpx_instrumented = True
    except Exception as e:
        log.warning("otel instrumentation unavailable: %s", e)


def tracer() -> trace.Tracer:
    return trace.get_tracer("agentic")


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[trace.Span]:
    with tracer().start_as_current_span(name) as sp:
        for k, v in attrs.items():
            if v is not None:
                sp.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else str(v))
        try:
            yield sp
        except Exception as e:
            sp.record_exception(e)
            sp.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            raise


def llm_attrs(sp: trace.Span, *, model: str, alias: str, prompt_tokens: int, completion_tokens: int,
              cost: float, cached: bool, finish_reason: str) -> None:
    sp.set_attribute("gen_ai.system", model.split("/")[0])
    sp.set_attribute("gen_ai.request.model", model)
    sp.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    sp.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
    sp.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    sp.set_attribute("agentic.llm.alias", alias)
    sp.set_attribute("agentic.llm.cost_usd", cost)
    sp.set_attribute("agentic.llm.cached", cached)


def inject_headers() -> dict[str, str]:
    h: dict[str, str] = {}
    propagate.inject(h)
    return h


@contextmanager
def extract_context(headers: dict[str, str]) -> Iterator[None]:
    token = otel_context.attach(propagate.extract(headers))
    try:
        yield
    finally:
        otel_context.detach(token)


def current_trace_id() -> str | None:
    sc = trace.get_current_span().get_span_context()
    return format(sc.trace_id, "032x") if sc.is_valid else None
