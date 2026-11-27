"""Structured JSON logs carrying request/task/tenant context and OTel trace ids."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

_ctx: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar("log_ctx", default=MappingProxyType({}))


def bind(**kw: Any) -> None:
    _ctx.set({**_ctx.get(), **kw})


def clear() -> None:
    _ctx.set({})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **_ctx.get(),
        }
        try:
            from opentelemetry import trace

            sc = trace.get_current_span().get_span_context()
            if sc.is_valid:
                payload["trace_id"] = format(sc.trace_id, "032x")
                payload["span_id"] = format(sc.span_id, "016x")
        except Exception:  # noqa: S110
            pass
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "aiokafka", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
