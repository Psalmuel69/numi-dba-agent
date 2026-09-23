"""Structured logging + tracing setup (spec §42).

Two hard rules enforced here:
  1. Secrets and raw database query results never reach a log line — callers
     pass structured, already-redacted fields, and `_SENSITIVE_KEYS` is a
     last-line-of-defense scrub for anything that slips through.
  2. Every log line is structured (key=value / JSON), never an f-string, so
     it can be correlated by `request_id` / `conversation_id` / etc.
"""

from __future__ import annotations

import logging
import re

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from structlog.types import EventDict, WrappedLogger

_SENSITIVE_KEYS = re.compile(
    r"(password|secret|token|api[_-]?key|credential|connection[_-]?string|"
    r"authorization|access[_-]?key)",
    re.IGNORECASE,
)


def _redact_sensitive(_logger: WrappedLogger, _method_name: str, event_dict: EventDict) -> EventDict:
    for key in list(event_dict.keys()):
        if _SENSITIVE_KEYS.search(key):
            event_dict[key] = "***REDACTED***"
    return event_dict


def configure_logging(service_name: str, log_level: str = "INFO") -> None:
    logging.basicConfig(level=log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_sensitive,
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service_name)


def configure_tracing(service_name: str, *, console_export: bool = False) -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if console_export:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def get_tracer(name: str):
    return trace.get_tracer(name)
