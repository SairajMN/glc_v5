"""Telemetry: OTel GenAI spans around every provider call."""

from __future__ import annotations

from glc.telemetry.otel import (
    SPAN_CHAT,
    SPAN_EMBED,
    SpanHandle,
    Telemetry,
    capture_content,
    chat_span,
    get_telemetry,
    init_telemetry,
    reset_telemetry,
)

__all__ = [
    "SPAN_CHAT",
    "SPAN_EMBED",
    "SpanHandle",
    "Telemetry",
    "capture_content",
    "chat_span",
    "get_telemetry",
    "init_telemetry",
    "reset_telemetry",
]
