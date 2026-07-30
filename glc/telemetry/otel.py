"""OpenTelemetry GenAI spans for every provider call.

glc_v3 had no tracing at all — a SQLite ledger and `/v1/status` polling. That
answers "what did today cost" and cannot answer "why did *this* request cost
that", which is the question a waterfall answers in one glance.

Three deliberate choices:

**Pinned attribute names.** The GenAI semantic conventions are formally
pre-stable, and in June 2026 they moved into a dedicated
`semantic-conventions-genai` repo with no tagged release. So the attribute keys
are string constants here, not imports from a package that may rename them
under us. When they stabilise, this is the one file to change.

**Content capture off by default.** Prompts and completions are the highest-PII
payload the gateway handles, and a trace backend is usually a different trust
boundary from the gateway. Turn it on per deployment with
`GLC_OTEL_CAPTURE_CONTENT=1` and know why you did.

**No-op when unconfigured.** With no exporter endpoint set, spans still exist
(so instrumentation code has one path, not two) but go nowhere. Tests need no
collector, and a missing OTLP endpoint can never take the gateway down.

Configuration is environment variables, which is the OTel convention and what
every collector sidecar already expects:

    OTEL_EXPORTER_OTLP_ENDPOINT   http://localhost:4318   -> export OTLP/HTTP
    OTEL_SERVICE_NAME             glc-gateway
    OTEL_EXPORTER_OTLP_PROTOCOL   http/protobuf (default) | grpc
    GLC_OTEL_CONSOLE              1  -> also print spans as JSON to stdout
    GLC_OTEL_CAPTURE_CONTENT      1  -> attach prompt/completion text (PII)
    GLC_OTEL_ENABLED              0  -> disable entirely
    GLC_OTEL_RECENT               how many finished traces to keep in memory (default 50)
    GLC_TRACE_UI                  base URL of the trace UI to deep-link into

The endpoint is the OTel *base* URL, as every collector sidecar expects:
`http://localhost:4318` for OTLP/HTTP (the `/v1/traces` path is appended here, and
not appended twice if you already wrote it), `localhost:4317` for gRPC. Jaeger v2
receives OTLP natively on both, so no separate Jaeger exporter is needed.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

# ── OTel GenAI semantic conventions (pinned; see module docstring) ───────────
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_TOKENS = "gen_ai.usage.cache_read_input_tokens"
GEN_AI_USAGE_CACHE_WRITE_TOKENS = "gen_ai.usage.cache_creation_input_tokens"
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"

#: Cost is not in the GenAI conventions (they stop at tokens), so it is
#: namespaced under the gateway rather than squatting on `gen_ai.*`.
GLC_COST_USD = "glc.cost.usd"
GLC_COST_INPUT_USD = "glc.cost.input_usd"
GLC_COST_OUTPUT_USD = "glc.cost.output_usd"
GLC_COST_CACHE_READ_USD = "glc.cost.cache_read_usd"
GLC_COST_PRICE_SOURCE = "glc.cost.price_source"
GLC_COST_PROJECTED_USD = "glc.cost.projected_usd"
GLC_BUDGET_PRINCIPAL = "glc.budget.principal"
GLC_BUDGET_LIMIT_USD = "glc.budget.limit_usd"
GLC_BUDGET_SPENT_USD = "glc.budget.spent_usd"
GLC_BUDGET_REMAINING_USD = "glc.budget.remaining_usd"
GLC_BUDGET_ADMITTED = "glc.budget.admitted"
GLC_ROUTING_ROLE = "glc.routing.role"
GLC_ROUTING_TIER = "glc.routing.tier"
GLC_ROUTING_TIER_CLASSIFIED = "glc.routing.tier_classified"
GLC_ROUTING_ESCALATIONS = "glc.routing.escalations"
GLC_ROUTING_CONFIDENCE = "glc.routing.confidence"
GLC_CACHE_HIT = "glc.cache.hit"
GLC_CACHE_KIND = "glc.cache.kind"
GLC_CACHE_SIMILARITY = "glc.cache.similarity"
GLC_CACHE_TOKENS_SAVED = "glc.cache.tokens_saved"
GLC_CACHE_USD_SAVED = "glc.cache.usd_saved"
GLC_LEDGER_ROW_ID = "glc.ledger.row_id"

#: Principal dimensions are also emitted with the FinOps-friendly names a trace
#: backend can group on directly.
PRINCIPAL_ATTR = {
    "tenant": "glc.principal.tenant",
    "project": "glc.principal.project",
    "user": "glc.principal.user",
    "agent": "glc.principal.agent",
    "session": "glc.principal.session",
}

SPAN_CHAT = "chat"
SPAN_EMBED = "embeddings"
SPAN_CREATE_AGENT = "create_agent"
SPAN_INVOKE_AGENT = "invoke_agent"
SPAN_EXECUTE_TOOL = "execute_tool"

DEFAULT_SERVICE_NAME = "glc-gateway"

#: Jaeger's own default UI port. Only used to *derive* a link when the operator
#: has not named one; an unset, underivable UI means the dashboard says so
#: rather than linking somewhere that does not exist.
DEFAULT_TRACE_UI_PORT = 16686
DEFAULT_RECENT_TRACES = 50


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def otlp_protocol() -> str:
    """`grpc` or `http/protobuf`. Endpoint shape wins over the env var."""
    declared = (
        (os.getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or "")
        .strip()
        .lower()
    )
    return "grpc" if declared == "grpc" else "http/protobuf"


def trace_ui_base() -> str | None:
    """Where a human goes to look at a trace id.

    `GLC_TRACE_UI` if the operator set one. Otherwise derived from the OTLP
    endpoint host, because an OTLP receiver on `somehost:4318` is almost always
    a Jaeger whose UI is on `somehost:16686`. Returns None when there is nothing
    honest to link to.
    """
    explicit = (os.getenv("GLC_TRACE_UI") or os.getenv("GLC_JAEGER_UI") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    endpoint = (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or ""
    ).strip()
    if not endpoint:
        return None
    host = re.sub(r"^\w+://", "", endpoint).split("/", 1)[0].rsplit(":", 1)[0]
    return f"http://{host}:{DEFAULT_TRACE_UI_PORT}" if host else None


def trace_url(trace_id: str | None) -> str | None:
    """A deep link to one trace, or None if no UI is known."""
    base = trace_ui_base()
    return f"{base}/trace/{trace_id}" if base and trace_id else None


def capture_content() -> bool:
    """Whether prompt/completion text may be attached to spans. Off unless set."""
    return _truthy(os.getenv("GLC_OTEL_CAPTURE_CONTENT"))


def telemetry_enabled() -> bool:
    return not (os.getenv("GLC_OTEL_ENABLED", "1").strip() == "0")


# ── span handle ─────────────────────────────────────────────────────────────


class SpanHandle:
    """Thin wrapper over an OTel span (or over nothing at all).

    Every method is safe when there is no live span, so instrumentation at the
    call site is unconditional and the "is tracing on" question is answered
    exactly once, here.
    """

    __slots__ = ("_span", "attributes")

    def __init__(self, span: Any = None):
        self._span = span
        #: Mirror of everything set, so tests and `/v1/*` responses can assert
        #: on span content without needing an exporter or a collector.
        self.attributes: dict[str, Any] = {}

    @property
    def recording(self) -> bool:
        return self._span is not None and getattr(self._span, "is_recording", lambda: False)()

    def set(self, key: str, value: Any) -> SpanHandle:
        if value is None:
            return self
        self.attributes[key] = value
        if self._span is not None:
            try:
                self._span.set_attribute(key, value)
            except Exception:  # pragma: no cover - never let telemetry break a call
                pass
        return self

    def set_many(self, mapping: dict[str, Any]) -> SpanHandle:
        for k, v in mapping.items():
            self.set(k, v)
        return self

    # ── domain-shaped setters ────────────────────────────────────────────────

    def set_request(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        operation: str = SPAN_CHAT,
    ) -> SpanHandle:
        self.set(GEN_AI_OPERATION_NAME, operation)
        # `gen_ai.provider.name` wants the vendor, not the key slot: gemini_3 is
        # the third Gemini key, not a third vendor.
        if provider:
            self.set(GEN_AI_PROVIDER_NAME, provider.split("_", 1)[0])
            self.set("glc.provider.instance", provider)
        self.set(GEN_AI_REQUEST_MODEL, model)
        self.set(GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
        self.set(GEN_AI_REQUEST_TEMPERATURE, temperature)
        return self

    def set_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        response_model: str | None = None,
        finish_reason: str | None = None,
    ) -> SpanHandle:
        self.set(GEN_AI_USAGE_INPUT_TOKENS, int(input_tokens or 0))
        self.set(GEN_AI_USAGE_OUTPUT_TOKENS, int(output_tokens or 0))
        if cache_read_tokens:
            self.set(GEN_AI_USAGE_CACHE_READ_TOKENS, int(cache_read_tokens))
        if cache_write_tokens:
            self.set(GEN_AI_USAGE_CACHE_WRITE_TOKENS, int(cache_write_tokens))
        self.set(GEN_AI_RESPONSE_MODEL, response_model)
        if finish_reason:
            self.set(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
        return self

    def set_cost(self, breakdown: dict | None = None, total_usd: float | None = None) -> SpanHandle:
        b = breakdown or {}
        self.set(GLC_COST_USD, float(total_usd if total_usd is not None else b.get("total_usd", 0.0)))
        self.set(GLC_COST_INPUT_USD, b.get("input_usd"))
        self.set(GLC_COST_OUTPUT_USD, b.get("output_usd"))
        self.set(GLC_COST_CACHE_READ_USD, b.get("cache_read_usd"))
        self.set(GLC_COST_PRICE_SOURCE, b.get("price_source"))
        return self

    def set_principal(self, principal: Any) -> SpanHandle:
        present = principal.present() if hasattr(principal, "present") else dict(principal or {})
        for dim, value in present.items():
            attr = PRINCIPAL_ATTR.get(dim)
            if attr:
                self.set(attr, value)
        return self

    def set_admission(self, admission: Any) -> SpanHandle:
        if admission is None:
            return self
        self.set(GLC_BUDGET_ADMITTED, bool(admission.allowed))
        self.set(GLC_COST_PROJECTED_USD, float(admission.projected_usd))
        binding = admission.breached or (admission.checked[0] if admission.checked else None)
        if binding is not None:
            self.set(GLC_BUDGET_PRINCIPAL, binding.principal)
            self.set(GLC_BUDGET_LIMIT_USD, binding.limit_usd)
            self.set(GLC_BUDGET_SPENT_USD, binding.spent_usd)
            self.set(GLC_BUDGET_REMAINING_USD, binding.remaining_usd)
        return self

    def set_content(self, messages: Any = None, completion: Any = None) -> SpanHandle:
        """Attach prompt/completion text. Silently does nothing unless
        GLC_OTEL_CAPTURE_CONTENT is set — the PII default is off."""
        if not capture_content():
            return self
        import json as _json

        if messages is not None:
            self.set(GEN_AI_INPUT_MESSAGES, _json.dumps(messages, default=str)[:32000])
        if completion is not None:
            self.set(GEN_AI_OUTPUT_MESSAGES, _json.dumps(completion, default=str)[:32000])
        return self

    def record_exception(self, exc: BaseException) -> SpanHandle:
        self.set("error.type", type(exc).__name__)
        if self._span is not None:
            try:
                from opentelemetry.trace import Status, StatusCode

                self._span.record_exception(exc)
                self._span.set_status(Status(StatusCode.ERROR, str(exc)[:300]))
            except Exception:  # pragma: no cover
                pass
        return self

    def set_error(self, message: str) -> SpanHandle:
        self.set("error.message", message[:300])
        if self._span is not None:
            try:
                from opentelemetry.trace import Status, StatusCode

                self._span.set_status(Status(StatusCode.ERROR, message[:300]))
            except Exception:  # pragma: no cover
                pass
        return self


# ── provider setup ──────────────────────────────────────────────────────────


class Telemetry:
    """Owns the tracer provider. `init()` is idempotent."""

    def __init__(self) -> None:
        self.tracer = None
        self.provider = None
        self.exporters: list[str] = []
        self.service_name = DEFAULT_SERVICE_NAME
        self.endpoint: str | None = None
        self._initialised = False
        #: Every span finished in this process, as plain dicts. Populated only
        #: when an in-memory exporter is requested (tests, proofs).
        self.captured: list[dict] = []
        #: Bounded ring of the traces this process emitted. Present whenever the
        #: tracer is active; the dashboard's trace list reads it.
        self.recent: _RecentTraceExporter | None = None

    @property
    def active(self) -> bool:
        return self.tracer is not None

    @property
    def protocol_label(self) -> str:
        """`grpc` when the endpoint is a bare host:port or the env says so."""
        if self.endpoint and not self.endpoint.startswith(("http://", "https://")):
            return "grpc"
        return "grpc" if otlp_protocol() == "grpc" else "http"

    @property
    def traces_endpoint(self) -> str | None:
        """The URL spans are actually POSTed to (HTTP) or dialled on (gRPC)."""
        if not self.endpoint:
            return None
        base = self.endpoint.rstrip("/")
        if self.protocol_label == "grpc":
            return base
        return base if base.endswith("/v1/traces") else f"{base}/v1/traces"

    def _otlp_exporter(self):
        """OTLP/HTTP or OTLP/gRPC. Jaeger v2 speaks both natively."""
        if self.protocol_label == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcSpanExporter,
            )

            return GrpcSpanExporter(endpoint=self.endpoint, insecure=True)
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=self.traces_endpoint)

    # ── what the dashboard reads ───────────────────────────────────────────────

    def recent_traces(self, limit: int = 25) -> list[dict]:
        return self.recent.summaries(limit=limit) if self.recent else []

    def trace_detail(self, trace_id: str) -> dict | None:
        return self.recent.detail(trace_id) if self.recent else None

    def init(self, force: bool = False, in_memory: bool = False) -> Telemetry:
        if self._initialised and not force:
            return self
        self._initialised = True
        self.exporters = []
        self.service_name = os.getenv("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME)
        self.endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        )
        console = _truthy(os.getenv("GLC_OTEL_CONSOLE"))
        in_memory = in_memory or _truthy(os.getenv("GLC_OTEL_IN_MEMORY"))

        if not telemetry_enabled():
            self.tracer, self.recent = None, None
            self.exporters = ["disabled"]
            return self
        if not (self.endpoint or console or in_memory):
            # Nothing to export to. Stay a no-op rather than building a
            # provider that buffers spans forever.
            self.tracer, self.recent = None, None
            self.exporters = ["none"]
            return self
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )
        except ImportError:  # pragma: no cover - deps are declared, but never crash
            self.tracer = None
            self.exporters = ["unavailable: opentelemetry-sdk not installed"]
            return self

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": os.getenv("GLC_VERSION", "0.4.0"),
                }
            )
        )
        if self.endpoint:
            try:
                provider.add_span_processor(BatchSpanProcessor(self._otlp_exporter()))
                self.exporters.append(f"otlp-{self.protocol_label}:{self.traces_endpoint}")
            except Exception as e:  # pragma: no cover - unreachable collector
                self.exporters.append(f"otlp-{self.protocol_label}-failed:{type(e).__name__}")
        if console:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            self.exporters.append("console")
        if in_memory:
            provider.add_span_processor(SimpleSpanProcessor(_ListExporter(self.captured)))
            self.exporters.append("in-memory")
        # Always on when tracing is on: bounded, local, and what the dashboard reads.
        self.recent = _RecentTraceExporter(
            int(os.getenv("GLC_OTEL_RECENT", str(DEFAULT_RECENT_TRACES)) or DEFAULT_RECENT_TRACES)
        )
        provider.add_span_processor(SimpleSpanProcessor(self.recent))

        self.provider = provider
        # Never call trace.set_tracer_provider twice — OTel warns and ignores.
        try:
            trace.set_tracer_provider(provider)
        except Exception:  # pragma: no cover
            pass
        self.tracer = provider.get_tracer("glc.gateway", "0.4.0")
        return self

    def shutdown(self) -> None:
        if self.provider is not None:
            try:
                self.provider.shutdown()
            except Exception:  # pragma: no cover
                pass

    def flush(self, timeout_ms: int = 5000) -> None:
        if self.provider is not None:
            try:
                self.provider.force_flush(timeout_ms)
            except Exception:  # pragma: no cover
                pass

    def describe(self) -> dict:
        return {
            "active": self.active,
            "service_name": self.service_name,
            "endpoint": self.endpoint,
            "traces_endpoint": self.traces_endpoint,
            "protocol": self.protocol_label if self.endpoint else None,
            "exporters": self.exporters,
            "capture_content": capture_content(),
            "captured_spans": len(self.captured),
            "trace_ui": trace_ui_base(),
            "recent_traces": len(self.recent.traces) if self.recent else 0,
        }

    @contextmanager
    def span(self, name: str = SPAN_CHAT, **attributes):
        """Start a span. Yields a `SpanHandle` that works either way."""
        if self.tracer is None:
            handle = SpanHandle(None)
            handle.set_many(attributes)
            yield handle
            return
        with self.tracer.start_as_current_span(name) as span:
            handle = SpanHandle(span)
            handle.set_many(attributes)
            try:
                yield handle
            except BaseException as e:
                handle.record_exception(e)
                raise


def _span_as_dict(s: Any) -> dict:
    """One finished SDK span, flattened for JSON."""
    parent = getattr(s, "parent", None)
    return {
        "name": s.name,
        "trace_id": f"{s.context.trace_id:032x}",
        "span_id": f"{s.context.span_id:016x}",
        "parent_span_id": f"{parent.span_id:016x}" if parent else None,
        "start_time": s.start_time,
        "end_time": s.end_time,
        "duration_ns": (s.end_time or 0) - (s.start_time or 0),
        "status": getattr(s.status, "status_code", None) and s.status.status_code.name,
        "attributes": dict(s.attributes or {}),
    }


class _RecentTraceExporter:
    """Keeps the last N traces in memory so the gateway can show its own work.

    A trace backend is the right place to *query* traces and the wrong place to
    be a hard dependency of the dashboard: with the collector down there would be
    no trace list at all, and the panel would have to invent one. So the gateway
    keeps a small bounded ring of what it just emitted — real spans, real ids, the
    same ids Jaeger holds — and the dashboard deep-links into the UI for the
    waterfall it cannot draw itself.
    """

    def __init__(self, limit: int = DEFAULT_RECENT_TRACES):
        self.limit = max(1, int(limit))
        self.traces: OrderedDict[str, dict] = OrderedDict()

    def export(self, spans):  # noqa: D102
        from opentelemetry.sdk.trace.export import SpanExportResult

        for s in spans:
            row = _span_as_dict(s)
            trace = self.traces.get(row["trace_id"])
            if trace is None:
                trace = {"trace_id": row["trace_id"], "spans": []}
                self.traces[row["trace_id"]] = trace
                while len(self.traces) > self.limit:
                    self.traces.popitem(last=False)
            trace["spans"].append(row)
            self.traces.move_to_end(row["trace_id"])
        return SpanExportResult.SUCCESS

    def summaries(self, limit: int | None = None) -> list[dict]:
        """Newest first: one row per trace, with the numbers a table wants."""
        rows = []
        for trace in reversed(self.traces.values()):
            spans = trace["spans"]
            root = min(spans, key=lambda s: s["start_time"] or 0)
            attrs: dict[str, Any] = {}
            for s in spans:  # child attributes win; the provider call is the interesting one
                attrs.update(s["attributes"])
            start = min((s["start_time"] or 0) for s in spans)
            end = max((s["end_time"] or 0) for s in spans)
            rows.append(
                {
                    "trace_id": trace["trace_id"],
                    "root_span": root["name"],
                    "spans": len(spans),
                    "started_at": start / 1e9 if start else None,
                    "duration_ms": (end - start) / 1e6 if end and start else None,
                    "status": "ERROR" if any(s["status"] == "ERROR" for s in spans) else "OK",
                    "provider": attrs.get(GEN_AI_PROVIDER_NAME),
                    "provider_instance": attrs.get("glc.provider.instance"),
                    "model": attrs.get(GEN_AI_RESPONSE_MODEL) or attrs.get(GEN_AI_REQUEST_MODEL),
                    "input_tokens": attrs.get(GEN_AI_USAGE_INPUT_TOKENS),
                    "output_tokens": attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS),
                    "cost_usd": attrs.get(GLC_COST_USD),
                    "role": attrs.get(GLC_ROUTING_ROLE),
                    "tier": attrs.get(GLC_ROUTING_TIER),
                    "cache_hit": attrs.get(GLC_CACHE_HIT),
                    "budget_admitted": attrs.get(GLC_BUDGET_ADMITTED),
                    "url": trace_url(trace["trace_id"]),
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows

    def detail(self, trace_id: str) -> dict | None:
        """Every span of one trace, parents resolvable — enough for a waterfall."""
        trace = self.traces.get(trace_id)
        if trace is None:
            return None
        spans = sorted(trace["spans"], key=lambda s: s["start_time"] or 0)
        return {"trace_id": trace_id, "url": trace_url(trace_id), "spans": spans}

    def shutdown(self):  # noqa: D102
        return None

    def force_flush(self, timeout_millis: int = 30000):  # noqa: D102, ARG002
        return True


class _ListExporter:
    """Minimal SpanExporter that appends finished spans to a list as dicts."""

    def __init__(self, sink: list[dict]):
        self.sink = sink

    def export(self, spans):  # noqa: D102
        from opentelemetry.sdk.trace.export import SpanExportResult

        for s in spans:
            self.sink.append(
                {
                    "name": s.name,
                    "trace_id": f"{s.context.trace_id:032x}",
                    "span_id": f"{s.context.span_id:016x}",
                    "duration_ns": (s.end_time or 0) - (s.start_time or 0),
                    "status": getattr(s.status, "status_code", None) and s.status.status_code.name,
                    "attributes": dict(s.attributes or {}),
                }
            )
        return SpanExportResult.SUCCESS

    def shutdown(self):  # noqa: D102
        return None

    def force_flush(self, timeout_millis: int = 30000):  # noqa: D102, ARG002
        return True


_telemetry: Telemetry | None = None


def get_telemetry() -> Telemetry:
    global _telemetry
    if _telemetry is None:
        _telemetry = Telemetry().init()
    return _telemetry


def init_telemetry(force: bool = False, in_memory: bool = False) -> Telemetry:
    global _telemetry
    if _telemetry is None:
        _telemetry = Telemetry()
    return _telemetry.init(force=force, in_memory=in_memory)


def reset_telemetry() -> None:
    """Drop the singleton (tests switching exporter config)."""
    global _telemetry
    if _telemetry is not None:
        _telemetry.shutdown()
    _telemetry = None


@contextmanager
def chat_span(name: str = SPAN_CHAT, **attributes):
    """Module-level convenience: `with chat_span() as span: ...`."""
    with get_telemetry().span(name, **attributes) as handle:
        yield handle
