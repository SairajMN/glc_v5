"""OTel spans: GenAI attribute names, cost attributes, content-capture default,
and the no-op-when-unconfigured guarantee."""

from __future__ import annotations

import pytest

from glc.economics import pricing as P
from glc.telemetry import otel as O


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "GLC_OTEL_CONSOLE",
        "GLC_OTEL_IN_MEMORY",
        "GLC_OTEL_CAPTURE_CONTENT",
        "GLC_OTEL_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    O.reset_telemetry()
    P.reload_pricing()
    yield
    O.reset_telemetry()


# ── the no-op path ──────────────────────────────────────────────────────────


def test_unconfigured_is_a_noop_and_tests_need_no_collector():
    t = O.init_telemetry(force=True)
    assert t.active is False
    assert t.exporters == ["none"]
    with t.span(O.SPAN_CHAT) as span:
        span.set_request(provider="gemini_1", model="m")
        span.set_usage(input_tokens=1, output_tokens=2)
    # attributes are still mirrored, so instrumentation has exactly one path
    assert span.attributes[O.GEN_AI_USAGE_INPUT_TOKENS] == 1


def test_explicitly_disabled():
    import os

    os.environ["GLC_OTEL_ENABLED"] = "0"
    os.environ["GLC_OTEL_IN_MEMORY"] = "1"
    t = O.init_telemetry(force=True)
    assert t.active is False
    assert t.exporters == ["disabled"]


def test_exception_inside_a_span_does_not_escape_changed():
    t = O.init_telemetry(force=True, in_memory=True)
    with pytest.raises(RuntimeError, match="boom"):
        with t.span(O.SPAN_CHAT):
            raise RuntimeError("boom")


# ── a real span with real attributes ────────────────────────────────────────


def test_span_carries_gen_ai_usage_and_cost():
    t = O.init_telemetry(force=True, in_memory=True)
    assert t.active is True
    assert "in-memory" in t.exporters

    cost = P.breakdown("gemini_1", "gemini-3.1-flash-lite", input_tokens=1_000_000, output_tokens=1_000_000)
    with t.span(O.SPAN_CHAT) as span:
        span.set_request(provider="gemini_3", model="gemini-3.1-flash-lite", max_tokens=512, temperature=0.2)
        span.set_usage(
            input_tokens=1_000_000, output_tokens=1_000_000, response_model="gemini-3.1-flash-lite"
        )
        span.set_cost(cost.as_dict())
    t.flush()

    assert len(t.captured) == 1
    a = t.captured[0]["attributes"]
    assert t.captured[0]["name"] == "chat"
    # OTel GenAI semantic conventions
    assert a[O.GEN_AI_OPERATION_NAME] == "chat"
    assert a[O.GEN_AI_PROVIDER_NAME] == "gemini"  # vendor, not the key slot
    assert a["glc.provider.instance"] == "gemini_3"
    assert a[O.GEN_AI_REQUEST_MODEL] == "gemini-3.1-flash-lite"
    assert a[O.GEN_AI_REQUEST_MAX_TOKENS] == 512
    assert a[O.GEN_AI_USAGE_INPUT_TOKENS] == 1_000_000
    assert a[O.GEN_AI_USAGE_OUTPUT_TOKENS] == 1_000_000
    assert a[O.GEN_AI_RESPONSE_MODEL] == "gemini-3.1-flash-lite"
    # computed cost
    assert a[O.GLC_COST_USD] == pytest.approx(1.75)
    assert a[O.GLC_COST_PRICE_SOURCE] == "model"


def test_attribute_names_match_the_published_conventions():
    """Pinned strings — this test is the tripwire if someone 'tidies' them."""
    assert O.GEN_AI_PROVIDER_NAME == "gen_ai.provider.name"
    assert O.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
    assert O.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert O.GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
    assert O.SPAN_CHAT == "chat"


def test_principal_dimensions_become_span_attributes():
    from glc.economics.meter import Principal

    t = O.init_telemetry(force=True, in_memory=True)
    with t.span(O.SPAN_CHAT) as span:
        span.set_principal(Principal(tenant="acme", user="u1", agent="researcher"))
    t.flush()
    a = t.captured[0]["attributes"]
    assert a["glc.principal.tenant"] == "acme"
    assert a["glc.principal.user"] == "u1"
    assert a["glc.principal.agent"] == "researcher"
    assert "glc.principal.session" not in a  # absent dimensions are not invented


def test_budget_admission_lands_on_the_span():
    from glc.economics.budget import Admission, BudgetStatus

    st = BudgetStatus(
        principal="session:s",
        dimension="session",
        value="s",
        limit_usd=1.0,
        spent_usd=0.9,
        period="day",
        period_start=0.0,
    )
    t = O.init_telemetry(force=True, in_memory=True)
    with t.span(O.SPAN_CHAT) as span:
        span.set_admission(Admission(allowed=True, projected_usd=0.05, checked=[st]))
    t.flush()
    a = t.captured[0]["attributes"]
    assert a[O.GLC_BUDGET_ADMITTED] is True
    assert a[O.GLC_BUDGET_PRINCIPAL] == "session:s"
    assert a[O.GLC_BUDGET_REMAINING_USD] == pytest.approx(0.1)
    assert a[O.GLC_COST_PROJECTED_USD] == pytest.approx(0.05)


def test_error_status_is_recorded():
    t = O.init_telemetry(force=True, in_memory=True)
    with t.span(O.SPAN_CHAT) as span:
        span.set_error("upstream 500")
    t.flush()
    assert t.captured[0]["status"] == "ERROR"


# ── PII default ─────────────────────────────────────────────────────────────


def test_content_capture_is_off_by_default():
    assert O.capture_content() is False
    t = O.init_telemetry(force=True, in_memory=True)
    with t.span(O.SPAN_CHAT) as span:
        span.set_content(
            messages=[{"role": "user", "content": "my social security number is 000-00-0000"}],
            completion="secret",
        )
    t.flush()
    a = t.captured[0]["attributes"]
    assert O.GEN_AI_INPUT_MESSAGES not in a
    assert O.GEN_AI_OUTPUT_MESSAGES not in a


def test_content_capture_opt_in(monkeypatch):
    monkeypatch.setenv("GLC_OTEL_CAPTURE_CONTENT", "1")
    assert O.capture_content() is True
    t = O.init_telemetry(force=True, in_memory=True)
    with t.span(O.SPAN_CHAT) as span:
        span.set_content(messages=[{"role": "user", "content": "hello"}], completion="hi")
    t.flush()
    a = t.captured[0]["attributes"]
    assert "hello" in a[O.GEN_AI_INPUT_MESSAGES]
    assert "hi" in a[O.GEN_AI_OUTPUT_MESSAGES]


# ── exporter selection ──────────────────────────────────────────────────────


def test_otlp_endpoint_selects_the_http_exporter(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    t = O.init_telemetry(force=True)
    assert t.active is True
    assert any(e.startswith("otlp-http:") for e in t.exporters)
    assert t.endpoint == "http://localhost:4318"


def test_unreachable_collector_does_not_break_a_span(monkeypatch):
    """An OTLP endpoint nothing is listening on must not fail a chat call."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    t = O.init_telemetry(force=True)
    with t.span(O.SPAN_CHAT) as span:
        span.set_usage(input_tokens=1, output_tokens=1)
    t.flush()  # export fails in the background; nothing raises here
    assert span.attributes[O.GEN_AI_USAGE_INPUT_TOKENS] == 1


def test_describe_reports_the_configuration(monkeypatch):
    monkeypatch.setenv("OTEL_SERVICE_NAME", "glc-test")
    t = O.init_telemetry(force=True, in_memory=True)
    d = t.describe()
    assert d["service_name"] == "glc-test"
    assert d["capture_content"] is False
    assert d["active"] is True
