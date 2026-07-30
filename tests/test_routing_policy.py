"""Routing policy: role changes the tier, HUGE is servable, cost/quality
ordering, and cascade escalation only on a low-confidence signal."""

from __future__ import annotations

import pytest

from glc.economics import pricing as P
from glc.routing import LIMITS
from glc.routing import policy as PO


@pytest.fixture(autouse=True)
def _fresh():
    P.reload_pricing()
    PO.reset_policy()
    yield
    PO.reset_policy()


def _pol(tmp_path=None, monkeypatch=None, text: str | None = None) -> PO.RoutingPolicy:
    if text is None:
        return PO.reload_policy()
    p = tmp_path / "routing.yaml"
    p.write_text(text)
    monkeypatch.setenv("GLC_ROUTING_YAML", str(p))
    return PO.reload_policy()


class _Prov:
    def __init__(self, model):
        self.model = model


def _providers(**kw):
    return {name: _Prov(model) for name, model in kw.items()}


# ── the shipped policy ──────────────────────────────────────────────────────


def test_shipped_tiers_match_v3_rings():
    """Backward compatibility: routing.yaml's TINY and LARGE rings are v3's
    TIER_TO_ORDER verbatim, so default routing does not move."""
    from glc.routes.chat import TIER_TO_ORDER

    pol = _pol()
    assert pol.tiers["TINY"].order == TIER_TO_ORDER["TINY"]
    assert pol.tiers["LARGE"].order == TIER_TO_ORDER["LARGE"]


def test_huge_tier_exists_and_declares_a_context_floor():
    pol = _pol()
    assert "HUGE" in pol.tiers
    assert pol.tiers["HUGE"].min_ctx == 8000
    assert pol.ladder == ["TINY", "LARGE", "HUGE"]


# ── role changes routing (the headline for build 3) ─────────────────────────


def test_decision_role_floors_a_tiny_classification_up_to_large():
    pol = _pol()
    d = pol.tier_for("decision", "TINY")
    assert d.tier == "LARGE"
    assert d.classified_tier == "TINY"
    assert d.clamped is True
    assert "floors at LARGE" in d.reason


def test_perception_role_caps_a_huge_classification_down_to_large():
    pol = _pol()
    d = pol.tier_for("perception", "HUGE")
    assert d.tier == "LARGE"
    assert d.clamped is True
    assert "caps at LARGE" in d.reason


def test_same_prompt_two_roles_two_tiers():
    """The single clearest statement of the feature: identical classifier
    output, different role, different model tier."""
    pol = _pol()
    assert pol.tier_for("perception", "TINY").tier == "TINY"
    assert pol.tier_for("decision", "TINY").tier == "LARGE"


def test_memory_role_passes_the_classification_through_unchanged():
    pol = _pol()
    for tier in ("TINY", "LARGE", "HUGE"):
        d = pol.tier_for("memory", tier)
        assert d.tier == tier
        assert d.clamped is False


def test_unknown_role_degrades_to_the_default_rather_than_erroring():
    pol = _pol()
    d = pol.tier_for("perceptoin", "TINY")  # typo
    assert d.role == pol.default_role
    assert d.tier == "TINY"


def test_no_classification_uses_the_role_default():
    pol = _pol()
    d = pol.tier_for("decision", None)
    assert d.tier == "LARGE"
    assert "role default" in d.reason or d.clamped


def test_adding_a_role_needs_no_python_edit(tmp_path, monkeypatch):
    pol = _pol(
        tmp_path,
        monkeypatch,
        """
version: 1
ladder: [SMALL, BIG]
tiers:
  SMALL: {order: [ollama]}
  BIG: {order: [gemini]}
default_role: whatever
roles:
  whatever: {default_tier: SMALL}
  auditor: {default_tier: BIG, min_tier: BIG, escalate: false}
""",
    )
    assert pol.tier_for("auditor", "SMALL").tier == "BIG"
    assert pol.role_spec("auditor").escalate is False
    assert "auditor" in pol.known_roles()


def test_a_role_pointing_at_an_undeclared_tier_is_a_loud_error(tmp_path, monkeypatch):
    with pytest.raises(PO.RoutingConfigError):
        _pol(
            tmp_path,
            monkeypatch,
            "version: 1\nladder: [A]\ntiers:\n  A: {order: []}\nroles:\n  r: {default_tier: NOPE}\n",
        )


# ── candidate ordering ──────────────────────────────────────────────────────


def test_base_names_expand_to_gemini_key_pool_instances():
    """The v3 bug this fixes: `gemini` never matched `gemini_1`, so Gemini was
    silently dropped from every auto-routed call."""
    pol = _pol()
    avail = _providers(gemini_1="gemini-3.1-flash", gemini_2="gemini-3.1-flash", groq="openai/gpt-oss-120b")
    ordered, _ = pol.order_for("LARGE", avail, objective="order")
    assert "gemini_1" in ordered and "gemini_2" in ordered
    assert ordered[:2] == ["gemini_1", "gemini_2"]  # LARGE lists gemini first


def test_huge_tier_rejects_providers_whose_context_is_too_small():
    pol = _pol()
    avail = _providers(gemini_1="gemini-3.1-flash", github="openai/gpt-4.1-mini")
    ordered, rejected = pol.order_for("HUGE", avail, limits=LIMITS, est_tokens=500_000)
    assert "gemini_1" in ordered  # 1M context
    assert "github" not in ordered  # 8k context
    assert any("max_ctx" in r["reason"] for r in rejected)


def test_huge_tier_can_reject_everything_and_say_why():
    pol = _pol()
    avail = _providers(github="openai/gpt-4.1-mini")
    ordered, rejected = pol.order_for("HUGE", avail, limits=LIMITS, est_tokens=900_000)
    assert ordered == []
    assert rejected and "max_ctx" in rejected[0]["reason"]


def test_cost_objective_puts_the_cheapest_first():
    pol = _pol()
    avail = _providers(
        gemini_1="gemini-3.1-pro",  # 2.00 / 12.00
        github="microsoft/Phi-4-mini-instruct",  # 0 / 0
        groq="openai/gpt-oss-120b",  # 0.15 / 0.75
    )
    ordered, _ = pol.order_for("TINY", avail, objective="cost")
    assert ordered[0] == "github"
    assert ordered[-1] == "gemini_1"


def test_quality_objective_puts_the_best_first():
    pol = _pol()
    avail = _providers(gemini_1="gemini-3.1-pro", github="microsoft/Phi-4-mini-instruct")
    ordered, _ = pol.order_for("TINY", avail, objective="quality")
    assert ordered[0] == "gemini_1"


def test_tradeoff_dial_moves_the_choice(tmp_path, monkeypatch):
    pol = _pol()
    avail = _providers(gemini_1="gemini-3.1-pro", groq="openai/gpt-oss-120b")
    cheap_first, _ = pol.order_for("TINY", avail, objective="cost_quality", tradeoff=0.0)
    quality_first, _ = pol.order_for("TINY", avail, objective="cost_quality", tradeoff=1.0)
    assert cheap_first[0] != quality_first[0]
    assert quality_first[0] == "gemini_1"


def test_pin_first_wins_over_the_objective(tmp_path, monkeypatch):
    pol = _pol(
        tmp_path,
        monkeypatch,
        """
version: 1
ladder: [T]
tiers:
  T: {order: [gemini, ollama]}
roles:
  r: {default_tier: T}
default_role: r
selection:
  objective: quality
  pin_first: ["ollama"]
""",
    )
    avail = _providers(gemini_1="gemini-3.1-pro", ollama="qwen3:4b")
    ordered, _ = pol.order_for("T", avail)
    assert ordered[0] == "ollama"


# ── confidence and cascade ──────────────────────────────────────────────────


def test_a_good_answer_scores_full_confidence_and_does_not_escalate():
    pol = _pol()
    a = pol.assess({"text": "A perfectly reasonable and complete answer.", "stop_reason": "end_turn"})
    assert a.score == 1.0
    assert a.signals == []
    go, why = pol.should_escalate("decision", "TINY", a, 0)
    assert go is False
    assert "threshold" in why


def test_an_empty_answer_escalates():
    pol = _pol()
    a = pol.assess({"text": "", "tool_calls": [], "stop_reason": "end_turn"})
    assert a.score == 0.0
    assert PO.TRIGGER_EMPTY in a.signals
    go, why = pol.should_escalate("decision", "TINY", a, 0)
    assert go is True
    assert "TINY -> LARGE" in why


def test_a_schema_failure_escalates():
    pol = _pol()
    a = pol.assess({"text": "not json", "schema_failed": True, "stop_reason": "end_turn"})
    assert PO.TRIGGER_SCHEMA in a.signals
    assert pol.should_escalate("memory", "TINY", a, 0)[0] is True


def test_a_provider_failure_escalates():
    pol = _pol()
    a = pol.assess(None, error=RuntimeError("upstream 500"))
    assert a.score == 0.0
    assert PO.TRIGGER_FAILURE in a.signals


def test_truncation_is_scored_but_above_the_default_threshold():
    """stop_reason=max_tokens is a weaker signal than an empty answer: it
    usually means the caller under-budgeted max_tokens, not that the model
    failed. The shipped threshold deliberately does not escalate it."""
    pol = _pol()
    a = pol.assess({"text": "a long answer cut off mid-sen", "stop_reason": "max_tokens"})
    assert a.signals == ["truncated_response"]
    assert a.score == pytest.approx(0.55)
    assert a.score > pol.confidence_threshold
    assert pol.should_escalate("decision", "TINY", a, 0)[0] is False


def test_a_tool_call_with_no_text_is_not_an_empty_answer():
    pol = _pol()
    a = pol.assess({"text": "", "tool_calls": [{"id": "1", "name": "f"}], "stop_reason": "tool_use"})
    assert a.score == 1.0


def test_hedge_markers_are_config_and_ship_empty(tmp_path, monkeypatch):
    """A marker list is a guess about one task class; a wrong guess escalates
    good answers, which is the exact failure mode this session is about."""
    pol = _pol()
    assert (pol.escalation.get("confidence") or {}).get("hedge_markers") == []
    tuned = _pol(
        tmp_path,
        monkeypatch,
        """
version: 1
ladder: [T, L]
tiers:
  T: {order: []}
  L: {order: []}
roles:
  r: {default_tier: T}
default_role: r
escalation:
  enabled: true
  max_escalations: 1
  confidence_threshold: 0.8
  triggers: [low_confidence]
  confidence:
    hedge_markers: ["i cannot determine"]
    hedge_penalty: 0.5
    default: 1.0
""",
    )
    a = tuned.assess({"text": "I cannot determine the answer.", "stop_reason": "end_turn"})
    assert a.score == pytest.approx(0.5)
    assert tuned.should_escalate("r", "T", a, 0)[0] is True


def test_escalation_respects_max_escalations():
    pol = _pol()
    a = pol.assess({"text": "", "stop_reason": "end_turn"})
    assert pol.should_escalate("decision", "TINY", a, 0)[0] is True
    go, why = pol.should_escalate("decision", "TINY", a, pol.max_escalations)
    assert go is False
    assert "max_escalations" in why


def test_a_role_with_escalate_false_never_escalates():
    pol = _pol()
    a = pol.assess({"text": "", "stop_reason": "end_turn"})
    go, why = pol.should_escalate("triage", "TINY", a, 0)
    assert go is False
    assert "escalate: false" in why


def test_the_top_of_the_ladder_cannot_escalate():
    pol = _pol()
    a = pol.assess({"text": "", "stop_reason": "end_turn"})
    go, why = pol.should_escalate("memory", "HUGE", a, 0)
    assert go is False
    assert "top of the ladder" in why


def test_escalation_can_be_disabled_wholesale(tmp_path, monkeypatch):
    pol = _pol(
        tmp_path,
        monkeypatch,
        "version: 1\nladder: [T]\ntiers:\n  T: {order: []}\nroles:\n  r: {default_tier: T}\n"
        "default_role: r\nescalation:\n  enabled: false\n",
    )
    a = pol.assess({"text": "", "stop_reason": "end_turn"})
    assert pol.should_escalate("r", "T", a, 0) == (False, "escalation disabled in routing.yaml")


def test_next_tier_walks_the_ladder():
    pol = _pol()
    assert pol.next_tier("TINY") == "LARGE"
    assert pol.next_tier("LARGE") == "HUGE"
    assert pol.next_tier("HUGE") is None


def test_tier_to_order_view_keeps_the_v3_shape():
    pol = _pol()
    t2o = pol.tier_to_order()
    assert set(t2o) == {"TINY", "LARGE", "HUGE"}
    assert isinstance(t2o["TINY"], list)


def test_empty_tiers_is_a_loud_error(tmp_path, monkeypatch):
    with pytest.raises(PO.RoutingConfigError):
        _pol(tmp_path, monkeypatch, "version: 1\ntiers: {}\n")


def test_ladder_referencing_an_undeclared_tier_is_a_loud_error(tmp_path, monkeypatch):
    with pytest.raises(PO.RoutingConfigError):
        _pol(tmp_path, monkeypatch, "version: 1\nladder: [A, GHOST]\ntiers:\n  A: {order: []}\n")
