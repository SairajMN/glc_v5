"""Per-model pricing: resolution order, cache/batch multipliers, and the
guarantee that v3's per-provider numbers still come out of the v3 API."""

from __future__ import annotations

import pytest

from glc.economics import pricing as P


@pytest.fixture(autouse=True)
def _fresh_table():
    P.reload_pricing()
    yield
    P.reload_pricing()


def test_exact_model_beats_provider_fallback():
    p = P.price_for("gemini_1", "gemini-3.1-flash-lite")
    assert p.source == "model"
    assert (p.input_usd_per_mtok, p.output_usd_per_mtok) == (0.25, 1.50)


def test_a_vendor_prefixed_model_resolves_to_its_bare_entry():
    """Providers disagree about prefixing the vendor: OpenRouter reports
    `zai/zai-glm-4.7` where Cerebras reports `zai-glm-4.7`. The bare entry in
    pricing.yaml must cover both without a duplicate row."""
    bare = P.price_for("cerebras", "zai-glm-4.7")
    assert bare.source == "model"
    prefixed = P.price_for("openrouter", "zai/zai-glm-4.7")
    assert prefixed.source == "model_suffix"
    assert prefixed.input_usd_per_mtok == bare.input_usd_per_mtok == 0.50


def test_glob_pattern_prices_an_unlisted_point_release():
    p = P.price_for("gemini_1", "gemini-3.9-flash-lite-preview")
    assert p.source.startswith("pattern:")
    assert p.input_usd_per_mtok == 0.25


def test_unknown_model_on_known_provider_uses_v3_provider_rate():
    p = P.price_for("groq", "some-model-nobody-priced")
    assert p.source == "provider"
    assert (p.input_usd_per_mtok, p.output_usd_per_mtok) == (0.15, 0.75)


def test_unknown_provider_is_unpriced_not_invented():
    p = P.price_for("no_such_vendor", "no-such-model")
    assert p.source == "unpriced"
    assert p.priced is False
    assert p.input_usd_per_mtok == 0.0


def test_gemini_key_pool_instances_share_the_vendor_price():
    """gemini_3 is the third key, not a third vendor."""
    a = P.price_for("gemini_1", "gemini-3.1-flash")
    b = P.price_for("gemini_7", "gemini-3.1-flash")
    assert a.input_usd_per_mtok == b.input_usd_per_mtok
    assert b.provider == "gemini"


def test_cost_splits_the_four_token_classes():
    c = P.breakdown(
        "anthropic",
        "claude-sonnet-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    assert c.input_usd == pytest.approx(2.00)
    assert c.output_usd == pytest.approx(10.00)
    # cache reads are 0.1x the input rate
    assert c.cache_read_usd == pytest.approx(0.20)
    # cache writes are 1.25x (5-minute TTL)
    assert c.cache_write_usd == pytest.approx(2.50)
    assert c.total_usd == pytest.approx(14.70)


def test_batch_multiplier_halves_both_rates():
    plain = P.cost_usd("anthropic", "claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000)
    batched = P.cost_usd(
        "anthropic", "claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000, batch=True
    )
    assert plain == pytest.approx(30.0)
    assert batched == pytest.approx(15.0)


def test_projection_charges_output_at_max_tokens():
    """A controller that admits on the average is not a controller."""
    projected = P.project_usd("gemini_1", "gemini-3.1-flash-lite", 1_000_000, 1_000_000)
    assert projected == pytest.approx(0.25 + 1.50)


def test_projection_safety_factor_scales_the_reservation():
    base = P.project_usd("gemini_1", "gemini-3.1-flash-lite", 100_000, 1_000)
    padded = P.project_usd("gemini_1", "gemini-3.1-flash-lite", 100_000, 1_000, safety_factor=2.0)
    assert padded == pytest.approx(base * 2)


def test_quality_scores_are_data_and_ordered_sanely():
    lite = P.quality_for("gemini_1", "gemini-3.1-flash-lite")
    pro = P.quality_for("gemini_1", "gemini-3.1-pro")
    assert 0.0 < lite < pro <= 1.0


def test_adding_a_model_needs_no_python_edit(tmp_path, monkeypatch):
    """The no-hardcoding claim, tested: a brand-new model priced only in YAML
    resolves and costs correctly."""
    y = tmp_path / "pricing.yaml"
    y.write_text(
        "version: 1\n"
        "models:\n"
        "  totally-new-model-9000:\n"
        "    provider: newvendor\n"
        "    input: 3.0\n"
        "    output: 9.0\n"
        "    cache_read_multiplier: 0.1\n"
        "    quality: 0.99\n"
    )
    monkeypatch.setenv("GLC_PRICING_YAML", str(y))
    P.reload_pricing()
    p = P.price_for("newvendor", "totally-new-model-9000")
    assert p.source == "model"
    assert p.quality == 0.99
    assert P.cost_usd(
        "newvendor", "totally-new-model-9000", input_tokens=1_000_000, output_tokens=1_000_000
    ) == pytest.approx(12.0)


def test_malformed_pricing_yaml_raises_rather_than_reporting_zero():
    with pytest.raises(ValueError):
        P.PricingTable.from_dict({"models": [1, 2, 3]})
    with pytest.raises(ValueError):
        P.PricingTable.from_dict({"patterns": [{"input": 1.0}]})


# ── v3 backward compatibility ───────────────────────────────────────────────


def test_v3_estimate_usd_signature_and_numbers_unchanged():
    from glc import pricing as v3

    # Exactly the numbers glc_v3's hardcoded table produced.
    assert v3.estimate_usd("groq", 1_000_000, 1_000_000) == pytest.approx(0.90)
    assert v3.estimate_usd("gemini", 1_000_000, 1_000_000) == 0.0
    assert v3.estimate_usd("cerebras", 1_000_000, 0) == pytest.approx(0.50)
    assert v3.estimate_usd("unknown_provider", 1_000_000, 1_000_000) == 0.0


def test_v3_pricing_dict_still_reads_like_a_dict():
    from glc import pricing as v3

    assert set(v3.PRICING_USD_PER_MTOK) >= {"gemini", "groq", "cerebras", "ollama"}
    assert v3.PRICING_USD_PER_MTOK["groq"] == (0.15, 0.75)
    assert "groq" in v3.PRICING_USD_PER_MTOK


def test_v3_estimate_usd_with_model_gives_the_per_model_rate():
    """The opt-in upgrade path: same function, one extra argument."""
    from glc import pricing as v3

    per_provider = v3.estimate_usd("gemini", 1_000_000, 1_000_000)
    per_model = v3.estimate_usd("gemini", 1_000_000, 1_000_000, model="gemini-3.1-pro")
    assert per_provider == 0.0
    assert per_model == pytest.approx(14.0)
