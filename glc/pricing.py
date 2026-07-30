"""Backward-compatible v3 pricing surface.

v3 exported exactly two names — `PRICING_USD_PER_MTOK` and
`estimate_usd(provider, in_tokens, out_tokens)` — and `/v1/cost/by_agent`
called the second one. Both still work and still mean the same thing.

The real table now lives in `glc.economics.pricing`, keyed by model rather than
provider. `estimate_usd()` gained an optional `model` argument: pass it and you
get the per-model rate, omit it and you get the v3 per-provider rate, which is
why the inherited `/v1/cost/by_agent` numbers do not move. New code should call
`glc.economics.pricing` directly.
"""

from __future__ import annotations

from glc.economics import pricing as _p


class _ProviderPricingView(dict):
    """`dict[provider] -> (input, output)` view over pricing.yaml.

    Kept as a mapping because v3 callers indexed it and iterated it. It reads
    the YAML `providers:` block, so editing the YAML moves this too.
    """

    def _refresh(self) -> None:
        table = _p.load_pricing()
        super().clear()
        for name, entry in table.providers.items():
            super().__setitem__(name, (float(entry.get("input", 0.0)), float(entry.get("output", 0.0))))

    def __getitem__(self, k):  # pragma: no cover - trivial
        self._refresh()
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._refresh()
        return super().get(k, default)

    def keys(self):
        self._refresh()
        return super().keys()

    def items(self):
        self._refresh()
        return super().items()

    def values(self):
        self._refresh()
        return super().values()

    def __iter__(self):
        self._refresh()
        return super().__iter__()

    def __len__(self):
        self._refresh()
        return super().__len__()

    def __contains__(self, k):
        self._refresh()
        return super().__contains__(k)


PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = _ProviderPricingView()


def estimate_usd(
    provider: str,
    in_tokens: int,
    out_tokens: int,
    model: str | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> float:
    """USD for a call. Rounded to 6 dp, as v3 was.

    With `model` omitted this resolves through the `providers:` fallback in
    pricing.yaml, which holds v3's numbers verbatim — so existing reports are
    byte-identical. With `model` supplied you get the per-model rate.
    """
    table = _p.load_pricing()
    if model is None:
        entry = table.providers.get(_p._base_provider(provider))
        if entry is None:
            return 0.0
        p_in = float(entry.get("input", 0.0))
        p_out = float(entry.get("output", 0.0))
        return round((in_tokens / 1e6) * p_in + (out_tokens / 1e6) * p_out, 6)
    return round(
        table.cost(
            provider,
            model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            batch=batch,
        ).total_usd,
        6,
    )
