"""Per-model pricing, loaded from `pricing.yaml`.

v3 priced per provider (`glc/pricing.py`), which reports the wrong number the
moment a provider serves two models — and every provider wired into this
gateway serves several. This module keys on the model and keeps the v3
provider table as the last-resort fallback, so a model nobody has priced yet
reports exactly what v3 reported for it. No price is ever invented.

Three cost shapes matter for the economics work, and all three are data:

* **cache reads** are cheaper than fresh input tokens (0.1x on Anthropic and
  OpenAI, 0.25x on Gemini explicit caching),
* **cache writes** can be *more* expensive than fresh input (1.25x for a 5-min
  Anthropic TTL, 2.0x for an hour),
* **batch** submission halves both rates on every vendor that offers it.

The one function the budget controller cares about is `project_usd()`: what a
call *could* cost before it runs. It has to assume the worst case
(`max_tokens` of output) because a controller that admits on the average and
refuses on the actual is not a controller.
"""

from __future__ import annotations

import fnmatch
import threading
from dataclasses import dataclass, field
from pathlib import Path

from glc import config as _config

MTOK = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """Resolved rates for one (provider, model) pair."""

    model: str
    provider: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_multiplier: float = 1.0
    cache_write_multiplier: float = 1.0
    batch_multiplier: float = 1.0
    quality: float | None = None
    #: how the entry was found — "model" | "model_suffix" | "pattern:<glob>"
    #: | "provider" | "unpriced". Surfaced on the API so a $0.00 report can
    #: always be traced to either a real free tier or a missing entry.
    source: str = "unpriced"

    @property
    def priced(self) -> bool:
        return self.source != "unpriced"


@dataclass
class CostBreakdown:
    """What a call cost, split by the four billable token classes."""

    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_read_usd: float = 0.0
    cache_write_usd: float = 0.0
    total_usd: float = 0.0
    price_source: str = "unpriced"
    model: str = ""
    provider: str = ""

    def as_dict(self) -> dict:
        return {
            "input_usd": round(self.input_usd, 8),
            "output_usd": round(self.output_usd, 8),
            "cache_read_usd": round(self.cache_read_usd, 8),
            "cache_write_usd": round(self.cache_write_usd, 8),
            "total_usd": round(self.total_usd, 8),
            "price_source": self.price_source,
            "model": self.model,
            "provider": self.provider,
        }


def _base_provider(provider: str) -> str:
    """`gemini_3` is key #3 of the Gemini pool, not a distinct vendor."""
    return provider.split("_", 1)[0] if provider else ""


def _f(d: dict, key: str, default: float) -> float:
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"pricing.yaml: {key}={v!r} is not a number") from e


@dataclass
class PricingTable:
    """An immutable-after-load view of pricing.yaml."""

    models: dict[str, dict] = field(default_factory=dict)
    patterns: list[dict] = field(default_factory=list)
    providers: dict[str, dict] = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    unpriced: dict = field(default_factory=lambda: {"input": 0.0, "output": 0.0})
    path: str = ""

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict, path: str = "") -> PricingTable:
        models = data.get("models") or {}
        if not isinstance(models, dict):
            raise ValueError("pricing.yaml: `models` must be a mapping of model-name -> rates")
        patterns = data.get("patterns") or []
        if not isinstance(patterns, list):
            raise ValueError("pricing.yaml: `patterns` must be a list of {match, input, output}")
        for p in patterns:
            if not isinstance(p, dict) or "match" not in p:
                raise ValueError(f"pricing.yaml: pattern entry {p!r} needs a `match` glob")
        return cls(
            models={str(k): (v or {}) for k, v in models.items()},
            patterns=patterns,
            providers=data.get("providers") or {},
            defaults=data.get("defaults") or {},
            unpriced=data.get("unpriced_default") or {"input": 0.0, "output": 0.0},
            path=path,
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> PricingTable:
        p = Path(path) if path else _config.pricing_yaml_path()
        return cls.from_dict(_config.load_yaml(p), str(p))

    # ── resolution ───────────────────────────────────────────────────────────

    def _mk(self, entry: dict, model: str, provider: str, source: str) -> ModelPrice:
        d = self.defaults
        return ModelPrice(
            model=model,
            provider=provider,
            input_usd_per_mtok=_f(entry, "input", 0.0),
            output_usd_per_mtok=_f(entry, "output", 0.0),
            cache_read_multiplier=_f(entry, "cache_read_multiplier", _f(d, "cache_read_multiplier", 1.0)),
            cache_write_multiplier=_f(entry, "cache_write_multiplier", _f(d, "cache_write_multiplier", 1.0)),
            batch_multiplier=_f(entry, "batch_multiplier", _f(d, "batch_multiplier", 1.0)),
            quality=float(entry["quality"]) if entry.get("quality") is not None else None,
            source=source,
        )

    def price_for(self, provider: str, model: str | None) -> ModelPrice:
        base = _base_provider(provider)
        model = (model or "").strip()

        if model and model in self.models:
            return self._mk(self.models[model], model, base, "model")

        # `openai/gpt-4.1-mini` and `gpt-4.1-mini` are the same model billed the
        # same way; providers disagree about whether to prefix the vendor.
        if model and "/" in model:
            suffix = model.rsplit("/", 1)[1]
            if suffix in self.models:
                return self._mk(self.models[suffix], model, base, "model_suffix")

        if model:
            for pat in self.patterns:
                if fnmatch.fnmatch(model, str(pat["match"])):
                    return self._mk(pat, model, base, f"pattern:{pat['match']}")

        if base in self.providers:
            return self._mk(self.providers[base], model or base, base, "provider")

        return self._mk(self.unpriced, model or base, base, "unpriced")

    def quality_for(self, provider: str, model: str | None, default: float = 0.5) -> float:
        q = self.price_for(provider, model).quality
        return default if q is None else q

    # ── costing ──────────────────────────────────────────────────────────────

    def cost(
        self,
        provider: str,
        model: str | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        batch: bool = False,
    ) -> CostBreakdown:
        """Actual cost of a completed call.

        `input_tokens` is the count of *fresh* input tokens. Providers differ on
        whether their reported input count already includes cached tokens; the
        gateway's own ledger stores them in separate columns, so the caller
        passes them separately and nothing is double-billed here.
        """
        pr = self.price_for(provider, model)
        bm = pr.batch_multiplier if batch else 1.0
        in_rate = pr.input_usd_per_mtok * bm
        out_rate = pr.output_usd_per_mtok * bm
        b = CostBreakdown(
            input_usd=max(0, input_tokens) / MTOK * in_rate,
            output_usd=max(0, output_tokens) / MTOK * out_rate,
            cache_read_usd=max(0, cache_read_tokens) / MTOK * in_rate * pr.cache_read_multiplier,
            cache_write_usd=max(0, cache_write_tokens) / MTOK * in_rate * pr.cache_write_multiplier,
            price_source=pr.source,
            model=pr.model,
            provider=pr.provider,
        )
        b.total_usd = b.input_usd + b.output_usd + b.cache_read_usd + b.cache_write_usd
        return b

    def project_usd(
        self,
        provider: str,
        model: str | None,
        est_input_tokens: int,
        max_output_tokens: int,
        batch: bool = False,
        safety_factor: float = 1.0,
    ) -> float:
        """Worst-case cost of a call that has not happened yet.

        Deliberately pessimistic: output is charged at the full `max_tokens` the
        caller asked for, because that is the most the provider is allowed to
        bill. Admitting on an optimistic projection is how a "budget" ends up
        being advisory.
        """
        c = self.cost(
            provider,
            model,
            input_tokens=est_input_tokens,
            output_tokens=max_output_tokens,
            batch=batch,
        )
        return round(c.total_usd * max(0.0, safety_factor), 8)

    # ── reporting ────────────────────────────────────────────────────────────

    def describe(self) -> dict:
        return {
            "path": self.path,
            "model_count": len(self.models),
            "pattern_count": len(self.patterns),
            "provider_fallbacks": sorted(self.providers),
            "defaults": self.defaults,
        }


# ── process-wide singleton, reloadable ──────────────────────────────────────

_lock = threading.Lock()
_table: PricingTable | None = None


def load_pricing(path: Path | str | None = None) -> PricingTable:
    """Return the shared table, loading it on first use."""
    global _table
    with _lock:
        if _table is None or path is not None:
            _table = PricingTable.load(path)
        return _table


def reload_pricing(path: Path | str | None = None) -> PricingTable:
    """Drop the cached table and re-read from disk (SIGHUP / tests)."""
    global _table
    with _lock:
        _table = PricingTable.load(path)
        return _table


def price_for(provider: str, model: str | None) -> ModelPrice:
    return load_pricing().price_for(provider, model)


def cost_usd(
    provider: str,
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> float:
    return round(
        load_pricing()
        .cost(
            provider,
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            batch=batch,
        )
        .total_usd,
        8,
    )


def breakdown(
    provider: str,
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> CostBreakdown:
    return load_pricing().cost(
        provider,
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        batch=batch,
    )


def project_usd(
    provider: str,
    model: str | None,
    est_input_tokens: int,
    max_output_tokens: int,
    batch: bool = False,
    safety_factor: float = 1.0,
) -> float:
    return load_pricing().project_usd(
        provider,
        model,
        est_input_tokens,
        max_output_tokens,
        batch=batch,
        safety_factor=safety_factor,
    )


def quality_for(provider: str, model: str | None, default: float = 0.5) -> float:
    return load_pricing().quality_for(provider, model, default)
