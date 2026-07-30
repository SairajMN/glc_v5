"""Role, cost/quality and cascade policy, loaded from `routing.yaml`.

What glc_v3 did: a router LLM classified the prompt as TINY / LARGE / HUGE, the
tier was looked up in a hardcoded dict, HUGE returned 503, and the `role` the
caller declared was written to the ledger and otherwise ignored.

What this module adds, all as data:

**Role changes the model.** Each role declares a default tier plus optional
floor and ceiling. A `decision` call floored at LARGE gets a LARGE model even
when the classifier says TINY; a `perception` call capped at LARGE cannot
escalate into the frontier tier no matter what the classifier says. This is the
config pattern the industry converged on in 2026 — you do not pick a model, you
pick a model per role.

**HUGE is servable.** The tier declares a `min_ctx`, candidates are filtered by
their actual `max_ctx`, and the refusal (when it comes) names the context limit
instead of pointing at an unimplemented summariser.

**Cost/quality ordering.** The tier's ring can be re-sorted by blended $/Mtok,
by quality score, or by a tradeoff dial between them, using the numbers already
in `pricing.yaml`.

**Cascade escalation.** Confidence is scored from structural signals — empty
text, a schema failure, truncation, configured hedge markers — never from a
second LLM call. A judge call to check a cheap answer frequently costs more than
the expensive answer would have, which is the trap that makes naive cascades
lose money.

The confidence signals are deliberately conservative and the hedge-marker list
ships empty: a marker list is a guess about one task class, and a wrong guess
escalates good answers, which is the exact cost-per-call-down /
cost-per-resolved-task-up failure this session is about.
"""

from __future__ import annotations

import fnmatch
import threading
from dataclasses import dataclass, field
from pathlib import Path

from glc import config as _config

#: Escalation triggers a caller/tester can name.
TRIGGER_EMPTY = "empty_response"
TRIGGER_FAILURE = "provider_failure"
TRIGGER_LOW_CONFIDENCE = "low_confidence"
TRIGGER_SCHEMA = "schema_failure"


class RoutingConfigError(ValueError):
    """routing.yaml is malformed."""


@dataclass(frozen=True)
class TierSpec:
    name: str
    order: list[str] = field(default_factory=list)
    min_ctx: int = 0
    description: str = ""
    #: When true, only the providers named in `order` may serve this tier.
    #: Without it `order_for` appends every other configured provider as a
    #: last-resort tail, which is right for a failover ring and wrong for a
    #: capability rung: a rung that can silently be served by a different
    #: model is not a rung. It is also the only way to keep a quarantined
    #: provider out of the tail.
    strict: bool = False
    #: Per-tier override of `selection.objective` / `selection.tradeoff`. A
    #: rung that names one model wants `order`, whatever the global dial says.
    objective: str | None = None
    tradeoff: float | None = None
    #: Which ladder this tier belongs to. Set from `ladder` / `ladders`.
    ladder: str = ""


@dataclass(frozen=True)
class RoleSpec:
    name: str
    default_tier: str
    min_tier: str | None = None
    max_tier: str | None = None
    escalate: bool = True

    def as_dict(self) -> dict:
        return {
            "role": self.name,
            "default_tier": self.default_tier,
            "min_tier": self.min_tier,
            "max_tier": self.max_tier,
            "escalate": self.escalate,
        }


@dataclass
class TierDecision:
    """Why this tier, and what the classifier had said."""

    role: str
    tier: str
    classified_tier: str | None = None
    reason: str = ""
    clamped: bool = False

    def as_dict(self) -> dict:
        return {
            "role": self.role,
            "tier": self.tier,
            "classified_tier": self.classified_tier,
            "reason": self.reason,
            "clamped": self.clamped,
        }


@dataclass
class ConfidenceAssessment:
    """A structural read on whether an answer is worth keeping."""

    score: float
    signals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"score": round(self.score, 4), "signals": list(self.signals)}


class RoutingPolicy:
    """Immutable-after-load view of routing.yaml."""

    def __init__(self, data: dict, path: str = ""):
        self.path = path
        self.raw = data
        ladder = data.get("ladder") or []
        tiers_raw = data.get("tiers") or {}
        if not tiers_raw:
            raise RoutingConfigError("routing.yaml: `tiers` is empty; at least one tier is required")
        if not ladder:
            ladder = list(tiers_raw)
        unknown = [t for t in ladder if t not in tiers_raw]
        if unknown:
            raise RoutingConfigError(f"routing.yaml: ladder names undeclared tiers {unknown}")
        self.ladder: list[str] = list(ladder)
        #: Named secondary ladders. `ladder` above stays the classifier's
        #: TINY/LARGE/HUGE ring for backward compatibility; a second, named
        #: ladder can order a different set of tiers by cost. `rank`,
        #: `next_tier` and `prev_tier` resolve within whichever ladder a tier
        #: belongs to, so one gateway can hold both without either seeing the
        #: other's rungs.
        self.ladders: dict[str, list[str]] = {"default": list(self.ladder)}
        for lname, names in (data.get("ladders") or {}).items():
            names = list(names or [])
            unknown_l = [t for t in names if t not in tiers_raw]
            if unknown_l:
                raise RoutingConfigError(f"routing.yaml: ladders.{lname} names undeclared tiers {unknown_l}")
            if lname == "default":
                raise RoutingConfigError("routing.yaml: `ladders.default` is reserved for `ladder`")
            self.ladders[str(lname)] = names
        #: tier name -> the ladder it lives on. A tier on two ladders is a
        #: config error, because "one rung up" would be ambiguous.
        self._ladder_of: dict[str, str] = {}
        for lname, names in self.ladders.items():
            for t in names:
                if t in self._ladder_of and self._ladder_of[t] != lname:
                    raise RoutingConfigError(
                        f"routing.yaml: tier {t!r} is on two ladders "
                        f"({self._ladder_of[t]!r} and {lname!r}); escalation would be ambiguous"
                    )
                self._ladder_of[t] = lname
        #: Providers config says must not be used, with the reason why. They are
        #: dropped from every tier and reported as a rejection, so a broken
        #: endpoint is benched by editing this file rather than by editing code.
        self.unavailable: dict[str, str] = {
            str(k): str(v) for k, v in (data.get("unavailable") or {}).items()
        }
        self.backoff: dict = data.get("backoff") or {}
        self.tiers: dict[str, TierSpec] = {}
        for name, spec in tiers_raw.items():
            spec = spec or {}
            self.tiers[name] = TierSpec(
                name=name,
                order=list(spec.get("order") or []),
                min_ctx=int(spec.get("min_ctx") or 0),
                description=str(spec.get("description") or ""),
                strict=bool(spec.get("strict", False)),
                objective=(str(spec["objective"]) if spec.get("objective") else None),
                tradeoff=(float(spec["tradeoff"]) if spec.get("tradeoff") is not None else None),
                ladder=self._ladder_of.get(name, ""),
            )
        self.default_role = str(data.get("default_role") or (list(data.get("roles") or {}) or ["default"])[0])
        self.roles: dict[str, RoleSpec] = {}
        for name, spec in (data.get("roles") or {}).items():
            spec = spec or {}
            for key in ("default_tier", "min_tier", "max_tier"):
                v = spec.get(key)
                if v is not None and v not in self.tiers:
                    raise RoutingConfigError(
                        f"routing.yaml: role {name!r} {key}={v!r} is not a declared tier"
                    )
            self.roles[name] = RoleSpec(
                name=name,
                default_tier=str(spec.get("default_tier") or self.ladder[0]),
                min_tier=spec.get("min_tier"),
                max_tier=spec.get("max_tier"),
                escalate=bool(spec.get("escalate", True)),
            )
        self.selection: dict = data.get("selection") or {}
        self.escalation: dict = data.get("escalation") or {}
        self.budget: dict = data.get("budget") or {}

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path | str | None = None) -> RoutingPolicy:
        p = Path(path) if path else _config.routing_yaml_path()
        return cls(_config.load_yaml(p), str(p))

    # ── tier ordering helpers ────────────────────────────────────────────────

    def ladder_of(self, tier: str) -> list[str]:
        """The ladder `tier` lives on — the classifier ladder unless a named
        ladder in `ladders:` claims it."""
        return self.ladders.get(self._ladder_of.get(tier, "default"), self.ladder)

    def rank(self, tier: str) -> int:
        """Position on the tier's own ladder. Unknown tiers sort last."""
        rungs = self.ladder_of(tier)
        return rungs.index(tier) if tier in rungs else len(rungs)

    def role_spec(self, role: str | None) -> RoleSpec:
        """Resolve a role name, falling back to the configured default rather
        than raising: an unknown role should downgrade to sane routing, not
        take the request down."""
        if role and role in self.roles:
            return self.roles[role]
        if self.default_role in self.roles:
            return self.roles[self.default_role]
        return RoleSpec(name=role or "default", default_tier=self.ladder[0])

    def known_roles(self) -> list[str]:
        return list(self.roles)

    # ── the role -> tier decision ────────────────────────────────────────────

    def tier_for(self, role: str | None, classified_tier: str | None = None) -> TierDecision:
        """Apply the role's floor/ceiling to whatever the classifier decided.

        This is the function that makes `role` matter. With no role configured
        it returns the classified tier unchanged, which is exactly v3.
        """
        spec = self.role_spec(role)
        role_ladder = self._ladder_of.get(spec.default_tier, "default")
        tier: str | None = None
        reason = ""
        if classified_tier in self.tiers:
            if self._ladder_of.get(classified_tier, "default") == role_ladder:
                tier, reason = classified_tier, "classifier"
            else:
                # The classifier only ever names rungs of its own ladder. A role
                # whose rungs are a different ladder still wants the classifier's
                # verdict, so carry it across by POSITION: "smallest of three"
                # stays "smallest of three" whichever ladder it lands on.
                src = self.ladder_of(classified_tier)
                dst = self.ladders.get(role_ladder) or self.ladder
                idx = src.index(classified_tier)
                pos = 0 if len(src) < 2 else min(len(dst) - 1, round(idx * (len(dst) - 1) / (len(src) - 1)))
                tier = dst[pos]
                reason = f"classifier said {classified_tier}, mapped onto the {role_ladder} ladder"
        if tier is None:
            tier, reason = spec.default_tier, f"role default ({spec.name})"
        clamped = False
        if spec.min_tier and self.rank(tier) < self.rank(spec.min_tier):
            tier, clamped = spec.min_tier, True
            reason = f"role {spec.name} floors at {spec.min_tier} (classifier said {classified_tier})"
        if spec.max_tier and self.rank(tier) > self.rank(spec.max_tier):
            tier, clamped = spec.max_tier, True
            reason = f"role {spec.name} caps at {spec.max_tier} (classifier said {classified_tier})"
        return TierDecision(
            role=spec.name,
            tier=tier,
            classified_tier=classified_tier,
            reason=reason,
            clamped=clamped,
        )

    # ── candidate selection ──────────────────────────────────────────────────

    def _blended_cost(self, provider: str, model: str | None) -> float:
        from glc.economics import pricing as _pricing

        pr = _pricing.price_for(provider, model)
        w = float(self.selection.get("output_weight", 0.35))
        return pr.input_usd_per_mtok * (1 - w) + pr.output_usd_per_mtok * w

    def _quality(self, provider: str, model: str | None) -> float:
        from glc.economics import pricing as _pricing

        return _pricing.quality_for(provider, model, default=0.5)

    def order_for(
        self,
        tier: str,
        available: dict,
        limits: dict | None = None,
        est_tokens: int = 0,
        objective: str | None = None,
        tradeoff: float | None = None,
    ) -> tuple[list[str], list[dict]]:
        """Return (ordered candidate names, rejected-with-reason).

        `available` maps provider-instance name -> provider object (so
        `gemini_1`, `gemini_2`, … are separate candidates, as v3 intended).
        Ordering starts from the tier's declared ring, then optionally re-sorts
        by cost/quality. Candidates whose `max_ctx` cannot hold `est_tokens` are
        rejected with a reason instead of silently failing later.
        """
        spec = self.tiers.get(tier)
        if spec is None:
            return list(available), [{"tier": tier, "reason": "tier not declared in routing.yaml"}]

        rejected: list[dict] = []

        def _base(name: str) -> str:
            """`gemini_3` -> `gemini`: the name config talks about."""
            head = name.rsplit("_", 1)
            return head[0] if len(head) == 2 and head[1].isdigit() else name

        # Providers config has benched. Dropped first so neither the ring nor
        # the last-resort tail can reach them.
        benched = set()
        for name in available:
            reason = self.unavailable.get(_base(name)) or self.unavailable.get(name)
            if reason:
                benched.add(name)
                rejected.append({"provider": name, "reason": f"unavailable: {reason}"})

        # Expand the declared base names to live instances, preserving order.
        ordered: list[str] = []
        for base in spec.order:
            for name in available:
                if name in benched:
                    continue
                if name == base or name.startswith(base + "_"):
                    if name not in ordered:
                        ordered.append(name)
        # Anything configured but not in the tier's ring still beats no answer —
        # unless the tier is `strict`, which is how a capability rung guarantees
        # the model it names is the model that answers.
        if not spec.strict:
            for name in available:
                if name not in ordered and name not in benched:
                    ordered.append(name)
        else:
            for name in available:
                if name not in ordered and name not in benched:
                    rejected.append({"provider": name, "reason": f"skipped:not in strict tier {tier} ring"})

        need_ctx = max(int(spec.min_ctx), int(est_tokens))
        if limits and need_ctx:
            keep = []
            for name in ordered:
                max_ctx = int((limits.get(name) or {}).get("max_ctx", 0))
                if max_ctx and max_ctx < need_ctx:
                    rejected.append(
                        {
                            "provider": name,
                            "reason": f"skipped:max_ctx {max_ctx} < required {need_ctx}",
                        }
                    )
                else:
                    keep.append(name)
            ordered = keep

        obj = str(objective or spec.objective or self.selection.get("objective", "order"))
        if obj != "order" and ordered:
            if tradeoff is None:
                tradeoff = spec.tradeoff
            t = float(self.selection.get("tradeoff", 0.5) if tradeoff is None else tradeoff)
            costs = {n: self._blended_cost(n, getattr(available[n], "model", None)) for n in ordered}
            quals = {n: self._quality(n, getattr(available[n], "model", None)) for n in ordered}
            cmax = max(costs.values()) or 1.0

            def score(n: str) -> float:
                norm_cost = costs[n] / cmax if cmax else 0.0
                if obj == "cost":
                    return -costs[n]
                if obj == "quality":
                    return quals[n]
                return t * quals[n] - (1 - t) * norm_cost

            base_rank = {n: i for i, n in enumerate(ordered)}
            ordered.sort(key=lambda n: (-score(n), base_rank[n]))

        pins = list(self.selection.get("pin_first") or [])
        if pins:
            pinned = [n for n in ordered if any(fnmatch.fnmatch(n, p) for p in pins)]
            rest = [n for n in ordered if n not in pinned]
            ordered = pinned + rest
        return ordered, rejected

    # ── cascade ──────────────────────────────────────────────────────────────

    @property
    def escalation_enabled(self) -> bool:
        return bool(self.escalation.get("enabled", False))

    @property
    def max_escalations(self) -> int:
        return int(self.escalation.get("max_escalations", 0))

    @property
    def confidence_threshold(self) -> float:
        return float(self.escalation.get("confidence_threshold", 0.0))

    def triggers(self) -> set[str]:
        return set(self.escalation.get("triggers") or [])

    def next_tier(self, tier: str) -> str | None:
        """One rung up the tier's ladder, or None at the top."""
        rungs = self.ladder_of(tier)
        r = self.rank(tier)
        if r + 1 < len(rungs):
            return rungs[r + 1]
        return None

    def prev_tier(self, tier: str) -> str | None:
        """One rung DOWN the tier's ladder, or None at the bottom.

        The downgrade direction. Budget pressure walks this, so a run under
        pressure lands on a genuinely cheaper model rather than on the same
        model asked to think less.
        """
        rungs = self.ladder_of(tier)
        r = self.rank(tier)
        if 0 < r < len(rungs):
            return rungs[r - 1]
        return None

    def assess(self, result: dict | None, error: Exception | str | None = None) -> ConfidenceAssessment:
        """Score an answer from structural signals only.

        Returns the *lowest* applicable score, because one damning signal is
        enough — a schema failure does not become acceptable by also being long.
        """
        cfg = self.escalation.get("confidence") or {}
        signals: list[str] = []
        scores: list[float] = []

        if error is not None:
            return ConfidenceAssessment(0.0, [TRIGGER_FAILURE])
        if result is None:
            return ConfidenceAssessment(0.0, [TRIGGER_EMPTY])

        text = str(result.get("text") or "")
        if result.get("schema_failed"):
            signals.append(TRIGGER_SCHEMA)
            scores.append(float(cfg.get("schema_failure", 0.1)))
        if not text.strip() and not result.get("tool_calls"):
            signals.append(TRIGGER_EMPTY)
            scores.append(float(cfg.get("empty_response", 0.0)))
        else:
            min_chars = int(cfg.get("min_response_chars", 0))
            if min_chars and len(text.strip()) < min_chars and not result.get("tool_calls"):
                signals.append("short_response")
                scores.append(float(cfg.get("short_response", 0.3)))
        if result.get("stop_reason") == "max_tokens":
            signals.append("truncated_response")
            scores.append(float(cfg.get("truncated_response", 0.45)))
        markers = [str(m).lower() for m in (cfg.get("hedge_markers") or [])]
        if markers:
            low = text.lower()
            hit = next((m for m in markers if m in low), None)
            if hit:
                signals.append(f"hedge:{hit}")
                scores.append(1.0 - float(cfg.get("hedge_penalty", 0.35)))

        default = float(cfg.get("default", 1.0))
        return ConfidenceAssessment(min(scores) if scores else default, signals)

    def should_escalate(
        self,
        role: str | None,
        tier: str,
        assessment: ConfidenceAssessment,
        escalations_so_far: int = 0,
    ) -> tuple[bool, str]:
        """Decide whether to pay for a bigger model. Returns (yes/no, why)."""
        if not self.escalation_enabled:
            return False, "escalation disabled in routing.yaml"
        if not self.role_spec(role).escalate:
            return False, f"role {self.role_spec(role).name} has escalate: false"
        if escalations_so_far >= self.max_escalations:
            return False, f"max_escalations ({self.max_escalations}) reached"
        nxt = self.next_tier(tier)
        if nxt is None:
            return False, f"{tier} is the top of the ladder"
        triggers = self.triggers()
        fired = [s for s in assessment.signals if s in triggers or s.split(":", 1)[0] in triggers]
        low = assessment.score < self.confidence_threshold
        if low and TRIGGER_LOW_CONFIDENCE in triggers:
            fired.append(TRIGGER_LOW_CONFIDENCE)
        if not fired:
            return False, (
                f"confidence {assessment.score:.2f} >= threshold {self.confidence_threshold:.2f}"
                if not low
                else "no configured trigger fired"
            )
        return True, f"escalate {tier} -> {nxt} on {sorted(set(fired))} (confidence {assessment.score:.2f})"

    # ── reporting ────────────────────────────────────────────────────────────

    def tier_to_order(self) -> dict[str, list[str]]:
        """v3-shaped view, so `/v1/routers` keeps reporting `tier_to_order`.

        Scoped to the classifier ladder on purpose: a v3 client reads this dict
        as "the tiers that exist" and has no idea what a capability rung is.
        `describe()` reports every ladder for anything that does.
        """
        return {name: list(self.tiers[name].order) for name in self.ladder if name in self.tiers}

    def describe(self) -> dict:
        return {
            "path": self.path,
            "ladder": self.ladder,
            "ladders": self.ladders,
            "unavailable": self.unavailable,
            "backoff": self.backoff,
            "tiers": {
                n: {
                    "order": s.order,
                    "min_ctx": s.min_ctx,
                    "description": s.description,
                    "strict": s.strict,
                    "objective": s.objective,
                    "ladder": s.ladder,
                }
                for n, s in self.tiers.items()
            },
            "default_role": self.default_role,
            "roles": {n: s.as_dict() for n, s in self.roles.items()},
            "selection": self.selection,
            "escalation": self.escalation,
            "budget": self.budget,
        }


_policy: RoutingPolicy | None = None
_lock = threading.Lock()


def load_policy(path: Path | str | None = None) -> RoutingPolicy:
    global _policy
    with _lock:
        if _policy is None or path is not None:
            _policy = RoutingPolicy.load(path)
        return _policy


def reload_policy(path: Path | str | None = None) -> RoutingPolicy:
    global _policy
    with _lock:
        _policy = RoutingPolicy.load(path)
        return _policy


def reset_policy() -> None:
    global _policy
    with _lock:
        _policy = None
