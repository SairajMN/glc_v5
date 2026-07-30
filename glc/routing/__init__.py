"""Routing package.

`glc/routing.py` became `glc/routing/core.py` in v4 so the policy layer could
sit beside it. Every name the v3 module exported is re-exported here verbatim,
so `from glc.routing import Router, LIMITS, SHORTCUTS, ...` keeps working and
`glc.routing.LIMITS` stays the *same dict object* that
`providers.build_providers()` mutates when it expands the Gemini pool.

New in v4: `glc.routing.policy` — role/quality/cost tier selection and cascade
escalation, all loaded from `routing.yaml`.
"""

from __future__ import annotations

from glc.routing.core import (
    DEFAULT_ROUTER_ORDER,
    LIMITS,
    MAX_GEMINI_KEYS,
    SHORTCUTS,
    RateState,
    Router,
    RouterPool,
    resolve,
)
from glc.routing.policy import (
    RoutingPolicy,
    load_policy,
    reload_policy,
)

__all__ = [
    "DEFAULT_ROUTER_ORDER",
    "LIMITS",
    "MAX_GEMINI_KEYS",
    "SHORTCUTS",
    "RateState",
    "Router",
    "RouterPool",
    "RoutingPolicy",
    "load_policy",
    "reload_policy",
    "resolve",
]
