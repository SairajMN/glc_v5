"""Agent economics: pricing, attribution, and the spend kill-switch.

Three modules, one per question:

* `pricing`  — what does a token cost, per *model* (`pricing.yaml`)
* `meter`    — who is it billed to, across five principal dimensions
* `budget`   — may this call happen at all (`budgets.yaml`)

The order matters. `budget` cannot admit without `pricing` to project a cost,
and `meter` cannot attribute without a principal to attribute to.
"""

from __future__ import annotations

from glc.economics.budget import (
    BUDGET_EXCEEDED,
    Admission,
    BudgetConfigError,
    BudgetController,
    BudgetPolicy,
    BudgetStatus,
    get_controller,
    reload_controller,
)
from glc.economics.meter import DIMENSIONS, Meter, Principal, Usage, get_meter
from glc.economics.pricing import (
    CostBreakdown,
    ModelPrice,
    PricingTable,
    load_pricing,
    reload_pricing,
)

__all__ = [
    "BUDGET_EXCEEDED",
    "DIMENSIONS",
    "Admission",
    "BudgetConfigError",
    "BudgetController",
    "BudgetPolicy",
    "BudgetStatus",
    "CostBreakdown",
    "Meter",
    "ModelPrice",
    "Principal",
    "PricingTable",
    "Usage",
    "get_controller",
    "get_meter",
    "load_pricing",
    "reload_controller",
    "reload_pricing",
]
