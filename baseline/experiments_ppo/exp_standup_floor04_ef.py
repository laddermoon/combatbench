"""A/B variant of `standup_floor04`: constant explore_factor via hook.

`explore_factor` scales the rollout-side sampling sigma
(σ_eff = σ · exp(e·ln3)); it does not touch the trained σ.  It is read
from ``self.explore_factor`` inside ``build_jobs`` — the loop calls
``param_overrides(update)`` before ``build_jobs`` each update, so
mutating it there is the supported injection point for scheduling.

This variant runs a constant e (class attribute ``EF``) from u1 so a
single run isolates the effect of wider rollout exploration on the
approach-phase slope (hypothesis: escape speed is bounded by direction
quality per unit KL once the KL budget saturates mid-phase).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04EF(StandupFloor04):
    name = "standup_floor04_ef"

    EF: float = 0.5  # σ_eff = σ × exp(0.5·ln3) ≈ σ×1.73

    def param_overrides(self, update: int) -> Optional[Mapping[str, Any]]:
        self.explore_factor = self.EF
        return super().param_overrides(update)


EXPERIMENT_CLASS = StandupFloor04EF
