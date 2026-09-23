"""Scheduled explore_factor variant of `standup_floor04`.

ef=0.5 constant confirmed positive on all three seeds (esc u335/u277/
u351 vs same-seed baselines).  This arm decays EF linearly from 0.5 at
u1 to 0 at u300 — testing whether the benefit lives in the early
direction-quality-limited phase, and whether late-phase wide
exploration is wasted precision.  Esc < u335 → late-phase ef was pure
cost; esc > u335 → ef helps throughout.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04EFDecay(StandupFloor04):
    name = "standup_floor04_efdecay"

    EF0: float = 0.5        # σ_eff = σ × exp(0.5·ln3) ≈ σ×1.73 at u1
    DECAY_END: int = 300    # linear decay to 0 by this update

    def param_overrides(self, update: int) -> Optional[Mapping[str, Any]]:
        frac = max(0.0, 1.0 - update / self.DECAY_END)
        self.explore_factor = self.EF0 * frac
        return super().param_overrides(update)


EXPERIMENT_CLASS = StandupFloor04EFDecay
