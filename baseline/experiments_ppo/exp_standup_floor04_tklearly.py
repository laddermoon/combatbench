"""Early-phase target_kl boost variant of `standup_floor04`.

Baseline observation: early updates only use ~30% of the KL budget
(kl_mean ≈ 0.016 vs cap 0.05) while mid-phase saturates near the cap.
Constant target_kl=0.10 was tested and failed — it fell behind after
~u160, exactly when the baseline naturally saturates the budget.

This arm raises the cap ONLY during the idle-budget window
(target_kl=0.10 for u<=150, else the base 0.05), testing whether the
early idle KL can be converted into displacement without paying the
mid-phase cost.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04TKLEarly(StandupFloor04):
    name = "standup_floor04_tklearly"

    TKL_EARLY: float = 0.10   # doubled cap while the budget is idle
    TKL_UNTIL: int = 150      # restore base target_kl after this update

    def param_overrides(self, update: int) -> Optional[Mapping[str, Any]]:
        base = dict(super().param_overrides(update) or {})
        if update <= self.TKL_UNTIL:
            base["target_kl"] = self.TKL_EARLY
        return base or None


EXPERIMENT_CLASS = StandupFloor04TKLEarly
