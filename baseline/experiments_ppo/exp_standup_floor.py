"""A/B variant of `standup`: uncertainty floor ENABLED (0.3).

Identical to `standup` in every other respect (same seed=42, same
network, same PPO params).  The single treatment variable is the
uncertainty floor:

  standup        : uncertainty_floor=0.0  (off — σ free to collapse)
  standup_floor  : uncertainty_floor=0.3  (base-class default)

Hypothesis: in the no-floor run, σ collapses late in training
(std 0.37→0.17, U 0.46→0.19), which inflates the Gaussian score
(∇log π ∝ 1/σ²), producing growing grad_norm / KL / ratio_max churn
once the task saturates.  With floor=0.3 the hinge loss should pin U
near 0.3 and break that amplifier.

Coef uses the base-class default 0.01 — it was moot in `standup`
(floor=0 disables the term entirely), so this is part of the "floor
on" treatment, not a second independent variable.
"""
from __future__ import annotations

from .exp_standup import Standup


class StandupFloor(Standup):
    name = "standup_floor"

    # --- Exploration: enable the uncertainty floor ---
    uncertainty_floor: float = 0.3
    uncertainty_coef: float = 0.01


EXPERIMENT_CLASS = StandupFloor
