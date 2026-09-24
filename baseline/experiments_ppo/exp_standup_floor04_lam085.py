"""GAE-lambda variant of `standup_floor04`: gae_lambda 0.95 -> 0.85.

Lower lambda shrinks the advantage horizon toward TD — higher bias
toward the critic's value, lower variance in the advantage estimate.
If per-update gradient direction is noise-limited (the running
hypothesis), a lower-variance adv signal should buy more accurate
displacement per unit KL; if the bias dominates instead, it degrades.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04Lam085(StandupFloor04):
    name = "standup_floor04_lam085"

    _gae_lambda: float = 0.85  # base is 0.95


EXPERIMENT_CLASS = StandupFloor04Lam085
