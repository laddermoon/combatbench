"""GAE-lambda bracket of `standup_floor04`: gae_lambda 0.95 -> 0.75.

lam085_s42 escaped u362 vs baseline u387 (-25u): the lower-variance adv
hypothesis got first support.  This arm probes further down the same
axis — if lam075 is better still, keep descending; if worse, 0.85 is
the sweet spot and the lambda dimension closes (0.95 base / 0.85 / 0.75
response curve complete).
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04Lam075(StandupFloor04):
    name = "standup_floor04_lam075"

    _gae_lambda: float = 0.75  # base is 0.95, lam085 is 0.85


EXPERIMENT_CLASS = StandupFloor04Lam075
