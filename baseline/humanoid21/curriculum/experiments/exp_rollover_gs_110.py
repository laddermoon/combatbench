"""Rollover ablation — Generalized shaping γ_s=1.1 (strong SNR dilution).

r_t = 1.1·φ(t) - φ(t-1)

γ_s > 1 → k = (1.1-0.99)/0.01 = 11.0 → advantage 11× amplified.
Holding bonus: 0.1·φ per step at top → strong exploit pressure.
But SNR heavily diluted (holding term O(0.1) vs Delta's O(0)).
Tests H2: much slower than Delta, low ev, high entropy.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverGS110Experiment(RolloverBase):
    name = "rollover_gs_110"
    reward_mode = "generalized_shaping"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "shaping_gamma": 1.1,
    }


EXPERIMENT = RolloverGS110Experiment()
