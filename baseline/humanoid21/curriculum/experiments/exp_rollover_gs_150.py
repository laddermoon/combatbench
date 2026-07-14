"""Rollover ablation — Generalized shaping γ_s=1.5 (extreme, approaching Dense).

r_t = 1.5·φ(t) - φ(t-1)

γ_s >> 1 → k = (1.5-0.99)/0.01 = 51.0 → advantage 51× amplified.
Holding bonus: 0.5·φ per step → extreme exploit.
SNR near-zero (holding term dominates, reward ≈ Dense).
Tests H2: convergence very slow, ev near Dense's 0.493, entropy very high.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverGS150Experiment(RolloverBase):
    name = "rollover_gs_150"
    reward_mode = "generalized_shaping"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "shaping_gamma": 1.5,
    }


EXPERIMENT = RolloverGS150Experiment()
