"""Rollover ablation — Generalized shaping γ_s=0.95 (anti-learning, mild).

r_t = 0.95·φ(t) - φ(t-1)

γ_s < γ (0.99) → k = (0.95-0.99)/0.01 = -4.0 → advantage reversed.
Tests H1: strategy should learn to MINIMIZE φ (success → 0).
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverGS095Experiment(RolloverBase):
    name = "rollover_gs_095"
    reward_mode = "generalized_shaping"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "shaping_gamma": 0.95,
    }


EXPERIMENT = RolloverGS095Experiment()
