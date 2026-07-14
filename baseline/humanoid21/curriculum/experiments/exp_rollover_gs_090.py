"""Rollover ablation — Generalized shaping γ_s=0.90 (anti-learning).

r_t = 0.90·φ(t) - φ(t-1)

γ_s < γ (0.99) → k = (0.90-0.99)/0.01 = -9.0 → advantage reversed.
Tests H1: strategy should learn to MINIMIZE φ (success → 0).
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverGS090Experiment(RolloverBase):
    name = "rollover_gs_090"
    reward_mode = "generalized_shaping"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "shaping_gamma": 0.90,
    }


EXPERIMENT = RolloverGS090Experiment()
