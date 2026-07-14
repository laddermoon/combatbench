"""Rollover ablation — Generalized shaping γ_s=0.998 (exploration regime).

r_t = 0.998·φ(t) - φ(t-1)

γ_s ∈ (γ, 1) → k = (0.998-0.99)/0.01 = 0.8 → same-direction, weaker.
Holding tax: (1-0.998)·φ = 0.002·φ per step at top → fallback cheaper.
Tests H2/H3: higher entropy, larger fallback gap, slower than Delta.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverGS0998Experiment(RolloverBase):
    name = "rollover_gs_0998"
    reward_mode = "generalized_shaping"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "shaping_gamma": 0.998,
    }


EXPERIMENT = RolloverGS0998Experiment()
