"""Rollover ablation — Delta + continuous dense base (c=-0.002).

r_t = [φ(t) - φ(t-1)] + (-0.002)·φ(t)

Delta's high-SNR progress signal plus a small negative continuous base
reward proportional to φ.  The negative term mildly penalizes staying
at high φ, encouraging slightly more exploration than pure Delta.

Coefficient c=-0.002 is the smaller of two negative sweeps {-0.005, -0.002}.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverDeltaDenseNeg002Experiment(RolloverBase):
    name = "rollover_delta_dense_neg002"
    reward_mode = "delta_plus_dense"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "dense_base_coef": -0.002,
    }


EXPERIMENT = RolloverDeltaDenseNeg002Experiment()
