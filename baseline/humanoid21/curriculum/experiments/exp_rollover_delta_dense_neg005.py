"""Rollover ablation — Delta + continuous dense base (c=-0.005).

r_t = [φ(t) - φ(t-1)] + (-0.005)·φ(t)

Delta's high-SNR progress signal plus a negative continuous base reward
proportional to φ.  The negative term penalizes staying at high φ,
encouraging continued exploration rather than exploitation of the
solved state.  This is the exploration-enhancing end of the c-sweep.

Coefficient c=-0.005 is the smaller of two negative sweeps {-0.005, -0.002}.
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverDeltaDenseNeg005Experiment(RolloverBase):
    name = "rollover_delta_dense_neg005"
    reward_mode = "delta_plus_dense"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "dense_base_coef": -0.005,
    }


EXPERIMENT = RolloverDeltaDenseNeg005Experiment()
