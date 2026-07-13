"""Rollover ablation — Delta + continuous dense base (c=0.02).

r_t = [φ(t) - φ(t-1)] + 0.02·φ(t)

Delta's high-SNR progress signal plus a continuous base reward
proportional to φ.  The continuous term provides constant marginal pull
toward higher φ even at the top, targeting Delta's weakness of
"satisficing at φ≈0.96" (avg_pot=0.962 < 1.0).

Coefficient c=0.02 is the largest of three sweeps {0.005, 0.01, 0.02}.
At higher c the dense component starts to dominate, potentially shifting
the advantage signal from the robust reward term toward the fragile
value bootstrap (approaching Dense's low-SNR regime).
"""
from __future__ import annotations

from baseline.humanoid21.curriculum.experiments.rollover_base import RolloverBase


class RolloverDeltaDense02Experiment(RolloverBase):
    name = "rollover_delta_dense_02"
    reward_mode = "delta_plus_dense"
    custom_config = {
        **RolloverBase.DEFAULT_CUSTOM_CONFIG,
        "dense_base_coef": 0.02,
    }


EXPERIMENT = RolloverDeltaDense02Experiment()
