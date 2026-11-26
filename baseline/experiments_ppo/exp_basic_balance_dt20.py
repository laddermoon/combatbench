"""basic_balance @ dt=20ms — PPO control arm for the smbptt reproduction.

Identical task/reward semantics to `basic_balance` (same blueprint family,
same r_fall/r_cross channels and φ² gating), but materialized with
phy_steps_per_action=10 (dt=20ms) and max_steps=1500 — exactly the env the
smbptt BPTT reproduction trains on. Purpose: decide whether the ~80-step
real-len plateau observed under world-model imagination is a property of the
task or of the model bottleneck.
"""
from __future__ import annotations

from baseline.experiments_ppo.exp_basic_balance import BasicBalance


class BasicBalanceDT20(BasicBalance):

    name = "basic_balance_dt20"

    env_blueprint = "basic_balance_v2_phi_dual_dt20_env.yaml"
    max_steps: int = 1500
    episodes_per_update: int = 256


EXPERIMENT_CLASS = BasicBalanceDT20
