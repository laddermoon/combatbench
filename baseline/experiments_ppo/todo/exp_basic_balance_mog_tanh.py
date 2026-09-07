"""basic_balance with MoGTanhMLPPolicy (policy family ③).

Same reward/truncation/eval logic as BasicBalance, but uses the
mixture-of-diagonal-Gaussians policy (K=3) instead of the baseline
TanhGaussianMLPPolicy.
"""
from __future__ import annotations

from .exp_basic_balance import BasicBalance


class BasicBalanceMoGTanh(BasicBalance):

    name = "basic_balance_mog_tanh"
    actor_blueprint = "init_policy_mog_tanh.yaml"
    max_updates = 500


EXPERIMENT_CLASS = BasicBalanceMoGTanh
