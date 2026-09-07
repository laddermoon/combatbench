"""basic_balance with RealNVPTanhMLPPolicy (policy family ④).

Same reward/truncation/eval logic as BasicBalance, but uses the
RealNVP normalizing flow policy (4 coupling layers) instead of the
baseline TanhGaussianMLPPolicy.
"""
from __future__ import annotations

from .exp_basic_balance import BasicBalance


class BasicBalanceRealNVPTanh(BasicBalance):

    name = "basic_balance_realnvp_tanh"
    actor_blueprint = "init_policy_realnvp_tanh.yaml"
    max_updates = 500


EXPERIMENT_CLASS = BasicBalanceRealNVPTanh
