"""basic_balance with LowRankGaussianMLPPolicy (policy family ②).

Same reward/truncation/eval logic as BasicBalance, but uses the
low-rank covariance Gaussian policy (rank=4) instead of the baseline
TanhGaussianMLPPolicy.
"""
from __future__ import annotations

from .exp_basic_balance import BasicBalance


class BasicBalanceLowRankGaussian(BasicBalance):

    name = "basic_balance_low_rank_gaussian"
    actor_blueprint = "init_policy_low_rank_gaussian.yaml"
    max_updates = 500


EXPERIMENT_CLASS = BasicBalanceLowRankGaussian
