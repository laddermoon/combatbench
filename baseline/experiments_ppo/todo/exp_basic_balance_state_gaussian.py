"""basic_balance with StateGaussianMLPPolicy (policy family ①).

Same reward/truncation/eval logic as BasicBalance, but uses the
state-dependent diagonal Gaussian policy instead of the baseline
TanhGaussianMLPPolicy.
"""
from __future__ import annotations

from .exp_basic_balance import BasicBalance


class BasicBalanceStateGaussian(BasicBalance):

    name = "basic_balance_state_gaussian"
    actor_blueprint = "init_policy_state_gaussian.yaml"
    max_updates = 500


EXPERIMENT_CLASS = BasicBalanceStateGaussian
