"""basic_balance_step OU arm: FixedSigmaGaussianMLPPolicy with OU noise.

Same reward/truncation/eval logic as BasicBalanceStep, but uses the
FixedSigmaGaussianMLPPolicy with OU exploration enabled:
  - noise_tau_steps = 10  (≈ 0.5 seconds at 20Hz, gait half-cycle)
  - noise_scale = 0.3     (raw-space shift std, comparable to σ ≈ 0.2)

This is the treatment arm of the OU-vs-white-noise A/B comparison.
The control arm (exp_basic_balance_step_ctrl.py) uses the same policy
family with noise_scale=0.0, isolating the effect of temporal correlation.
"""
from __future__ import annotations

from .exp_basic_balance_step import BasicBalanceStep


class BasicBalanceStepOU(BasicBalanceStep):

    name = "basic_balance_step_ou"
    actor_blueprint = "init_policy_fixed_sigma_gaussian.yaml"

    # OU enabled — temporally correlated exploration noise.
    noise_tau_steps: float = 10.0
    noise_scale: float = 0.3


EXPERIMENT_CLASS = BasicBalanceStepOU
