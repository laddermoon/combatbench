"""basic_balance_step control arm: FixedSigmaGaussianMLPPolicy, OU disabled.

Same reward/truncation/eval logic as BasicBalanceStep, but uses the
FixedSigmaGaussianMLPPolicy (which supports OU exploration) with
``noise_scale=0.0`` — i.e. OU is disabled and the policy behaves
identically to the baseline TanhGaussianMLPPolicy.

This is the control arm of the OU-vs-white-noise A/B comparison.
Using the same policy family as the OU arm (but with OU disabled)
isolates the effect of temporal correlation from any difference
between FixedSigmaGaussianMLPPolicy and TanhGaussianMLPPolicy.
"""
from __future__ import annotations

from .exp_basic_balance_step import BasicBalanceStep


class BasicBalanceStepNoiseCtrl(BasicBalanceStep):

    name = "basic_balance_step_ctrl"
    actor_blueprint = "init_policy_fixed_sigma_gaussian.yaml"

    # OU disabled — white noise, identical to baseline behavior.
    noise_tau_steps: float = 0.0
    noise_scale: float = 0.0


EXPERIMENT_CLASS = BasicBalanceStepNoiseCtrl
