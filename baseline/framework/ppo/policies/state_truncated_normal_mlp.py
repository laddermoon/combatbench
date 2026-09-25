"""StateTruncatedNormalPolicy — truncated normal with state-dependent σ.

Identical to :class:`TruncatedNormalPolicy` in every respect except one:
σ is a function of the observation, not a global ``nn.Parameter``.

    mean        = tanh(head_mean(trunk(obs)))            ∈ (-1, 1)
    σ(obs)      = exp(clamp(head_logstd(trunk(obs))))    > 0, per-state

The distribution is ``Normal(mean, σ(obs))`` truncated to [-1, 1] and
renormalized — same math, same sampling, same log_prob, same
explore_factor scaling, same uncertainty definition U = 1/(2×peak) as
the global-σ baseline.  All of that is inherited unchanged; this class
only overrides the σ source (``_policy_params``), the network
architecture (``__init__``), and the stats dict (adds
``sigma_state_std``).
Only the parameterization of σ differs, so A/B against
``TruncatedNormalPolicy`` isolates exactly one variable:
state-dependence of exploration width.

σ deliberately carries no business bounds — the head output is only
clamped to the same numerical safety range (±20) the baseline applies
to its global ``log_std``.  This keeps U → 1 (uniform) reachable and
keeps the comparison clean.  See DESIGN_state_truncated_normal.md.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from baseline.framework.ppo.policies.truncated_normal_mlp import (
    _DistParams,
    _LOG_STD_SAFE_MAX,
    _LOG_STD_SAFE_MIN,
    TruncatedNormalPolicy,
)

__all__ = [
    "StateTruncatedNormalPolicy",
]


class StateTruncatedNormalPolicy(TruncatedNormalPolicy):
    """Truncated normal policy on [-1, 1] with state-dependent σ.

    Head layout: ``[mean | log_std]`` — (B, 2D) split into (B, D) each.
    """

    _POLICY_CLASS = "StateTruncatedNormalPolicy"
    _EXPORTED_CLASS = "ExportedStateTruncNormPolicy"
    _EXPORT_TEMPLATE = "_export_template_state_truncnorm.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ):
        nn.Module.__init__(self)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        # Trunk shape matches TruncatedNormalPolicy's first two layers so
        # the parameter delta vs baseline is attributable only to the
        # wider head (hidden → 2·action_dim instead of → action_dim).
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Head outputs raw_mean | raw_log_std (each action_dim wide).
        self.head = nn.Linear(hidden_dim, 2 * action_dim)
        self._init_head()

        # Device is derived from parameters via the @property below, so
        # .to(device) / .cuda() / DataParallel all keep it in sync (P0-4).
        self.to(torch.device(device))

        # Per-policy RNG — same contract as the parent's __init__.
        self._gen = torch.Generator(device=self.device)
        self._last_seed: Optional[int] = None

    def _init_head(self) -> None:
        """Init so σ(obs) = e⁻¹ ≈ 0.368 everywhere at step 0.

        σ half: weights zeroed, bias = -1.0 → raw_log_std ≡ -1.0,
        matching TruncatedNormalPolicy's ``log_std`` init exactly.  This
        makes the degenerate-equivalence test (copy baseline weights,
        get bit-identical outputs) possible.

        Mean half keeps default PyTorch Linear init — same distribution
        as the baseline's final layer init.
        """
        d = self.action_dim
        with torch.no_grad():
            self.head.weight[d:, :].zero_()
            self.head.bias[d:].fill_(-1.0)

    # ------------------------------------------------------------------
    # The ONLY semantic difference: σ comes from the head, not a param
    # ------------------------------------------------------------------

    def _policy_params(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One trunk+head pass → (mean, policy_sigma), both (B, action_dim).

        ``policy_sigma`` is the *unscaled* σ — explore_factor is applied
        on top by :meth:`effective_sigma`.
        """
        out = self.head(self.trunk(obs))
        raw_mean, raw_log_std = out.split(self.action_dim, dim=-1)
        mean = torch.tanh(raw_mean)
        log_std = torch.clamp(
            raw_log_std, _LOG_STD_SAFE_MIN, _LOG_STD_SAFE_MAX,
        )
        return mean, log_std.exp()

    def _build_stats(
        self,
        uncertainty: torch.Tensor,
        params: _DistParams,
    ) -> Dict[str, float]:
        """Parent stats + ``sigma_state_std``: spread of σ across the batch.

        ~0 means the σ head is (near-)constant and the policy is
        behaving like the global-σ baseline.
        """
        stats = super()._build_stats(uncertainty, params)
        with torch.no_grad():
            stats["sigma_state_std"] = float(
                params.policy_sigma.std().item()
            )
        return stats

    def _export_extra(self) -> Dict[str, Any]:
        """Same identity metadata, with std_source = state."""
        extra = super()._export_extra()
        extra["std_source"] = "state"
        return extra
